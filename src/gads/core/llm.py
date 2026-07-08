import os
import json
import re
import hashlib
import instructor
from litellm import acompletion
from dotenv import load_dotenv
from contextvars import ContextVar
from typing import Optional, Dict, Any

# Load environment variables from .env
load_dotenv()

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-1234")

# Global context for trace propagation (project_id, task_id, workflow_id)
# Value should be a Dict[str, Any]
trace_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar("trace_context", default=None)

# Configure instructor with LiteLLM async completion
client = instructor.patch(create=acompletion, mode=instructor.Mode.JSON_SCHEMA)

def _tier_of(model: str) -> Optional[str]:
    """Reverse-lookup the capability tier of a model name (T4 for local_model)."""
    try:
        from gads.core.registry import TIER_MAPPING
        for tier, models in TIER_MAPPING.items():
            if model in models:
                return tier
    except Exception:
        pass
    return None

def _prepare_llm_kwargs(model: str, messages: list, kwargs: dict):
    """Shared setup for every LiteLLM call: provider/base_url/api_key, token
    budget, trace-metadata injection, timeouts, and local-model hardening.
    Mutates `kwargs` in place; returns the metadata dict (or None) so callers
    can relabel `completion_path` on repair/fallback sub-calls."""
    # CRITICAL: We pass custom_llm_provider="openai" to allow raw model names (like 'local_model')
    # without requiring a prefix that confuses the proxy server.
    if "custom_llm_provider" not in kwargs:
        kwargs["custom_llm_provider"] = "openai"

    if "base_url" not in kwargs:
        kwargs["base_url"] = LITELLM_BASE_URL
    if "api_key" not in kwargs:
        kwargs["api_key"] = LITELLM_MASTER_KEY
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 8192

    # Inject observability metadata from global context
    ctx = trace_context.get()
    meta = None
    if ctx:
        # Hash of the rendered messages: makes "cloud success vs local failure on
        # the same prompt" an exact join instead of fuzzy matching (plan 010).
        messages_hash = hashlib.sha256(
            json.dumps(messages, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        meta = {
            # existing_trace_id attaches the generation WITHOUT letting LiteLLM
            # upsert/rename the trace (trace_id overwrites trace attributes).
            "existing_trace_id": str(ctx.get("project_id")),
            "session_id": str(ctx.get("project_id")),
            "parent_observation_id": str(ctx.get("parent_observation_id")) if ctx.get("parent_observation_id") else None,
            "generation_name": str(ctx.get("agent_name", "agent_call")),
            "trace_user_id": str(ctx.get("user_id", "default_user")),
            # Non-reserved keys below land in the generation's metadata.
            "task_id": str(ctx.get("task_id")) if ctx.get("task_id") else None,
            "stage": ctx.get("stage"),
            "attempt": ctx.get("attempt"),
            "escalation_count": ctx.get("escalation_count"),
            "model_tier": _tier_of(model),
            "prompt_version": ctx.get("prompt_version"),
            "completion_path": "first_shot",
            "messages_hash": messages_hash,
        }
        meta = {k: v for k, v in meta.items() if v is not None}

        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {}

        kwargs["extra_body"]["metadata"] = meta

        if "extra_headers" not in kwargs:
            kwargs["extra_headers"] = {}

        kwargs["extra_headers"].update({
            "x-langfuse-trace-id": str(ctx.get("project_id")),
            "x-langfuse-session-id": str(ctx.get("project_id")),
            "x-langfuse-tags": f"agent:{ctx.get('agent_name')}"
        })

        print(f"  [LLM] Injecting Trace Metadata (Headers + Body): {ctx.get('project_id')}", flush=True)

    # Set a robust default timeout if none specified
    if "timeout" not in kwargs:
        kwargs["timeout"] = 60.0

    # LOCAL MODEL HARDENING:
    # 1. Increase repetition penalty to prevent infinite loops (like standard_deviation repetition).
    # 2. Use a smaller max_tokens for safety.
    # 3. Override timeout: local models are slow (planning tasks can take 3+ minutes).
    if model == "local_model":
        if "extra_body" not in kwargs: kwargs["extra_body"] = {}
        kwargs["extra_body"].update({
            "repetition_penalty": 1.1,
            "temperature": 0.1 # Stricter output for local models
        })
        # 8192, not 4096: the old clamp truncated legitimate long generations
        # (Coder writing a full training step, Planner JSON) mid-string —
        # finish_reason='length' → unparseable JSON → fatal in local mode.
        # Runaway rambling is bounded by repetition_penalty + the 600s timeout.
        kwargs["max_tokens"] = min(kwargs.get("max_tokens", 8192), 8192)
        kwargs["timeout"] = 600.0  # 60s default kills local planning; 192s observed in prod

    return meta


_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


async def get_code_completion(model: str, messages: list, stream_callback=None, **kwargs) -> str:
    """Plain-text completion that returns a Python code string (RAW CODE MODE).

    Small local models degenerate when forced to escape an entire script inside
    a JSON string field — observed: gemma-4-12b burning the full 8K budget on a
    single derailed CoderOutput generation, twice in one run. Asking for fenced
    plain code removes the escaping pressure entirely; the caller wraps the
    result into its schema programmatically.
    """
    meta = _prepare_llm_kwargs(model, messages, kwargs)
    if meta is not None:
        meta["completion_path"] = "raw_code"

    # Accumulate BOTH channels: LM Studio routes gemma-style thinking into
    # reasoning_content, and depending on the generation the entire output —
    # code included — can land there while `content` stays empty. The
    # structured path already reads both; this must too.
    full_content = ""
    full_reasoning = ""
    if stream_callback:
        resp = await acompletion(model=model, messages=messages, stream=True, **kwargs)
        async for chunk in resp:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", "") or ""
            reasoning = ""
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning = delta.reasoning_content
            elif hasattr(delta, "provider_specific_fields") and delta.provider_specific_fields:
                reasoning = delta.provider_specific_fields.get("reasoning_content", "") or ""
            full_content += content
            full_reasoning += reasoning
            active = content if content else reasoning
            if active:
                await stream_callback(active)
    else:
        resp = await acompletion(model=model, messages=messages, **kwargs)
        msg = resp.choices[0].message
        full_content = msg.content or ""
        full_reasoning = getattr(msg, "reasoning_content", "") or ""
        if not full_reasoning:
            psf = getattr(msg, "provider_specific_fields", None) or {}
            full_reasoning = psf.get("reasoning_content", "") or ""

    def _fenced_code(text: str) -> Optional[str]:
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        fenced = _CODE_FENCE_RE.findall(cleaned)
        if fenced:
            # Largest fenced block is the program; smaller ones are usually examples.
            return max(fenced, key=len).strip() or None
        # Unclosed fence (finish before the closing ```): take everything after
        # the last opening fence.
        m = list(re.finditer(r"```(?:python|py)?\s*\n", cleaned))
        if m:
            return cleaned[m[-1].end():].strip() or None
        return None

    # Prefer whichever channel actually contains a fence — content first, then
    # reasoning (the model sometimes puts its whole answer, code included, in
    # the reasoning channel while content stays empty or prose-only).
    for candidate in (full_content, full_reasoning):
        code = _fenced_code(candidate)
        if code:
            return code
    # No fence anywhere: the model followed "code only" literally. Use the
    # longer channel as-is (the executor's sanitizer strips stray fence lines).
    unfenced = max(
        (re.sub(r'<think>.*?</think>', '', t, flags=re.DOTALL).strip() for t in (full_content, full_reasoning)),
        key=len
    )
    if unfenced:
        return unfenced
    # Never return an empty program — empty code "succeeds" in the sandbox and
    # fails only at the metrics probe, a far more confusing failure than an
    # explicit error here.
    raise ValueError(
        f"Raw code mode: model returned no usable code "
        f"(content={len(full_content)} chars, reasoning={len(full_reasoning)} chars)."
    )


async def get_structured_completion(model: str, response_model, messages: list, stream_callback=None, **kwargs):
    """
    Wrapper around litellm to get validated Pydantic objects.
    Supports streaming reasoning tokens to a callback, stripping <think> tags,
    and automatic repair via `instructor` if manual extraction fails.
    Injects Langfuse/LiteLLM metadata from trace_context for observability.
    """
    meta = _prepare_llm_kwargs(model, messages, kwargs)

    def _mark_path(path: str):
        # Label repair/fallback sub-calls so first-shot and repaired completions
        # are distinguishable in the trace. `meta` is the same dict referenced by
        # kwargs["extra_body"]; LiteLLM serializes it at call time.
        if meta is not None:
            meta["completion_path"] = path

    if stream_callback:
        print(f"  [LLM] Streaming enabled for {model}...", flush=True)
        try:
            resp = await acompletion(
                model=model,
                messages=messages,
                stream=True,
                **kwargs
            )
            
            full_content = ""
            full_reasoning = ""
            
            async for chunk in resp:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", "") or ""
                
                reasoning = ""
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning = delta.reasoning_content
                elif hasattr(delta, "provider_specific_fields") and delta.provider_specific_fields:
                    reasoning = delta.provider_specific_fields.get("reasoning_content", "")
                
                full_content += content
                full_reasoning += reasoning
                
                active_token = reasoning if reasoning else content
                if active_token:
                    await stream_callback(active_token)
            
            cleaned_text = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL)
            
            # Find all JSON-like blocks and try them from largest to smallest
            json_blocks = re.findall(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}', cleaned_text, re.DOTALL)
            if not json_blocks:
                # Fallback to simple greedy regex
                json_match = re.search(r'(\{.*\})', cleaned_text, re.DOTALL)
                json_blocks = [json_match.group(0)] if json_match else []

            for block in sorted(json_blocks, key=len, reverse=True):
                try:
                    data = json.loads(block)
                    return response_model(**data)
                except Exception:
                    continue

            # Partial JSON recovery: scan backwards for last complete key-value pair
            m_start = re.search(r'\{', cleaned_text)
            if m_start:
                raw_partial = cleaned_text[m_start.start():]
                for i in range(len(raw_partial) - 1, max(0, len(raw_partial) - 512), -1):
                    if raw_partial[i] in (',', '{'):
                        try:
                            data = json.loads(raw_partial[:i] + '}')
                            result = response_model(**data)
                            print(f"  [LLM] Streaming partial JSON recovery succeeded.", flush=True)
                            return result
                        except Exception:
                            continue

            print(f"  [LLM] No valid JSON found in {len(json_blocks)} candidate blocks. Attempting Instructor repair.", flush=True)
            
            repair_messages = list(messages)
            repair_messages.append({"role": "assistant", "content": full_content})
            repair_messages.append({"role": "user", "content": "Your previous response was malformed. Please return ONLY a valid JSON object matching the required schema. Ensure you do not include any conversational text or markdown formatting."})

            try:
                _mark_path("stream_repair")
                return await client(
                    model=model,
                    response_model=response_model,
                    messages=repair_messages,
                    max_retries=1,
                    **kwargs
                )
            except Exception as e:
                # Add raw completion to the error for visibility in the Task log
                raise ValueError(f"JSON Parsing Error: {str(e)}\nRaw Output: {full_content[:500]}...")
            
        except Exception as e:
            print(f"  [LLM] Streaming failed: {e}", flush=True)
            raise e
            
    try:
        return await client(
            model=model,
            response_model=response_model,
            messages=messages,
            max_retries=0, 
            **kwargs
        )
    except Exception as e:
        print(f"  [LLM] Instructor parsing failed. Attempting robust manual extraction for {model}...", flush=True)

        # Partial JSON recovery: when max_tokens truncates mid-field, scan backwards for
        # the last complete key-value pair and close the JSON object there.
        try:
            cause = getattr(e, '__cause__', None) or getattr(e, '__context__', None)
            last_comp = getattr(cause, 'last_completion', None) or getattr(e, 'last_completion', None)
            if last_comp:
                raw = getattr(last_comp.choices[0].message, 'content', '') or ''
                raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
                m = re.search(r'\{', raw)
                if m:
                    raw = raw[m.start():]
                    for i in range(len(raw) - 1, max(0, len(raw) - 512), -1):
                        if raw[i] in (',', '{'):
                            try:
                                data = json.loads(raw[:i] + '}')
                                result = response_model(**data)
                                print(f"  [LLM] Partial JSON recovery succeeded.", flush=True)
                                return result
                            except Exception:
                                continue
        except Exception as recovery_ex:
            print(f"  [LLM] Partial JSON recovery failed: {recovery_ex}", flush=True)

        fallback_messages = list(messages)
        schema_str = json.dumps(response_model.model_json_schema())
        fallback_messages.append({
            "role": "user",
            "content": f"CRITICAL INSTRUCTION: Return ONLY a raw JSON object conforming EXACTLY to this schema:\n{schema_str}\n\nDo NOT include any <think> tags, reasoning blocks, or markdown formatting."
        })

        try:
            _mark_path("manual_fallback")
            raw_resp = await acompletion(
                model=model,
                messages=fallback_messages,
                **kwargs
            )
            
            msg = raw_resp.choices[0].message
            content = getattr(msg, "content", "") or ""
            reasoning = ""
            
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                reasoning = msg.reasoning_content
            elif hasattr(msg, "provider_specific_fields") and msg.provider_specific_fields:
                reasoning = msg.provider_specific_fields.get("reasoning_content", "")
                
            text = content if content.strip() else reasoning
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)
                
            data = json.loads(text)
            return response_model(**data)
        except Exception as fallback_error:
            print(f"  [LLM] Fallback extraction failed: {fallback_error}", flush=True)
            raise e
