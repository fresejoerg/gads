import os
import json
import re
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

async def get_structured_completion(model: str, response_model, messages: list, stream_callback=None, **kwargs):
    """
    Wrapper around litellm to get validated Pydantic objects.
    Supports streaming reasoning tokens to a callback, stripping <think> tags,
    and automatic repair via `instructor` if manual extraction fails.
    Injects Langfuse/LiteLLM metadata from trace_context for observability.
    """
    if "base_url" not in kwargs:
        kwargs["base_url"] = LITELLM_BASE_URL
    if "api_key" not in kwargs:
        kwargs["api_key"] = LITELLM_MASTER_KEY
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 8192

    # Inject observability metadata from global context
    ctx = trace_context.get()
    if ctx:
        # standard LiteLLM/Langfuse metadata keys
        # We create a CLEAN, independent copy to avoid any potential circularity
        meta = {
            "trace_id": str(ctx.get("project_id")),
            "session_id": str(ctx.get("project_id")),
            "parent_observation_id": str(ctx.get("parent_observation_id")) if ctx.get("parent_observation_id") else None,
            "generation_name": str(ctx.get("agent_name", "agent_call")),
            "user_id": str(ctx.get("user_id", "default_user")),
        }
        meta = {k: v for k, v in meta.items() if v is not None}
        
        # When using instructor/client (OpenAI SDK), metadata MUST be in extra_body 
        # to be forwarded by LiteLLM Proxy. Top-level kwargs are often stripped by the SDK.
        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {}
        
        # We don't put it in both places to avoid LiteLLM merging logic loops
        kwargs["extra_body"]["metadata"] = meta

        # ALSO inject via HEADERS (Most reliable for LiteLLM Proxy callbacks)
        if "extra_headers" not in kwargs:
            kwargs["extra_headers"] = {}
        
        kwargs["extra_headers"].update({
            "x-langfuse-trace-id": str(ctx.get("project_id")),
            "x-langfuse-session-id": str(ctx.get("project_id")),
            "x-langfuse-tags": f"agent:{ctx.get('agent_name')}"
        })
        
        print(f"  [LLM] Injecting Trace Metadata (Headers + Body): {ctx.get('project_id')}", flush=True)

        try:
            # Crucial: Flush the Langfuse SDK buffer before calling the LLM.
            from gads.core.server import langfuse_client
            print(f"  [LLM] Triggering Langfuse flush... (Tasks in queue: {langfuse_client.task_manager._queue.qsize()})", flush=True)
            langfuse_client.flush()
            print("  [LLM] Langfuse flush complete.", flush=True)
        except Exception as e:
            print(f"  [LLM] Langfuse flush failed/skipped: {type(e).__name__} - {e}", flush=True)

    if stream_callback:
        print(f"  [LLM] Streaming enabled for {model}...", flush=True)
        try:
            resp = await acompletion(
                model=model,
                messages=messages,
                custom_llm_provider="openai",
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
                
                # Emit whichever stream is active. (For models that put <think> inside content, we just stream content)
                active_token = reasoning if reasoning else content
                if active_token:
                    await stream_callback(active_token)
            
            # Clean up <think> tags if they ended up inside full_content
            cleaned_text = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL)
            
            # Extract JSON block
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = json.loads(json_str)
                    return response_model(**data)
                except Exception as e:
                    print(f"  [LLM] Manual JSON parse failed: {e}. Attempting Instructor repair.", flush=True)
            else:
                print(f"  [LLM] No JSON found in output. Attempting Instructor repair.", flush=True)
            
            # Fallback repair pass
            repair_messages = list(messages)
            repair_messages.append({"role": "assistant", "content": full_content})
            repair_messages.append({"role": "user", "content": "Your previous response was malformed. Please return ONLY a valid JSON object matching the required schema."})
            
            return await client(
                model=model,
                response_model=response_model,
                messages=repair_messages,
                custom_llm_provider="openai",
                max_retries=1,
                **kwargs
            )
            
        except Exception as e:
            print(f"  [LLM] Streaming failed: {e}", flush=True)
            raise e
            
    # Non-streaming path
    try:
        # Fast-fail so we can drop to our manual fallback immediately if instructor fails
        return await client(
            model=model,
            response_model=response_model,
            messages=messages,
            custom_llm_provider="openai",
            max_retries=0, 
            **kwargs
        )
    except Exception as e:
        print(f"  [LLM] Instructor parsing failed. Attempting robust manual extraction for {model}...", flush=True)
        
        fallback_messages = list(messages)
        schema_str = json.dumps(response_model.model_json_schema())
        fallback_messages.append({
            "role": "user",
            "content": f"CRITICAL INSTRUCTION: Return ONLY a raw JSON object conforming EXACTLY to this schema:\n{schema_str}\n\nDo NOT include any <think> tags, reasoning blocks, or markdown formatting."
        })
        
        try:
            raw_resp = await acompletion(
                model=model,
                messages=fallback_messages,
                custom_llm_provider="openai",
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
            
            # Clean up <think> tags before parsing
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            
            # Find the JSON block
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)
                
            data = json.loads(text)
            return response_model(**data)
        except Exception as fallback_error:
            print(f"  [LLM] Fallback extraction failed: {fallback_error}", flush=True)
            raise e
