"""
GADS Distillation Capture — every Coder generation, and every accepted task, on disk.

Always-on collection for local-model distillation (approach_docs/031 §3, 035). Two
append-only JSONL streams under a gitignored directory, mirroring the error ledger's
"it happened / it was resolved" shape, because acceptance is not known where the attempt
happens (the executor generates and runs; the server validates the contract later):

  attempts-YYYY-MM.jsonl  one row per Coder generation (executor.run_task): the rendered
                          prompt CORE, the raw model output, the post-sanitizer code that
                          actually ran, and what happened to it.
  accepts-YYYY-MM.jsonl   one row per task the workflow ACCEPTED (ExecutionHub.complete_task):
                          the final model_used (fallback prefixes included), the executed
                          code, and the run context needed for labels and splits.
  calls-YYYY-MM.jsonl     one row per call of every OTHER model stage (BaseAgent.run: Router,
                          SpecDrafter, Planner, PlanCritique, Synthesizer, Critique, ...): the
                          exact messages and the structured output. Stages differ in how
                          good an automatic label is. The builder only turns stages with a
                          trustworthy label into training data (the Router, against the
                          spec's declared taxonomy and recipe), but capture is cheap and
                          labels can be defined later, so every stage is kept.

scripts/build_distill_dataset.py joins them (accepted attempt = the attempt whose executed
code matches an accept row) into a training snapshot.

Why the prompt CORE and not the rendered messages: the Coder renders one core into two
formats. Cloud models get a JSON-envelope instruction; `local_model` gets raw-code mode
(coder.render_messages). A student must be trained on the format it is SERVED, so the
builder re-renders every example, cloud ones included, into the local format with the
Coder's own renderer. Storing the core makes that exact rather than a string surgery.

Payloads contain prompts, which carry data profiles (column names, value ranges, low-
cardinality category values). They are derived data about the user's datasets, which is why
the directory is gitignored (research/finetune/).

Best-effort by contract: every write is wrapped. Capture failing must never fail a task.
Disable with GADS_DISTILL_CAPTURE=false.
"""
import hashlib
import json
import os
import time
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

# Extra capture context for callers outside a workflow (e.g. scripts/eval_routing.py sets
# the spec). Deliberately separate from llm.trace_context: setting that without a project
# would stamp project_id=None onto the call's trace metadata.
capture_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar("capture_context", default=None)
_SPEC_CACHE: Dict[str, Optional[str]] = {}

CAPTURE_DIR = os.environ.get("GADS_DISTILL_CAPTURE_DIR", "research/finetune/capture")


def enabled() -> bool:
    return os.environ.get("GADS_DISTILL_CAPTURE", "true").lower() != "false"


def code_sha(code: Optional[str]) -> str:
    return hashlib.sha256((code or "").strip().encode()).hexdigest()[:16]


def _append(stream: str, record: Dict[str, Any]) -> None:
    if not enabled():
        return
    try:
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        path = os.path.join(CAPTURE_DIR, f"{stream}-{time.strftime('%Y-%m')}.jsonl")
        record = {"ts": time.time(), **record}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:  # never let collection break execution
        print(f"  [DistillCapture] Warning: could not write {stream}: {e}", flush=True)


def _spec_for(project_id: Optional[str]) -> Optional[str]:
    """The spec a project was launched from (cached; None for ad-hoc projects)."""
    if not project_id:
        return None
    if project_id not in _SPEC_CACHE:
        spec = None
        try:
            import uuid
            from sqlmodel import Session
            from gads.core.database import engine
            from gads.core.models import Project
            with Session(engine) as s:
                p = s.get(Project, uuid.UUID(str(project_id)))
                spec = ((p.last_state_json or {}).get("spec_filename")) if p else None
        except Exception:
            spec = None
        _SPEC_CACHE[project_id] = spec
    return _SPEC_CACHE[project_id]


def _context() -> Dict[str, Any]:
    try:
        from gads.core.llm import trace_context
        ctx = trace_context.get() or {}
    except Exception:
        ctx = {}
    out = {k: ctx.get(k) for k in ("project_id", "task_id", "attempt", "escalation_count",
                                   "prompt_version", "engine_id", "recipe_id", "stage")}
    extra = capture_context.get() or {}
    out["spec"] = extra.get("spec") or _spec_for(out.get("project_id"))
    out["source"] = extra.get("source", "workflow")
    return out


def record_attempt(*, model: str, render: Optional[Dict[str, str]], raw_output: Optional[str],
                   executed_code: Optional[str], outcome: str, error: Optional[str] = None,
                   task_id: Optional[str] = None, task_description: Optional[str] = None,
                   recipe_id: Optional[str] = None) -> None:
    """One Coder generation. `outcome`: executed | exec_error | state_drift | no_program.

    `executed` means the executor accepted it (ran clean, no drift). The server may still
    reject it on the contract, so the label comes from the accepts stream, not from here.
    """
    try:
        ctx = _context()
        _append("attempts", {
            **ctx,
            "task_id": task_id or ctx.get("task_id"),
            "recipe_id": recipe_id or ctx.get("recipe_id"),
            "model": model,
            "task_description": (task_description or "")[:300],
            "system": (render or {}).get("system"),
            "user_core": (render or {}).get("user_core"),
            # The skills block exactly as substituted into `system`, so a build can render
            # the same example WITHOUT skills (the skills-into-weights test, 035 §7).
            "skills_context": (render or {}).get("skills_context"),
            "raw_output": raw_output,
            "executed_code": executed_code,
            "executed_sha": code_sha(executed_code) if executed_code else None,
            "outcome": outcome,
            "error": (error or "")[:1000] or None,
        })
    except Exception as e:
        print(f"  [DistillCapture] Warning: attempt not recorded: {e}", flush=True)


def record_call(*, stage: str, model: str, messages: List[Dict[str, Any]],
                output: Optional[Dict[str, Any]], error: Optional[str] = None) -> None:
    """One call of a non-Coder model stage (BaseAgent.run), successful or not."""
    try:
        _append("calls", {
            **_context(),
            "stage_name": stage,
            "model": model,
            "messages": messages,
            "output": output,
            "error": (error or "")[:1000] or None,
        })
    except Exception as e:
        print(f"  [DistillCapture] Warning: call not recorded: {e}", flush=True)


def record_accept(task: Any, result: Dict[str, Any], project: Any = None) -> None:
    """A task the workflow accepted. Called from ExecutionHub.complete_task for code tasks."""
    try:
        code = result.get("code")
        if not (code or "").strip():
            return
        pc = task.postcondition_json or {}
        meta = (getattr(project, "last_state_json", None) or {}) if project is not None else {}
        routing_mode = run_mode = None
        try:
            from gads.core import registry
            routing_mode = getattr(registry, "get_routing_mode", lambda: None)()
            run_mode = getattr(registry, "get_run_mode", lambda: None)()
        except Exception:
            pass
        ctx = _context()
        _append("accepts", {
            "project_id": str(task.project_id),
            "task_id": str(task.id),
            "instruction_id": str(task.instruction_id) if getattr(task, "instruction_id", None) else None,
            "assigned_to": task.assigned_to,
            "model_used": result.get("model_used"),
            "mode": result.get("mode", "workflow"),
            "executed_sha": code_sha(code),
            "executed_code": code,
            "spec": meta.get("spec_filename"),
            "dial": meta.get("dial"),
            "recipe_id": ctx.get("recipe_id"),
            "recipe_node_id": pc.get("recipe_node_id"),
            "required_variables": pc.get("required_variables") or [],
            "required_metrics": pc.get("required_metrics") or [],
            "prompt_version": ctx.get("prompt_version"),
            "engine_id": ctx.get("engine_id") or meta.get("engine_id"),
            "routing_mode": routing_mode,
            "run_mode": run_mode,
        })
    except Exception as e:
        print(f"  [DistillCapture] Warning: accept not recorded: {e}", flush=True)
