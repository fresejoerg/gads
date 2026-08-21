"""Offline check of research vs production run mode (executor native-first short-circuit).

research   — the model attempts every node; a native is only a post-exhaustion safety net,
             which is what keeps pass@model meaningful.
production — a node declaring a native uses it directly and the model is never asked.

    PYTHONPATH=src uv run python scripts/test_run_mode.py
"""
import asyncio
import sys
import uuid

sys.path.insert(0, "src")

from gads.core.executor import ExecutionManager
from gads.tools.sandbox import ExecutionResult


def manager():
    m = ExecutionManager.__new__(ExecutionManager)
    m.authoritative_state = {}
    m.protected_state = {}
    m.file_schemas = {}
    m.coder = type("C", (), {"model": "local_model", "model_str": "local_model"})()
    m.calls = {"native": 0, "coder": 0}

    async def fake_native(name, call, project_id, session_id):
        m.calls["native"] += 1
        return m._native_returns

    m._run_native_fallback = fake_native
    return m


OK = ExecutionResult(stdout="native ran", stderr="", error=None,
                     execution_time_ms=10, kernel_state={})


async def main():
    # 1. production + native declared → native runs, model never asked
    m = manager()
    m._native_returns = OK
    res, tag = await m.run_task(
        "some node", project_id=uuid.uuid4(), session_id="s",
        fallback_native="gads_evaluate_holdout", fallback_call="_e = gads_evaluate_holdout()",
        run_mode="production", max_attempts=0)
    assert tag == "native_primary:gads_evaluate_holdout", tag
    assert m.calls["native"] == 1 and m.calls["coder"] == 0
    print("production + native      ->", tag, "(model not asked)")

    # 2. production + NO native declared → must fall through to the model path.
    #    max_attempts=0 makes the retry loop a no-op, so reaching the end proves the
    #    short-circuit did not fire.
    m = manager()
    m._native_returns = OK
    res, tag = await m.run_task(
        "judgment node", project_id=uuid.uuid4(), session_id="s",
        run_mode="production", max_attempts=0)
    assert not str(tag).startswith("native_primary"), tag
    assert m.calls["native"] == 0
    print("production, no native    -> model path (judgment nodes stay model-generated)")

    # 3. research + native declared → native must NOT pre-empt the model
    m = manager()
    m._native_returns = OK
    res, tag = await m.run_task(
        "some node", project_id=uuid.uuid4(), session_id="s",
        fallback_native="gads_evaluate_holdout", fallback_call="_e = gads_evaluate_holdout()",
        run_mode="research", fallback_mode="none", max_attempts=0)
    assert not str(tag).startswith("native_primary"), tag
    assert m.calls["native"] == 0, "research mode must never run the native first"
    print("research + native        -> model-first preserved, native untouched")

    # 4. production where the native itself fails → fall through to the model, not a dead node
    m = manager()
    m._native_returns = None
    res, tag = await m.run_task(
        "some node", project_id=uuid.uuid4(), session_id="s",
        fallback_native="gads_evaluate_holdout", fallback_call="_e = gads_evaluate_holdout()",
        run_mode="production", max_attempts=0)
    assert not str(tag).startswith("native_primary"), tag
    assert m.calls["native"] == 1
    print("production, native fails -> falls through to the model")

    # 5. production + native + model_required → the model MUST still be asked.
    #    This is the third category: nodes whose deliverable is the reasoning itself
    #    (the defended shortlist, the model card). The native stays a rescue, never
    #    the primary.
    m = manager()
    m._native_returns = OK
    res, tag = await m.run_task(
        "THE REASONING STEP", project_id=uuid.uuid4(), session_id="s",
        fallback_native="gads_default_shortlist", fallback_call="_c = gads_default_shortlist()",
        run_mode="production", model_required=True, max_attempts=0)
    assert not str(tag).startswith("native_primary"), tag
    assert m.calls["native"] == 0, "model_required must forbid native-first substitution"
    print("production + model_required -> model still asked (native not substituted)")

    # 6. research mode ignores the flag — it is already model-first everywhere
    m = manager()
    m._native_returns = OK
    res, tag = await m.run_task(
        "THE REASONING STEP", project_id=uuid.uuid4(), session_id="s",
        fallback_native="gads_default_shortlist", fallback_call="_c = gads_default_shortlist()",
        run_mode="research", model_required=True, max_attempts=0)
    assert m.calls["native"] == 0
    print("research + model_required   -> unchanged (already model-first)")

    print("\nALL CHECKS PASSED")


asyncio.run(main())
