"""
GADS Follow-Up Lane — analyst-in-the-loop instructions on a completed run.

The autonomous pipeline answers the objective it was given. The next question is a human
one, asked after reading the result: "do error analysis on the false negatives", "plot recall
against tumor grade". Routing that through `run_agent_workflow` is the wrong shape — it
re-runs SpecDrafter → Router → Planner → PlanCritique → … → Reporting on a cold kernel, and a
matched recipe recompiles and re-executes the whole DAG, redoing work the user already
accepted.

This lane is deliberately short:

    instruction -> ONE task -> (rehydrate kernel) -> Coder -> execute -> artifacts

Skipped on purpose: SpecDrafter, Router, Planner, PlanCritique, CompletenessVerifier,
Critique. **The user is the planner and the critic here** — that is what makes it interactive.

What it still inherits, because it calls the same `ExecutionManager.run_task`: adaptive
retries with accumulated error feedback, the cross-run error-ledger pitfalls prior,
keyword-routed native-node injection, the code sanitizers, the hallucination guard, and the
opt-in native/cloud fallback (approach_docs/019).

Knowledge application is graded by how much it constrains the user's request:
  * **skills** — always (keyword + semantic matching on the objective),
  * **native nodes** — automatically, via the executor's keyword routing,
  * **recipes** — opt-in and single-node only; this lane never compiles a DAG.

Measurement hygiene: follow-up tasks are user-directed, so they are tagged `mode="followup"`
in `result_json` and must never be counted toward an autonomous run's `pass@model`
(approach_docs/019, 020).
"""
import asyncio
import base64
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlmodel import Session, select

from gads.core.database import engine
from gads.core.models import Artifact, Instruction, Project, Task
from gads.core.execution_hub import ExecutionHub
from gads.core.executor import ExecutionManager
from gads.core.history_renderer import HistoryRenderer
from gads.core.introspection import summarize_artifact


def _skills_context(registry, objective: str) -> Optional[str]:
    """Skill bodies for a follow-up objective.

    A follow-up has no curated `attached_skills`, so both matchers apply: keyword triggers
    plus the semantic index (the same 'uncurated task' branch the main loop uses). The
    sandbox constraints are mandatory for every Coder call.
    """
    ids = {s.id for s, hits in registry.find_skills_scored(objective) if hits >= 2}
    for skill, score in registry.find_skills_semantic(objective):
        ids.add(skill.id)
    ids.add("sandbox_environment")
    loaded = [registry.skills[i] for i in ids if i in registry.skills]
    if not loaded:
        return None
    bodies = "\n\n".join(f"#### {s.id}\n{s.content}" for s in loaded)
    print(f"  [Follow-up] Applied skills: {[s.id for s in loaded]}", flush=True)
    return f"### LOADED SKILL GUIDANCE\n{bodies}"


def _recipe_node_hint(registry, recipe_ref: str) -> str:
    """Optional single-node recipe guidance: 'recipe_id#node_id' or 'recipe_id'.

    Deliberately injects ONE node's intent as advice — never a compiled DAG. Returns '' when
    nothing resolves, so a bad hint degrades to a plain follow-up instead of failing.
    """
    if not recipe_ref or recipe_ref in ("auto", "none"):
        return ""
    rid, _, node_id = recipe_ref.partition("#")
    recipe = registry.get_recipe(rid)
    if not recipe:
        return ""
    node = next((n for n in recipe.dag if n.id == node_id), None) if node_id else None
    if node is None:
        return ""
    return (f"\n\n### RECIPE GUIDANCE (from {rid}#{node.id})\n{node.intent}\n"
            "Apply this as guidance for the user's request; do not run the rest of the recipe.")


async def run_followup(project_id: uuid.UUID, instruction_id: uuid.UUID, task_id: uuid.UUID,
                       objective: str, rehydrate: bool = True,
                       use_recipe: str = "auto") -> Dict[str, Any]:
    """Execute one user-directed follow-up instruction against a project's live kernel."""
    # Imported lazily: server imports this module, so a top-level import would be circular.
    from gads.core import server as srv
    from gads.core.kernel_state import ensure_kernel_state, snapshot_kernel, meaningful_variables

    workspace_dir = f"{srv.WORKSPACE_ROOT}/{project_id}"
    session_id = str(project_id)
    executor = ExecutionManager()

    with Session(engine) as s:
        proj = s.get(Project, project_id)
        if proj and proj.last_state_json and "__schemas__" in proj.last_state_json:
            executor.file_schemas = proj.last_state_json["__schemas__"]

    # 1. Kernel — rehydrate so the follow-up can build on the run's fitted state.
    kernel: Dict[str, Any] = {"status": "skipped", "variables": {}}
    if rehydrate:
        srv.PINNED_SESSIONS.add(project_id)
        kernel = await ensure_kernel_state(executor.sandbox, project_id, session_id,
                                           workspace_dir=workspace_dir)
    else:
        kernel = {"status": "live", "variables":
                  meaningful_variables(await snapshot_kernel(executor.sandbox, project_id, session_id))}

    # 2. Context — same ingredients the main loop gives the Coder.
    # NOTE: the kernel snapshot goes in via `state_summary`, NOT `executor.authoritative_state`.
    # The latter holds {name: {"type","shape",...}} dicts consumed by the RuntimeOracle;
    # seeding it with snapshot strings would raise on `var_info.get(...)`. The executor's
    # "Calling ... with 0 variables" line refers to that structure and is cosmetic here.
    import json as _json
    state_summary = _json.dumps(kernel.get("variables", {}), indent=2)
    with Session(engine) as s:
        all_tasks = s.exec(select(Task).where(Task.project_id == project_id)
                           .order_by(Task.created_at.asc())).all()
        idx = next((i for i, t in enumerate(all_tasks) if t.id == task_id), len(all_tasks) - 1)
        try:
            HistoryRenderer.build_coder_context(all_tasks, idx)
        except Exception:
            pass
    skills_ctx = (_skills_context(srv.registry, objective) or "") + _recipe_node_hint(srv.registry, use_recipe)

    # 3. Streaming + heartbeat, mirroring the main execution loop.
    tid_str = str(task_id)
    srv.LIVE_STREAMS[tid_str] = {"reasoning": "", "stdout": ""}

    async def _reasoning(token: str):
        srv.LIVE_STREAMS[tid_str]["reasoning"] += token

    async def _stdout(text: str):
        srv.LIVE_STREAMS[tid_str]["stdout"] = text

    async def _heartbeat():
        while True:
            await asyncio.sleep(60)
            try:
                with Session(engine) as _s:
                    ExecutionHub(_s).heartbeat(task_id)
            except Exception:
                pass

    files_before = {f["name"] for f in srv._get_recursive_files(workspace_dir)}
    hb = asyncio.create_task(_heartbeat())
    try:
        res, model_used = await executor.run_task(
            objective,
            project_id=project_id,
            session_id=session_id,
            skills_context=skills_ctx or None,
            task_id=task_id,
            stdout_callback=_stdout,
            stream_callback=_reasoning,
            state_summary=state_summary,
        )
    finally:
        hb.cancel()
        try:
            await hb
        except (asyncio.CancelledError, Exception):
            pass

    # 4. Persist outcome + register any new artifacts against THIS instruction.
    new_files = [n for n in
                 ({f["name"] for f in srv._get_recursive_files(workspace_dir)} - files_before)
                 if n != "final_dashboard.html"]
    with Session(engine) as s:
        hub = ExecutionHub(s)
        if res.error:
            hub.fail_task(task_id, res.error.get("evalue", "Unknown error"),
                          result={"stdout": res.stdout, "code": res.code,
                                  "model_used": model_used, "mode": "followup"})
            s.commit()
            return {"status": "failed", "error": res.error.get("evalue"),
                    "kernel": kernel, "new_files": [], "model_used": model_used}

        summaries = [summarize_artifact(os.path.join(workspace_dir, n)) for n in new_files
                     if os.path.exists(os.path.join(workspace_dir, n))]
        hub.complete_task(task_id, {
            "stdout": res.stdout, "code": res.code, "model_used": model_used,
            # Excluded from pass@model: this work was user-directed, not autonomous.
            "mode": "followup", "instruction_id": str(instruction_id),
            "orchestrator_summary": "; ".join(summaries) or "Follow-up completed with no new files.",
        })

        for nf in new_files:
            full = os.path.join(workspace_dir, nf)
            try:
                if nf.lower().endswith(".png"):
                    with open(full, "rb") as fh:
                        b64 = base64.b64encode(fh.read()).decode("utf-8")
                    art = Artifact(project_id=project_id, type="plot",
                                   description=f"Follow-up artifact: {nf}",
                                   content_json={"image_base64": b64,
                                                 "instruction_id": str(instruction_id),
                                                 "followup": True},
                                   agent_id="FollowUp")
                elif nf.lower().endswith(".json") and not nf.endswith(".meta.json"):
                    from gads.core.introspection import harden_json_artifact
                    harden_json_artifact(full)
                    art = Artifact(project_id=project_id, type="json_plot",
                                   description=f"Follow-up interactive: {nf}",
                                   content_json={"filename": nf,
                                                 "instruction_id": str(instruction_id),
                                                 "followup": True},
                                   agent_id="FollowUp")
                else:
                    continue
                s.add(art)
                s.commit()
                hub.create_outbox_event("ARTIFACT_CREATED", {
                    "type": art.type, "description": art.description,
                    "content_json": art.content_json, "project_id": str(project_id)})
            except Exception:
                pass
        s.commit()

    print(f"  [Follow-up] ✓ '{objective[:60]}' — {len(new_files)} new file(s), "
          f"kernel={kernel.get('status')}", flush=True)
    return {"status": "completed", "kernel": kernel, "new_files": new_files,
            "model_used": model_used, "stdout": (res.stdout or "")[-4000:]}


async def run_followup_wrapper(project_id: uuid.UUID, instruction_id: uuid.UUID,
                               task_id: uuid.UUID, objective: str, rehydrate: bool,
                               use_recipe: str):
    """Background entrypoint: guarantees the task never sticks in 'running' and releases
    the duplicate-launch guard."""
    from gads.core import server as srv
    try:
        await run_followup(project_id, instruction_id, task_id, objective, rehydrate, use_recipe)
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            with Session(engine) as s:
                ExecutionHub(s).fail_task(task_id, f"Follow-up failed: {e}")
                s.commit()
        except Exception:
            pass
    finally:
        srv.ACTIVE_WORKFLOWS.discard(project_id)
