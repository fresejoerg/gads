"""
GADS Kernel State — inspect and rehydrate a project's sandbox kernel.

A project's IPython session holds the state a run built up (fitted models, splits, feature
matrices). That state does NOT survive the run: the session is reset at the start of every
workflow invocation, and the stale-session sweep resets any project with no active tasks.
So a completed project is, by default, cold — which makes follow-up work on a finished
analysis impossible without redoing the whole thing.

This module restores it, cheapest path first:

  1. **Probe** the live session. If the expected variables are already there, do nothing.
  2. **Replay** otherwise: re-execute the project's successful task code, in order, in one
     sandbox call. This is the same code `notebook_exporter.export_python_script` writes to
     `workflow_execution.py` — filtered to COMPLETED tasks only (a failed task's code is
     useful for an audit export, but must never be replayed into a live kernel).
  3. **Report** the resulting namespace so callers (and the Coder) know what is available.

Honesty about what this is: replayed state is *reconstructed*, not *restored*. Recipes fix
random seeds, but a nondeterministic step or a mutated source dataset can rebuild something
subtly different — callers label it `replayed`, never `restored`.

Fail-soft by contract: rehydration is an optimization. Any failure returns a `cold` result
so the caller proceeds with an empty kernel (and tells the Coder to reload from workspace
files) rather than blocking the user's request. See approach_docs/020.
"""
import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from gads.core.database import engine
from gads.core.models import Task

# Agents that never produce replayable analysis code.
_SYSTEM_AGENTS = {
    "System", "Router", "Planner", "SpecDrafter",
    "Synthesizer", "Critique", "PlanCritique", "Auditor",
    "CompletenessVerifier", "DataAnalyzer", "DataSampler",
}

# Namespace probe. Mirrors the executor's live-state snapshot so the shape callers see here
# is identical to the one the Coder is shown mid-run.
SNAPSHOT_CODE = """
import json as _json_ks
_summary_ks = {}
for _v_ks in dir():
    if _v_ks.startswith('_'): continue
    try:
        _o_ks = globals()[_v_ks]
        import pandas as _pd_ks, numpy as _np_ks
        if isinstance(_o_ks, _pd_ks.DataFrame):
            _summary_ks[_v_ks] = f"DataFrame ({_o_ks.shape[0]}x{_o_ks.shape[1]}) - Columns: {list(_o_ks.columns)}"
        elif isinstance(_o_ks, (list, dict, _np_ks.ndarray)):
            _summary_ks[_v_ks] = f"{type(_o_ks).__name__} (len: {len(_o_ks)})"
        elif isinstance(_o_ks, (str, int, float, bool)):
            _summary_ks[_v_ks] = _o_ks
        elif hasattr(_o_ks, 'predict') and hasattr(_o_ks, 'fit'):
            _summary_ks[_v_ks] = f"Model ({type(_o_ks).__name__})"
        else:
            _summary_ks[_v_ks] = f"{type(_o_ks).__module__}.{type(_o_ks).__name__}"
    except Exception: pass
print("GADS_STATE_SNAPSHOT:" + _json_ks.dumps(_summary_ks))
"""

# Emit shim: replayed task code calls gads_emit_insight(), which normally comes from the
# executor's telemetry preamble. During replay we only want the side effects (fitted state),
# not the insights — they were already captured and reported by the original run.
_EMIT_SHIM = """
if '_gads_insights' not in globals(): _gads_insights = []
def gads_emit_insight(artifact, insight, evidence=""):
    _gads_insights.append({"artifact": artifact, "insight": insight, "evidence": evidence})
"""


# Names every fresh sandbox session already has (IPython builtins + the container's
# bootstrap imports). They are NOT analysis state: counting them as "live" would make a
# cold kernel look rehydrated and silently skip the replay it needs.
_BOOTSTRAP_NAMES = {
    "In", "Out", "exit", "quit", "get_ipython", "open", "os", "sys", "json",
    "np", "pd", "pl", "plt", "px", "pio", "sns", "matplotlib", "warnings",
}


def meaningful_variables(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Snapshot entries that represent actual analysis state.

    Drops session bootstrap names and bare module objects, so callers can distinguish "the
    kernel holds this run's data" from "the kernel is freshly booted".
    """
    if not snapshot:
        return {}
    return {
        k: v for k, v in snapshot.items()
        if k not in _BOOTSTRAP_NAMES and str(v) != "builtins.module"
    }


def _parse_snapshot(stdout: str) -> Optional[Dict[str, Any]]:
    if "GADS_STATE_SNAPSHOT:" not in (stdout or ""):
        return None
    try:
        raw = stdout.split("GADS_STATE_SNAPSHOT:")[1].strip().split("\n")[0]
        out = json.loads(raw)
        return out if isinstance(out, dict) else None
    except Exception:
        return None


async def snapshot_kernel(sandbox, project_id: uuid.UUID, session_id: Optional[str] = None,
                          timeout: float = 15.0) -> Optional[Dict[str, Any]]:
    """Return {var_name: description} for the live session, or None if unavailable."""
    session_id = session_id or str(project_id)
    try:
        res = await asyncio.wait_for(
            sandbox.execute(SNAPSHOT_CODE, project_id=project_id, session_id=session_id),
            timeout=timeout)
        return _parse_snapshot(res.stdout)
    except Exception as e:
        print(f"  [KernelState] Snapshot failed: {e}", flush=True)
        return None


def replayable_tasks(project_id: uuid.UUID) -> List[Task]:
    """COMPLETED, non-system tasks that carry code, oldest first.

    Deliberately stricter than `notebook_exporter._get_execution_tasks`, which also includes
    failed tasks (correct for an audit export, wrong for rebuilding live state). Tasks
    satisfied by a prior attempt's resume are skipped — their effect is already represented
    by the attempt that actually ran the code.
    """
    with Session(engine) as session:
        tasks = session.exec(
            select(Task).where(Task.project_id == project_id).order_by(Task.created_at)
        ).all()
        out = []
        for t in tasks:
            rj = t.result_json or {}
            if (t.status == "completed"
                    and t.assigned_to not in _SYSTEM_AGENTS
                    and not rj.get("resumed_from_prior_attempt")
                    and (rj.get("code") or "").strip()):
                out.append(t)
        return out


def replay_code_from_workspace(workspace_dir: str) -> Optional[str]:
    """Recover replayable code from the workspace's `workflow_execution.py`.

    Fallback for when the DB task rows are unavailable but the workspace survives — not
    hypothetical: the GADS tables live in the LiteLLM-owned Postgres database and have been
    dropped wholesale by its migrations, taking every project/task row with them while the
    workspaces stayed intact. Since "load any project from the archive" must survive that,
    the on-disk export is treated as a first-class source.

    Parses the export's task blocks and keeps only those marked `Status: completed`; drops
    the file's own standalone shim (this module injects its own).
    """
    import os
    import re
    path = os.path.join(workspace_dir or "", "workflow_execution.py")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return None

    blocks, keep = re.split(r"^# ={60,}$", text, flags=re.M), []
    # Blocks alternate: [preamble][header][code][header][code]... — a header block carries
    # "# TASK n:" and the status line; the code follows in the next segment.
    for i, seg in enumerate(blocks):
        if "# TASK " in seg and "Status:" in seg:
            status = re.search(r"Status:\s*(\w+)", seg)
            if status and status.group(1).strip().lower() == "completed" and i + 1 < len(blocks):
                code = blocks[i + 1].strip()
                if code:
                    keep.append(code)
    if not keep:
        return None
    print(f"  [KernelState] Recovered {len(keep)} completed task block(s) from "
          f"workflow_execution.py (DB rows unavailable).", flush=True)
    return "\n\n".join(keep)


def build_replay_code(project_id: uuid.UUID,
                      workspace_dir: Optional[str] = None) -> Optional[str]:
    """Assemble the replay script for a project, or None if there is nothing to replay.

    Prefers DB task rows (authoritative, status-filtered); falls back to the workspace
    export so an archived project stays replayable even if its rows are gone.
    """
    tasks = replayable_tasks(project_id)
    if tasks:
        body = "\n\n".join(
            f"# --- replay {i}/{len(tasks)}: {(t.description or '')[:70]} ---\n"
            + (t.result_json or {}).get("code", "").strip()
            for i, t in enumerate(tasks, 1)
        )
    else:
        body = replay_code_from_workspace(workspace_dir) if workspace_dir else None
        if not body:
            return None
    # Native definitions the replayed code references (same routing the executor uses).
    native = ""
    try:
        from gads.knowledge.native import preamble_for_code
        native, names = preamble_for_code(body)
        if names:
            print(f"  [KernelState] Replay needs native preamble(s): {', '.join(names)}", flush=True)
    except Exception as e:
        print(f"  [KernelState] Warning: native preamble unavailable for replay: {e}", flush=True)

    # Archived code can reference natives that are no longer in any always-on preamble —
    # e.g. a node later DEMOTED to a fallback-only helper (approach_docs/019). The knowledge
    # base moves; the archive must still replay. Append the source of any registered native
    # the code calls but the routed preamble does not define.
    try:
        from gads.knowledge.native import NATIVE_SOURCE
        extra = [src for name, src in NATIVE_SOURCE.items()
                 if name in body and f"def {name}" not in native]
        if extra:
            missing = [n for n in NATIVE_SOURCE if n in body and f"def {n}" not in native]
            print(f"  [KernelState] Adding {len(extra)} non-preamble native(s) for replay: "
                  f"{', '.join(missing)}", flush=True)
            native += "\n" + "\n\n".join(extra) + "\n"
    except Exception as e:
        print(f"  [KernelState] Warning: could not resolve extra natives: {e}", flush=True)
    return native + _EMIT_SHIM + "\n" + body


async def ensure_kernel_state(sandbox, project_id: uuid.UUID, session_id: Optional[str] = None,
                              required_vars: Optional[List[str]] = None,
                              force: bool = False, timeout: float = 900.0,
                              workspace_dir: Optional[str] = None) -> Dict[str, Any]:
    """Make a project's kernel usable, replaying prior task code only if needed.

    Returns a dict:
      status       'live'     — session already held the state (nothing executed)
                   'replayed' — prior task code was re-executed to rebuild it
                   'cold'     — replay failed or errored; kernel is unusable/empty
                   'empty'    — nothing to replay (no completed tasks with code)
      variables    {name: description} snapshot after the operation
      replayed_tasks / seconds / error as applicable

    Never raises: a rehydration failure must not block the caller's request.
    """
    session_id = session_id or str(project_id)
    result: Dict[str, Any] = {"status": "cold", "variables": {}, "replayed_tasks": 0,
                              "seconds": 0.0, "project_id": str(project_id)}

    # 1. Is it already live?
    live = await snapshot_kernel(sandbox, project_id, session_id)
    if live is not None and not force:
        state = meaningful_variables(live)   # ignore bootstrap imports — see above
        satisfied = set(required_vars or []) <= set(state) if required_vars else bool(state)
        if satisfied:
            result.update(status="live", variables=state)
            print(f"  [KernelState] Session already holds state ({len(state)} variables) — no replay.",
                  flush=True)
            return result

    # 2. Replay.
    code = build_replay_code(project_id, workspace_dir=workspace_dir)
    if not code:
        result.update(status="empty", variables=meaningful_variables(live))
        print("  [KernelState] Nothing to replay (no completed tasks with code).", flush=True)
        return result

    n_tasks = len(replayable_tasks(project_id)) or code.count("# --- replay ") or 1
    print(f"  [KernelState] ⟳ Replaying {n_tasks} completed task(s) to rebuild kernel state — "
          f"this re-runs the original computation and may take a while.", flush=True)
    t0 = time.time()
    try:
        res = await asyncio.wait_for(
            sandbox.execute(code, project_id=project_id, session_id=session_id), timeout=timeout)
    except Exception as e:
        result.update(status="cold", error=f"{type(e).__name__}: {e}",
                      seconds=round(time.time() - t0, 1))
        print(f"  [KernelState] ⚠ Replay failed ({e}) — continuing with a cold kernel.", flush=True)
        return result
    elapsed = round(time.time() - t0, 1)

    if res.error:
        # Partial replay still leaves useful state; report what survived rather than nothing.
        after = meaningful_variables(await snapshot_kernel(sandbox, project_id, session_id))
        result.update(status="cold" if not after else "replayed", variables=after,
                      replayed_tasks=n_tasks, seconds=elapsed,
                      error=f"{res.error.get('ename')}: {str(res.error.get('evalue'))[:200]}")
        print(f"  [KernelState] ⚠ Replay errored after {elapsed}s "
              f"({result['error'][:100]}); {len(after)} variable(s) present.", flush=True)
        return result

    after = meaningful_variables(await snapshot_kernel(sandbox, project_id, session_id))
    result.update(status="replayed" if after else "cold", variables=after,
                  replayed_tasks=n_tasks, seconds=elapsed)
    print(f"  [KernelState] ✓ Replayed {n_tasks} task(s) in {elapsed}s — "
          f"{len(after)} variable(s) live.", flush=True)
    return result
