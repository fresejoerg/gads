"""Delegation-dial formalism (approach_docs/013): which rung of the autonomy ladder
does a project run sit on, and the run ledger that accumulates the rung × engine grid.

The ladder (higher rung = more decisions moved from the model into curated artifacts):

  D0  full delegation — no recipe, no spec hints; the model frames, plans, realizes
  D1  framed          — spec hints fix the framing (target column / features / filters)
  D2  advised         — recipe suggested but deviatable: a Router match below the
                        deterministic-compile confidence threshold is passed to the
                        Planner as advisory context (a knowledge base to draw on) rather
                        than compiled verbatim, and the Planner drafts its own plan.
                        Selection provenance (pinned|routed|advised|drafted) is recorded
                        separately.
  D3  directed        — compiled plan; methodology fixed at decision level (invariants)
  D4  patterned       — + curated skills fix the code patterns per task
  D5  mechanized      — + a native kernel function replaces the step's mechanical core

Rungs are per-task; the PROJECT rung is the minimum over its tasks — the run is only
as prescribed as its least-prescribed task (weakest link gets the most freedom).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

RUNG_ORDER = ["D0", "D1", "D2", "D3", "D4", "D5"]

# Native kernel functions that replace a whole methodology step (NOT mere helpers like
# gads_emit_insight/gads_calibrate_threshold — those assist a step the model still
# writes). A node whose intent mandates one of these is mechanized.
NATIVE_STEP_FUNCTIONS = {"gads_causal_estimate_ate"}

LEDGER_PATH = os.getenv("GADS_DIAL_LEDGER", "research/dial_ledger.jsonl")


def node_rung(node: Dict[str, Any]) -> str:
    """Rung of a single compiled recipe node."""
    intent = node.get("intent", "") or ""
    if any(fn in intent for fn in NATIVE_STEP_FUNCTIONS):
        return "D5"
    curated = [s for s in (node.get("attached_skills") or []) if s != "sandbox_environment"]
    return "D4" if curated else "D3"


def compiled_plan_dial(nodes: List[Dict[str, Any]], selection: str) -> Dict[str, Any]:
    """Dial info for a plan compiled from recipe DAG nodes.

    selection: 'pinned' (spec named the recipe) or 'routed' (Router classified).
    """
    task_rungs = {
        (n.get("id") or f"node_{i}"): node_rung(n) for i, n in enumerate(nodes)
    }
    project = min(task_rungs.values(), key=RUNG_ORDER.index) if task_rungs else "D3"
    return {"rung": project, "task_rungs": task_rungs, "selection": selection}


def advisory_plan_dial(recipe_id: str, confidence: float) -> Dict[str, Any]:
    """Dial info for a Planner-drafted plan that received a weak-matched recipe as
    advisory context (D2 — 'advised': recipe suggested but deviatable). No per-task
    `task_rungs` — the Planner drafted its own tasks, not the recipe's DAG nodes, so
    there is nothing to score node-by-node the way a compiled plan's nodes are."""
    return {
        "rung": "D2",
        "task_rungs": {},
        "selection": "advised",
        "recipe_id": recipe_id,
        "recipe_confidence": confidence,
    }


def drafted_plan_dial(spec_hints: Dict[str, Any]) -> Dict[str, Any]:
    """Dial info for an LLM-drafted plan (no recipe). Skills may still be attached by
    the Planner or discovered, but with the methodology and plan model-derived the
    project cannot sit above D1."""
    framed = bool(
        (spec_hints or {}).get("target_column")
        or (spec_hints or {}).get("feature_columns")
        or (spec_hints or {}).get("filters")
    )
    return {"rung": "D1" if framed else "D0", "task_rungs": {}, "selection": "drafted"}


def node_attribution(tasks: List[Any]) -> Dict[str, Any]:
    """pass@model vs pass@model+fallback (approach_docs/019) for one run's tasks.

    The single implementation shared by the dial ledger (server.py) and the run scorecard
    (gads.eval). `tasks` are Task rows (or objects with .status/.result_json/
    .postcondition_json/.description), in creation order, all attempts; only completed and
    failed ones are considered.

    How much of the run the assigned model did itself vs. what a fallback (native/cloud)
    had to rescue. The two stay SEPARATE: a fallback-assisted pass must never read as a
    model pass, or the delegation-dial measurement is contaminated. Deduped per recipe node
    (last completed instance wins across attempts).

    A node that EXHAUSTED its retries and never completed in any attempt counts as
    attempted-not-passed. Counting completed tasks only made pass@model survivorship-biased:
    a run that died at node 7 of 10 recorded 6/6 = 1.0 (4653ffae, 50bd2bf3, 22b8557a).
    Downstream nodes that never ran carry no code and stay out of the denominator; the
    model was never asked to do them.
    """
    by_node: Dict[str, str] = {}
    failed_nodes: Dict[str, str] = {}
    for t in tasks:
        if t.status not in ("completed", "failed"):
            continue
        rj = t.result_json or {}
        mu = str(rj.get("model_used", ""))
        node = (t.postcondition_json or {}).get("recipe_node_id") or t.description
        if t.status == "completed":
            if rj.get("code") or mu.startswith(("native_fallback:", "cloud_fallback:")):
                by_node[node] = mu
        # A failed task with ANY execution record was attempted, including one whose
        # generation never parsed and so stored empty code (50bd2bf3). Tasks the exhaustion
        # cleanup marked as never-run carry no result_json at all.
        elif rj and not mu.startswith("native_primary:"):
            failed_nodes[node] = mu
    n_failed = len([n for n in failed_nodes if n not in by_node])
    n_exec = len(by_node)
    n_native_fb = sum(1 for m in by_node.values() if m.startswith("native_fallback:"))
    n_cloud_fb = sum(1 for m in by_node.values() if m.startswith("cloud_fallback:"))
    # Production mode ran these natively BY POLICY: the model was never asked, so they are
    # neither a model pass nor a rescue. Counted apart from both and excluded from the
    # pass@model denominator; a node the model never attempted is no evidence either way.
    n_native_primary = sum(1 for m in by_node.values() if m.startswith("native_primary:"))
    n_fb = n_native_fb + n_cloud_fb
    n_attempted = n_exec - n_native_primary + n_failed
    n_model = n_exec - n_native_primary - n_fb
    return {
        "by_node": by_node,
        "failed_by_node": {n: m for n, m in failed_nodes.items() if n not in by_node},
        "exec_nodes": n_exec,
        "attempted_nodes": n_attempted,
        "failed_nodes": n_failed,
        "native_primary": n_native_primary,
        "model_pass": n_model,
        "fallback_pass": n_fb,
        "native_fallback": n_native_fb,
        "cloud_fallback": n_cloud_fb,
        "pass_at_model": round(n_model / n_attempted, 3) if n_attempted else None,
    }


ROUTING_LEDGER_PATH = "research/routing_ledger.jsonl"


def append_routing_ledger(record: Dict[str, Any], path: Optional[str] = None) -> None:
    """Append one routing decision record (approach_docs/024 §4).

    Separate from the dial ledger on purpose: a routing decision is made once per run, up
    front, and is worth recording even for runs that later fail or are cancelled — the dial
    ledger is only written on completion, so routing evidence would be lost exactly for the
    runs most likely to have been misrouted.

    Never raises: telemetry must not be able to fail a workflow.
    """
    try:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        target = path or ROUTING_LEDGER_PATH
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  [Routing] {record.get('verdict')} | labels="
              f"{record.get('task_type')}/{record.get('data_modality')} | "
              f"chosen={record.get('chosen_recipe') or 'none'} | "
              f"oracle={len(record.get('oracle_candidates') or [])} candidate(s)", flush=True)
    except Exception as e:
        print(f"  [Routing] Warning: ledger append failed: {e}", flush=True)


def append_ledger(record: Dict[str, Any], path: Optional[str] = None) -> None:
    """Append one run record to the dial ledger (JSONL). Never raises — the ledger is
    telemetry; it must not be able to fail a workflow."""
    try:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        target = path or LEDGER_PATH
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  [Dial] Ledger: {record.get('rung')} × {record.get('routing_mode')} → "
              f"{record.get('outcome')} ({record.get('project_id', '')[:8]})", flush=True)
    except Exception as e:
        print(f"  [Dial] Warning: ledger append failed: {e}", flush=True)
