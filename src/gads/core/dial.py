"""Delegation-dial formalism (approach_docs/013): which rung of the autonomy ladder
does a project run sit on, and the run ledger that accumulates the rung × engine grid.

The ladder (higher rung = more decisions moved from the model into curated artifacts):

  D0  full delegation — no recipe, no spec hints; the model frames, plans, realizes
  D1  framed          — spec hints fix the framing (target column / features / filters)
  D2  advised         — RESERVED: recipe suggested but deviatable. Operationally
                        unreachable today: a Router-matched recipe is compiled exactly
                        like a pinned one, so "advice" collapses into D3+. Selection
                        provenance (pinned|routed|drafted) is recorded separately.
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
