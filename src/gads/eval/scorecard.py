"""The run scorecard: one eval object per GADS execution, code-only, deterministic.

    evaluate(rec)          RunRecord -> eval object (dict)
    evaluate_run(pid)      load + evaluate
    write_scorecard(ev)    research/evals/<project_id>.json (overwritten on re-score)

Verdict (user's rule, 2026-11-11): PASS iff every CRITICAL rubric passes, i.e. the run
reached final reporting and, on the final attempt, every node executed and met its
contract's required_metrics and required_variables. Any critical fail -> FAIL. No critical
fail but some critical rubric not evaluable -> INCONCLUSIVE. A run still in progress, or a
project whose workflow was never launched or whose workflow rows are gone, is NOT_EVALUABLE. A run abandoned mid-run
(backend died) is FAIL with run.void_suspected=true. There is no weighted composite
score: dimensions report pass/fail counts.

Deliberately NOT rubrics: the Critique's approval and the CompletenessVerifier's verdict.
Both are LLM judgments; they appear under `llm_judgments` as metadata only. That includes
the dial ledger's `outcome`, which requires Critique approval.

Bump RUBRIC_VERSION whenever a rubric's meaning changes, so scorecards from different
versions are never compared as if they were the same instrument.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gads.core.dial import node_attribution
from gads.eval import benchmark as bench
from gads.eval.record import RunRecord, load_run
from gads.eval.rubrics import FAIL, NA, PASS, Check, node_checks, run_checks, terminal_state

RUBRIC_VERSION = "1.0.0"
EVALS_DIR = Path(os.getenv("GADS_EVALS_DIR", "research/evals"))
DIAL_LEDGER = Path(os.getenv("GADS_DIAL_LEDGER", "research/dial_ledger.jsonl"))


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else None
    except Exception:
        return None


def _dial_row(project_id: str, path: Path = DIAL_LEDGER) -> Optional[Dict[str, Any]]:
    """The run's dial-ledger record (last one wins), or None (the ledger is only written
    for runs that reached the end of the pipeline)."""
    if not path.exists():
        return None
    row = None
    with open(path) as f:
        for line in f:
            if project_id in line:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("project_id") == project_id:
                    row = d
    return row


def _usage(tasks) -> Dict[str, Any]:
    """Tokens and cost over the main workflow. Resumed nodes copy their prior attempt's
    result_json (usage included), so they are skipped to avoid double counting. Calls the
    gateway could not price are reported as such: cost is then a lower bound, not $0."""
    tot = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0,
           "unpriced_calls": 0}
    models = set()
    for t in tasks:
        rj = t.result_json or {}
        if rj.get("resumed_from_prior_attempt"):
            continue
        u = rj.get("usage") or {}
        for k in ("calls", "prompt_tokens", "completion_tokens", "unpriced_calls"):
            tot[k] += int(u.get(k) or 0)
        tot["cost_usd"] += float(u.get("cost_usd") or 0.0)
        models.update(u.get("models") or [])
    tot["cost_usd"] = round(tot["cost_usd"], 6)
    tot["cost_complete"] = tot["unpriced_calls"] == 0
    tot["models"] = sorted(models)
    return tot


def _invocation(model_used: str) -> str:
    for p in ("native_fallback", "cloud_fallback", "native_primary"):
        if model_used.startswith(p + ":"):
            return p
    return "model" if model_used else "none"


def _rollup(checks: List[Check]) -> Dict[str, Dict[str, int]]:
    dims: Dict[str, Dict[str, int]] = {}
    for c in checks:
        d = dims.setdefault(c.dimension, {PASS: 0, FAIL: 0, NA: 0})
        d[c.status] += 1
    return dims


def verdict(checks: List[Check], state: str) -> str:
    if state in ("in_progress", "never_started", "record_missing"):
        return "not_evaluable"
    crit = [c for c in checks if c.critical]
    if any(c.status == FAIL for c in crit):
        return "fail"
    if any(c.status == NA for c in crit):
        return "inconclusive"
    return "pass"


def evaluate(rec: RunRecord, include_benchmark: bool = True) -> Dict[str, Any]:
    state = terminal_state(rec)
    in_progress = state == "in_progress"
    fa = rec.final_attempt

    checks: List[Check] = run_checks(rec, state)
    nodes: List[Dict[str, Any]] = []
    for t in (fa.exec_tasks if fa else []):
        nc = node_checks(rec, t, in_progress)
        checks.extend(nc)
        rj = t.result_json or {}
        mu = str(rj.get("model_used") or "")
        u = rj.get("usage") or {}
        nodes.append({
            "node": t.node_id,
            "task_id": t.id,
            "status": t.status,
            # Who wrote the contract: a recipe node's contract is curated; a drafted plan's
            # was written by the Planner LLM, so passing it is weaker evidence.
            "contract_source": "recipe" if (t.postcondition_json or {}).get("recipe_node_id")
                               else "planner_drafted" if t.postcondition_json else "none",
            "assigned_to": t.assigned_to,
            "model_used": mu or None,
            "invocation": _invocation(mu),
            "resumed": bool(rj.get("resumed_from_prior_attempt")),
            "generations": u.get("calls"),       # Coder calls incl. retries
            "cost_usd": u.get("cost_usd"),
            "checks": {PASS: sum(c.status == PASS for c in nc), FAIL: sum(c.status == FAIL for c in nc),
                       NA: sum(c.status == NA for c in nc)},
        })

    # Earlier attempts: history, not scored (their failures are superseded by the replan)
    history = []
    for a in rec.attempts[:-1]:
        failed = [{"node": t.node_id, "error": (t.error or "")[:200]}
                  for t in a.exec_tasks if t.status == "failed" and t.result_json]
        history.append({"attempt": a.index, "exec_nodes": len(a.exec_tasks),
                        "completed": sum(t.status == "completed" for t in a.exec_tasks),
                        "failed": failed})

    # Benchmark layer, when the run's spec is a benchmark spec
    bench_block = None
    bdir = bench.find_benchmark(rec.spec_filename) if include_benchmark else None
    if bdir is not None and not in_progress:
        expected = json.loads((bdir / "expected.json").read_text())
        sections = bench.score_sections(expected, rec.workspace)
        bchecks = [c for cs in sections.values() for c in cs]
        n_ok = sum(ok for _, ok, _ in bchecks)
        bench_block = {
            "benchmark_id": expected.get("benchmark_id", bdir.name),
            "verdict": "pass" if n_ok == len(bchecks) else "fail",
            "passed": n_ok, "total": len(bchecks),
        }
        for dim, cs in sections.items():
            for name, ok, det in cs:
                checks.append(Check(f"benchmark.{dim}.{name}", "benchmark", PASS if ok else FAIL, det))

    dial = _dial_row(rec.project_id)
    critique = rec.stage_tasks("Critique")
    cv = rec.stage_tasks("CompletenessVerifier")
    attribution = node_attribution(rec.tasks)
    attribution.pop("by_node", None)
    attribution.pop("failed_by_node", None)

    v = verdict(checks, state)
    crit = [c for c in checks if c.critical]
    return {
        "schema": "gads.run_eval",
        "rubric_version": RUBRIC_VERSION,
        "project_id": rec.project_id,
        "name": rec.name,
        "verdict": v,
        "critical": {PASS: sum(c.status == PASS for c in crit), FAIL: sum(c.status == FAIL for c in crit),
                     NA: sum(c.status == NA for c in crit)},
        "dimensions": _rollup(checks),
        "benchmark": bench_block,
        "run": {
            "terminal_state": state,
            # The process died mid-run: a harness failure, not evidence about the model
            # (approach_docs/033 §4b). The verdict is still FAIL; consumers filter on this.
            "void_suspected": state == "abandoned",
            "workflow_attempts": len(rec.attempts),
            "recipe_id": fa.recipe_id if fa else None,
            "plan": "recipe_compiled" if (fa and fa.recipe_sections) else "planner_drafted" if fa else None,
            "exec_nodes": len(nodes),
            "created_at": rec.created_at,
            "excluded_followup_tasks": rec.excluded_followup_tasks,
            "executions_on_project": rec.executions,   # >1: relaunched; the latest is scored
        },
        "fingerprint": {
            # Run-time harness state, as far as the run recorded it. routing_mode/run_mode
            # are only in the dial ledger, which exists for runs that reached the end.
            "spec": rec.spec_filename,
            "recipe_id": fa.recipe_id if fa else None,
            "recipe_version": rec.applied_recipe_version(),
            "engine_id": rec.last_state.get("engine_id") or (dial or {}).get("engine_id"),
            "routing_mode": (dial or {}).get("routing_mode"),
            "run_mode": (dial or {}).get("run_mode"),
            "rung": ((rec.last_state.get("dial") or {}).get("rung")) or (dial or {}).get("rung"),
        },
        "attribution": attribution,       # pass@model; same function as the dial ledger
        "usage": _usage(rec.tasks),
        "nodes": nodes,
        "prior_attempts": history,
        "llm_judgments": {
            # Recorded for context only. NOT rubrics, never part of the verdict.
            "critique_attempts": len(critique),
            "completeness_verifier_ran": bool(cv),
            "dial_ledger_outcome": (dial or {}).get("outcome"),
        },
        "checks": [c.to_dict() for c in checks],
        "scored": {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": _git_commit(),
        },
    }


def evaluate_run(project_id, include_benchmark: bool = True) -> Dict[str, Any]:
    return evaluate(load_run(project_id), include_benchmark=include_benchmark)


def write_scorecard(ev: Dict[str, Any], out_dir: Path = EVALS_DIR) -> Path:
    """Write the card, unless only the `scored` stamp would change: the files are in git,
    and a re-score that finds nothing new must not rewrite every tracked file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ev['project_id']}.json"
    if path.exists():
        try:
            old = json.loads(path.read_text())
            if {k: v for k, v in old.items() if k != "scored"} == \
               {k: v for k, v in ev.items() if k != "scored"}:
                return path
        except Exception:
            pass
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ev, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def score_and_write(project_id) -> Optional[Path]:
    """The pipeline hook: evaluate one run and write its scorecard. Never raises: the
    scorecard is measurement and must not be able to fail or delay a workflow."""
    try:
        ev = evaluate_run(project_id)
        path = write_scorecard(ev)
        print(f"  [Eval] {ev['verdict'].upper()} | critical {ev['critical'][PASS]}/"
              f"{sum(ev['critical'].values())} | {path}", flush=True)
        return path
    except Exception as e:
        print(f"  [Eval] Warning: scorecard failed for {str(project_id)[:8]}: {e}", flush=True)
        return None
