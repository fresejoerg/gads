#!/usr/bin/env python3
"""Regression tests for the code-only run scorecard (src/gads/eval/). No LLM, no DB.

Builds synthetic RunRecords and pins the rules the backfill surfaced:
  - attempts split at Planner tasks; only the final attempt is scored
  - a completed reporting task beats leftover "pending" rows from superseded attempts
  - never-started / record-missing runs are not_evaluable; abandoned runs are void-flagged
  - dependent checks of a node that did not complete are not_evaluable, not extra fails
  - verdict: any critical fail -> fail; critical n/a only -> inconclusive
  - resumed nodes are not double-counted in usage
  - attribution is dial.node_attribution (the dial ledger's own function)

    PYTHONPATH=src uv run python scripts/test_eval_scorecard.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gads.eval.record import INIT_TASK_PREFIX, RunRecord, TaskView, partition_attempts  # noqa: E402
from gads.eval.rubrics import terminal_state  # noqa: E402
from gads.eval.scorecard import evaluate  # noqa: E402

_n = 0


def T(assigned, status="completed", desc="", pc=None, rj=None, error=None):
    global _n
    _n += 1
    return TaskView(id=f"t{_n}", assigned_to=assigned, status=status, description=desc or f"task {_n}",
                    created_at=f"2026-11-11T00:00:{_n:02d}", postcondition_json=pc,
                    result_json=rj, error=error)


def init():
    return T("System", desc=INIT_TASK_PREFIX + " and re-syncing kernel...")


def report(status="completed"):
    return T("System", status=status, desc="Publishing final dashboard and research reports...")


def planner(sections=None):
    return T("Planner", rj={"recipe_id": "r", "recipe_sections": sections} if sections else {})


def node(nid, status="completed", metrics=None, captured=None, variables=None, rj_extra=None, error=None):
    pc = {"output_type": "value", "recipe_node_id": nid}
    if metrics:
        pc["required_metrics"] = metrics
    if variables:
        pc["required_variables"] = variables
    rj = None
    if status in ("completed", "failed"):
        rj = {"code": "x = 1", "model_used": "local_model", "stdout": "ok",
              "metrics_captured": captured if captured is not None else {},
              "semantic_insights": [], "usage": {"calls": 1, "cost_usd": 0.5}}
        rj.update(rj_extra or {})
    return T("local_model", status=status, pc=pc, rj=rj, error=error)


def record(tasks, narrative=None, kernel=None, files=()):
    ws = Path(tempfile.mkdtemp(prefix="gads_eval_test_"))
    for f in files:
        (ws / f).write_text("{}" if f.endswith(".json") else "x")
    state = {"spec_filename": "nonexistent_spec.md"}
    state.update({k: {"type": "int", "value": "1"} for k in (kernel or [])})
    return RunRecord(project_id="00000000-0000-0000-0000-000000000000", name="t", objective="o",
                     narrative=narrative, created_at="", updated_at="", last_state=state,
                     tasks=tasks, attempts=partition_attempts(tasks), workspace=ws)


DELIVERABLES = ("final_dashboard.html", "research_report.md", "workflow_execution.py", "metrics.json")
failures = []


def check(name, cond, info=""):
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  [{info}]" if info and not cond else ""))
    if not cond:
        failures.append(name)


def status_of(ev, cid, node_id=None):
    for c in ev["checks"]:
        if c["id"] == cid and (node_id is None or c.get("node") == node_id):
            return c["status"]
    return None


print("clean single-attempt pass")
rec = record([init(), planner([{"id": "a"}, {"id": "b"}]),
              node("a", metrics=["m"], captured={"m": 0.9}, variables=["x"]),
              node("b", variables=["y"]), report()],
             kernel=["x", "y"], files=DELIVERABLES)
ev = evaluate(rec, include_benchmark=False)
check("verdict pass", ev["verdict"] == "pass", ev["verdict"])
check("attribution = dial ledger function", ev["attribution"]["model_pass"] == 2
      and ev["attribution"]["pass_at_model"] == 1.0, ev["attribution"])
check("recipe nodes planned", status_of(ev, "run.recipe_nodes_planned") == "pass")

print("superseded attempt: earlier pendings do not make the run 'in progress'")
rec = record([init(), planner(), node("a", status="failed", error="boom"), node("b", status="pending"),
              planner(), node("a"), node("b"), report()], files=DELIVERABLES)
check("terminal state completed", terminal_state(rec) == "completed", terminal_state(rec))
ev = evaluate(rec, include_benchmark=False)
check("final attempt scored only", ev["run"]["workflow_attempts"] == 2 and ev["run"]["exec_nodes"] == 2)
check("prior attempt recorded as history", ev["prior_attempts"][0]["failed"][0]["node"] == "a")
check("verdict pass", ev["verdict"] == "pass", ev["verdict"])

print("failed node: dependents not_evaluable, verdict fail")
rec = record([init(), planner(), node("a", status="failed", metrics=["m"], variables=["x"], error="KeyError"),
              report()], files=DELIVERABLES)
ev = evaluate(rec, include_benchmark=False)
check("executed fails", status_of(ev, "node.executed", "a") == "fail")
check("metric n/a", status_of(ev, "node.metric.m", "a") == "not_evaluable")
check("variable n/a", status_of(ev, "node.variable.x", "a") == "not_evaluable")
check("verdict fail", ev["verdict"] == "fail")

print("missing metric / non-finite metric / missing variable fail")
rec = record([init(), planner(),
              node("a", metrics=["m", "n"], captured={"n": float("nan")}, variables=["x", "gone"]),
              report()], kernel=["x"], files=DELIVERABLES)
ev = evaluate(rec, include_benchmark=False)
check("absent metric fails", status_of(ev, "node.metric.m") == "fail")
check("NaN metric fails", status_of(ev, "node.metric.n") == "fail")
check("bound variable passes", status_of(ev, "node.variable.x") == "pass")
check("unbound variable fails", status_of(ev, "node.variable.gone") == "fail")

print("old run without metrics_captured -> inconclusive, not fail")
rec = record([init(), planner(), node("a", metrics=["m"], rj_extra={"metrics_captured": None}), report()],
             files=DELIVERABLES)
ev = evaluate(rec, include_benchmark=False)
check("metric n/a", status_of(ev, "node.metric.m") == "not_evaluable")
check("verdict inconclusive", ev["verdict"] == "inconclusive", ev["verdict"])

print("terminal states")
check("never started", terminal_state(record([])) == "never_started")
ev = evaluate(record([]), include_benchmark=False)
check("never started -> not_evaluable", ev["verdict"] == "not_evaluable", ev["verdict"])
rec = record([T("local_model", desc="follow-up only")])
rec.main_workflow_found = False
check("record missing", terminal_state(rec) == "record_missing")
rec = record([init(), planner(), node("a"), node("b", status="failed", error="x")])
ev = evaluate(rec, include_benchmark=False)
check("abandoned", ev["run"]["terminal_state"] == "abandoned" and ev["run"]["void_suspected"])
check("abandoned: deliverables n/a, not extra fails",
      status_of(ev, "run.file.final_dashboard.html") == "not_evaluable")
check("cancelled", terminal_state(record([init(), planner(), report()], narrative="[CANCELLED] user"))
      == "cancelled")
rec = record([init(), planner(), node("a", status="running")])
check("in progress", terminal_state(rec) == "in_progress")
check("in progress -> not_evaluable", evaluate(rec, include_benchmark=False)["verdict"] == "not_evaluable")

print("integrity")
rec = record([init(), planner(), node("a", metrics=["m"], captured={"m": 1.0},
                                       rj_extra={"stdout": "Simulating data because none was found"}),
              report()], files=DELIVERABLES)
(rec.workspace / "metrics.json").write_text('{"m": 2.0}')
ev = evaluate(rec, include_benchmark=False)
check("hallucination phrase flagged", status_of(ev, "node.no_hallucination_phrases") == "fail")
check("metrics.json mismatch flagged", status_of(ev, "run.metrics_consistent") == "fail")
check("integrity fails are not critical", ev["verdict"] == "pass", ev["verdict"])

print("usage: resumed nodes not double counted")
resumed = node("a", rj_extra={"resumed_from_prior_attempt": True})
rec = record([init(), planner(), node("a"), node("b", status="failed", error="x"),
              planner(), resumed, node("b"), report()], files=DELIVERABLES)
ev = evaluate(rec, include_benchmark=False)
check("cost counts 3 executions, not 4", abs(ev["usage"]["cost_usd"] - 1.5) < 1e-9, ev["usage"])
check("resumed node marked", any(n["resumed"] for n in ev["nodes"]))

print("required_variables: replanned drafted plan (kernel reset, snapshot not)")
drafted = lambda st="completed": T("local_model", status=st, desc="drafted step",
                                   pc={"output_type": "value", "required_variables": ["x"]},
                                   rj={"code": "x=1", "model_used": "local_model"} if st != "pending" else None,
                                   error=None if st == "completed" else "boom")
rec = record([init(), planner(), drafted("failed"), planner(), drafted(), report()],
             kernel=["x"], files=DELIVERABLES)
ev = evaluate(rec, include_benchmark=False)
check("stale-able name is not_evaluable", status_of(ev, "node.variable.x") == "not_evaluable")
check("verdict inconclusive, not pass", ev["verdict"] == "inconclusive", ev["verdict"])
rec = record([init(), planner([{"id": "a"}]), node("a", status="failed", error="e"),
              planner([{"id": "a"}]), node("a", variables=["x"]), report()],
             kernel=["x"], files=DELIVERABLES)
check("recipe plan keeps its kernel: name counts",
      status_of(evaluate(rec, include_benchmark=False), "node.variable.x") == "pass")

print("write_scorecard: a re-score that changes only the stamp leaves the file alone")
from gads.eval.scorecard import write_scorecard  # noqa: E402
out = Path(tempfile.mkdtemp(prefix="gads_eval_out_"))
ev = evaluate(record([init(), planner(), node("a"), report()], files=DELIVERABLES), include_benchmark=False)
p = write_scorecard(ev, out)
before = p.read_text()
ev["scored"] = {"at": "later", "commit": "other"}
write_scorecard(ev, out)
check("unchanged card not rewritten", p.read_text() == before)
ev["verdict"] = "fail"
write_scorecard(ev, out)
check("changed card rewritten", '"verdict": "fail"' in p.read_text())

print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all checks passed'}")
sys.exit(1 if failures else 0)
