"""Code-only rubrics over a RunRecord. No LLM calls, no judgment calls.

Every rubric returns pass / fail / not_evaluable. `not_evaluable` means the record lacks
the evidence (an older run predating a field, a node that never got far enough, a run
still in progress); it never counts toward a pass or a fail.

`critical` rubrics decide the run verdict (scorecard.verdict): the run reached its
terminal "completed" state and, on the final attempt, every node executed and met its
contract's required_metrics and required_variables. Everything else is reported but does
not flip the verdict.

Contract fields and how each is checked here, vs. at run time:

  required_metrics    captured in result_json.metrics_captured as a finite number.
                      Same evidence the runtime captured.
  required_variables  present in the END-OF-SESSION kernel snapshot (last_state_json, the
                      executor's authoritative_state: it accumulates every name bound during
                      the run and omits callables). The runtime output gate checked recipe
                      nodes right after they ran; this is post-hoc evidence. On a replanned
                      drafted plan (kernel reset, snapshot not) presence is not attributable
                      and the check is not_evaluable.
  required_insights   substring match over the node's emitted insights, the same match
                      the runtime uses. Runtime: soft warning. Here: a reported fail.
  required_columns    deterministic normalized-name match (case/punctuation-insensitive)
                      against DataFrame columns in the end-of-session kernel. STRICTER than
                      the runtime, which asks an LLM whether each concept is "captured by"
                      some column (execution_hub._semantic_validate_columns). That LLM check
                      is exactly what this scorecard must not use.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from gads.core.stdout_clean import HALLUCINATION_TOKENS
from gads.eval.record import RunRecord, TaskView

PASS, FAIL, NA = "pass", "fail", "not_evaluable"


@dataclass
class Check:
    id: str
    dimension: str            # execution | contract | integrity | deliverables | benchmark
    status: str               # pass | fail | not_evaluable
    detail: str
    critical: bool = False
    node: Optional[str] = None
    evidence: Optional[str] = None   # where the verdict came from, when not obvious

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _clip(s: Optional[str], n: int = 240) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _is_finite_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return math.isfinite(v)
    if isinstance(v, str):
        try:
            return math.isfinite(float(v))
        except ValueError:
            return False
    return False


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# --------------------------------------------------------------------------------------
# Terminal state
# --------------------------------------------------------------------------------------
def terminal_state(rec: RunRecord) -> str:
    """completed | halted | cancelled | in_progress | never_started | abandoned |
    record_missing.

    `completed` means the pipeline reached final reporting, NOT that the Critique approved
    the synthesis: that approval is an LLM judgment and stays out of the rubrics."""
    if not rec.main_workflow_found:
        return "record_missing"     # only follow-up tasks survive; no workflow to score
    narr = rec.narrative or ""
    if narr.startswith("[CANCELLED]"):
        return "cancelled"
    if narr.startswith("[HALTED]"):
        return "halted"
    # Reporting first: a SUCCESSFUL workflow leaves earlier attempts' never-run tasks
    # "pending" forever (server.py only sweeps leftovers when the workflow failed), so
    # pending rows alone do not mean the run is still going (56dbecc4).
    reporting = [t for t in rec.tasks if t.assigned_to == "System"
                 and t.description.startswith("Publishing final dashboard")]
    if reporting and reporting[-1].status == "completed":
        return "completed"
    if any(t.status in ("pending", "running") for t in rec.tasks):
        return "in_progress"
    if not rec.tasks:
        return "never_started"      # project row exists, workflow never launched
    # Terminal tasks, no reporting, no halt/cancel narrative: the process stopped mid-run
    # (e.g. the 2026-08-21 backend restarts, approach_docs/033 §4b "void runs").
    return "abandoned"


# --------------------------------------------------------------------------------------
# Node rubrics (contract dimension), final attempt only
# --------------------------------------------------------------------------------------
def node_checks(rec: RunRecord, t: TaskView, in_progress: bool) -> List[Check]:
    node = t.node_id
    pc = t.postcondition_json or {}
    rj = t.result_json or {}
    out: List[Check] = []

    # Executed
    if t.status == "completed":
        how = "resumed from a prior attempt" if rj.get("resumed_from_prior_attempt") else "completed"
        out.append(Check("node.executed", "contract", PASS, how, critical=True, node=node))
    elif in_progress and t.status in ("pending", "running"):
        out.append(Check("node.executed", "contract", NA, f"run in progress ({t.status})",
                         critical=True, node=node))
    elif t.status == "bypassed":
        out.append(Check("node.executed", "contract", FAIL,
                         "bypassed (runtime estimate too long; handover bundle instead of execution)",
                         critical=True, node=node))
    else:
        never_ran = not rj
        out.append(Check("node.executed", "contract", FAIL,
                         ("never ran: " if never_ran else "failed: ") + _clip(t.error),
                         critical=True, node=node))
    done = t.status == "completed"
    why_na = f"node did not complete ({t.status})"

    # required_metrics
    captured = rj.get("metrics_captured")
    for m in pc.get("required_metrics") or []:
        cid = f"node.metric.{m}"
        if not done:
            out.append(Check(cid, "contract", NA, why_na, critical=True, node=node))
        elif captured is None:
            out.append(Check(cid, "contract", NA, "result has no metrics_captured (run predates metric capture)",
                             critical=True, node=node))
        elif m not in captured:
            out.append(Check(cid, "contract", FAIL, "not captured", critical=True, node=node))
        elif not _is_finite_number(captured[m]):
            out.append(Check(cid, "contract", FAIL, f"captured but not a finite number: {captured[m]!r}",
                             critical=True, node=node))
        else:
            out.append(Check(cid, "contract", PASS, f"= {captured[m]}", critical=True, node=node))

    # required_variables (end-of-session kernel). The snapshot is the executor's
    # authoritative_state, which only GROWS within a run (dict.update, never a delete). On a
    # replanned DRAFTED plan the kernel was reset but the snapshot still holds the superseded
    # attempts' names, so presence cannot be attributed to the final attempt. Recipe plans
    # keep their kernel across replans (resume path), so their snapshot is the real kernel.
    kvars = rec.kernel_vars
    fa = rec.final_attempt
    kernel_reset_since = bool(fa and fa.index > 1 and not fa.recipe_sections)
    for v in pc.get("required_variables") or []:
        cid = f"node.variable.{v}"
        if not done:
            out.append(Check(cid, "contract", NA, why_na, critical=True, node=node))
        elif not kvars:
            out.append(Check(cid, "contract", NA, "no kernel snapshot on the project", critical=True, node=node))
        elif v in kvars and kernel_reset_since:
            out.append(Check(cid, "contract", NA,
                             "in the snapshot, but the kernel was reset for this replanned drafted plan "
                             "and the snapshot keeps earlier attempts' names: not attributable",
                             critical=True, node=node, evidence="end_of_session_kernel"))
        elif v in kvars:
            out.append(Check(cid, "contract", PASS, f"bound ({kvars[v].get('type')})", critical=True,
                             node=node, evidence="end_of_session_kernel"))
        else:
            out.append(Check(cid, "contract", FAIL,
                             "never appeared in any kernel snapshot this run (the snapshot keeps every "
                             "name ever bound; it omits only callables)",
                             critical=True, node=node, evidence="end_of_session_kernel"))

    # required_insights (runtime soft-fail; here a reported, non-critical check)
    insights = rj.get("semantic_insights")
    emitted = [str(i.get("artifact", "")).lower() for i in (insights or [])] + \
              [str(i.get("insight", "")).lower() for i in (insights or [])]
    for req in pc.get("required_insights") or []:
        cid = f"node.insight.{req}"
        if not done:
            out.append(Check(cid, "contract", NA, why_na, node=node))
        elif insights is None:
            out.append(Check(cid, "contract", NA, "result has no semantic_insights", node=node))
        elif any(str(req).lower() in e for e in emitted):
            out.append(Check(cid, "contract", PASS, "emitted", node=node))
        else:
            out.append(Check(cid, "contract", FAIL, "not emitted (runtime only warns)", node=node))

    # required_columns (dataframe contracts), deterministic name match
    if pc.get("output_type") == "dataframe" and pc.get("required_columns"):
        cols = {_norm(c) for v in kvars.values() if v.get("type") == "DataFrame"
                for c in (v.get("columns") or [])}
        for c in pc["required_columns"]:
            cid = f"node.column.{c}"
            if not done:
                out.append(Check(cid, "contract", NA, why_na, node=node))
            elif not cols:
                out.append(Check(cid, "contract", NA, "no DataFrame in the end-of-session kernel", node=node))
            elif _norm(c) in cols:
                out.append(Check(cid, "contract", PASS, "column present", node=node,
                                 evidence="end_of_session_kernel"))
            else:
                out.append(Check(cid, "contract", FAIL,
                                 "no DataFrame column with this name (exact normalized match; the runtime's "
                                 "LLM column check is more lenient)", node=node,
                                 evidence="end_of_session_kernel"))

    # Hallucination guard (integrity): stdout phrases meaning the code faked or skipped data
    if done:
        low = str(rj.get("stdout") or "").lower()
        hits = [tok for tok in HALLUCINATION_TOKENS if tok in low]
        out.append(Check("node.no_hallucination_phrases", "integrity", FAIL if hits else PASS,
                         f"stdout contains {hits}" if hits else "clean", node=node))
    return out


# --------------------------------------------------------------------------------------
# Run rubrics
# --------------------------------------------------------------------------------------
REQUIRED_DELIVERABLES = ("final_dashboard.html", "research_report.md", "workflow_execution.py")


def run_checks(rec: RunRecord, state: str) -> List[Check]:
    out: List[Check] = []
    detail = {"completed": "reached final reporting",
              "halted": _clip(rec.narrative), "cancelled": _clip(rec.narrative),
              "in_progress": "tasks still pending/running",
              "never_started": "no tasks: the workflow was never launched",
              "record_missing": "the main workflow's task rows are gone (only follow-ups remain)",
              "abandoned": "stopped mid-run: no reporting, no halt/cancel record (suspected void run)",
              }[state]
    out.append(Check("run.completed", "execution",
                     PASS if state == "completed"
                     else NA if state in ("in_progress", "never_started", "record_missing") else FAIL,
                     detail, critical=True))

    fa = rec.final_attempt
    n_exec = len(fa.exec_tasks) if fa else 0
    if state in ("never_started", "record_missing"):
        out.append(Check("run.has_plan", "execution", NA, detail, critical=True))
    else:
        out.append(Check("run.has_plan", "execution", PASS if n_exec else FAIL,
                         f"final attempt {fa.index} has {n_exec} execution node(s)" if fa
                         else "no Planner task: the run never produced a plan", critical=True))

    # Recipe coverage: every compiled recipe node has a task in the final attempt
    sections = fa.recipe_sections if fa else None
    if sections:
        have = {t.node_id for t in fa.exec_tasks}
        missing = [s.get("id") for s in sections if s.get("id") and s.get("id") not in have]
        out.append(Check("run.recipe_nodes_planned", "execution", FAIL if missing else PASS,
                         f"no task for {missing}" if missing else f"all {len(sections)} recipe nodes planned"))
    else:
        out.append(Check("run.recipe_nodes_planned", "execution", NA, "plan was not compiled from a recipe"))

    # Deliverables are written by final reporting; for a run that never got there their
    # absence is a consequence already counted by run.completed, not a second failure.
    if state != "completed":
        for f in REQUIRED_DELIVERABLES:
            out.append(Check(f"run.file.{f}", "deliverables", NA, f"run did not complete ({state})"))
    else:
        for f in REQUIRED_DELIVERABLES:
            ok = rec.workspace_file(f) is not None
            out.append(Check(f"run.file.{f}", "deliverables", PASS if ok else FAIL,
                             "present" if ok else "missing"))

    # metrics.json: required when any node's contract declares metrics; and consistent
    # with what the nodes captured.
    declared = any((t.postcondition_json or {}).get("required_metrics") for t in (fa.exec_tasks if fa else []))
    mj = rec.metrics_json()
    if declared and state == "completed":
        out.append(Check("run.file.metrics.json", "deliverables", PASS if mj is not None else FAIL,
                         "present" if mj is not None else "missing or unparseable"))
    if mj is not None and fa:
        mismatches, compared = [], 0
        for t in fa.exec_tasks:
            for k, v in ((t.result_json or {}).get("metrics_captured") or {}).items():
                if k in mj and _is_finite_number(v) and _is_finite_number(mj[k]):
                    compared += 1
                    if not math.isclose(float(v), float(mj[k]), rel_tol=1e-9, abs_tol=1e-12):
                        mismatches.append(f"{k}: node {v} vs metrics.json {mj[k]}")
        out.append(Check("run.metrics_consistent", "integrity",
                         FAIL if mismatches else PASS if compared else NA,
                         "; ".join(mismatches[:5]) if mismatches
                         else f"{compared} captured metric(s) match metrics.json" if compared
                         else "no captured metric appears in metrics.json"))
    return out
