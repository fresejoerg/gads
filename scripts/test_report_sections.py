"""Offline check of the recipe-driven dashboard composition (core/report_sections.py).

No database, no sandbox, no LLM: fabricates task rows and asserts the sections come out in
DAG order with the right status, attribution and collapse decisions, then renders the real
Jinja template and the Markdown twin over them.

    PYTHONPATH=src uv run python scripts/test_report_sections.py
"""
import sys, uuid, json, os, types
from datetime import datetime, timedelta
sys.path.insert(0, "src")

from gads.core import report_sections as rs

class T:
    def __init__(self, desc, assigned, status, pc=None, res=None, err=None, age=0):
        self.id = uuid.uuid4(); self.description = desc; self.assigned_to = assigned
        self.status = status; self.postcondition_json = pc or {}; self.result_json = res or {}
        self.error = err; self.created_at = datetime(2026, 8, 19) + timedelta(minutes=age)

skeleton = [
    {"id": "load_prepared_data", "intent": "Load the modelling table and bind X_train. Second sentence.", "produces": ["X_train"], "report": {}},
    {"id": "shortlist_candidates", "intent": "THE REASONING STEP — nominate candidates.", "produces": ["candidates"], "report": {"title": "Reasoned Shortlist"}},
    {"id": "holdout_evaluation", "intent": "Evaluate on the held-out split.", "produces": ["macro_f1"], "report": {}},
    {"id": "performance_report", "intent": "Write the model card.", "produces": ["model_card"], "report": {}},
]

planner = T("Planner (deterministic)", "Planner", "completed",
            res={"recipe_sections": skeleton, "recipe_id": "tabular_supervised.selection.classification"})

t1 = T("Load the modelling table and bind X_train.\n\n[SPEC HINTS: OBJECTIVE: x]\n\n[RECIPE INVARIANTS — these rules are MANDATORY for every step:\n- foo\n]",
       "qwen", "completed", pc={"recipe_node_id": "load_prepared_data"},
       res={"model_used": "qwen", "stdout": "shapes ok", "metrics_captured": {"n_train_rows": 32561},
            "semantic_insights": [], "artifact_files": []}, age=1)

# failed first attempt, then a successful replan — the completed one must win
t2a = T("shortlist", "qwen", "failed", pc={"recipe_node_id": "shortlist_candidates"}, err="KeyError", age=2)
t2b = T("shortlist", "qwen", "completed", pc={"recipe_node_id": "shortlist_candidates"},
        res={"model_used": "qwen", "stdout": "4 candidates",
             "semantic_insights": [{"artifact": "candidates", "insight": "Chose RF over XGBoost: 32k rows, 14 features.", "evidence": "rows_per_feature=2325"}],
             "artifact_files": ["shortlist.json"]}, age=3)

t3 = T("holdout", "qwen", "completed", pc={"recipe_node_id": "holdout_evaluation"},
       res={"model_used": "native_fallback:gads_evaluate_holdout", "stdout": "eval done",
            "usage": {"calls": 3, "prompt_tokens": 12000, "completion_tokens": 2500,
                      "reasoning_tokens": 900, "total_tokens": 14500, "cost_usd": 0.0234,
                      "models": ["gemini-3.7-flash"], "cost_source": "computed",
                      "unpriced_calls": 0},
            "orchestrator_summary": "Created roc.json | Metrics captured: macro_f1=0.8212, roc_auc=0.9255",
            "artifact_files": ["roc.json"]}, age=4)
# performance_report never ran

tasks = [planner, t1, t2a, t2b, t3]

secs = rs.build_sections(tasks)
cards = [
    {"description": "Interactive: roc.json", "type": "json_plot", "json_data": '{"data":[],"layout":{}}',
     "task_id": str(t3.id), "source_filename": "roc.json", "caption": "ROC", "ctx_text": "", "img_b64": None, "html_content": None, "filename": None},
    {"description": "Orphan chart", "type": "static_plot", "img_b64": "AAA", "task_id": None,
     "source_filename": None, "caption": "?", "ctx_text": "", "json_data": None, "html_content": None, "filename": None},
]
orphans = rs.attach_cards(secs, cards)
rs.apply_section_notes(secs, [{"node_id": "shortlist_candidates", "text": "Random Forest was preferred."}])
rs.finalize(secs)

print("=== SECTIONS ===")
for s in secs:
    print(f"[{s['index']}] {s['title']:<24} status={s['status']:<13} collapsed={s['collapsed']} "
          f"cards={len(s['cards'])} metrics={s['metrics']} fb={s['fallback']} note={bool(s['note'])}")
print("orphans:", len(orphans))
assert [s["node_id"] for s in secs] == [n["id"] for n in skeleton], "DAG order must be preserved"
assert secs[1]["title"] == "Reasoned Shortlist"
assert secs[1]["status"] == "completed", "completed replan must beat the failed attempt"
assert secs[2]["metrics"] == {"macro_f1": "0.8212", "roc_auc": "0.9255"}
assert secs[2]["fallback"] == "native"
assert len(secs[2]["cards"]) == 1
assert secs[3]["status"] == "not_executed"
assert secs[0]["collapsed"] is False, "a node that captured metrics has substance"
assert secs[3]["collapsed"] is True, "a node with no evidence at all collapses"
assert "SPEC HINTS" not in secs[0]["intent"] and "INVARIANTS" not in secs[0]["intent"]
assert len(orphans) == 1

# --- usage plumbing ---
assert secs[2]["usage"]["total_tokens"] == 14500, secs[2]["usage"]
assert secs[1]["usage"] is None, "a node with no recorded LLM call reports None, not zero"
totals = rs.run_totals(tasks)
assert totals["calls"] == 3 and totals["total_tokens"] == 14500, totals
assert abs(totals["cost_usd"] - 0.0234) < 1e-9, totals
from gads.core.usage import format_cost, format_tokens
assert format_cost(0.0234) == "$0.02" and format_cost(0.0003) == "$0.0003"
assert format_cost(None) == "n/a" and format_cost(0) == "$0.00"
assert format_tokens(14500) == "14.5k" and format_tokens(0) == "0"
print("usage plumbing OK:", totals["calls"], "calls,", format_tokens(totals["total_tokens"]),
      "tokens,", format_cost(totals["cost_usd"]))

print("\n=== PROMPT BRIEF ===")
print(rs.format_for_prompt(secs)[:900])

# --- render the real template ---
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader("src/gads/templates"))
html = env.get_template("dashboard.html.j2").render(
    project_id="test", narrative="N", takeaways=["a"], cards=cards, sections=secs,
    orphan_cards=orphans, key_metrics={"macro_f1": "0.8212"}, followups=[],
    run_usage=totals, fmt_cost=rs.format_cost_value, fmt_tokens=rs.format_tokens_value)
out = os.path.join(os.environ.get("GADS_TEST_OUT", "/tmp"), "dash.html")
open(out, "w").write(html)
for needle in ["How This Result Was Produced", "Reasoned Shortlist", "not executed",
               "native fallback", "Additional Artifacts", "Random Forest was preferred",
               "Total run cost", "$0.02", "14.5k", "no LLM call", "900 reasoning"]:
    assert needle in html, f"missing from HTML: {needle}"
print(f"\nHTML OK ({len(html)} bytes) -> {out}")

# --- markdown twin ---
from gads.core.reporting import _build_markdown
md = _build_markdown("test", "N", ["a"], {"macro_f1": "0.8212"}, secs, orphans, totals)
assert "## Run Cost" in md and "$0.02" in md, "markdown must carry the run cost too"
print("\n=== MARKDOWN (excerpt) ===")
print(md[md.index("## Methodology"):][:1200])

# --- drafted-plan lane (no skeleton) ---
drafted = [T("Do EDA on the file", "qwen", "completed", res={"stdout": "x"}, age=1),
           T("Train a model", "qwen", "failed", err="boom", age=2)]
d = rs.finalize(rs.build_sections(drafted))
assert [s["title"] for s in d] == ["Step 1", "Step 2"], d
print("\ndrafted-plan fallback OK:", [(s["title"], s["status"]) for s in d])

# --- legacy project (no sections at all) ---
legacy = rs.build_sections([])
assert legacy == []
html2 = env.get_template("dashboard.html.j2").render(
    project_id="t", narrative="N", takeaways=[], cards=cards, sections=[], orphan_cards=[],
    key_metrics={}, followups=[])
assert "Supporting Data Visualizations" in html2
print("legacy fallback OK")
print("\nALL CHECKS PASSED")

# --- drafted plan across a replan: 2 steps x 2 attempts must yield 2 sections ---
replanned = [
    T("Do EDA on the file", "qwen", "failed", err="boom", age=1),
    T("Train a model", "qwen", "pending", age=2),
    T("Do EDA on the file", "qwen", "completed", res={"stdout": "ok"}, age=3),
    T("Train a model", "qwen", "completed", res={"stdout": "ok"}, age=4),
]
r = rs.finalize(rs.build_sections(replanned))
assert len(r) == 2, [s["title"] for s in r]
assert all(s["status"] == "completed" for s in r), [(s["title"], s["status"]) for s in r]
print("replanned drafted plan OK:", [(s["title"], s["status"]) for s in r])
