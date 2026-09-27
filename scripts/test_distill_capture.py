"""Offline check of the distillation capture path (core/distill_capture.py) and the builder.

Simulates what a run writes: a local task that fails once then is accepted, a cloud task
accepted first time, a native-fallback completion, and a cloud task whose code the sanitizer
rewrote. Then checks that scripts/build_distill_dataset.py labels, pairs and filters them.
No sandbox, LLM or DB needed; writes only to a temp capture dir.

    PYTHONPATH=src uv run python scripts/test_distill_capture.py
"""
import os
import sys
import tempfile
import types
import uuid

TMP = tempfile.mkdtemp(prefix="distill_capture_")
os.environ["GADS_DISTILL_CAPTURE_DIR"] = TMP
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import collections
import contextlib
import io
import json

from gads.agents.workers.coder import CodeGeneratorAgent, render_messages, RAW_SYSTEM_SUFFIX
from gads.core import distill_capture as dc
from gads.core.executor import ExecutionManager

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  — {detail}" if not cond and detail else ""))
    if not cond:
        FAILS.append(name)


def manager(model):
    m = ExecutionManager.__new__(ExecutionManager)
    m.coder = CodeGeneratorAgent(model=model)
    m.coder.model_str = model
    return m


def task(node):
    return types.SimpleNamespace(id=uuid.uuid4(), project_id=uuid.uuid4(), instruction_id=None,
                                 assigned_to="x", postcondition_json={
                                     "recipe_node_id": node, "required_variables": ["result"]})


def render(n):
    return {"system": f"SYSTEM PROMPT {n}", "user_core": f"TASK: step {n}\n\n"}


print("capture writes")
# 1. local: fails, then accepted
t1, m1 = task("fit"), manager("local_model")
m1.coder.last_render = render(1)
m1._capture_attempt("exec_error", "result = df.fitt()", "result = df.fitt()",
                    "AttributeError: fitt", t1.id, "step 1", "r")
m1._capture_attempt("executed", "result = df.fit()", "result = df.fit()", None, t1.id, "step 1", "r")
dc.record_accept(t1, {"code": "result = df.fit()", "model_used": "local_model"})
# 2. cloud: accepted first time
t2, m2 = task("plot"), manager("gemini-3.7-flash")
m2.coder.last_render = render(2)
m2._capture_attempt("executed", "result = 2", "result = 2", None, t2.id, "step 2", "r")
dc.record_accept(t2, {"code": "result = 2", "model_used": "gemini-3.7-flash"})
# 3. native fallback completion: no model attempt accepted
t3 = task("native")
dc.record_accept(t3, {"code": "result = gads_native()", "model_used": "native_fallback:gads_native"})
# 4. cloud, sanitizer rewrote the code (not whitespace/imports)
t4, m4 = task("rewrite"), manager("claude-haiku-4.5")
m4.coder.last_render = render(4)
m4._capture_attempt("executed", "result = y_prob[:, 1]", "result = y_prob.iloc[:, 1]", None,
                    t4.id, "step 4", "r")
dc.record_accept(t4, {"code": "result = y_prob.iloc[:, 1]", "model_used": "claude-haiku-4.5"})
# 5. capture disabled writes nothing
os.environ["GADS_DISTILL_CAPTURE"] = "false"
dc.record_accept(task("off"), {"code": "x = 1", "model_used": "local_model"})
os.environ["GADS_DISTILL_CAPTURE"] = "true"

files = sorted(os.listdir(TMP))
check("two monthly streams written", any(f.startswith("attempts-") for f in files)
      and any(f.startswith("accepts-") for f in files), str(files))

import build_distill_dataset as b

stats = collections.Counter()
examples, pairs = b.from_capture(stats)
by_task = {e["task_id"]: e for e in examples}
print("builder")
check("local accepted -> student_success", by_task.get(str(t1.id), {}).get("cls") == "student_success")
check("local failure -> one student_pair", [p["cls"] for p in pairs] == ["student_pair"], str(pairs))
check("pair's rejected side is the failed code", pairs and pairs[0]["rejected"] == "result = df.fitt()")
check("cloud accepted -> teacher_demo", by_task.get(str(t2.id), {}).get("cls") == "teacher_demo")
check("native completion excluded", str(t3.id) not in by_task and stats["excluded_native"] == 1)
check("sanitizer rewrite flagged", by_task.get(str(t4.id), {}).get("sanitizer") == "rewritten")
check("identical code flagged identical", by_task.get(str(t2.id), {}).get("sanitizer") == "identical")
check("disabled capture wrote nothing", stats["capture_accepts"] == 4, str(stats))

print("rendering")
msgs = render_messages("S", "U", raw=True)
check("student format is raw-code mode", msgs[0]["content"] == "S" + RAW_SYSTEM_SUFFIX)
check("cloud-format prompt recovers its core",
      b.core_from_rendered(render_messages("S", "U", raw=False)) == ("S", "U"))
check("raw-format prompt recovers its core",
      b.core_from_rendered(render_messages("S", "U", raw=True)) == ("S", "U"))

print("stage calls (BaseAgent.run) and Router labels")
import asyncio
import gads.core.llm as llm
from gads.agents.router import DataScienceRouter, RouterInput, RouterOutput

_answers = iter([
    RouterOutput(task_type="regression.survival", data_modality="tabular",
                 matched_recipe_id="survival_analysis.cox_regression", confidence=0.9),
    RouterOutput(task_type="classification.binary", data_modality="tabular",
                 matched_recipe_id="binary_classification.tabular.standard", confidence=0.8),
])


async def _fake_completion(model, response_model, messages, stream_callback=None, **kw):
    return next(_answers)

llm.get_structured_completion = _fake_completion      # BaseAgent.run imports it per call
dc.capture_context.set({"spec": "gbsg2_survival_cox.md", "source": "test"})
from gads.core.knowledge import KnowledgeRegistry
with contextlib.redirect_stdout(io.StringIO()):
    _reg = KnowledgeRegistry("src/gads/knowledge/recipes")
router = DataScienceRouter(model="gemini-3.7-flash")
for _ in range(2):
    asyncio.run(router.run(RouterInput(objective="Which factors drive recurrence?",
                                       available_recipes=_reg.get_recipes_summary())))
calls = b.read_jsonl(f"{TMP}/calls-*.jsonl")
check("both Router calls captured with spec", len(calls) == 2
      and all(c["stage_name"] == "Router" and c["spec"] == "gbsg2_survival_cox.md" for c in calls),
      str([(c.get("stage_name"), c.get("spec")) for c in calls]))
stats = collections.Counter()
rex = b.router_examples(stats, cap_per_spec=5)
check("correct Router call -> one example, wrong one rejected",
      len(rex) == 1 and stats["router_wrong"] == 1, f"{len(rex)} examples, {dict(stats)}")
check("Router example is teacher_demo with the served messages",
      rex and rex[0]["cls"] == "teacher_demo" and rex[0]["messages"][0]["role"] == "system")
check("Coder calls are NOT duplicated into the calls stream",
      not any(c["stage_name"] == "CodeGenerator" for c in calls))

print("skills-ablated snapshot")
t5, m5 = task("skills"), manager("gemini-3.7-flash")
m5.coder.last_render = {"system": "PROMPT\nSKILL: use gads_x()\nEND", "user_core": "TASK: s5\n\n",
                        "skills_context": "SKILL: use gads_x()"}
m5._capture_attempt("executed", "result = gads_x()", "result = gads_x()", None, t5.id, "s5", "r")
dc.record_accept(t5, {"code": "result = gads_x()", "model_used": "gemini-3.7-flash"})
b.OUT_ROOT = os.path.join(TMP, "datasets")
b.PINNED_HOLDOUT = set()
b.benchmark_specs = lambda *a, **k: set()
sys.argv = ["build", "--no-backfill", "--min-train", "0", "--holdout-pct", "0"]
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    b.main()
snap = os.path.join(b.OUT_ROOT, "latest")
ns = [json.loads(l) for l in open(os.path.join(snap, "sft_train_noskills.jsonl"))]
row5 = [r for r in ns if r["task_id"] == str(t5.id)]
from gads.agents.workers.coder import NO_SKILLS_TEXT
check("skills block replaced by the no-skills placeholder",
      row5 and "SKILL: use gads_x()" not in row5[0]["messages"][0]["content"]
      and NO_SKILLS_TEXT in row5[0]["messages"][0]["content"])
full = [json.loads(l) for l in open(os.path.join(snap, "sft_train.jsonl"))]
check("the regular snapshot keeps the skills",
      any("SKILL: use gads_x()" in r["messages"][0]["content"] for r in full))
rt = [json.loads(l) for l in open(os.path.join(snap, "router_train.jsonl"))]
check("router_train.jsonl written with an assistant JSON target",
      len(rt) == 1 and json.loads(rt[0]["messages"][-1]["content"])["task_type"] == "regression.survival")

print()
if FAILS:
    sys.exit(f"{len(FAILS)} check(s) failed: {FAILS}")
print("all checks passed")
