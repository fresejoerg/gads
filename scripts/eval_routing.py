"""Routing quality evaluation (approach_docs/024 phase 1).

Runs ONLY the Router stage over the tagged spec corpus — one LLM call per spec, no sandbox,
no planner, no execution — and scores it against the ground truth the specs already carry:

  * selection   — `matched_recipe_id` vs the spec's pinned `recipe_id`
  * coverage    — did the deterministic oracle cover the Router's own labels?
  * calibration — is `confidence` predictive of being right?

Two caveats that must travel with any number this prints:

  1. The specs were written alongside the recipes, so they are IN-DISTRIBUTION. This measures
     regression and relative change between models/prompts — not generalisation to novel
     objectives.
  2. The corpus is heavily skewed (37 of 54 tagged specs are causal.effect_estimation), so
     overall accuracy is close to meaningless: answering "causal" to everything scores ~69%.
     Per-class and macro-averaged figures are the honest summary, which is what is reported.

Usage:
    PYTHONPATH=src uv run python scripts/eval_routing.py            # all specs
    PYTHONPATH=src uv run python scripts/eval_routing.py --limit 8  # quick smoke
    PYTHONPATH=src uv run python scripts/eval_routing.py --out research/routing_eval.jsonl
"""
import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, "src")

import yaml

from gads.agents.router import DataScienceRouter, RouterInput
from gads.core.knowledge import KnowledgeRegistry
from gads.core.registry import resolve_stage_model

# ——— VOCABULARY BRIDGE ————————————————————————————————————————————————————————————
# DEBT, and deliberately visible: the spec taxonomy, the Router's prompt enum and recipe
# `applies_when` are three different vocabularies (approach_docs/024 §1). Phase 0 replaces
# this table with one controlled vocabulary derived from taxonomy.yaml. Until then every
# number below is measured through this translation, so a mapping error reads as a Router
# error. `None` means "the Router has no term for this" — those specs are UNSCORABLE on
# classification rather than counted wrong, because the model cannot express the right answer.
TAXONOMY_TO_ROUTER = {
    "classification.binary": "binary_classification",
    "classification.multiclass": None,          # Router enum has no multiclass term
    "causal.effect_estimation": "causal_inference",
    "regression.survival": "regression",
    "clustering.partitional": "clustering",
    "ranking.learning_to_rank": None,           # no term
    "recommendation.collaborative": None,       # no term
    "analytics.kpi_metrics": None,              # no term
    "data_preparation": None,                   # no term
    "eda": "eda",
}
MODALITY_TO_ROUTER = {
    "tabular": "tabular",
    "text": "text",
    "image": "image",
    "relational": "tabular",
}

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)


def load_specs(spec_dir="specs"):
    out = []
    for name in sorted(os.listdir(spec_dir)):
        if not name.endswith(".md"):
            continue
        raw = open(os.path.join(spec_dir, name)).read()
        m = FRONTMATTER.search(raw)
        if not m:
            continue
        try:
            fm = yaml.safe_load(m.group(1)) or {}
        except Exception:
            continue
        tax = fm.get("taxonomy") or {}
        task = tax.get("task")
        task = task[0] if isinstance(task, list) and task else task
        mod = tax.get("modality")
        mod = mod[0] if isinstance(mod, list) and mod else mod
        out.append({
            "spec": name,
            "objective": (m.group(2) or "").strip(),
            "true_recipe": fm.get("recipe_id"),
            "true_task": task,
            "true_modality": mod,
        })
    return out


async def evaluate(specs, registry, model):
    router = DataScienceRouter(model=model)
    rows = []
    for i, sp in enumerate(specs, 1):
        rec = dict(sp)
        try:
            res = await asyncio.wait_for(
                router.run(RouterInput(objective=sp["objective"],
                                       available_recipes=registry.get_recipes_summary())),
                timeout=180.0)
            out = res.content
            rec.update({
                "pred_recipe": out.matched_recipe_id,
                "pred_task": out.task_type,
                "pred_modality": out.data_modality,
                "confidence": out.confidence,
                "error": None,
            })
        except Exception as e:
            rec.update({"pred_recipe": None, "pred_task": None, "pred_modality": None,
                        "confidence": None, "error": str(e)[:200]})
            print(f"  [{i}/{len(specs)}] {sp['spec']}: ERROR {str(e)[:80]}", flush=True)
            rows.append(rec)
            continue

        oracle = registry.find_matches(
            {"task_type": rec["pred_task"], "data_modality": rec["pred_modality"]},
            objective=sp["objective"])
        rec["oracle_candidates"] = [r.id for r in oracle]
        rec["verdict"] = registry.classify_routing(rec["pred_recipe"], oracle)

        mark = "?"
        if rec["true_recipe"]:
            mark = "OK " if rec["pred_recipe"] == rec["true_recipe"] else "MISS"
        print(f"  [{i}/{len(specs)}] {mark} {sp['spec'][:38]:<38} "
              f"pred={str(rec['pred_recipe'])[:34]:<34} {rec['verdict']}", flush=True)
        rows.append(rec)
    return rows


def report(rows):
    print("\n" + "=" * 78)
    print("ROUTING EVALUATION")
    print("=" * 78)

    errs = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]
    print(f"specs evaluated: {len(ok)}   router errors: {len(errs)}")

    # ——— selection accuracy (specs that pin a recipe) ———
    pinned = [r for r in ok if r.get("true_recipe")]
    hits = [r for r in pinned if r["pred_recipe"] == r["true_recipe"]]
    print(f"\nSELECTION  (specs pinning a recipe: {len(pinned)})")
    print(f"  exact match: {len(hits)}/{len(pinned)} "
          f"({(len(hits)/len(pinned)*100 if pinned else 0):.1f}%)")
    near = [r for r in pinned
            if r["pred_recipe"] != r["true_recipe"] and r["true_recipe"] in (r.get("oracle_candidates") or [])]
    print(f"  missed but the oracle DID cover the true recipe: {len(near)} "
          f"(recoverable by the deterministic fallback)")

    # per-class, because the corpus is skewed
    by_task = defaultdict(lambda: [0, 0])
    for r in pinned:
        key = r.get("true_task") or "untagged"
        by_task[key][1] += 1
        if r["pred_recipe"] == r["true_recipe"]:
            by_task[key][0] += 1
    print("\n  per true task (n>=1):")
    accs = []
    for key, (h, n) in sorted(by_task.items(), key=lambda kv: -kv[1][1]):
        acc = h / n if n else 0
        accs.append(acc)
        print(f"    {key:<30} {h:>3}/{n:<3} {acc*100:5.1f}%")
    if accs:
        print(f"  MACRO-AVERAGED selection accuracy: {sum(accs)/len(accs)*100:.1f}%  "
              f"(the honest number — overall is dominated by one class)")

    # ——— classification, only where the Router CAN express the answer ———
    scorable = [r for r in ok
                if r.get("true_task") and TAXONOMY_TO_ROUTER.get(r["true_task"]) is not None]
    unscorable = [r for r in ok
                  if r.get("true_task") and TAXONOMY_TO_ROUTER.get(r["true_task"]) is None]
    t_hits = [r for r in scorable if r["pred_task"] == TAXONOMY_TO_ROUTER[r["true_task"]]]
    print(f"\nCLASSIFICATION  (task_type)")
    print(f"  scorable: {len(scorable)}   correct: {len(t_hits)} "
          f"({(len(t_hits)/len(scorable)*100 if scorable else 0):.1f}%)")
    if unscorable:
        missing = sorted({r["true_task"] for r in unscorable})
        print(f"  UNSCORABLE: {len(unscorable)} spec(s) whose true task the Router enum "
              f"cannot express — {', '.join(missing)}")
        print(f"    These recipes are unreachable by classification; only a spec pin can "
              f"select them. This is a routing DEFECT, not a model error (024 §1).")

    # ——— verdict mix ———
    print("\nVERDICTS (against the Router's own labels)")
    counts = defaultdict(int)
    for r in ok:
        counts[r.get("verdict", "?")] += 1
    for v in ("routed_consistent", "routed_off_manifest", "drafted_miss", "drafted_gap"):
        print(f"  {v:<22} {counts.get(v, 0)}")
    if counts.get("drafted_gap"):
        print("    NOTE drafted_gap here means 'a gap GIVEN THE ROUTER'S LABELS'. A "
              "misclassification produces a phantom gap; phase 3 (semantic near-miss) is "
              "what separates the two.")

    # ——— calibration ———
    conf_rows = [r for r in pinned if r.get("confidence") is not None]
    if conf_rows:
        right = [r["confidence"] for r in conf_rows if r["pred_recipe"] == r["true_recipe"]]
        wrong = [r["confidence"] for r in conf_rows if r["pred_recipe"] != r["true_recipe"]]
        print("\nCALIBRATION")
        print(f"  mean confidence when RIGHT: "
              f"{(sum(right)/len(right)) if right else float('nan'):.3f}  (n={len(right)})")
        print(f"  mean confidence when WRONG: "
              f"{(sum(wrong)/len(wrong)) if wrong else float('nan'):.3f}  (n={len(wrong)})")
        if right and wrong and (sum(right)/len(right)) - (sum(wrong)/len(wrong)) < 0.05:
            print("  ⚠ confidence barely separates right from wrong — it is not a usable "
                  "basis for a 'draft when unsure' policy.")
    print("=" * 78)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=None, help="Router model (default: resolved stage model)")
    ap.add_argument("--out", default="research/routing_eval.jsonl")
    args = ap.parse_args()

    registry = KnowledgeRegistry("src/gads/knowledge/recipes")
    specs = [s for s in load_specs() if s["true_recipe"] or s["true_task"]]
    if args.limit:
        specs = specs[:args.limit]
    model = args.model or resolve_stage_model("Router", "local_model")
    print(f"Evaluating {len(specs)} spec(s) against {len(registry.recipes)} recipes "
          f"| router model: {model}\n")

    rows = await evaluate(specs, registry, model)
    report(rows)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps({"ts": stamp, "router_model": model, **r}) + "\n")
        print(f"\nwrote {len(rows)} record(s) to {args.out}")


asyncio.run(main())
