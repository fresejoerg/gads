"""Routing quality evaluation (approach_docs/024 phase 1).

Runs ONLY the Router stage over the tagged spec corpus — one LLM call per spec, no sandbox,
no planner, no execution — and scores it against the ground truth the specs already carry:

  * selection   — `matched_recipe_id` vs the spec's pinned `recipe_id`, EXCLUDING specs
                  pinned to a pin-only research instrument (a delegation-dial arm, an AAH
                  grounding rung). Those pins encode an operator's chosen experimental
                  condition, not a routing target the objective could ever imply, so
                  scoring them as misses measures the experiment design, not the Router.
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

# ——— ONE CONTROLLED VOCABULARY ————————————————————————————————————————————————————
# There is no longer a hand-written bridge table here. The spec taxonomy, the Router's
# labels and recipe `applies_when` are all normalized through `taxonomy.yaml`
# (approach_docs/024 §1), so a number below is a routing measurement rather than a
# measurement of a translation table. `canonical_task` returns None only for a label the
# vocabulary genuinely does not contain — those specs are UNSCORABLE, not counted wrong.
from gads.core import taxonomy as tx

# Consecutive Router failures after which the run aborts rather than logging an
# outage as a result. Distinct reasons would be data; the same transport error
# repeating is a dead provider.
FAILURE_LIMIT = 5

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
        # `task` and `modality` are MANY-valued facets. Scoring only the first value
        # punishes a correct answer: amlb_segment declares `[image, tabular]` for
        # image-region features flattened into a table, and "tabular" — the right answer
        # for what the run actually reads — was being marked wrong against "image".
        def _all(v):
            return [str(x) for x in v] if isinstance(v, list) else ([str(v)] if v else [])
        tasks, mods = _all(tax.get("task")), _all(tax.get("modality"))
        out.append({
            "spec": name,
            "objective": (m.group(2) or "").strip(),
            "true_recipe": fm.get("recipe_id"),
            "true_task": tasks[0] if tasks else None,
            "true_tasks": tasks,
            "true_modality": mods[0] if mods else None,
            "true_modalities": mods,
        })
    return out


async def evaluate(specs, registry, model):
    router = DataScienceRouter(model=model)
    rows = []
    consecutive_failures = 0
    for i, sp in enumerate(specs, 1):
        rec = dict(sp)
        # One retry: the LiteLLM transport intermittently raises with an EMPTY message,
        # and a dropped connection is not a routing result. Recording it as one made six
        # specs read as misclassifications in the 2026-08-21 batch. `error` is never the
        # empty string now, so a falsy check cannot mistake a failure for a success.
        last_exc = None
        for attempt in (1, 2):
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
                last_exc = None
                consecutive_failures = 0
                break
            except Exception as e:
                last_exc = e
                if attempt == 1:
                    print(f"  [{i}/{len(specs)}] {sp['spec']}: transport error "
                          f"({type(e).__name__}: {str(e)[:60] or 'no message'}) — retrying",
                          flush=True)
                    await asyncio.sleep(3)
        if last_exc is not None:
            detail = f"{type(last_exc).__name__}: {str(last_exc)[:160] or '(no message)'}"
            rec.update({"pred_recipe": None, "pred_task": None, "pred_modality": None,
                        "confidence": None, "error": detail})
            print(f"  [{i}/{len(specs)}] {sp['spec']}: ERROR {detail[:90]}", flush=True)
            rows.append(rec)
            # CIRCUIT BREAKER. When gemini-3.7-flash went down upstream on 2026-08-21 this
            # loop spent 35 minutes on 6 specs (180s timeout x 2 attempts each) on its way
            # to a five-hour run whose output would have been all errors. A run where the
            # provider is unreachable is not a routing measurement, and pressing on turns a
            # provider outage into a corrupted batch in the ledger. Fail loudly instead.
            consecutive_failures += 1
            if consecutive_failures >= FAILURE_LIMIT:
                raise SystemExit(
                    f"\nABORTED after {consecutive_failures} consecutive failures — "
                    f"'{model}' looks unreachable, not badly calibrated. Last error: {detail}\n"
                    f"Nothing was written to the ledger. Check the provider and re-run.")
            continue

        pinned_recipe = registry.get_recipe(sp["true_recipe"]) if sp.get("true_recipe") else None
        rec["arm_variant"] = bool(pinned_recipe and registry.is_pin_only(pinned_recipe.applies_when))

        oracle = registry.find_matches(
            {"task_type": rec["pred_task"], "data_modality": rec["pred_modality"]},
            objective=sp["objective"])
        rec["oracle_candidates"] = [r.id for r in oracle]
        rec["verdict"] = registry.classify_routing(rec["pred_recipe"], oracle)

        mark = "?"
        if rec["arm_variant"]:
            mark = "ARM "
        elif rec["true_recipe"]:
            mark = "OK " if rec["pred_recipe"] == rec["true_recipe"] else "MISS"
        print(f"  [{i}/{len(specs)}] {mark} {sp['spec'][:38]:<38} "
              f"pred={str(rec['pred_recipe'])[:34]:<34} {rec['verdict']}", flush=True)
        rows.append(rec)
    return rows


def _declared(row, facet):
    """Every value a spec declared for a many-valued facet, tolerating older ledger
    rows that only carry the singular key."""
    vals = row.get(f"true_{facet}s") or []
    if not vals:
        one = row.get(f"true_{facet}")
        vals = [one] if one else []
    return [v for v in vals if v]


def report(rows):
    print("\n" + "=" * 78)
    print("ROUTING EVALUATION")
    print("=" * 78)

    errs = [r for r in rows if r.get("error") is not None]
    ok = [r for r in rows if r.get("error") is None]
    print(f"specs evaluated: {len(ok)}   router errors: {len(errs)}")

    # ——— selection accuracy (specs that pin a ROUTABLE recipe) ———
    arms = [r for r in ok if r.get("arm_variant")]
    pinned = [r for r in ok if r.get("true_recipe") and not r.get("arm_variant")]
    hits = [r for r in pinned if r["pred_recipe"] == r["true_recipe"]]
    print(f"\nSELECTION  (specs pinning a routable recipe: {len(pinned)})")
    if arms:
        print(f"  excluded: {len(arms)} spec(s) pinned to a pin-only research instrument "
              f"(dial arm / AAH rung) — the pin is an experimental condition, not a "
              f"routing target")
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

    # ——— classification, compared through the one controlled vocabulary ———
    scorable = [r for r in ok if tx.canonical_task(r.get("true_task"))]
    unscorable = [r for r in ok if r.get("true_task") and not tx.canonical_task(r["true_task"])]
    exact_specs = {r["spec"] for r in scorable
                   if any(tx.canonical_task(r["pred_task"]) == tx.canonical_task(t)
                          for t in _declared(r, "task"))}
    exact = [r for r in scorable if r["spec"] in exact_specs]
    family = [r for r in scorable if r["spec"] not in exact_specs
              and any(tx.tasks_overlap(r["pred_task"], t) for t in _declared(r, "task"))]
    family_specs = {r["spec"] for r in family}
    print(f"\nCLASSIFICATION  (task_type, compared canonically)")
    print(f"  scorable: {len(scorable)}   exact: {len(exact)} "
          f"({(len(exact)/len(scorable)*100 if scorable else 0):.1f}%)   "
          f"right family, wrong subtype: {len(family)}")
    wrong = [r for r in scorable if r["spec"] not in exact_specs | family_specs]
    for r in wrong[:12]:
        print(f"    MISLABEL {r['spec'][:38]:<38} "
              f"true={'|'.join(_declared(r, 'task'))[:28]:<28} pred={str(r['pred_task'])[:26]}")
    if unscorable:
        missing = sorted({r["true_task"] for r in unscorable})
        print(f"  UNSCORABLE: {len(unscorable)} spec(s) whose true task is not in the "
              f"vocabulary at all — {', '.join(missing)}")
        print(f"    Fix taxonomy.yaml; do not count these against the model.")

    # modality, same treatment
    m_scorable = [r for r in ok if tx.canonical_modality(r.get("true_modality"))]
    m_hit_specs = {r["spec"] for r in m_scorable
                   if any(tx.canonical_modality(r["pred_modality"]) == tx.canonical_modality(m)
                          for m in _declared(r, "modality"))}
    m_hits = [r for r in m_scorable if r["spec"] in m_hit_specs]
    print(f"\nCLASSIFICATION  (data_modality)")
    print(f"  scorable: {len(m_scorable)}   correct: {len(m_hits)} "
          f"({(len(m_hits)/len(m_scorable)*100 if m_scorable else 0):.1f}%)")
    for r in m_scorable:
        if r["spec"] not in m_hit_specs:
            print(f"    MISLABEL {r['spec'][:38]:<38} "
                  f"true={'|'.join(_declared(r, 'modality'))[:14]:<14} "
                  f"pred={str(r['pred_modality'])[:14]}")

    # ——— verdict mix ———
    print("\nVERDICTS (against the Router's own labels)")
    counts = defaultdict(int)
    for r in ok:
        counts[r.get("verdict", "?")] += 1
    for v in ("routed_consistent", "routed_off_manifest", "drafted_miss", "drafted_gap"):
        print(f"  {v:<22} {counts.get(v, 0)}")
    if counts.get("drafted_gap"):
        print("    NOTE drafted_gap still means 'a gap GIVEN THE ROUTER'S LABELS' — a "
              "misclassification produces a phantom gap. Cross-check it against the "
              "CLASSIFICATION block above: a gap under a label that block scored correct "
              "is a real gap.")

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
    ap.add_argument("--tag", default=None,
                    help="Stamped on every row. Use it to distinguish arms of one "
                         "experiment (e.g. pre_vocab / post_vocab) — router_model alone "
                         "cannot tell two code versions apart.")
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
                f.write(json.dumps({"ts": stamp, "router_model": model,
                                    "tag": args.tag, **r}) + "\n")
        print(f"\nwrote {len(rows)} record(s) to {args.out}")


asyncio.run(main())
