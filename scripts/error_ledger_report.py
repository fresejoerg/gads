#!/usr/bin/env python3
"""
Recipe-hardening report over the cross-run error ledger (research/error_ledger.jsonl).

Ranks, per recipe step, the structural errors seen most often across ALL runs — the signal
for where a recipe's intent or an attached skill needs tightening. This is the async,
human-in-the-loop end of the loop: the ledger observes failures during execution; this
report surfaces the patterns worth hardening (exactly how the .iloc-on-ndarray guidance was
added by hand). Read-only.

Usage:
    PYTHONPATH=src uv run python scripts/error_ledger_report.py
    PYTHONPATH=src uv run python scripts/error_ledger_report.py --recipe binary_classification.tabular.standard
"""
import argparse
from collections import defaultdict

from gads.core.error_ledger import aggregate, LEDGER_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", help="filter to one recipe_id")
    ap.add_argument("--min-count", type=int, default=1, help="hide reasons below this count")
    args = ap.parse_args()

    err, res, samples, intents = aggregate()
    if not err:
        print(f"No structural errors recorded yet in {LEDGER_PATH}.")
        return

    # group reasons under each (recipe_id, node)
    by_step = defaultdict(list)
    for (key, reason), count in err.items():
        by_step[key].append((count, reason))

    steps = sorted(by_step.items(), key=lambda kv: -sum(c for c, _ in kv[1]))
    print(f"=== GADS recipe-hardening report ({LEDGER_PATH}) ===\n")
    for (recipe_id, node), reasons in steps:
        if args.recipe and recipe_id != args.recipe:
            continue
        total = sum(c for c, _ in reasons)
        resolved = res.get((recipe_id, node), 0)
        intent = intents.get((recipe_id, node), "")
        print(f"▶ {recipe_id}  ·  step: {node}")
        if intent:
            print(f"    intent: {intent[:110]}")
        print(f"    {total} structural failure(s) across runs; {resolved} eventually recovered")
        for count, reason in sorted(reasons, reverse=True):
            if count < args.min_count:
                continue
            sample = samples.get(((recipe_id, node), reason), "")
            flag = "  ⚠ HARDEN" if count >= 3 and resolved == 0 else ""
            print(f"      {count:>3}×  {reason}{flag}")
            if sample:
                print(f"           e.g. {sample[:120]}")
        print()


if __name__ == "__main__":
    main()
