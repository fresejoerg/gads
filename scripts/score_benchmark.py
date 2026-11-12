#!/usr/bin/env python3
"""Score a finished GADS workspace against a benchmark's expected results.

Quantifies run quality on the two research metrics (research/JOURNAL.md):
reproducibility (metric values vs. canonical), completeness (artifact manifest)
and methodological appropriateness (code-level pattern checks). Stdlib only;
reads the workspace directory — no DB or backend required. The checks live in
gads.eval.benchmark, shared with the per-run scorecard (scripts/eval_run.py).

Usage:
    python scripts/score_benchmark.py \
        --benchmark research/benchmarks/fraud_autogluon_v1 \
        --workspace /path/to/workspaces/<project_id> \
        [--mode local] [--json out.jsonl]

Exit code 0 iff every check passes.
"""
import argparse
import json
import sys
from pathlib import Path

# The checks live in gads.eval.benchmark (shared with the per-run scorecard); this script is
# the CLI wrapper and keeps its historical output and results.jsonl format unchanged.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gads.eval.benchmark import score_sections  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", required=True, help="benchmark directory containing expected.json")
    ap.add_argument("--workspace", required=True, help="finished project workspace directory")
    ap.add_argument("--mode", default=None, help="routing mode label for the JSON record")
    ap.add_argument("--json", default=None, help="append a machine-readable result line to this file")
    ap.add_argument(
        "--metrics-only", action="store_true",
        help="grade on the metric golds alone, skipping artifact and methodology checks. "
             "For cross-API-surface comparisons (approach_docs/014): the same estimand is "
             "realized through different libraries, so a methodology block naming one of them "
             "(e.g. requires gads_causal_estimate_ate, forbids CausalModel) cannot be held "
             "constant across arms and would grade the API surface rather than the answer.",
    )
    args = ap.parse_args()

    bench = Path(args.benchmark)
    workspace = Path(args.workspace)
    expected = json.loads((bench / "expected.json").read_text())

    sections = score_sections(expected, workspace, metrics_only=args.metrics_only)

    total = passed = 0
    print(f"\n═══ {expected['benchmark_id']} · workspace {workspace.name[:8]}"
          + (f" · mode={args.mode}" if args.mode else "") + " ═══")
    for dim, checks in sections.items():
        n_ok = sum(1 for _, ok, _ in checks if ok)
        print(f"\n{dim.upper()}  [{n_ok}/{len(checks)}]")
        for name, ok, detail in checks:
            print(f"  {'✓' if ok else '✗'} {name}: {detail}")
        total += len(checks)
        passed += n_ok

    verdict = "PASS" if passed == total else "FAIL"
    print(f"\n═══ {verdict}: {passed}/{total} checks ═══\n")

    if args.json:
        record = {
            "benchmark_id": expected["benchmark_id"],
            "workspace": workspace.name,
            "mode": args.mode,
            "metrics_only": args.metrics_only,
            "passed": passed,
            "total": total,
            "verdict": verdict,
            "dimensions": {d: [{"check": n, "ok": ok, "detail": det} for n, ok, det in cs]
                           for d, cs in sections.items()},
        }
        with open(args.json, "a") as f:
            f.write(json.dumps(record) + "\n")

    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
