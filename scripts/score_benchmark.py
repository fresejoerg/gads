#!/usr/bin/env python3
"""Score a finished GADS workspace against a benchmark's expected results.

Quantifies run quality on the two research metrics (research/JOURNAL.md):
reproducibility (metric values vs. canonical), completeness (artifact manifest)
and methodological appropriateness (code-level pattern checks). Stdlib only;
reads the workspace directory — no DB or backend required.

Usage:
    python scripts/score_benchmark.py \
        --benchmark research/benchmarks/fraud_autogluon_v1 \
        --workspace /path/to/workspaces/<project_id> \
        [--mode local] [--json out.jsonl]

Exit code 0 iff every check passes.
"""
import argparse
import json
import re
import sys
from pathlib import Path


def check_metrics(expected: dict, workspace: Path):
    results = []
    mfile = workspace / "metrics.json"
    observed = {}
    if mfile.exists():
        try:
            observed = json.loads(mfile.read_text())
        except Exception as e:
            results.append(("metrics.json parse", False, str(e)))
            return results
    else:
        results.append(("metrics.json exists", False, "file missing"))
        return results

    for name, spec in expected.get("metrics", {}).items():
        if name not in observed:
            results.append((f"metric {name}", False, "not in metrics.json"))
            continue
        got, want = observed[name], spec["value"]
        if spec.get("exact"):
            ok = got == want
            detail = f"got {got!r}, want exactly {want!r}"
        else:
            tol = spec.get("tol", 0.0)
            ok = abs(float(got) - float(want)) <= tol
            detail = f"got {got}, want {want} ± {tol}"
        results.append((f"metric {name}", ok, detail))
    return results


def check_artifacts(expected: dict, workspace: Path):
    results = []
    for f in expected.get("artifacts", {}).get("required", []):
        results.append((f"artifact {f}", (workspace / f).exists(), "present" if (workspace / f).exists() else "MISSING"))
    return results


def check_methodology(expected: dict, workspace: Path):
    results = []
    meth = expected.get("methodology", {})
    code_file = workspace / "workflow_execution.py"
    if not code_file.exists():
        results.append(("workflow_execution.py exists", False, "file missing — cannot check methodology"))
        return results
    code = code_file.read_text(errors="replace")
    for pat in meth.get("required_patterns", []):
        ok = re.search(pat, code) is not None
        results.append((f"code has /{pat}/", ok, "found" if ok else "NOT FOUND"))
    for pat in meth.get("forbidden_patterns", []):
        m = re.search(pat, code)
        results.append((f"code lacks /{pat}/", m is None, "clean" if m is None else f"VIOLATION: {m.group(0)[:60]!r}"))
    return results


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

    sections = {"reproducibility": check_metrics(expected, workspace)}
    if not args.metrics_only:
        sections["completeness"] = check_artifacts(expected, workspace)
        sections["methodology"] = check_methodology(expected, workspace)

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
