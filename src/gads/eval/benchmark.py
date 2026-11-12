"""Benchmark checks: a finished workspace against a benchmark's `expected.json`.

Moved here from scripts/score_benchmark.py (which is now a thin CLI wrapper with identical
output) so the per-run scorecard (gads.eval.scorecard) and the benchmark scorer share one
implementation. Stdlib only, reads the workspace directory, no DB or backend required.

Each check returns a list of `(name, ok, detail)` tuples, the shape score_benchmark.py has
always printed and appended to results.jsonl.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

Check = Tuple[str, bool, str]

BENCHMARKS_DIR = Path("research/benchmarks")


def check_metrics(expected: dict, workspace: Path) -> List[Check]:
    results: List[Check] = []
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
        comparator = spec.get("comparator")
        if comparator:
            # Threshold checks (e.g. "must beat the seasonal-naive MASE"): `value` is the bound.
            ops = {"lt": (float.__lt__, "<"), "le": (float.__le__, "<="),
                   "gt": (float.__gt__, ">"), "ge": (float.__ge__, ">=")}
            if comparator not in ops:
                results.append((f"metric {name}", False, f"unknown comparator {comparator!r}"))
                continue
            op, sym = ops[comparator]
            ok = op(float(got), float(want))
            detail = f"got {got}, want {sym} {want}"
        elif spec.get("exact"):
            ok = got == want
            detail = f"got {got!r}, want exactly {want!r}"
        else:
            tol = spec.get("tol") or 0.0
            ok = abs(float(got) - float(want)) <= tol
            detail = f"got {got}, want {want} ± {tol}"
        results.append((f"metric {name}", ok, detail))
    return results


def check_artifacts(expected: dict, workspace: Path) -> List[Check]:
    results: List[Check] = []
    for f in expected.get("artifacts", {}).get("required", []):
        results.append((f"artifact {f}", (workspace / f).exists(), "present" if (workspace / f).exists() else "MISSING"))
    return results


def check_methodology(expected: dict, workspace: Path) -> List[Check]:
    results: List[Check] = []
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


def score_sections(expected: dict, workspace: Path, metrics_only: bool = False) -> dict:
    """The three benchmark dimensions, keyed as score_benchmark.py has always keyed them."""
    sections = {"reproducibility": check_metrics(expected, workspace)}
    if not metrics_only:
        sections["completeness"] = check_artifacts(expected, workspace)
        sections["methodology"] = check_methodology(expected, workspace)
    return sections


def find_benchmark(spec_filename: Optional[str], root: Path = BENCHMARKS_DIR) -> Optional[Path]:
    """The benchmark directory whose expected.json names this spec, or None.

    Matched on `expected.json["spec_file"]`, the spec filename a run records in
    `last_state_json["spec_filename"]`. Every benchmark names a distinct spec (checked
    2026-11-11), so the match is unique; if that ever stops holding, this returns None
    rather than guessing.
    """
    if not spec_filename or not root.is_dir():
        return None
    hits = []
    for exp in sorted(root.glob("*/expected.json")):
        try:
            if json.loads(exp.read_text()).get("spec_file") == spec_filename:
                hits.append(exp.parent)
        except Exception:
            continue
    return hits[0] if len(hits) == 1 else None
