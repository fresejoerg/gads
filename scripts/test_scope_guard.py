"""Regression guard for the executor's scope gate (core/executor._scope_violations) — no LLM calls.

The gate rejects a node's generated code before execution if it binds, at module level, a
variable that a LATER node of a recipe plan produces. Two layers:

1. Synthetic cases (always run): what must be flagged, and what must not (function-local
   names, attribute/subscript writes, names the node owns).
2. Replay (runs when the gitignored distillation capture is present): the node-1 attempts of
   the 2026-10-02 forecasting runs. All local attempts must be flagged and all cloud attempts
   must pass. This is the measurement that motivated the gate.

    PYTHONPATH=src uv run python scripts/test_scope_guard.py
"""
import glob
import json
import sys

sys.path.insert(0, "src")

from gads.core.executor import _scope_violations  # noqa: E402

FORECAST_DOWNSTREAM = {n: "a later step" for n in
                       ("ts_df", "predictor_ts", "prediction_length", "best_model_mase",
                        "seasonal_naive_mase", "forecasts")}
failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  — ' + detail) if detail and not ok else ''}")
    if not ok:
        failures.append(label)


print("\n1. synthetic cases")
cases = [
    ("assign downstream name", "predictor_ts = 1", {"predictor_ts"}),
    ("tuple unpack", "ts_df, x = f()", {"ts_df"}),
    ("augassign", "best_model_mase += 1", {"best_model_mase"}),
    ("for target", "for forecasts in []: pass", {"forecasts"}),
    ("with target", "with open('f') as ts_df: pass", {"ts_df"}),
    ("walrus", "(prediction_length := 14)", {"prediction_length"}),
    ("nested in if", "if True:\n    predictor_ts = 2", {"predictor_ts"}),
    ("function-local is fine", "def f():\n    predictor_ts = 1\n    return predictor_ts", set()),
    ("class body is fine", "class A:\n    ts_df = None", set()),
    ("attribute write is fine", "obj.predictor_ts = 1", set()),
    ("subscript write is fine", "d['ts_df'] = 1", set()),
    ("reading is fine", "print(best_model_mase)", set()),
    ("own outputs are fine", "timestamp_col = 'month'\ninferred_freq = 'MS'", set()),
    ("syntax error is not a scope verdict", "def (:", set()),
]
for label, code, want in cases:
    got = set(_scope_violations(code, FORECAST_DOWNSTREAM))
    check(label, got == want, f"got {sorted(got)}, want {sorted(want)}")
check("empty forbidden set never flags", _scope_violations("predictor_ts = 1", {}) == {})

print("\n2. replay of captured node-1 attempts (forecasting, 2026-10-02)")
runs = {"77654023": "cloud", "15af1437": "cloud", "fea39645": "local", "8abde526": "local"}
rows = []
for path in glob.glob("research/finetune/capture/attempts-*.jsonl"):
    with open(path) as f:
        rows += [json.loads(line) for line in f
                 if any(p in line for p in runs) and '"Profile the time-series' in line]
if not rows:
    print("  SKIP  no distillation capture on this machine")
else:
    for kind in ("cloud", "local"):
        mine = [r for r in rows if runs.get(r.get("project_id", "")[:8]) == kind
                and r.get("outcome") != "no_program" and r.get("executed_code")]
        flagged = [r for r in mine if _scope_violations(r["executed_code"], FORECAST_DOWNSTREAM)]
        if kind == "cloud":
            check(f"cloud attempts pass ({len(mine)})", mine and not flagged,
                  f"{len(flagged)}/{len(mine)} flagged")
        else:
            check(f"local attempts flagged ({len(mine)})", mine and len(flagged) == len(mine),
                  f"{len(flagged)}/{len(mine)} flagged")

print("\n" + "=" * 70)
print(f"FAILED ({len(failures)})" if failures else "All scope-guard checks passed.")
sys.exit(1 if failures else 0)
