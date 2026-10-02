"""Analytic reference for m4_hourly_forecast_v1 — expected values computed independently of
GADS (approach_docs/033 §3a, `provenance: analytic`). Same structure as
air_passengers_v1/reference.py.

Protocol: the one GADS's forecasting recipe applies, so the numbers are comparable:
  * horizon h = the recipe's rule, ~10% of the median series length, min 1 → 70 for 700 hours
  * holdout   = the last h observations of EACH series; everything before is the fit window
  * MASE      = per series, holdout MAE / in-sample MAE of the seasonal-naive predictor
                (m = 24) on that series' fit window, then the mean over series. This is
                AutoGluon's eval_metric="MASE" for a panel with equal horizons; the
                seasonal-naive figure matches the SeasonalNaive row of AutoGluon's own
                leaderboard on the M4 cloud run (2.0187).
  * sMAPE     = mean over all holdout points of 2|y - ŷ| / (|y| + |ŷ|) × 100

Reference models are textbook defaults, fitted per series without tuning:
seasonal_naive, naive, auto_ets (AutoETS, season 24) and auto_theta (AutoTheta, season 24).
AutoARIMA with a 24-hour season on 50 series is left out because of its runtime.

Imports nothing from GADS. Usage (pinned to the sandbox's versions):
    uv run --no-project --with statsforecast==2.0.1 \\
        python research/benchmarks/m4_hourly_forecast_v1/reference.py [--write]
`--write` updates expected.json's `reference` block. Pass thresholds in `metrics` are set by
hand from it and justified in notes.md.
"""
import argparse
import json
import os
import platform

import numpy as np
import pandas as pd

DATA = os.path.join(os.environ.get("GADS_DATASETS_ROOT", "/home/joergf/datasets"),
                    "m4", "m4_hourly_train.csv")
HERE = os.path.dirname(os.path.abspath(__file__))
M = 24


def horizon(lengths) -> int:
    return max(1, int(round(0.10 * float(np.median(lengths)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="update expected.json's reference block")
    args = ap.parse_args()

    df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values(["item_id", "timestamp"])
    groups = {k: g["target"].to_numpy(dtype=float) for k, g in df.groupby("item_id", sort=True)}
    h = horizon([len(v) for v in groups.values()])

    import statsforecast
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, AutoTheta

    # Drop each series' last h rows (pandas-3 safe: groupby.apply no longer keeps the key column).
    fit = df[df.groupby("item_id").cumcount(ascending=False) >= h]
    sf = StatsForecast(models=[AutoETS(season_length=M), AutoTheta(season_length=M)], freq="h", n_jobs=1)
    fc = sf.forecast(df=fit.rename(columns={"item_id": "unique_id", "timestamp": "ds", "target": "y"}), h=h)

    per_model = {k: {"mase": [], "smape_terms": []} for k in ("seasonal_naive", "naive", "auto_ets", "auto_theta")}
    for sid, y in groups.items():
        y_fit, y_test = y[:-h], y[-h:]
        scale = np.mean(np.abs(y_fit[M:] - y_fit[:-M]))
        f = fc[fc["unique_id"] == sid]
        preds = {
            "seasonal_naive": np.array([y_fit[len(y_fit) - M + (i % M)] for i in range(h)]),
            "naive": np.repeat(y_fit[-1], h),
            "auto_ets": f["AutoETS"].to_numpy(),
            "auto_theta": f["AutoTheta"].to_numpy(),
        }
        for k, p in preds.items():
            per_model[k]["mase"].append(np.mean(np.abs(y_test - p)) / scale)
            denom = np.abs(y_test) + np.abs(p)
            per_model[k]["smape_terms"].extend(np.where(denom > 0, 2 * np.abs(y_test - p) / denom, 0.0))

    models = {k: {"mase": round(float(np.mean(v["mase"])), 4),
                  "smape": round(float(np.mean(v["smape_terms"]) * 100), 3)} for k, v in per_model.items()}
    reference = {
        "protocol": {"n_series": len(groups), "n_obs_per_series": int(np.median([len(v) for v in groups.values()])),
                     "horizon": h, "season_length": M, "aggregation": "mean of per-series MASE"},
        "models": models,
        "best_classical": min((k for k in models if k not in ("seasonal_naive", "naive")),
                              key=lambda k: models[k]["mase"]),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "statsforecast": statsforecast.__version__},
    }
    print(json.dumps(reference, indent=1))

    if args.write:
        path = os.path.join(HERE, "expected.json")
        with open(path) as f:
            exp = json.load(f)
        exp["reference"] = reference
        with open(path, "w") as f:
            json.dump(exp, f, indent=1)
            f.write("\n")
        print(f"updated reference block in {path}")


if __name__ == "__main__":
    main()
