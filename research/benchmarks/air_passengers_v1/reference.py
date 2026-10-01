"""Analytic reference for air_passengers_v1 — computes the benchmark's expected values
independently of GADS (approach_docs/033 §3a, `provenance: analytic`).

Protocol: the same one GADS's forecasting recipe applies, so the numbers are comparable:
  * horizon h = the recipe's rule, ~10% of the series length, min 1  → 14 for 144 months
  * holdout   = the last h observations; everything before is the fit window
  * MASE      = holdout MAE / in-sample MAE of the seasonal-naive predictor (m = 12) on the
                fit window. This is what AutoGluon's eval_metric="MASE" reports for monthly
                data, and why "MASE < 1" is read as "beats seasonal-naive"
  * sMAPE     = mean(2|y - ŷ| / (|y| + |ŷ|)) × 100

Reference models are textbook methods with no tuning: they set a competent baseline, they
are not trying to win.
  * seasonal_naive   ŷ(t) = y(t − 12)
  * naive            ŷ(t) = last observed value (random walk)
  * auto_ets         statsforecast AutoETS(season_length=12)
  * auto_arima       statsforecast AutoARIMA(season_length=12)
  * airline_log      SARIMA(0,1,1)(0,1,1)12 on log(y), back-transformed (Box & Jenkins 1976)

Deliberately imports nothing from GADS (recipes, natives, skills): a bug shared by GADS and
its own reference would pass silently.

Usage (pinned to the sandbox's versions, isolated from both GADS and the sandbox):
    uv run --no-project --with statsforecast==2.0.1 --with statsmodels==0.14.6 \\
        python research/benchmarks/air_passengers_v1/reference.py [--write]
`--write` updates expected.json's `reference` block. The pass thresholds in `metrics` are
derived from it by hand and justified in notes.md, never rewritten automatically.
"""
import argparse
import json
import os
import platform

import numpy as np
import pandas as pd

DATA = os.path.join(os.environ.get("GADS_DATASETS_ROOT", "/home/joergf/datasets"),
                    "airpassengers", "air_passengers.csv")
HERE = os.path.dirname(os.path.abspath(__file__))
M = 12


def horizon(n: int) -> int:
    return max(1, int(round(0.10 * n)))


def mase(y_true, y_pred, y_fit):
    scale = np.mean(np.abs(y_fit[M:] - y_fit[:-M]))
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def smape(y_true, y_pred):
    return float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="update expected.json's reference block")
    args = ap.parse_args()

    df = pd.read_csv(DATA, parse_dates=["month"])
    y = df["passengers"].to_numpy(dtype=float)
    n, h = len(y), horizon(len(y))
    y_fit, y_test = y[:-h], y[-h:]

    import statsforecast
    import statsmodels
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    preds = {
        "seasonal_naive": np.array([y_fit[len(y_fit) - M + (i % M)] for i in range(h)]),
        "naive": np.repeat(y_fit[-1], h),
    }
    sf_df = pd.DataFrame({"unique_id": "air", "ds": df["month"].iloc[:-h], "y": y_fit})
    sf = StatsForecast(models=[AutoETS(season_length=M), AutoARIMA(season_length=M)], freq="MS", n_jobs=1)
    fc = sf.forecast(df=sf_df, h=h)
    preds["auto_ets"] = fc["AutoETS"].to_numpy()
    preds["auto_arima"] = fc["AutoARIMA"].to_numpy()
    airline = SARIMAX(np.log(y_fit), order=(0, 1, 1), seasonal_order=(0, 1, 1, M)).fit(disp=False)
    preds["airline_log"] = np.exp(airline.forecast(h))

    models = {k: {"mase": round(mase(y_test, p, y_fit), 4), "smape": round(smape(y_test, p), 3)}
              for k, p in preds.items()}
    reference = {
        "protocol": {"n_obs": n, "horizon": h, "season_length": M,
                     "fit_window": f"{df['month'].iloc[0]:%Y-%m} … {df['month'].iloc[-h - 1]:%Y-%m}",
                     "holdout": f"{df['month'].iloc[-h]:%Y-%m} … {df['month'].iloc[-1]:%Y-%m}"},
        "models": models,
        "best_classical": min((k for k in models if k not in ("seasonal_naive", "naive")),
                              key=lambda k: models[k]["mase"]),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "statsforecast": statsforecast.__version__,
                     "statsmodels": statsmodels.__version__},
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
