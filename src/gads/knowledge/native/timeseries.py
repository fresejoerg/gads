"""Time-series forecasting natives: profiling, scoring and the AutoGluon fit/predict wrappers.

Why these are natives (approach_docs/019's rule: nativize invariant, single-right-answer
operations): every one of them failed *mechanically* in local runs on 2026-10-02:
  * profiling: frequency inference with invalid offset strings (`pd.Timedelta('D')`),
    `.abs()` on a scalar when computing `naive_mae`. These are 5 of 9 failed attempts.
  * scoring: reading the SeasonalNaive row with `leaderboard.loc['SeasonalNaive', ...]`
    (the leaderboard has a plain integer index), and forgetting that AutoGluon negates
    error metrics, so MASE = -score_val.
None of that is a judgment call. What stays model-written is the actual modelling work:
the long-format conversion, the fit call, the forecast and its plot, and the narrative.

The old AutoGluon time-series helpers lived as string literals with raising stubs in ml.py,
so NATIVE_SOURCE exported the STUB to the fallback path. They are real functions here now.

IMPORTANT: each function is injected verbatim via inspect.getsource, and the fallback path
injects exactly ONE function. Keep every function self-contained: imports inside, no
annotations, no calls to siblings.
"""


def gads_profile_timeseries(df, target_col=None, timestamp_col=None, item_id_col=None):
    """Profile a long-format time-series frame. Returns a dict; the input df is not mutated.

    Keys:
      df             copy of df with the timestamp parsed and, for a single series, a
                     constant `item_id` column added. Reassign it: `df = prof["df"]`
      timestamp_col  the datetime column (given, else the first datetime-typed or
                     date/time-named column)
      target_col     the value to forecast (given, else the single numeric non-id column)
      item_id_col    the series identifier (given, else a low-cardinality non-numeric
                     column, else the added constant `item_id`)
      inferred_freq  pandas offset alias, e.g. 'h', 'D', 'W-SUN', 'MS'. pd.infer_freq on the
                     longest series, else mapped from the median timestamp delta
      season_length  the usual seasonal period for that frequency (h→24, D→7, W→52, M→12,
                     Q→4, Y→1)
      n_series, n_rows, min_len, median_len, max_len
      naive_mae      mean absolute deviation of the target from its global mean. This is
                     the recipe's descriptive baseline. It is NOT the MASE yardstick: use
                     gads_forecast_scores for that
    """
    import numpy as np
    import pandas as pd

    out = df.copy()
    cols = list(out.columns)

    if timestamp_col is None:
        timestamp_col = next((c for c in cols if pd.api.types.is_datetime64_any_dtype(out[c])), None)
    if timestamp_col is None:
        # Whole-token match on the column name: a substring match would take `deposits` for `ds`.
        time_tokens = {"timestamp", "date", "datetime", "time", "month", "week", "day",
                       "period", "ds", "dt", "year", "hour"}
        def _tokens(name):
            return set("".join(ch if ch.isalnum() else " " for ch in str(name).lower()).split())
        timestamp_col = next((c for c in cols if _tokens(c) & time_tokens), None)
    if timestamp_col is None:
        raise ValueError(f"gads_profile_timeseries: no timestamp column found among {cols}")
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])

    if item_id_col is None:
        candidates = [c for c in cols if c not in (timestamp_col, target_col)
                      and not pd.api.types.is_numeric_dtype(out[c])
                      and 1 < out[c].nunique() <= max(1, len(out) // 2)]
        item_id_col = candidates[0] if candidates else None
    if item_id_col is None:
        item_id_col = "item_id"
        out[item_id_col] = "series_1"

    if target_col is None:
        numeric = [c for c in cols if c not in (timestamp_col, item_id_col)
                   and pd.api.types.is_numeric_dtype(out[c])]
        if len(numeric) != 1:
            raise ValueError(f"gads_profile_timeseries: pass target_col explicitly; numeric candidates: {numeric}")
        target_col = numeric[0]

    out = out.sort_values([item_id_col, timestamp_col]).reset_index(drop=True)
    lengths = out.groupby(item_id_col).size()
    longest = lengths.idxmax()
    ts = out.loc[out[item_id_col] == longest, timestamp_col]

    inferred_freq = None
    try:
        inferred_freq = pd.infer_freq(pd.DatetimeIndex(ts))
    except Exception:
        inferred_freq = None
    if inferred_freq is None:
        hours = ts.diff().dropna().median() / pd.Timedelta(hours=1)
        for limit, alias in ((1.01, "h"), (24.5, "D"), (7 * 24.5, "W"), (32 * 24, "MS"),
                             (93 * 24, "QS"), (367 * 24, "YS")):
            if hours <= limit:
                inferred_freq = alias
                break
        if inferred_freq is None:
            inferred_freq = "YS"
    base = "".join(ch for ch in str(inferred_freq).split("-")[0] if ch.isalpha()).upper()
    season_length = {"H": 24, "D": 7, "B": 5, "W": 52, "M": 12, "MS": 12, "ME": 12,
                     "Q": 4, "QS": 4, "QE": 4, "Y": 1, "YS": 1, "YE": 1, "A": 1, "AS": 1}.get(base, 1)

    y = out[target_col].astype(float)
    naive_mae = float(np.mean(np.abs(y - y.mean())))
    prof = {
        "df": out, "timestamp_col": timestamp_col, "target_col": target_col,
        "item_id_col": item_id_col, "inferred_freq": str(inferred_freq),
        "season_length": int(season_length), "n_series": int(lengths.size), "n_rows": int(len(out)),
        "min_len": int(lengths.min()), "median_len": float(lengths.median()), "max_len": int(lengths.max()),
        "naive_mae": naive_mae,
    }
    print(f"[gads_profile_timeseries] {prof['n_series']} series, {prof['n_rows']} rows, "
          f"length min/median/max {prof['min_len']}/{prof['median_len']:.0f}/{prof['max_len']}")
    print(f"[gads_profile_timeseries] timestamp={timestamp_col!r} target={target_col!r} "
          f"item_id={item_id_col!r} freq={prof['inferred_freq']} season_length={season_length} "
          f"naive_mae={naive_mae:.4f}")
    return prof


def gads_forecast_scores(predictor_ts, ts_df):
    """Validation scores of a fitted TimeSeriesPredictor, with the baseline it must beat.

    AutoGluon negates error metrics ("higher is better"), so MASE = -score_val. The
    leaderboard has a plain integer index with model names in the `model` column. Both
    facts tripped generated code, which is why this is a native.

    Returns: best_model, best_model_mase, seasonal_naive_mase (NaN if no SeasonalNaive row),
    beats_seasonal_naive, leaderboard (DataFrame, sorted best first, with a positive `mase`
    column).

    The correct test of "the model earns its keep" is best_model_mase < seasonal_naive_mase
    on the SAME window. It is NOT MASE < 1: MASE is scaled by the in-sample seasonal-naive
    error, and on trending data seasonal-naive itself scores > 1 out of sample.
    """
    lb = predictor_ts.leaderboard(ts_df, silent=True).copy()
    lb["mase"] = -lb["score_val"].astype(float)
    lb = lb.sort_values("mase").reset_index(drop=True)
    best_model = str(lb.loc[0, "model"])
    best_model_mase = float(lb.loc[0, "mase"])
    sn = lb[lb["model"] == "SeasonalNaive"]
    seasonal_naive_mase = float(sn["mase"].iloc[0]) if len(sn) else float("nan")
    beats = bool(best_model_mase < seasonal_naive_mase) if len(sn) else None
    print(f"[gads_forecast_scores] best={best_model} MASE={best_model_mase:.4f} | "
          f"SeasonalNaive MASE={seasonal_naive_mase:.4f} on the same window | "
          f"beats seasonal-naive: {beats}")
    print(lb[["model", "mase"]].to_string(index=False))
    return {"best_model": best_model, "best_model_mase": best_model_mase,
            "seasonal_naive_mase": seasonal_naive_mase, "beats_seasonal_naive": beats,
            "leaderboard": lb}


def gads_timeseries_fit(df, target_col, timestamp_col, item_id_col=None,
                        prediction_length=None, time_limit=120, presets="fast_training"):
    """Train an AutoGluon TimeSeriesPredictor end to end (conversion + fit + scores + persist).

    prediction_length defaults to the recipe rule: ~10% of the median series length, min 1.
    Returns: predictor_ts, ts_df, prediction_length, best_model, best_model_mase,
    seasonal_naive_mase, leaderboard_ts. MASE values are POSITIVE (AutoGluon's sign undone).
    """
    import joblib
    import pandas as pd
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    data = df.copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    if item_id_col is None or item_id_col not in data.columns:
        data["item_id"] = "series_1"
        item_id_col = "item_id"
    ts_df = TimeSeriesDataFrame.from_data_frame(
        data[[item_id_col, timestamp_col, target_col]], id_column=item_id_col, timestamp_column=timestamp_col)
    if prediction_length is None:
        median_len = float(ts_df.groupby(level=0).size().median())
        prediction_length = max(1, int(round(0.10 * median_len)))
        print(f"[gads_timeseries_fit] prediction_length={prediction_length} (~10% of median length {median_len:.0f})")

    predictor_ts = TimeSeriesPredictor(prediction_length=prediction_length, target=target_col,
                                       eval_metric="MASE", verbosity=0
                                       ).fit(ts_df, presets=presets, time_limit=time_limit)
    lb = predictor_ts.leaderboard(ts_df, silent=True).copy()
    lb["mase"] = -lb["score_val"].astype(float)
    lb = lb.sort_values("mase").reset_index(drop=True)
    sn = lb[lb["model"] == "SeasonalNaive"]
    best_model_mase = float(lb.loc[0, "mase"])
    seasonal_naive_mase = float(sn["mase"].iloc[0]) if len(sn) else float("nan")
    print(f"[gads_timeseries_fit] best={lb.loc[0, 'model']} MASE={best_model_mase:.4f} | "
          f"SeasonalNaive MASE={seasonal_naive_mase:.4f} on the same window")
    joblib.dump(predictor_ts, "model_timeseries.joblib")
    return {"predictor_ts": predictor_ts, "ts_df": ts_df, "prediction_length": int(prediction_length),
            "best_model": str(lb.loc[0, "model"]), "best_model_mase": best_model_mase,
            "seasonal_naive_mase": seasonal_naive_mase, "leaderboard_ts": lb}


def gads_timeseries_predict(predictor_ts, ts_df):
    """Generate quantile forecasts from a fitted TimeSeriesPredictor (mean + 0.1…0.9)."""
    forecasts = predictor_ts.predict(ts_df)
    print(f"[gads_timeseries_predict] Forecasts shape: {forecasts.shape}")
    return forecasts
