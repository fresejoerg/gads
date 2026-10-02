---
id: autogluon_timeseries
description: "AutoGluon TimeSeriesPredictor code patterns: long-format conversion, fit, scoring via gads_forecast_scores, quantile forecasts. Forecasting only."
triggers: ["TimeSeriesPredictor", "TimeSeriesDataFrame", "forecast", "forecasting", "time series forecast", "MASE", "prediction_length"]
---
# AutoGluon TimeSeriesPredictor — Canonical Patterns

Installed as `autogluon.timeseries`. Trains and ensembles statistical (ARIMA/ETS/Theta),
tree-based, and deep forecasting models in one call, handling train/validation splitting
and seasonality internally. **Never hand-roll ARIMA/Prophet pipelines.**

## Data must be long format: one row per (item_id, timestamp)

```python
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# Single series? Add a constant identifier first:
if item_id_col is None:
    df['item_id'] = 'series_1'
    item_id_col = 'item_id'

df[timestamp_col] = pd.to_datetime(df[timestamp_col])
ts_df = TimeSeriesDataFrame.from_data_frame(
    df[[item_id_col, timestamp_col, target_col]],
    id_column=item_id_col,
    timestamp_column=timestamp_col,
)
```

## Profiling: use the native, never infer the frequency by hand

```python
prof = gads_profile_timeseries(df, target_col=globals().get("target_column"))   # pre-loaded native
# keys: df (timestamp parsed, item_id added for a single series), timestamp_col, target_col,
#       item_id_col, inferred_freq ('h', 'D', 'W-SUN', 'MS', ...), season_length, n_series,
#       n_rows, min_len, median_len, max_len, naive_mae
```
Hand-written frequency maps were the most common local failure: `pd.Timedelta('D')` and
`pd.Timedelta('MS')` raise "unit abbreviation w/o a number".

## Fit

```python
predictor_ts = TimeSeriesPredictor(
    prediction_length=prediction_length,   # derive from data: ~10% of median series length, min 1
    target=target_col,
    eval_metric='MASE',
    verbosity=0,
).fit(ts_df, presets='fast_training', time_limit=120)

import joblib; joblib.dump(predictor_ts, 'model_timeseries.joblib')

scores = gads_forecast_scores(predictor_ts, ts_df)    # pre-loaded native
best_model_mase = scores["best_model_mase"]            # positive MASE (AutoGluon's sign undone)
seasonal_naive_mase = scores["seasonal_naive_mase"]    # SeasonalNaive on the SAME window
```

**Reproducibility caveat:** `time_limit`/`presets` make the trained ensemble depend on
wall-clock and machine load — repeat runs can select different models. If the recipe
invariants demand reproducible results, pin an explicit fixed model set via
`hyperparameters={...}` instead of a time budget (same principle as the deterministic
tabular portfolio).

## Forecast

```python
forecasts = predictor_ts.predict(ts_df)   # columns: mean + quantiles 0.1..0.9
print(forecasts.head(20))
```

Compare `best_model_mase` against `seasonal_naive_mase` from `gads_forecast_scores` (same
validation window): the model earns its keep only if it is lower. Never read the leaderboard
by hand: its index is a plain integer (`leaderboard.loc['SeasonalNaive']` raises KeyError), and
AutoGluon negates error metrics. Do not read "MASE < 1" as "beats seasonal-naive"
— MASE is scaled by the in-sample seasonal-naive error, and on trending data seasonal-naive itself
scores well above 1 out of sample (AirPassengers: 1.94).

## Scale limit

More than 200 distinct series: subsample to the 200 longest before fitting.
