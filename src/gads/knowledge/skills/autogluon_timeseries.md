---
id: autogluon_timeseries
description: "AutoGluon TimeSeriesPredictor code patterns: long-format conversion, frequency inference, fit, quantile forecasts. Forecasting only."
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

## Frequency inference (needed to sanity-check the data, not passed to fit)

```python
deltas = df.sort_values(timestamp_col)[timestamp_col].diff().dropna()
median_delta = deltas.median()   # ~1 day → 'D', ~7 days → 'W', ~30 days → 'ME', ~1 hour → 'h'
```

## Fit

```python
predictor_ts = TimeSeriesPredictor(
    prediction_length=prediction_length,   # derive from data: ~10% of median series length, min 1
    target=target_col,
    eval_metric='MASE',
    verbosity=0,
).fit(ts_df, presets='fast_training', time_limit=120)

leaderboard = predictor_ts.leaderboard(ts_df, silent=True)
best_model_mase = float(leaderboard.iloc[0]['score_val']) * -1   # AutoGluon reports negated scores
import joblib; joblib.dump(predictor_ts, 'model_timeseries.joblib')
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

MASE < 1 means the model beats the seasonal-naive baseline — always report that comparison.

## Scale limit

More than 200 distinct series: subsample to the 200 longest before fitting.
