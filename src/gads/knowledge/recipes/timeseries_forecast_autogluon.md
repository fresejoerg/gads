---
id: timeseries_forecast.autogluon.standard
version: 1.0.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [time_series_forecasting, forecasting, time_series]
  data_modality: [tabular, time_series]
  signals:
    - temporal_ordering_required: true
    - objective_contains: [forecast, predict future, next N, upcoming, trend]
  anti_signals:
    - task: causal_inference
    - task: causal_impact

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [autogluon, pandas]

# ——— DAG TEMPLATE ———
dag:
  - id: profile_time_series
    intent: >
      Profile the time series dataset:
      (1) Identify the timestamp column (datetime dtype or name contains 'date', 'time', 'timestamp').
      (2) Identify the target value column to forecast.
      (3) Identify the series identifier column (item_id / group / entity) — if none exists, the
          dataset is a single series; create a constant column: df['item_id'] = 'series_1'.
      (4) Parse the timestamp column to datetime if not already: pd.to_datetime().
      (5) Infer the frequency: compute the median time delta between consecutive rows.
          Map to pandas freq string: daily→'D', weekly→'W', monthly→'ME', hourly→'H'.
      (6) Count total rows, number of distinct series, and min/max/mean series length.
      (7) Store: n_series, median_series_length, inferred_freq, naive_mae (mean absolute deviation
          of the target from its own mean — baseline for comparison).
      Print a summary.
    worker_tier: T2
    produces: [timestamp_col, target_col, item_id_col, inferred_freq, n_series]
    postconditions:
      - "isinstance(timestamp_col, str)"
      - "isinstance(target_col, str)"
      - "inferred_freq is not None"
    required_metrics: [n_series, naive_mae]

  - id: prepare_timeseries_dataframe
    intent: >
      Convert the raw DataFrame to AutoGluon's TimeSeriesDataFrame format:
        from autogluon.timeseries import TimeSeriesDataFrame
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df[[item_id_col, timestamp_col, target_col] + covariate_cols],
            id_column=item_id_col,
            timestamp_column=timestamp_col
        )
      If the dataset has more than 200 distinct series, subsample to the 200 longest ones to keep
      training tractable. Print the resulting shape and a sample.
    depends_on: [profile_time_series]
    worker_tier: T2
    produces: [ts_df]
    postconditions:
      - "ts_df is not None"
      - "len(ts_df) > 0"

  - id: train_forecast_model
    intent: >
      Train the AutoGluon TimeSeriesPredictor:
        predictor_ts = TimeSeriesPredictor(
            prediction_length=prediction_length,  # steps ahead; default to 10% of median series length, min 1
            target=target_col,
            eval_metric='MASE',
            verbosity=0
        ).fit(
            ts_df,
            presets='fast_training',
            time_limit=120
        )
      Print the leaderboard. Store `best_model_mase` (lower is better; MASE < 1 means beating naive).
      Save: joblib.dump(predictor_ts, 'model_timeseries.joblib')
    depends_on: [prepare_timeseries_dataframe]
    worker_tier: T2
    produces: [predictor_ts, prediction_length, best_model_mase]
    postconditions:
      - "predictor_ts is not None"
      - "isinstance(best_model_mase, float)"
    required_metrics: [best_model_mase, prediction_length]

  - id: generate_forecasts_and_visualize
    intent: >
      Generate forecasts and visualise results:
      (1) `forecasts = predictor_ts.predict(ts_df)` — produces mean + quantile predictions.
      (2) For up to 4 series (or all if ≤4), plot historical values + forecast mean + 80% CI band.
          Save as `figure_1_forecast.json` (Plotly).
      (3) Print a summary table: series_id, last observed value, forecast mean for next period,
          trend direction (up/down/flat based on forecast slope).
      (4) Emit a `gads_emit_insight()` call summarising: prediction_length, best model name,
          MASE score vs naive baseline, and the direction of the dominant trend.
    depends_on: [train_forecast_model]
    worker_tier: T2
    postconditions:
      - "forecasts is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "USE AUTOGLUON: always use TimeSeriesPredictor — NEVER implement ARIMA, ETS, or Prophet from scratch."
  - "TIME LIMIT: always set time_limit=120. Never omit it."
  - "VERBOSITY: always set verbosity=0 in TimeSeriesPredictor()."
  - "ITEM_ID: every dataset needs an item_id column. If the data is a single series, add df['item_id'] = 'series_1' before converting."
  - "PREDICTION_LENGTH: default to 10% of median series length (minimum 1 step). Never hardcode a number without deriving it from the data."
  - "SAVE WITH JOBLIB: joblib.dump(predictor_ts, 'model_timeseries.joblib') — never pickle."
  - "MASE < 1 means the model beats the naive (seasonal) baseline — always report this comparison."
---

# Time Series Forecasting with AutoGluon

## Rationale
AutoGluon TimeSeriesPredictor trains and ensembles ARIMA, ETS, Theta, LightGBM-based, and DeepAR models in a single call, handling train/validation splitting and frequency detection internally. This eliminates the common Coder failure mode of building manual ARIMA or Prophet pipelines with incorrect frequency strings, missing seasonality parameters, or improper validation leakage.

## When to use
Use when the objective is to forecast future values of one or more time series. The spec needs only to identify the timestamp column, the value to forecast, and optionally a series grouping column. Use `causal_impact_timeseries.md` instead when the objective is to measure the causal effect of an intervention on a time series.

## Key Constraints
- Datasets with more than 200 series should subsample to the longest 200 for tractability.
- `prediction_length` must be derived from data characteristics, not hardcoded.
- `presets='fast_training'` is the default — sufficient for most use cases; use `'medium_quality'` if the spec asks for best accuracy.
