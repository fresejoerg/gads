---
id: timeseries_forecast.autogluon.standard
version: 1.1.0
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
      Profile the time-series structure:
      (1) Identify `timestamp_col` (datetime dtype or a date/time-like name) and parse
          it with pd.to_datetime.
      (2) Identify `target_col` — the value to forecast.
      (3) Identify `item_id_col` — the series identifier; if the data is a single
          series, create a constant identifier column.
      (4) Infer the frequency from the median delta between consecutive timestamps and
          store in `inferred_freq`.
      (5) Count rows, distinct series (`n_series`), and min/median/max series length.
      (6) Compute `naive_mae` — the mean absolute deviation of the target from its own
          mean — as the baseline any forecaster must beat.
      Print a summary of all of it.
    worker_tier: T2
    produces: [timestamp_col, target_col, item_id_col, inferred_freq, n_series]
    attached_skills: [autogluon_timeseries]
    postconditions:
      - "isinstance(timestamp_col, str)"
      - "isinstance(target_col, str)"
      - "inferred_freq is not None"
    required_metrics: [n_series, naive_mae]

  - id: prepare_timeseries_dataframe
    intent: >
      Convert to AutoGluon's long-format TimeSeriesDataFrame (`ts_df`) using the
      conversion pattern in the attached skill. If there are more than 200 distinct
      series, keep the 200 longest for tractability and say so. Print the resulting
      shape and a sample.
    depends_on: [profile_time_series]
    worker_tier: T2
    produces: [ts_df]
    attached_skills: [autogluon_timeseries]
    postconditions:
      - "ts_df is not None"
      - "len(ts_df) > 0"

  - id: train_forecast_model
    intent: >
      Train a TimeSeriesPredictor (fit pattern in the attached skill): derive
      `prediction_length` from the data (~10% of median series length, minimum 1 —
      never a hardcoded number), eval_metric MASE, time-budgeted presets. Print the
      leaderboard, store `best_model_mase`, and persist the predictor as
      model_timeseries.joblib.
    depends_on: [prepare_timeseries_dataframe]
    worker_tier: T2
    produces: [predictor_ts, prediction_length, best_model_mase]
    attached_skills: [autogluon_timeseries]
    postconditions:
      - "predictor_ts is not None"
      - "isinstance(best_model_mase, float)"
    required_metrics: [best_model_mase, prediction_length]

  - id: generate_forecasts_and_visualize
    intent: >
      Generate quantile forecasts (`forecasts = predictor_ts.predict(ts_df)`). For up
      to 4 series, plot history + forecast mean + 80% interval band; save as
      figure_1_forecast.json. Print a summary table per series: last observed value,
      next-period forecast, trend direction. Emit an insight summarizing
      prediction_length, the best model, MASE vs the naive baseline (MASE < 1 beats
      seasonal-naive), and the dominant trend.
    depends_on: [train_forecast_model]
    worker_tier: T2
    attached_skills: [autogluon_timeseries, visualization_best_practices]
    postconditions:
      - "forecasts is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "USE AUTOGLUON: always use TimeSeriesPredictor — NEVER implement ARIMA, ETS, or Prophet from scratch."
  - "TIME BUDGET: always set time_limit=120 with presets='fast_training'. NOTE: a wall-clock budget makes model selection machine-load-dependent — this recipe is for exploratory forecasting; reproducibility-critical runs must pin an explicit fixed model set instead."
  - "ITEM_ID: every dataset needs a series-identifier column; add a constant one for a single series."
  - "PREDICTION_LENGTH: derive from the data (~10% of median series length, min 1). Never hardcode."
  - "BASELINE COMPARISON: always report MASE against the naive baseline — MASE < 1 means the model earns its keep."
  - "PERSIST: save the fitted predictor as model_timeseries.joblib."
---

# Time Series Forecasting with AutoGluon

## Rationale
AutoGluon's TimeSeriesPredictor trains and ensembles statistical, tree-based, and deep
forecasting models in one call, handling validation splitting and seasonality
internally — eliminating the classic failure modes of hand-rolled ARIMA/Prophet
pipelines (wrong frequency strings, missing seasonality, validation leakage). The recipe
keeps the forecasting judgments — horizon derived from the data, baseline comparison,
uncertainty bands in the deliverable — and delegates API mechanics to the
`autogluon_timeseries` skill.

## When to use
Forecasting future values of one or more series. The spec needs the timestamp column,
the value to forecast, and optionally a grouping column. Use
`causal_effect.timeseries.causalimpact` instead when the question is the effect of an
intervention on a series.

## Key Constraints
- More than 200 series: subsample to the longest 200.
- `prediction_length` must be derived from data characteristics.
- Results are time-budgeted and therefore not bitwise-reproducible across machines —
  acceptable for exploration, not for benchmarks.
