---
name: "M4 Hourly demand forecasting (50 series)"
datasets:
  - m4/m4_hourly_train.csv
target_column: target
domain: time-series demand forecasting
recipe_id: timeseries_forecast.autogluon.standard
taxonomy:
  intent: predictive
  task: [forecasting.multivariate]
  modality: [time_series]
  domain: operations
  domain_detail: "M4 competition, Hourly split (Makridakis et al. 2018)"
  deliverable: [forecast_series]
  validation: [temporal_backtest]

---
Forecast future demand for each of the 50 hourly series in this panel.

Dataset: 35,000 rows, long format, from the M4 competition's Hourly split.
- `item_id`: series identifier (H1 … H50)
- `timestamp`: hourly, contiguous, 700 observations per series
- `target`: the observed value to forecast

The series carry strong daily seasonality (period 24) and differ by orders of
magnitude in scale, so per-series scaling matters and a pooled baseline will be
misleading.

Train a forecasting model, report MASE against the seasonal-naive baseline, and
visualise history plus the forecast interval for a sample of series. A model only
earns its keep if MASE < 1.
