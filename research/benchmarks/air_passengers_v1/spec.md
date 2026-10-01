---
name: "AirPassengers monthly forecasting (univariate)"
datasets:
  - airpassengers/air_passengers.csv
target_column: passengers
domain: airline passenger demand forecasting
recipe_id: timeseries_forecast.autogluon.standard
taxonomy:
  intent: predictive
  task: [forecasting.univariate]
  modality: [time_series]
  domain: operations
  domain_detail: "Box & Jenkins Series G — international airline passengers, 1949–1960"
  deliverable: [forecast_series]
  validation: [temporal_backtest]

---
Forecast monthly international airline passenger totals.

Dataset: 144 rows, one series — the classic AirPassengers data (Box & Jenkins, Series G).
- `month`: first day of each month, January 1949 to December 1960, no gaps
- `passengers`: monthly total, in thousands

The series has a clear upward trend and yearly seasonality whose amplitude grows with the
level of the series (multiplicative seasonality), so a forecaster has to capture both.

Train a forecasting model, evaluate it on a held-out final window, report MASE against the
seasonal-naive baseline, and visualise the history together with the forecast and its
prediction interval.
