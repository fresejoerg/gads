# air_passengers_v1 — provenance & tolerance rationale

**Established:** 2026-10-01 (SPRINT-10, coverage sprint). **Status:** the first benchmark for
`forecasting.univariate`, and the first in the repository whose metrics carry 033's
`provenance` field. Every pass criterion is `analytic`, computed by `reference.py` in this
directory, independently of GADS.
**Recipe:** `timeseries_forecast.autogluon.standard` (pinned in the spec).

## Why this benchmark exists

`forecasting.univariate` was L1: a recipe declared it, but no spec or benchmark exercised it.
The only forecasting benchmark (M4 Hourly) is a 50-series panel. AirPassengers is the textbook
univariate series: trend plus multiplicative yearly seasonality, a well-known best classical
model (the Box & Jenkins airline model), and small enough that every reference model fits in
seconds.

## Dataset

Box & Jenkins (1976), Series G: 144 monthly observations, 1949-01 … 1960-12, staged by
`scripts/stage_air_passengers.py` from the copy in `statsforecast` (no network). The staging
script asserts the value sum (40363), and `PROVENANCE.md` in the dataset folder pins the
file's sha256.

## Protocol (shared by the recipe and the reference)

- **Horizon 14:** the recipe's rule (~10% of series length, min 1) on 144 observations. It is
  deliberately not the conventional 12, because the recipe forbids hardcoded horizons. The
  reference uses the same 14 so that one window is compared.
- **Holdout:** the last 14 months (1959-11 … 1960-12). This is the window AutoGluon validates
  on, and the window the recipe's `best_model_mase` is scored on.
- **MASE** with m = 12: holdout MAE divided by the in-sample seasonal-naive MAE on the fit
  window, which matches AutoGluon's `eval_metric="MASE"` for monthly data.

## Reference results (`reference.py`, statsforecast 2.0.1 / statsmodels 0.14.6)

| model | holdout MASE | sMAPE |
|---|---|---|
| airline model, SARIMA(0,1,1)(0,1,1)12 on log | **0.335** | 2.20 |
| AutoARIMA (season 12) | 0.372 | 2.44 |
| AutoETS (season 12) | 1.272 | 8.14 |
| seasonal-naive | 1.941 | 13.88 |
| naive (random walk) | 2.249 | 14.41 |

Re-running `reference.py` reproduces these exactly. AutoETS is notably weak here; it most
likely selects an additive-seasonal form for a multiplicatively seasonal series. That is
recorded rather than tuned away, because the reference is a set of textbook defaults, not a
best effort.

## Pass criteria and why

- `n_series == 1` and `prediction_length == 14` are **exact**: both are pure functions of the
  data plus the recipe's rule.
- `best_model_mase < 1.9414` is the **seasonal-naive floor on the identical holdout**. It is
  deliberately a threshold, not a target value. AutoGluon's model set under the recipe's
  wall-clock budget varies with machine load (see m4_hourly_forecast_v1/notes.md), so an exact
  expected MASE would be a fiction.

**The floor is 1.94, not 1.0.** The forecasting recipe and the M4 benchmark both read "MASE < 1"
as "beats seasonal-naive". That is only true *in-sample*. Out of sample, on a trending series,
seasonal-naive falls further behind every year and scores well above 1 itself (1.94 here).
On a trending panel the M4 benchmark's `< 1.0` therefore demands more than beating
seasonal-naive. Recorded as a follow-up for M4 (re-anchor its threshold with its own reference
script).

## Quality grading (informational, not pass/fail)

A run that clears the floor is graded against the reference block on the same window:
≤ 0.40 classical-par · 0.40–1.27 adequate · 1.27–1.94 weak. The floor alone is necessary, not
sufficient. A model that ignores seasonality lands near 2 and fails; a model that captures
seasonality but not its multiplicative growth lands in the "weak" band.

One caveat when comparing: AutoGluon *selects* its best model on this same window, so its
`best_model_mase` is mildly optimistic, while the reference models are fitted without
looking at the holdout.

## What is not scored

- The forecast beyond 1960-12: there is no ground truth for it.
- Interval calibration: the recipe emits quantiles, but no coverage check exists yet. A
  candidate for `forecasting.probabilistic`.
