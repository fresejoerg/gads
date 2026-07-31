---
id: survival_analysis
description: "Survival / time-to-event fundamentals: the (duration, event) pair, right-censoring (never drop censored rows), Kaplan-Meier vs Cox vs ML, the C-index, and which GADS native nodes to call. Read before any time-to-event work."
triggers: ["survival analysis", "time to event", "time-to-event", "censoring", "censored", "right-censored", "kaplan-meier", "kaplan meier", "hazard", "hazard ratio", "cox", "proportional hazards", "duration", "survival curve", "survival function", "concordance index", "c-index", "churn time", "time until", "days until", "failure time", "event history"]
---
# Survival Analysis (Time-to-Event) — Fundamentals

Survival analysis models the **time until an event** (death, churn, failure, default,
relapse) when some subjects have **not yet had the event** by the end of observation —
they are *right-censored*. This is not ordinary regression and not classification.

## The two columns that define every survival problem
Every row needs a **duration** and an **event indicator**:
- **duration / time** (float ≥ 0): how long the subject was observed.
- **event** (bool / 0-1): `1`/`True` = the event happened at `duration`; `0`/`False` =
  censored (observation ended — end of study, dropout, still alive — event time unknown,
  known only to be *after* `duration`).

## The cardinal rule: NEVER drop censored rows
Censored subjects carry information ("survived at least this long"). Dropping them, or
treating `duration` as a plain regression target, biases every estimate downward and is
the single most common survival-analysis mistake. Keep them; the models below use the
censoring indicator explicitly.

Related traps:
- **Do not** turn it into "did the event happen (yes/no)" classification — that throws
  away *when* and mishandles censoring.
- **Immortal-time bias**: a covariate measured *after* time zero (or that guarantees
  survival up to some point) leaks the outcome. Define covariates at the origin.
- **Leakage**: any feature recorded at/after the event (e.g. "cause of death") is a golden
  feature. Exclude it.

## Which method / recipe to use
- **Inference — "which factors drive time-to-event, and by how much?"** → the
  `survival_analysis.cox_regression` recipe (lifelines): Kaplan-Meier curves, log-rank
  tests, Cox proportional-hazards with **hazard ratios** and a **PH-assumption check**.
  Use when you must *explain and defend* the effect of each covariate. See the
  `survival_lifelines` skill.
- **Prediction — "forecast individual risk / survival curves"** → the
  `survival_analysis.ml` recipe (scikit-survival): Random Survival Forest / Gradient
  Boosting with censoring-aware evaluation (IPCW C-index, time-dependent AUC, Integrated
  Brier Score). Use when ranking/scoring subjects matters more than interpretation. See
  the `survival_scikit_survival` skill.

## Evaluation, at a glance
- **Concordance index (C-index)**: probability the model ranks a higher-risk subject as
  failing sooner. `0.5` = random, `>0.7` = good discrimination, `1.0` = perfect. The
  survival analogue of AUC. Prefer the **IPCW** C-index under heavy censoring.
- **Integrated Brier Score (IBS)**: overall calibration+discrimination of the predicted
  survival curves over time. **Lower is better**; `0.25` ≈ uninformative.
- Never report plain accuracy/RMSE on `duration` — they ignore censoring.

## GADS native nodes — prefer them over hand-rolled code
These are pre-injected; call them directly (do not re-implement, do not import them):
- `gads_make_surv_target(df, time_col, event_col)` → the scikit-survival structured
  target array `(event: bool, time: float)`, with the boolean cast and censored-row
  handling done correctly. Building this array by hand is the #1 sksurv failure.
- `gads_evaluate_survival(model, X_train, y_train, X_test, y_test)` → IPCW C-index,
  time-dependent AUC, Integrated Brier Score in one call, at safe follow-up times; writes
  `survival_metrics.json`.
- `gads_cox_ph_report(df, duration_col, event_col, covariates=...)` → lifelines Cox fit +
  hazard ratios + the proportional-hazards assumption test; writes `cox_report.json`.

## Out of scope (v1)
Deep-learning survival (DeepSurv/pycox/TorchSurv) and competing risks (hazardous) are not
yet GADS recipes. For competing risks with `scikit-survival` installed, note the caveat: a
standard model treats competing events as censoring, which overestimates the survival
probability of the event of interest.
