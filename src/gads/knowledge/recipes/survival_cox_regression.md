---
id: survival_analysis.cox_regression
version: 1.2.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [survival_analysis, survival_regression, cox_regression, time_to_event, survival]
  data_modality: [tabular]
  signals:
    - objective_contains: [survival, time to event, time-to-event, hazard, censored, censoring, kaplan-meier, cox, "how long until", "time until", "days until", median survival, risk factors for time]
    - has_duration_and_event_columns: true
  anti_signals:
    - objective_contains: [forecast the time series, classify into categories]

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [pandas, matplotlib, lifelines]

# ——— DAG TEMPLATE ———
dag:
  - id: prepare_survival_data
    intent: "Load the dataset into `df`. Identify the DURATION column (time observed, float >= 0) and the EVENT column (1/True = event occurred, 0/False = right-censored) from the objective/hints; store their names as `time_col` and `event_col`. NEVER drop censored rows — censoring carries information. Exclude any leakage/immortal-time features (anything recorded at or after the event, e.g. cause-of-death). Print the number of events vs censored, the censoring rate, and the median observed time."
    worker_tier: T2
    produces: [df, time_col, event_col]
    attached_skills: [survival_analysis]
    postconditions:
      - "time_col is not None and event_col is not None"
      - "df[event_col].nunique() == 2"

  - id: kaplan_meier_logrank
    intent: "Describe survival before modeling. Using lifelines (see the survival_lifelines skill): fit a KaplanMeierFitter on df[time_col] with event_observed=df[event_col]; read the overall median survival from `kmf.median_survival_time_` (note the trailing underscore) and store it as `km_median`. Save the survival-curve figure. Then pick one meaningful categorical covariate (e.g. horTh), plot KM curves per group on one axis, and run a log-rank test across the groups (multivariate_logrank_test) — report its p-value and whether the curves differ significantly."
    depends_on: [prepare_survival_data]
    worker_tier: T2
    produces: [km_median]
    attached_skills: [survival_lifelines, visualization_best_practices]
    fallback_native: gads_kaplan_meier
    fallback_call: "km = gads_kaplan_meier(df, time_col=time_col, event_col=event_col); km_median = km['overall_median']"
    postconditions:
      - "km_median is not None"

  - id: cox_ph_model
    intent: "Fit the Cox model AND test its core assumption in one native call. Keep it minimal — the native already prints the hazard ratios, concordance, and PH violations and emits insights, so do NOT add manual print/reporting code. First one-hot encode the categorical covariates (horTh, menostat, tgrade) to numeric with pandas.get_dummies(drop_first=True) into a dataframe `enc` that still contains time_col and event_col, then:\n\ncox_report = gads_cox_ph_report(enc, duration_col=time_col, event_col=event_col)\n\nThat is the core task. Do NOT hand-roll the Cox fit or the proportional-hazards test."
    depends_on: [kaplan_meier_logrank]
    worker_tier: T2
    produces: [cox_report]
    attached_skills: [survival_lifelines]
    postconditions:
      - "'hazard_ratios' in cox_report"
      - "cox_report['concordance'] is not None"

  - id: hazard_ratio_report
    intent: "Turn the Cox fit into a defensible interpretation. Produce a forest plot of the hazard ratios (with 95% CIs, reference line at HR=1) and save it. In plain language, state for each significant covariate whether it raises (HR>1) or lowers (HR<1) the instantaneous event risk and by how much. If `cox_report['ph_violations']` is non-empty, explicitly flag that those covariates' hazard ratios are NOT constant over time (so the single HR is misleading) and recommend stratifying on them or adding a time-interaction — optionally refit with `strata=[...]` to demonstrate. Summarize the model's discrimination (concordance) and overall defensibility."
    depends_on: [cox_ph_model]
    worker_tier: T2
    attached_skills: [survival_lifelines, visualization_best_practices]
    postconditions:
      - "cox_report is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "NEVER drop right-censored rows or convert the problem to yes/no classification — censoring carries information and both destroy it."
  - "Random seeds must be fixed for reproducibility where any sampling/splitting occurs (random_state=42)."
  - "PROPORTIONAL-HAZARDS GATE: the Cox model must be passed through gads_cox_ph_report so the PH assumption is tested (Schoenfeld residuals). A violated PH assumption makes the reported hazard ratio misleading and MUST be surfaced, not ignored."
  - "LEAKAGE / IMMORTAL TIME: exclude any covariate measured at or after the event, or that guarantees survival up to some time — it leaks the outcome. Covariates are defined at time zero."
  - "HAZARD RATIOS: report exp(coef) with 95% CIs and p-values; a CI crossing 1 is not statistically significant. HR>1 = higher risk / shorter survival; HR<1 = protective."
---

# Cox Proportional-Hazards Survival Regression (interpretable time-to-event)

## Rationale
The standard rigorous workflow for **explaining** time-to-event outcomes with
right-censored data. Describe first with Kaplan-Meier and the log-rank test, then fit a
Cox proportional-hazards model whose **hazard ratios** can be read and defended — this is
the survival analogue of an interpretable regression baseline. The discipline that
separates sound survival analysis from a plausible-looking mistake is (1) keeping censored
observations and (2) actually **testing the proportional-hazards assumption** rather than
assuming it: a violated assumption silently invalidates the constant hazard ratio the model
reports. This recipe makes that test mandatory via the native `gads_cox_ph_report`.

## When to use
A duration column plus an event/censoring indicator, when the deliverable is
*understanding and defending which factors drive time-to-event and by how much* (medical
trials, churn-timing drivers, time-to-default, reliability). For **predicting** individual
risk or survival curves with maximum discrimination rather than interpretation, use
`survival_analysis.ml` (scikit-survival). For a plain forecast of an aggregate time series,
this is the wrong recipe — use a forecasting recipe.

## Key Constraints
- The duration and event columns must be identifiable from the objective or data; the event
  indicator must be binary (event vs censored).
- Categorical covariates must be encoded to numeric before the Cox fit.
- Hazard ratios are only interpretable as constant effects where the PH assumption holds.
