---
id: survival_lifelines
description: "lifelines for survival INFERENCE: Kaplan-Meier + log-rank, CoxPHFitter with hazard ratios, the proportional-hazards assumption test (and how to fix violations via stratification), AFT models, and calibration. Use the native gads_cox_ph_report for the Cox fit + PH gate."
triggers: ["lifelines", "coxphfitter", "kaplanmeierfitter", "kaplan-meier", "kaplan meier", "log-rank", "logrank", "hazard ratio", "proportional hazards", "check_assumptions", "cox regression", "aft model", "weibullaftfitter", "accelerated failure time", "baseline hazard", "survival regression"]
---
# Survival Inference with lifelines

`lifelines` is a pandas-native survival library. Data stays a DataFrame with a duration
column and an event column — **no structured array** (that is scikit-survival; do not mix
the two APIs).

## Kaplan-Meier + log-rank (describe before you model)
```python
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

kmf = KaplanMeierFitter()
kmf.fit(df["time"], event_observed=df["event"], label="all")
kmf.plot_survival_function()          # median survival: kmf.median_survival_time_

# Compare two groups (e.g. treated vs control):
mask = df["group"] == "treated"
res = logrank_test(df.loc[mask, "time"], df.loc[~mask, "time"],
                   event_observed_A=df.loc[mask, "event"],
                   event_observed_B=df.loc[~mask, "event"])
print(res.p_value)                    # < 0.05 → survival curves differ
```

## Cox proportional hazards — PREFER the native `gads_cox_ph_report`
It fits the Cox model AND runs the proportional-hazards assumption test in one call,
returning hazard ratios + CIs + p-values and the list of covariates that violate PH:
```python
report = gads_cox_ph_report(df, duration_col="time", event_col="event",
                            covariates=["age", "grade", "horTh"])   # omit covariates to use all cols
report["concordance"]        # C-index
report["hazard_ratios"]      # {cov: {hazard_ratio, ci_lower, ci_upper, p}}
report["ph_violations"]      # covariates whose HR is NOT constant over time
```
It writes `cox_report.json` and emits insights (concordance, significant HRs, any PH
violation) automatically. Do not re-implement the PH test by hand.

### Fitting Cox directly (when you need the fitter object)
```python
from lifelines import CoxPHFitter
cph = CoxPHFitter(penalizer=0.0)                 # penalizer>0 (e.g. 0.1) if it fails to converge
cph.fit(df[["age", "grade", "horTh", "time", "event"]],
        duration_col="time", event_col="event")
cph.print_summary()                              # coefs, exp(coef)=hazard ratio, p, CI
cph.concordance_index_                           # predictive accuracy (0.5=random)
cph.plot()                                       # forest plot of log-hazard-ratios
```

## Interpreting hazard ratios (the whole point of Cox)
`HR = exp(coef)` is the **multiplicative effect on the instantaneous event rate** per unit
of the covariate:
- `HR > 1` → higher risk / **shorter** survival (e.g. `HR=1.5` = 50% higher hazard).
- `HR < 1` → protective / **longer** survival (e.g. `HR=0.65` = 35% lower hazard).
- `HR = 1` → no effect. Report the 95% CI and p-value; a CI crossing 1 is not significant.

## The proportional-hazards assumption — the methodological gate
Cox assumes each covariate's hazard ratio is **constant over time**. If violated, the
single reported HR is misleading. `gads_cox_ph_report` flags violators via Schoenfeld
residuals (`ph_violations`). To fix a violation, **stratify** on the offending covariate
(estimates a separate baseline hazard per stratum, no single HR for it):
```python
cph.fit(df, duration_col="time", event_col="event", strata=["grade"])
```
Alternatives: add a time-varying interaction, or switch to an AFT model. Diagnostics-only
view: `cph.check_assumptions(df, p_value_threshold=0.05)` prints per-covariate detail.

## Accelerated Failure Time (AFT) — models time directly, no PH assumption
```python
from lifelines import WeibullAFTFitter          # also LogNormalAFTFitter, LogLogisticAFTFitter
aft = WeibullAFTFitter()
aft.fit(df, duration_col="time", event_col="event")
aft.print_summary()
aft.predict_median(df)                           # predicted median survival time per row
```
AFT coefficient sign is **opposite** to Cox: a positive coefficient means *longer*
survival (protective). Compare parametric models by `AIC_` (lower is better); Cox uses
`AIC_partial_`.

## Calibration
```python
from lifelines.calibration import survival_probability_calibration
survival_probability_calibration(cph, df, t0=median_followup)   # predicted vs observed at t0
```

## Pitfalls
- Do not one-hot then also pass the original categorical — pick one encoding; lifelines
  accepts numeric columns (encode strings first, or use `formula=`).
- Perfect separation / non-convergence → set `penalizer=0.1`.
- Don't feed a scikit-survival structured array to lifelines; it wants plain DataFrame
  columns.
