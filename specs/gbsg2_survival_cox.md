---
name: "GBSG2 Breast Cancer — Cox survival regression (hazard ratios + PH check)"
datasets:
  - survival/gbsg2.csv
recipe_id: survival_analysis.cox_regression
domain: breast cancer recurrence-free survival
taxonomy:
  intent: diagnostic
  task: [regression.survival]
  modality: [tabular]
  domain: healthcare
  domain_detail: "GBSG2 breast cancer trial; recurrence-free survival; interpretable hazard ratios + PH assumption test"
  deliverable: [estimate, report_narrative]
  validation: [statistical_inference]
---
Which factors drive time to breast-cancer recurrence, and how defensible is the model?

Dataset: 686 patients from the German Breast Cancer Study Group 2 trial
(`survival/gbsg2.csv`). This is right-censored time-to-event data:
- `time`: recurrence-free survival time in DAYS (the duration).
- `event`: 1 = recurrence or death occurred at `time`; 0 = right-censored (still
  recurrence-free at last follow-up). About 44% of patients had the event (299 of 686); the rest are censored.
- Covariates: `age`, `menostat` (menopausal status Pre/Post), `tsize` (tumor size, mm),
  `tgrade` (tumor grade I/II/III), `pnodes` (number of positive lymph nodes), `progrec`
  (progesterone receptor), `estrec` (estrogen receptor), `horTh` (hormonal therapy yes/no).

Do a proper interpretable survival analysis: describe the data with Kaplan-Meier curves
and a log-rank test across a meaningful group (e.g. hormonal therapy), then fit a Cox
proportional-hazards model and report the hazard ratios (with 95% CIs) for each covariate.

Critically, TEST the proportional-hazards assumption — do not just assume it — and report
any covariate whose hazard ratio is not constant over time, with a recommended remedy.
The deliverable is a defensible explanation of the recurrence-risk drivers plus an honest
assessment of whether the Cox model's assumptions actually hold.
