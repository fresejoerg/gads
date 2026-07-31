---
name: "GBSG2 Breast Cancer — ML survival prediction (Random Survival Forest)"
datasets:
  - survival/gbsg2.csv
recipe_id: survival_analysis.ml
domain: breast cancer recurrence risk prediction
taxonomy:
  intent: predictive
  task: [regression.survival]
  modality: [tabular]
  domain: healthcare
  domain_detail: "GBSG2 breast cancer trial; predict individual recurrence risk with censoring-aware evaluation"
  deliverable: [model_artifact, diagnostic_report]
  validation: [holdout_metric]
---
Predict which breast-cancer patients are at highest risk of early recurrence, and measure
how well the model discriminates.

Dataset: 686 patients from the German Breast Cancer Study Group 2 trial
(`survival/gbsg2.csv`). This is right-censored time-to-event data:
- `time`: recurrence-free survival time in DAYS (the duration).
- `event`: 1 = recurrence or death occurred at `time`; 0 = right-censored (still
  recurrence-free at last follow-up). About 44% of patients had the event (299 of 686); the rest are censored.
- Covariates: `age`, `menostat` (menopausal status Pre/Post), `tsize` (tumor size, mm),
  `tgrade` (tumor grade I/II/III), `pnodes` (number of positive lymph nodes), `progrec`
  (progesterone receptor), `estrec` (estrogen receptor), `horTh` (hormonal therapy yes/no).

Build a machine-learning survival model (Random Survival Forest) that scores each patient's
recurrence risk. Handle the censoring correctly — keep the censored patients and use the
structured time-to-event target. Evaluate with censoring-aware metrics (IPCW concordance
index, time-dependent AUC, Integrated Brier Score), not plain accuracy.

The deliverable is a risk-scoring model plus predicted survival curves for representative
low-, medium-, and high-risk patients, and an honest read of how well it ranks patients by
recurrence risk.
