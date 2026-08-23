---
name: "Adult Income (interpretable binary classification + audit)"
datasets:
  - amlb/adult.csv
target_column: class
domain: census income prediction
recipe_id: binary_classification.tabular.standard
sample_rows: 15000
taxonomy:
  intent: predictive
  task: [classification.binary]
  modality: [tabular]
  domain: public_sector
  domain_detail: "census income prediction (UCI Adult), interpretable baseline"
  deliverable: [model_artifact, report_narrative]
  validation: [holdout_metric]

---
Which adults earn more than $50K per year, and how defensible is the model?

Dataset: 48,842 census records (OpenML data_id 1590, the AMLB 'adult' task).
- `class`: '<=50K' or '>50K' (target, ~24% positive)
- 14 demographic and employment features (age, workclass, education, occupation,
  hours-per-week, capital gains/losses, native country, ...)

Build an **interpretable** logistic-regression baseline whose coefficients can be read
and defended: validate the target balance, preprocess (impute, encode, scale), do a
stratified split, fit `class_weight='balanced'` with a calibrated decision threshold,
and report the top coefficients by magnitude.

Then run the methodological-soundness audit on the fitted model and report what it
finds — the deliverable is an explainable model plus an honest assessment of whether it
is methodologically defensible (leakage, over/under-fitting, class imbalance, whether it
beats a baseline).
