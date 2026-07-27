---
name: "AMLB Adult Income (binary classification)"
datasets:
  - amlb/adult.csv
target_column: class
domain: census income prediction
recipe_id: tabular_automl.autogluon.deterministic
taxonomy:
  intent: predictive
  task: [classification.binary]
  modality: [tabular]
  domain: public_sector
  domain_detail: "census income prediction (UCI Adult)"
  deliverable: [model_artifact]
  validation: [holdout_metric]

---
Which adults earn more than $50K per year?

Dataset: 48,842 census records (OpenML data_id 1590, the AMLB 'adult' task).
- `class`: '<=50K' or '>50K' (target, ~24% positive)
- 14 demographic and employment features (age, workclass, education, occupation,
  hours-per-week, capital gains/losses, ...)

Train the best possible income classifier using AutoML with the deterministic
model portfolio. Report the ROC-AUC on the held-out test set and the top
predictive features.
