---
name: "SAP Subscription (binary classification, minimal guidance)"
datasets:
  - sap/data_train.csv
target_column: has_subscribed
taxonomy:
  intent: predictive
  task: [classification.binary]
  modality: [tabular]
  domain: general
  deliverable: [model_artifact]
  validation: [holdout_metric]

---
Train a binary classification model to predict the target `has_subscribed`. Make reasonable
data preprocessing and model selection choices.
