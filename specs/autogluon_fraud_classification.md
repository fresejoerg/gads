---
name: "Fraud Detection AutoML (AutoGluon)"
datasets:
  - creditcard.csv
target_column: Class
domain: financial fraud detection
recipe_id: tabular_automl.autogluon.standard
sample_rows: 50000
taxonomy:
  intent: predictive
  task: [classification.binary]
  modality: [tabular]
  domain: finance
  domain_detail: "credit-card fraud, severely imbalanced"
  deliverable: [model_artifact]
  validation: [holdout_metric]

---
Which credit card transactions are fraudulent?

Dataset: 284,807 transactions from European cardholders.
- `Class`: 0 = legitimate, 1 = fraudulent (target — severely imbalanced at ~0.17%)
- `Amount`: transaction value in currency units
- `Time`: seconds elapsed since first transaction in the dataset
- `V1`–`V28`: PCA-anonymised behavioural features

Train the best possible fraud detection model using AutoML. Report the ROC-AUC on the
held-out test set and the top predictive features. The model should handle the class
imbalance appropriately.
