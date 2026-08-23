---
name: "Adult census — reasoned model selection"
datasets:
  - amlb/adult.csv
target_column: class
domain: public_sector
sample_rows: 20000
taxonomy:
  intent: predictive
  task: [classification.binary]
  modality: [tabular]
  domain: public_sector
  domain_detail: "US census income (OpenML adult / AMLB)"
  deliverable: [model_artifact, report_narrative]
  validation: [holdout_metric]

---
Which model should we use to predict whether a person earns more than $50K a year, and why?

Compare candidate models on this dataset, choose the best algorithm with a defensible
rationale, run hyperparameter tuning on the winner, and report held-out performance
together with feature importance.

Dataset: 48,842 rows, 14 features (OpenML data_id 1590, the AMLB 'adult' task).
- `class`: binary income label (`<=50K` / `>50K`), roughly 24% positive
- 6 numeric features (age, education-num, capital-gain, capital-loss, hours-per-week, fnlwgt)
- 8 categorical features, the widest being `native-country` with 41 levels

The deliverable is a defended choice: which algorithm, what it beat and by how much
relative to fold noise, what tuning bought, and which features actually drive the
predictions.
