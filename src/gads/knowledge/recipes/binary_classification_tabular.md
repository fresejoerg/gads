---
id: binary_classification.tabular.standard
version: 1.0.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [binary_classification, classification]
  data_modality: [tabular]
  signals:
    - target_type: categorical_binary
  anti_signals:
    - temporal_ordering_required: true

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [sklearn, pandas, matplotlib]

# ——— DAG TEMPLATE ———
dag:
  - id: check_target_balance
    intent: "Verify target column distribution and identify class imbalance."
    worker_tier: T3
    postconditions:
      - "isinstance(output, dict)"
      - "'distribution' in output"
    skippable_if: "blackboard.has_fact('target_distribution_known')"

  - id: preprocessing_pipeline
    intent: "Handle missing values, encode categorical variables, and scale features."
    depends_on: [check_target_balance]
    worker_tier: T2
    produces: [X, y]
    postconditions:
      - "X.isna().sum().sum() == 0"
      - "len(X) == len(y)"

  - id: stratified_split
    intent: "Perform a stratified train/test split to preserve class ratios."
    depends_on: [preprocessing_pipeline]
    worker_tier: T3
    produces: [X_train, X_test, y_train, y_test]
    postconditions:
      - "abs(y_train.mean() - y_test.mean()) < 0.05"

  - id: train_baseline_model
    intent: "Train a Logistic Regression baseline and evaluate using ROC-AUC."
    depends_on: [stratified_split]
    worker_tier: T2
    produces: [baseline_model, baseline_auc]
    postconditions:
      - "baseline_auc > 0.5"

  - id: confusion_matrix_plot
    intent: "Generate and save a confusion matrix visualization."
    depends_on: [train_baseline_model]
    worker_tier: T2
    postconditions:
      - "blackboard.has_artifact('plot')"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "The target column must remain excluded from features (X)."
  - "Random seeds must be fixed for reproducibility (random_state=42)."
---

# Standard Binary Classification for Tabular Data

## Rationale
This recipe follows industry best practices for supervised classification. It emphasizes **data validation** before modeling, ensures class ratios are preserved via **stratification**, and establishes a **baseline performance** metric before attempting more complex architectures.

## When to use
Use this SOP when you have a clear binary target (Yes/No, 1/0, Survived/Died) and a set of independent features in a CSV or DataFrame.

## Key Constraints
- Target column must be specified.
- Titanic survival is a classic fit for this recipe.
