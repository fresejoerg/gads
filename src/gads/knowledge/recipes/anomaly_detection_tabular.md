---
id: anomaly_detection.tabular.isolation_forest
version: 1.1.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [anomaly_detection, outlier_detection]
  data_modality: [tabular]
  signals:
    - no_labeled_target: true
    - objective_contains: [anomaly, outlier, unusual, abnormal, deviation]
  anti_signals:
    - target_type: categorical_binary
    - task: classification

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [sklearn, pandas, matplotlib]

# ——— EXECUTION DAG ———
dag:
  - id: prepare_features
    intent: "Select the numerical features relevant to the objective and handle missing values (impute or drop with a stated rationale). Store the feature matrix in `X_features` and print its shape and the chosen columns."
    worker_tier: T3
    produces: [X_features]
    postconditions:
      - "X_features is not None"

  - id: scale_and_train
    intent: "Apply StandardScaler to the features, fit an IsolationForest (contamination='auto' unless the objective states an expected outlier rate; random_state=42), and append `anomaly_label` (-1 anomaly / 1 normal) and `anomaly_score` columns to df. Print the count and fraction of detected anomalies."
    worker_tier: T3
    depends_on: [prepare_features]
    produces: [anomaly_model, df]
    postconditions:
      - "'anomaly_label' in df.columns"
      - "'anomaly_score' in df.columns"

  - id: visualize_outliers
    intent: "Create a scatter plot (2D, using the two most informative features or the first two principal components) highlighting detected anomalies vs normal points. Save as Figure 1."
    worker_tier: T2
    depends_on: [scale_and_train]
    attached_skills: [visualization_best_practices]
    postconditions:
      - "'anomaly_label' in df.columns"

  - id: analyze_anomalies
    intent: "Inspect the top anomalies (lowest scores): summarize how their feature values deviate from the population (e.g. z-scores per feature), and emit an insight characterizing what makes them anomalous. If the objective mentions known/historical outliers, cross-reference against them."
    worker_tier: T2
    depends_on: [scale_and_train]
    postconditions:
      - "'anomaly_score' in df.columns"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "UNSUPERVISED ONLY: this recipe is for data WITHOUT a labeled target. If the dataset has ground-truth labels for the events of interest, a supervised recipe is the correct choice — flag this instead of proceeding."
  - "Scale features before fitting — isolation paths should not be dominated by one feature's units."
  - "Random seeds must be fixed (random_state=42)."
  - "Validate qualitatively: unsupervised detections must be characterized (which features deviate, by how much), never just counted."
---

# Anomaly Detection with Isolation Forest

## Rationale
Isolation Forest isolates outliers directly rather than profiling normal points: random
feature/split selection isolates anomalous points in fewer steps than normal ones. It is
the robust default for unsupervised outlier detection on numeric tabular data.

## When to use
The objective is to find unusual observations and **no labeled target exists**. Domains
like fraud, quality control, or intrusion are *not* automatically anomaly-detection
tasks — if labeled outcomes exist, supervised learning (e.g.
`tabular_automl.autogluon.standard`) will outperform unsupervised detection and should
be used instead.

## Key Parameters
- **contamination**: expected outlier fraction; `'auto'` when unknown, or the rate the
  objective states.
- **Labels**: `1` = normal, `-1` = anomaly.

## Best Practices
- Focus on numeric features where deviation is meaningful for the objective.
- Cross-reference detections with domain knowledge or known historical outliers when
  available — unsupervised results need external validation.
