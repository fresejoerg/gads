---
id: anomaly_detection.tabular.isolation_forest
version: 1.0.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [anomaly_detection, outlier_detection, fraud_detection, quality_control]
  data_modality: [tabular]

# ——— METHODOLOGICAL RATIONALE ———
rationale: >
  Isolation Forest is an unsupervised algorithm specifically designed to isolate outliers rather than profiling 
  normal points. It works by randomly selecting a feature and then randomly selecting a split value between 
  the maximum and minimum values of the selected feature. Outliers are typically isolated in fewer steps 
  than normal points.

# ——— PREREQUISITES ———
requires:
  variables: []
  capabilities: [numerical_analysis, unsupervised_learning]

# ——— EXECUTION DAG ———
dag:
  - id: prepare_features
    intent: "Load the dataset and select relevant numerical features. Handle missing values by imputation or removal."
    worker_tier: "T3"
    postcondition:
      output_type: "dataframe"
      required_columns: []

  - id: scale_and_train
    intent: "Apply StandardScaler to numerical features. Initialize IsolationForest with contamination='auto' (or user-specified) and fit it to the data."
    worker_tier: "T3"
    depends_on: [prepare_features]
    postcondition:
      output_type: "dataframe"
      required_columns: ["anomaly_label", "anomaly_score"]

  - id: visualize_outliers
    intent: "Create a scatter plot (2D or 3D) highlighting the detected anomalies (-1) vs normal points (1). Save as Figure 1."
    worker_tier: "T2"
    depends_on: [scale_and_train]
    attached_skills: [visualization_best_practices]

  - id: analyze_anomalies
    intent: "Inspect the top anomalies with the lowest scores and provide a qualitative summary of their characteristics."
    worker_tier: "T2"
    depends_on: [scale_and_train]

---
# Anomaly Detection with Isolation Forest

This recipe provides a robust workflow for identifying outliers in tabular data using scikit-learn's `IsolationForest`.

## Key Parameters
- **Contamination**: Represents the expected proportion of outliers (e.g., 0.05 for 5%). If unknown, use `'auto'`.
- **Labels**: 
    - `1`: Normal data point.
    - `-1`: Detected anomaly.

## Best Practices
- **Feature Selection**: Focus on numerical columns where anomalies are likely to deviate significantly from the mean/median.
- **Normalization**: While Isolation Forest is tree-based, scaling ensures all features contribute equally to the isolation path lengths.
- **Validation**: Unsupervised results MUST be cross-referenced with domain knowledge or "known" historical outliers if available.
