---
id: ordinal_classification.tabular.regression_wrapper
version: 1.1.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [ordinal_classification, ordered_multiclass, likert_scale_prediction, rating_prediction]
  data_modality: [tabular, unstructured_text]

# ——— METHODOLOGICAL RATIONALE ———
rationale: >
  Ordinal classification respects the inherent order between classes (e.g., 'Low' < 'Medium' < 'High'). 
  By using a 'Regression-to-Classification' wrapper, we treat the categories as a continuous gradient. 
  This ensures that being 'one class off' is penalized less than being 'completely off'. 
  For text predictors, we vectorize the content and reduce dimensionality to capture semantic patterns 
  while maintaining model efficiency.

# ——— PREREQUISITES ———
requires:
  variables: [target_column, ordered_labels]
  capabilities: [numerical_analysis, supervised_learning]

# ——— EXECUTION DAG ———
dag:
  - id: map_and_preprocess
    intent: "Map the ordered text labels to integers (0, 1, 2...). Split into train/test sets."
    worker_tier: "T3"
    postcondition:
      output_type: "dataframe"
      required_columns: []

  - id: vectorize_text_features
    intent: "If the predictor is text, use the local sentence transformer to embed the text. Apply PCA (e.g., n_components=50) to reduce dimensionality while preserving signal. Concatenate with other features if present."
    worker_tier: "T3"
    depends_on: [map_and_preprocess]
    attached_skills: [local_text_embedding]
    postcondition:
      output_type: "dataframe"
      required_columns: []

  - id: train_ordinal_regressor
    intent: "Apply StandardScaler to all features. Train a RandomForestRegressor on the processed data. Implement a wrapper that rounds and clips predictions to the [min, max] range of the mapped labels."
    worker_tier: "T3"
    depends_on: [vectorize_text_features]
    postcondition:
      output_type: "list"
      required_columns: []

  - id: validate_distance_metrics
    intent: "Evaluate the model using Mean Absolute Error (MAE) and Exact Accuracy. Generate a confusion matrix to visualize 'off-by-one' errors."
    worker_tier: "T2"
    depends_on: [train_ordinal_regressor]
    attached_skills: [visualization_best_practices]

  - id: persist_model_bundle
    intent: "Use joblib to save a dictionary containing the model, scaler, PCA object (if used), and label mappings into 'ordinal_model_bundle.joblib'."
    worker_tier: "T3"
    depends_on: [validate_distance_metrics]

---
# Ordinal Classification (Ordered Multi-Class)

This recipe implements an ordinal classification workflow using a regression-based approach. This is ideal for Likert scales, star ratings, or any categorized data where the "distance" between categories matters.

## Handling Text Predictors
If your features are textual (e.g., predicting a rating based on a review), the workflow automatically:
1.  **Embeds** the text using a local SentenceTransformer.
2.  **Reduces** dimensionality via PCA to prevent overfitting and improve training speed.
3.  **Standardizes** the compressed features before training the regressor.

## Key Logic: The Wrapper
Instead of a standard classifier, we use a **Regressor** and post-process the output:
```python
def predict_ordinal(model, X, min_val, max_val):
    preds = model.predict(X)
    return np.clip(np.round(preds), min_val, max_val).astype(int)
```

## Best Practices
- **Evaluation**: Priority should be given to **Mean Absolute Error (MAE)**. An MAE of 0.5 means your model is, on average, only half a category away from the truth.
- **Scaling**: Regressors are highly sensitive to feature scales. Always use the saved `StandardScaler` for inference.
- **Persistence**: Always bundle the `mapping`, `scaler`, and `PCA` objects with the model file using `joblib`.
