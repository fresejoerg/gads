---
id: ordinal_classification.tabular.regression_wrapper
version: 1.2.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [ordinal_classification, ordered_multiclass, likert_scale_prediction, rating_prediction]
  data_modality: [tabular, unstructured_text]

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: target_column
      kind: str
    - name: ordered_labels
      kind: list
  capabilities: [sklearn, pandas, numpy]

# ——— EXECUTION DAG ———
dag:
  - id: map_and_preprocess
    intent: "Map the ordered labels to consecutive integers (0, 1, 2, ...) preserving their order; store the mapping in `label_mapping`. Split 80/20 with random_state=42, stratified by the target."
    worker_tier: T3
    produces: [label_mapping, df_train, df_test]
    postconditions:
      - "label_mapping is not None"

  - id: vectorize_text_features
    intent: "If the predictors include text, embed it with the locally cached sentence transformer (pattern in the attached skill) and reduce dimensionality with PCA — choose n_components from the data (e.g. min(50, n_features, n_train // 10)) and print the retained variance. Concatenate with any numeric features."
    worker_tier: T3
    depends_on: [map_and_preprocess]
    attached_skills: [local_text_embedding]
    produces: [X_train, X_test]
    postconditions:
      - "X_train is not None"
    skippable_if: "no text predictor columns exist"

  - id: train_ordinal_regressor
    intent: "Apply StandardScaler (fit on train only). Train a regressor on the integer-mapped labels (RandomForestRegressor is the robust default; note that a proportional-odds/ordinal-logit model is the classical alternative when interpretability matters). Wrap predictions: round, then clip to [min, max] of the mapped labels, and store the wrapper logic in `predict_ordinal`."
    worker_tier: T3
    depends_on: [vectorize_text_features]
    attached_skills: [supervised_modeling]
    produces: [ordinal_model, scaler, predict_ordinal]
    postconditions:
      - "ordinal_model is not None"

  - id: validate_distance_metrics
    intent: "Evaluate with BOTH Mean Absolute Error (primary — distance between classes matters) and exact accuracy (secondary). Generate a confusion matrix to visualize off-by-one vs catastrophic errors. Compare MAE against the trivial predict-the-median baseline."
    worker_tier: T2
    depends_on: [train_ordinal_regressor]
    attached_skills: [visualization_best_practices]
    produces: [mae, exact_accuracy]
    postconditions:
      - "isinstance(mae, float)"
    required_metrics: [mae, exact_accuracy]

  - id: persist_model_bundle
    intent: "Save a joblib bundle containing the model, scaler, PCA object (if used), and label_mapping as 'ordinal_model_bundle.joblib' — everything inference needs, nothing less."
    worker_tier: T3
    depends_on: [validate_distance_metrics]
    postconditions:
      - "ordinal_model is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "ORDER IS SIGNAL: never treat ordered categories as unordered classes — the loss must penalize distance, which is why the regression wrapper (or an ordinal-logit model) is used."
  - "PRIMARY METRIC IS MAE on the integer-mapped labels; exact accuracy is secondary."
  - "Predictions must be rounded AND clipped to the valid label range."
  - "Scalers/PCA are fit on the training split only."
  - "Persist the mapping, scaler, and any projection together with the model — they are part of the model."
  - "Random seeds must be fixed (random_state=42)."
---

# Ordinal Classification (Ordered Multi-Class)

## Rationale
Ordinal targets ('Low' < 'Medium' < 'High', star ratings, Likert scales) carry order
information that plain classification discards: being one class off should cost less
than being three off. The regression-to-classification wrapper treats the categories as
a gradient — train a regressor on integer-mapped labels, then round and clip. The
classical alternative is a proportional-odds (ordinal logistic) model, preferable when
coefficient interpretability is required; the wrapper generalizes better to nonlinear
feature effects.

## Key Logic: The Wrapper
```python
def predict_ordinal(model, X, min_val, max_val):
    preds = model.predict(X)
    return np.clip(np.round(preds), min_val, max_val).astype(int)
```

## Best Practices
- **Evaluation**: MAE first — an MAE of 0.5 means predictions are on average half a
  category off. Always show the confusion matrix so off-by-one errors are visible.
- **Text predictors**: embed locally, reduce dimensionality (derive the dimension from
  the data, don't hardcode), standardize before the regressor.
- **Persistence**: bundle mapping + scaler + projection + model in one joblib file.
