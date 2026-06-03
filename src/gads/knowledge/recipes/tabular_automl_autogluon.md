---
id: tabular_automl.autogluon.standard
version: 1.0.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [binary_classification, multiclass_classification, regression, classification]
  data_modality: [tabular]
  signals:
    - objective_contains: [predict, classify, forecast, model, accuracy, performance]
  anti_signals:
    - temporal_ordering_required: true
    - task: causal_inference
    - task: anomaly_detection

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [autogluon, sklearn, pandas]

# ——— DAG TEMPLATE ———
dag:
  - id: eda_and_target_profile
    intent: >
      Profile the dataset and target column:
      (1) Print shape, dtypes, and null counts.
      (2) Compute target distribution — for classification: class counts and minority fraction;
          for regression: min/max/mean/std.
      (3) Identify and print: any high-cardinality text columns (>50 unique values, dtype object),
          datetime columns, ID-like columns (name contains 'id', 'index', 'key' — drop these before
          training).
      (4) Store naive_baseline metric: for classification, `majority_class_rate = target.value_counts(normalize=True).max()`;
          for regression, `mean_absolute_deviation = float((target - target.mean()).abs().mean())`.
      Print a summary table of variable roles (feature / target / drop).
    worker_tier: T2
    produces: [target_col, problem_type, drop_cols, naive_baseline]
    postconditions:
      - "isinstance(target_col, str)"
      - "problem_type in ('binary', 'multiclass', 'regression')"
    required_metrics: [naive_baseline]

  - id: train_automl_model
    intent: >
      Train an AutoGluon TabularPredictor on the dataset:
      (1) Drop ID-like columns identified in eda_and_target_profile.
      (2) Split into train/test using stratified split for classification (test_size=0.2,
          random_state=42), or random split for regression.
      (3) Fit the predictor:
            predictor = TabularPredictor(label=target_col, eval_metric=eval_metric, verbosity=0)
                          .fit(df_train, presets='good_quality', time_limit=120,
                               excluded_model_types=['NN_TORCH', 'FASTAI'])
          eval_metric: 'roc_auc' for binary, 'accuracy' for multiclass, 'rmse' for regression.
      (4) Evaluate on the test set using the leaderboard (more reliable than evaluate()):
            leaderboard = predictor.leaderboard(df_test, silent=True)
            test_score = float(leaderboard.iloc[0]['score_test'])
          Do NOT use predictor.evaluate(df_test)[predictor.eval_metric] — the key name
          may differ from eval_metric and will raise a KeyError.
      (5) Print leaderboard.head(8) to show model rankings.
      (6) Save predictor: joblib.dump(predictor, 'model.joblib')
      (7) Emit the score immediately so it is captured in metrics.json:
            gads_emit_insight('model_score', f'{predictor.eval_metric}={test_score:.4f}, naive_baseline={naive_baseline:.4f}')
    depends_on: [eda_and_target_profile]
    worker_tier: T2
    produces: [predictor, df_train, df_test, test_score, eval_metric]
    postconditions:
      - "predictor is not None"
      - "isinstance(test_score, float)"
    required_metrics: [test_score]

  - id: feature_importance_and_insights
    intent: >
      Extract and visualise model insights:
      (1) Compute feature importance:
            fi = predictor.feature_importance(df_test, subsample_size=1000, num_shuffle_sets=1, silent=True)
          subsample_size=1000 and num_shuffle_sets=1 are MANDATORY — without them the permutation
          importance runs over the full test set and will exceed the 600s sandbox timeout.
          fi is a DataFrame where the INDEX contains feature names and the column 'importance'
          contains the permutation importance scores. Access with fi.index and fi['importance'].
          Do NOT use fi.columns for feature names — they are in the index.
      (2) Plot a horizontal bar chart of the top 15 features by importance:
            fi_top = fi.head(15)
            fig = px.bar(x=fi_top['importance'], y=fi_top.index, orientation='h')
          Save as `figure_1_feature_importance.json`.
      (3) For classification: plot ROC curve (binary) or confusion matrix (multiclass).
          For confusion matrix use px.imshow(cm, text_auto=True) where cm = confusion_matrix(y_test, y_pred).
          Do NOT use px.heatmap() — it does not exist; use px.imshow() instead.
          Save as `figure_2_model_performance.json`.
      (4) For regression: plot predicted vs actual scatter. Save as `figure_2_predicted_vs_actual.json`.
      (5) Emit a `gads_emit_insight()` call summarising the top 3 features and the test score
          relative to the naive baseline.
    depends_on: [train_automl_model]
    worker_tier: T2
    produces: [feature_importance_df]
    postconditions:
      - "feature_importance_df is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "USE AUTOGLUON: always use TabularPredictor.fit() — NEVER build manual sklearn pipelines, encoders, or imputers. AutoGluon handles all preprocessing internally."
  - "TIME LIMIT: always set time_limit=120 in .fit(). Never omit it — unlimited training will exhaust the sandbox budget."
  - "VERBOSITY: always set verbosity=0 in TabularPredictor() to suppress log spam."
  - "EXCLUDE NEURAL NETS IN CPU SANDBOX: always pass excluded_model_types=['NN_TORCH', 'FASTAI'] unless the user explicitly requests neural network models."
  - "DROP ID COLUMNS: remove columns whose names contain 'id', 'index', 'key', or 'name' (case-insensitive) before fitting — they cause leakage."
  - "SAVE WITH JOBLIB: joblib.dump(predictor, 'model.joblib') — never use pickle (sandbox-blocked)."
  - "EVAL METRIC: roc_auc for binary, accuracy for multiclass, rmse for regression."
  - "Random state must be 42 everywhere."
---

# AutoML Classification & Regression with AutoGluon

## Rationale
This recipe replaces the manual sklearn pipeline approach (preprocessing → split → model selection → hyperparameter tuning → evaluate) with a single AutoGluon `.fit()` call. AutoGluon trains and ensembles LightGBM, CatBoost, XGBoost, Random Forest, and Extra Trees automatically, handling missing values, categorical encoding, and feature type inference without any Coder intervention. This makes the recipe maximally reliable for local LLMs — the Coder calls one function with three arguments rather than constructing a multi-step pipeline.

## When to use
Use for standard supervised learning on tabular data when the objective is to predict a target column as accurately as possible. The spec needs only to identify the target column and whether the task is classification or regression. Use the existing `binary_classification_tabular.md` recipe instead if interpretability, manual feature engineering, or a specific model architecture is required.

## Key Constraints
- Datasets over 500K rows should be subsampled to 200K before training to avoid OOM.
- The sandbox has no GPU — neural net models are excluded by default.
- `time_limit=120s` is the default; increase to 240s in the spec if the dataset is large.
