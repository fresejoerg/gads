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
      Profile the dataset. Use EXACTLY the code patterns below — do not vary the variable names.
      (1) Print shape, dtypes, null counts:
            print(df.shape); print(df.dtypes); print(df.isnull().sum())
      (2) Print target value counts:
            print(df[target_col].value_counts())
      (3) Determine problem type and store in `problem_type`:
            problem_type = 'binary' if df[target_col].nunique() == 2 else ('multiclass' if df[target_col].dtype == object else 'regression')
            print("problem_type:", problem_type)
      (4) Find and print ID-like columns to drop; store in `drop_cols`:
            drop_cols = [c for c in df.columns if any(x in c.lower() for x in ['id', 'index', 'key'])]
            print("drop_cols:", drop_cols)
      (5) Compute naive_baseline — store in a variable named EXACTLY `naive_baseline`:
            vc = df[target_col].value_counts(normalize=True)
            naive_baseline = float(vc.max())
            print("naive_baseline:", naive_baseline)
      Do NOT use pd.api.types.is_datetime64_ns — it does not exist.
      Do NOT use f-strings for printing naive_baseline — use plain print() as shown above.
      Do NOT wrap code in def main() or any function — all variables must be at global scope
      so the next task can read them from the kernel.
    worker_tier: T2
    produces: [target_col, problem_type, drop_cols, naive_baseline]
    postconditions:
      - "isinstance(target_col, str)"
      - "problem_type in ('binary', 'multiclass', 'regression')"
    required_metrics: [naive_baseline]

  - id: train_automl_model
    intent: >
      Train an AutoGluon TabularPredictor. Use EXACTLY the code below — do not vary the structure.
      DO NOT call pd.read_csv() — the DataFrame `df` is already in the kernel from the previous task.
      (1) Drop ID-like columns and set eval_metric:
            df_clean = df.drop(columns=drop_cols, errors='ignore')
            eval_metric = 'roc_auc' if problem_type == 'binary' else ('f1_macro' if problem_type == 'multiclass' else 'rmse')
      (2) Split — always use try/except to handle both classification and regression:
            try:
                df_train, df_test = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean[target_col])
            except ValueError:
                df_train, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)
      (3) Fit — put ALL arguments on ONE logical line inside the parentheses (no backslash continuation):
            predictor = TabularPredictor(label=target_col, eval_metric=eval_metric, verbosity=0).fit(df_train, presets='good_quality', time_limit=120, excluded_model_types=['NN_TORCH', 'FASTAI'])
      (4) Evaluate:
            leaderboard = predictor.leaderboard(df_test, silent=True)
            test_score = float(leaderboard.iloc[0]['score_test'])
            print(leaderboard.head(8))
      (5) Save and emit:
            joblib.dump(predictor, 'model.joblib')
            gads_emit_insight('model_score', f'{eval_metric}={test_score:.4f}, naive_baseline={naive_baseline:.4f}')
            if problem_type == 'binary' and naive_baseline > 0.9:
                from sklearn.metrics import average_precision_score
                y_prob = predictor.predict_proba(df_test)
                pr_auc = float(average_precision_score(df_test[target_col], y_prob.iloc[:, 1]))
                gads_emit_insight('secondary_metric', f'Average Precision (PR AUC)={pr_auc:.4f}')
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
            # For imbalanced datasets, use a stratified sample to ensure the minority class is represented:
            if problem_type == 'binary' and df_test[target_col].value_counts().min() < 100:
                fi_df = df_test.groupby(target_col, group_keys=False).apply(lambda x: x.sample(min(len(x), 500), random_state=42))
                fi = predictor.feature_importance(fi_df, num_shuffle_sets=1, silent=True)
            else:
                fi = predictor.feature_importance(df_test, subsample_size=1000, num_shuffle_sets=1, silent=True)
          subsample_size=1000 (or equivalent custom sample size) and num_shuffle_sets=1 are MANDATORY — without them the permutation
          importance runs over the full test set and will exceed the 600s sandbox timeout.
          fi is a DataFrame where the INDEX contains feature names and the column 'importance'
          contains the permutation importance scores. Access with fi.index and fi['importance'].
          Do NOT use fi.columns for feature names — they are in the index.
      (2) Plot a horizontal bar chart of the top 15 features by importance:
            fi_top = fi.head(15)
            fig = px.bar(x=fi_top['importance'], y=fi_top.index, orientation='h')
          Save as `figure_1_feature_importance.json`.
      (3) For classification: calibrate decision threshold and plot confusion matrix.
           y_prob = predictor.predict_proba(df_test)
           y_test = df_test[target_col]
           if problem_type == 'binary':
               cal = gads_calibrate_threshold(y_test, y_prob)
               best_t = cal['best_threshold']
               y_pred = (y_prob.iloc[:, 1] >= best_t).astype(int)
               gads_emit_insight('calibrated_threshold', f"Optimal threshold: {best_t:.4f}")
           else:
               y_pred = predictor.predict(df_test)
           cm = confusion_matrix(y_test, y_pred)
           fig2 = px.imshow(cm, text_auto=True)
           fig2.write_json('figure_2_model_performance.json')
           Do NOT use px.heatmap() — it does not exist; use px.imshow() instead.
          If computing ROC-AUC: predict_proba returns a DataFrame in AutoGluon.
          Use y_prob.iloc[:, 1] NOT y_prob[:, 1] (numpy-style indexing fails on DataFrames).
      (4) For regression: plot predicted vs actual scatter. Save as `figure_2_predicted_vs_actual.json`.
      (5) Emit insights — use simple variable references, NOT complex expressions inside f-strings:
            top3 = ', '.join(fi.sort_values('importance', ascending=False).head(3).index.tolist())
            gads_emit_insight('feature_summary', f'Top 3 features: {top3}. Test score: {test_score}')
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
  - "EVAL METRIC: roc_auc for binary, f1_macro for multiclass, rmse for regression."
  - "Random state must be 42 everywhere."
  - "THRESHOLD CALIBRATION: For binary classification, always calibrate the decision threshold on validation/test data using gads_calibrate_threshold() before evaluating predictions or confusion matrices."
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
