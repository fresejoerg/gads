---
id: tabular_automl.autogluon.standard
version: 1.2.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [binary_classification, multiclass_classification, regression, classification]
  data_modality: [tabular]
  signals:
    - objective_contains: [predict, classify, model, accuracy, performance]
  anti_signals:
    - temporal_ordering_required: true
    - task: causal_inference
    - task: anomaly_detection
    # This recipe's signals are deliberately broad, which made it swallow objectives whose
    # deliverable is a *defended* choice of algorithm rather than an accurate model. Those
    # belong to `tabular_supervised.selection.classification` (approach_docs/022 §6). The
    # distinction is the deliverable, not the data: "predict churn" is AutoML, "compare
    # models and tell me which to use and why" is model selection.
    - objective_contains: [compare models, model selection, which model, which algorithm,
                           choose a model, model comparison, best algorithm,
                           hyperparameter tuning, tune the model, justify the model,
                           why this model]

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
      Profile the dataset and establish the naive baseline any model must beat. Follow
      the canonical profile pattern from the attached skill, producing exactly these
      global variables: `target_col` (the target named in the objective/spec hints),
      `problem_type` (one of 'binary', 'multiclass', 'regression' — remember that a
      numeric target with few distinct values is a coded class label, not regression),
      `drop_cols` (identifier-like columns that would leak), and `naive_baseline`
      (majority-class fraction for classification). Print the dataset shape, dtypes,
      null counts, and target distribution as evidence.
    worker_tier: T2
    produces: [target_col, problem_type, drop_cols, naive_baseline]
    attached_skills: [tabular_profiling]
    postconditions:
      - "isinstance(target_col, str)"
      - "problem_type in ('binary', 'multiclass', 'regression')"
    required_metrics: [naive_baseline]

  - id: train_automl_model
    intent: >
      Train and evaluate the model. The DataFrame `df` and the profile variables are
      already in the kernel — reuse them. Steps (code patterns are in the attached
      skill):
      (1) Drop the identifier columns found by the profile into `df_clean`.
      (2) Set `eval_metric` by problem type: roc_auc (binary), f1_macro (multiclass),
          rmse (regression).
      (3) Split 80/20 with random_state=42, stratified by the target for classification.
      (4) Fit an AutoGluon TabularPredictor under the TIME BUDGET invariant (the exact
          fit arguments are in the invariants block).
      (5) Evaluate on the held-out test split via the leaderboard; store the top score
          as float `test_score` and print the leaderboard head.
      (6) Save the predictor to model.joblib and emit a model_score insight comparing
          test_score to naive_baseline. If the task is binary AND severely imbalanced
          (naive_baseline > 0.9), additionally report average precision (PR AUC) as a
          secondary insight — ROC-AUC alone overstates performance on rare positives.
    depends_on: [eda_and_target_profile]
    worker_tier: T2
    produces: [predictor, df_train, df_test, test_score, eval_metric]
    attached_skills: [autogluon_tabular]
    postconditions:
      - "predictor is not None"
      - "isinstance(test_score, float)"
    required_metrics: [test_score]

  - id: feature_importance_and_insights
    intent: >
      Explain the model. Reuse `predictor`, `df_test`, and `problem_type` from the
      kernel.
      (1) Compute permutation feature importance on the held-out test set into `fi`,
          subsampled per the attached skill so it finishes inside the sandbox budget
          (stratify the subsample when the minority class is small).
      (2) Plot the top 15 features as a horizontal bar chart; save as
          figure_1_feature_importance.json.
      (3) For classification: calibrate the decision threshold with
          gads_calibrate_threshold, build label-dtype-safe predictions, and save a
          confusion matrix as figure_2_model_performance.json. For regression: save a
          predicted-vs-actual scatter as figure_2_predicted_vs_actual.json.
      (4) Emit a feature_summary insight naming the top 3 features and test_score.
      Store the importance table in `feature_importance_df`.
    depends_on: [train_automl_model]
    worker_tier: T2
    produces: [feature_importance_df]
    attached_skills: [autogluon_tabular, visualization_best_practices]
    postconditions:
      - "feature_importance_df is not None"

  - id: diagnostic_curves
    report:
      title: ROC and Precision-Recall Curves
      summary: How the model trades false positives against false negatives, against both baselines.
    intent: >
      CLASSIFICATION ONLY — if `problem_type` is 'regression', print that curves do not apply
      and bind `curve_diagnostics = {}` and `average_precision = float("nan")` instead of
      plotting anything. Otherwise plot the ROC and precision-recall curves by calling the
      native:
        curve_diagnostics = gads_plot_classification_curves(df_test[target_col], predictor.predict_proba(df_test))
      Do NOT hand-roll the curves. The native handles the binary/multiclass split, the label
      dtype, the one-vs-rest expansion and the two baselines (the ROC chance diagonal and the
      PR no-skill line at the positive rate), and writes both figures as Plotly JSON the
      dashboard renders directly. Bind `average_precision` from the result. Then state, in one
      or two sentences, what the PR curve says that the ROC curve does not — under class
      imbalance the ROC baseline is fixed at 0.5 while the PR baseline moves with prevalence,
      which is the whole reason both are plotted.
    depends_on: [train_automl_model]
    worker_tier: T2
    produces: [curve_diagnostics, average_precision]
    attached_skills: [visualization_best_practices]
    skippable_if: "problem_type == 'regression'"
    fallback_native: gads_plot_classification_curves
    fallback_call: "curve_diagnostics = gads_plot_classification_curves(df_test[target_col], predictor.predict_proba(df_test)); average_precision = curve_diagnostics.get('average_precision')"
    postconditions:
      - "curve_diagnostics is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "USE AUTOGLUON: always use TabularPredictor.fit() — NEVER build manual sklearn pipelines, encoders, or imputers. AutoGluon handles all preprocessing internally."
  - "TIME BUDGET: always set time_limit=120 and presets='good_quality' in .fit(). Never omit the time limit — unlimited training exhausts the sandbox budget."
  - "EXCLUDE NEURAL NETS: always pass excluded_model_types=['NN_TORCH', 'FASTAI'] — the sandbox is CPU-only."
  - "VERBOSITY: always set verbosity=0 in TabularPredictor()."
  - "DROP ID COLUMNS: remove columns whose names contain 'id', 'index', 'key', or 'name' (case-insensitive) before fitting — they cause leakage. Print what was dropped."
  - "EVAL METRIC: roc_auc for binary, f1_macro for multiclass, rmse for regression."
  - "Random state must be 42 everywhere."
  - "THRESHOLD CALIBRATION: for binary classification, calibrate the decision threshold with gads_calibrate_threshold() before evaluating predictions or confusion matrices."
  - "PERSIST: save the fitted predictor as model.joblib."
---

# AutoML Classification & Regression with AutoGluon (time-budgeted)

## Rationale
A single AutoGluon `.fit()` call replaces the manual pipeline (preprocessing → split →
model selection → hyperparameter tuning → evaluation). AutoGluon trains and ensembles
gradient-boosted trees and forests automatically, handling missing values, categorical
encoding, and feature-type inference without Coder intervention — the most reliable
shape for small local models: one function, three arguments.

This variant explores the model space under a wall-clock budget. **That makes its
results machine-load-dependent: repeat runs of the same spec can produce different
scores.** When reproducibility matters — benchmarks, comparisons across runs or
engines — use `tabular_automl.autogluon.deterministic`, which differs only in pinning a
fixed model portfolio.

## When to use
One-off exploratory supervised learning on tabular data where squeezing out accuracy
within a time budget matters more than run-to-run repeatability. The spec needs only the
target column. Use `binary_classification_tabular` instead if interpretability, manual
feature engineering, or a specific model architecture is required.

## Key Constraints
- Datasets over 500K rows should be subsampled to 200K before training (OOM risk).
- The sandbox has no GPU — neural nets are excluded.
- `time_limit=120` is the default; raise to 240 via a prompt/spec hint for large data.
