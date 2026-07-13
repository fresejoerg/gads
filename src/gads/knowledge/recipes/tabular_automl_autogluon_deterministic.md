---
id: tabular_automl.autogluon.deterministic
version: 1.1.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [binary_classification, multiclass_classification, regression, classification]
  data_modality: [tabular]
  signals:
    - objective_contains: [reproducible, deterministic, benchmark, identical results]
    - objective_contains: [predict, classify, model, accuracy, performance]
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
      (4) Fit an AutoGluon TabularPredictor under the DETERMINISM invariant (fixed
          model portfolio — the exact fit arguments are in the invariants block).
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

# ——— GLOBAL INVARIANTS ———
invariants:
  - "USE AUTOGLUON: always use TabularPredictor.fit() — NEVER build manual sklearn pipelines, encoders, or imputers. AutoGluon handles all preprocessing internally."
  - "DETERMINISM: NEVER pass time_limit or presets to .fit(). Always pass EXACTLY hyperparameters={'GBM': {}, 'XGB': {}, 'RF': {}}, num_bag_folds=0, fit_weighted_ensemble=True — a fixed model portfolio makes the trained ensemble independent of wall-clock and machine load, so repeat runs reproduce identical scores."
  - "MODEL PORTFOLIO IS FIXED: do not add, remove, or configure models beyond the mandated hyperparameters dict — no neural nets, no CatBoost, no extra params."
  - "VERBOSITY: always set verbosity=0 in TabularPredictor()."
  - "DROP ID COLUMNS: remove columns whose names contain 'id', 'index', 'key', or 'name' (case-insensitive) before fitting — they cause leakage. Print what was dropped."
  - "EVAL METRIC: roc_auc for binary, f1_macro for multiclass, rmse for regression."
  - "Random state must be 42 everywhere."
  - "THRESHOLD CALIBRATION: for binary classification, calibrate the decision threshold with gads_calibrate_threshold() before evaluating predictions or confusion matrices."
  - "PERSIST: save the fitted predictor as model.joblib."
---

# Deterministic AutoML with AutoGluon (reproducible variant)

## Rationale
Variant of `tabular_automl.autogluon.standard` with one changed decision: the fit uses a
FIXED model portfolio instead of a wall-clock time budget. A time budget makes the
trained stack depend on machine load — identical generated code produced ROC-AUC 0.9571
vs 0.9693 across days (research/JOURNAL.md 2026-07-09, kept out of the prompt-injected
invariants deliberately). With a fixed portfolio and seeded splits, consecutive runs of
the same spec produce bitwise-identical scores, which is what `research/benchmarks/`
scoring assumes and what the platform's reproducibility metric demands.

Like the standard recipe, a single AutoGluon `.fit()` call replaces the manual pipeline
(preprocessing → split → model selection → tuning → evaluation), which is the most
reliable shape for small local models: one function, three arguments.

## When to use
Supervised learning on tabular data whenever results must be reproducible — benchmarks,
regulated analyses, any spec that will be re-run and compared. Trades a small amount of
accuracy (narrower model space than a long time budget) for exact repeatability. Use
`tabular_automl.autogluon.standard` when a one-off exploratory run should search a wider
space, or `binary_classification_tabular` when interpretability or a specific
architecture is required.

## Key Constraints
- Datasets over 500K rows should be subsampled to 200K before training (OOM risk).
- The sandbox has no GPU — the fixed portfolio is CPU-only by construction.
- No time budget: the GBM+XGB+RF portfolio (no bagging) trains well inside the sandbox
  timeout for ≤200K-row datasets. For larger data, cap rows via the spec's
  `sample_rows` — never reintroduce time_limit.
