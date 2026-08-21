---
id: binary_classification.tabular.standard
version: 1.4.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [binary_classification, classification]
  data_modality: [tabular]
  signals:
    - target_type: categorical_binary
    - objective_contains: [interpretable, coefficients, why, drivers, explainable]
  anti_signals:
    - temporal_ordering_required: true

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [sklearn, pandas, matplotlib, skore]

# ——— DAG TEMPLATE ———
dag:
  - id: check_target_balance
    intent: "Verify the target column's distribution: print value counts (absolute and normalized), identify the positive class, and flag class imbalance (minority under ~25%). Store the class distribution in `target_distribution`."
    worker_tier: T3
    produces: [target_distribution]
    postconditions:
      - "target_distribution is not None"

  - id: preprocessing_pipeline
    intent: "Build the feature matrix: handle missing values (impute or drop with a stated rationale), encode categorical variables, and scale continuous features. Produce `X` (features only — target excluded) and `y` (target). Print the resulting shapes and dtypes."
    depends_on: [check_target_balance]
    worker_tier: T2
    produces: [X, y]
    attached_skills: [supervised_modeling]
    postconditions:
      - "X.isna().sum().sum() == 0"
      - "len(X) == len(y)"

  - id: stratified_split
    intent: "Stratified 80/20 train/test split with random_state=42, preserving class ratios. Verify by printing the positive-class rate in train and test — they should differ by well under 5 percentage points."
    depends_on: [preprocessing_pipeline]
    worker_tier: T3
    produces: [X_train, X_test, y_train, y_test]
    postconditions:
      - "abs(pd.Series(y_train).value_counts(normalize=True).max() - pd.Series(y_test).value_counts(normalize=True).max()) < 0.05"

  - id: train_baseline_model
    intent: "Train a Logistic Regression baseline with class_weight='balanced' on the training split. Evaluate on the test split with ROC-AUC (store as `baseline_auc`). Calibrate the decision threshold on the test probabilities with gads_calibrate_threshold(y_test, y_prob) and store `best_threshold`. Report the top coefficients by absolute value — interpretability is the reason this recipe exists. IMPORTANT: `predict_proba` returns a NumPy array, not a DataFrame — take the positive-class column positionally with `y_prob = model.predict_proba(X_test)[:, 1]` (NEVER `.iloc[:, 1]`, which raises 'numpy.ndarray object has no attribute iloc')."
    depends_on: [stratified_split]
    worker_tier: T2
    produces: [baseline_model, baseline_auc, best_threshold]
    attached_skills: [supervised_modeling]
    postconditions:
      - "baseline_auc > 0.5"

  - id: methodological_audit
    intent: "Run the native methodological-soundness audit on the fitted baseline: `audit = gads_audit_model(baseline_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)`. This wraps skore's EstimatorReport to check for leakage, over/under-fitting, class imbalance, worse-than-baseline performance, and low-value features; it writes `model_checks.json` and emits one insight per flagged issue. Do NOT re-implement these checks — just call the native function on the model already trained. Report the number of issues found and summarize each issue in an insight."
    depends_on: [train_baseline_model]
    worker_tier: T2
    produces: [audit]
    attached_skills: [model_audit]
    postconditions:
      - "'issues' in audit"

  - id: confusion_matrix_plot
    intent: "Apply the calibrated threshold to produce label-dtype-safe predictions on the test split, then generate and save a confusion-matrix figure plus a short plain-language reading of the error trade-off."
    depends_on: [train_baseline_model]
    worker_tier: T2
    attached_skills: [visualization_best_practices]
    postconditions:
      - "best_threshold is not None"

  - id: diagnostic_curves
    report:
      title: ROC and Precision-Recall Curves
      summary: How the model trades false positives against false negatives, against both baselines.
    intent: >
      Plot the ROC and precision-recall curves for the final model by calling the native:
        curve_diagnostics = gads_plot_classification_curves(y_test, baseline_model.predict_proba(X_test))
      Do NOT hand-roll the curves. The native handles the binary/multiclass split, the
      label dtype, the one-vs-rest expansion and the two baselines (the ROC chance
      diagonal and the PR no-skill line at the positive rate), and writes both figures as
      Plotly JSON the dashboard renders directly. Bind `average_precision` from the
      result. Then state, in one or two sentences, what the PR curve says that the ROC
      curve does not — under class imbalance the ROC baseline is fixed at 0.5 while the
      PR baseline moves with prevalence, which is the whole reason both are plotted.
    depends_on: [train_baseline_model]
    worker_tier: T2
    produces: [curve_diagnostics, average_precision]
    required_metrics: [average_precision]
    attached_skills: [visualization_best_practices]
    fallback_native: gads_plot_classification_curves
    fallback_call: "curve_diagnostics = gads_plot_classification_curves(y_test, baseline_model.predict_proba(X_test)); average_precision = curve_diagnostics['average_precision']"
    postconditions:
      - "curve_diagnostics.get('roc_auc') is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "NO MONKEY-PATCHING: never reassign a method on an estimator class or instance (`SomeClassifier.fit = ...`, `model.predict = ...`). Retries re-run your code in the SAME kernel, so the second run captures the already-patched function as the 'original' and recurses until RecursionError — and a patched CLASS corrupts every later step including the pre-written natives. If a model cannot consume a column type, route columns inside the Pipeline with ColumnTransformer + make_column_selector(dtype_include=...)."
  - "The target column must remain excluded from features (X)."
  - "Random seeds must be fixed for reproducibility (random_state=42)."
  - "CLASS WEIGHTS: always set class_weight='balanced' in sklearn classifiers (LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier) to safeguard against class imbalance."
  - "THRESHOLD CALIBRATION: always calibrate the decision threshold with gads_calibrate_threshold(y_true, y_prob, metric='f1') before generating predictions, confusion matrices, or F1/accuracy metrics."
  - "BASELINE FIRST: establish and report the logistic-regression baseline before any more complex model is considered."
  - "METHODOLOGICAL AUDIT (native): the fitted model must be passed through gads_audit_model (skore EstimatorReport) — do NOT hand-roll leakage/overfit/baseline checks. The audit writes model_checks.json; its issue-severity findings are ground-truth evidence for whether the analysis is methodologically sound."
---

# Standard Binary Classification for Tabular Data (interpretable baseline)

## Rationale
Industry best practice for supervised classification when **interpretability** matters:
validate the data before modeling, preserve class ratios via stratification, establish a
linear baseline whose coefficients can be read and defended, and only then consider more
complex architectures. A calibrated decision threshold turns scores into decisions
honestly — the default 0.5 is rarely optimal under class imbalance.

## When to use
A clear binary target (Yes/No, 1/0, churned/retained) with independent features, when
the deliverable is an *explainable* model or a defensible baseline. For maximum
predictive accuracy with no interpretability requirement, use
`tabular_automl.autogluon.standard` (or the deterministic variant when results must be
reproducible).

## Key Constraints
- Target column must be specified or unambiguous from the objective.
- Class labels may be strings — never assume 0/1 when computing rates or mapping
  thresholded probabilities back to labels.
