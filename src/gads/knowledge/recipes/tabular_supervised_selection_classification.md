---
id: tabular_supervised.selection.classification
version: 1.1.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [binary_classification, multiclass_classification, classification]
  data_modality: [tabular]
  signals:
    - objective_contains: [compare models, model selection, choose a model, which model,
                           model comparison, best algorithm, hyperparameter tuning,
                           tune the model, tune hyperparameters, feature importance,
                           justify the model, reasoned choice, why this model,
                           candidate models]
  anti_signals:
    - objective_contains: [automl, autogluon, leaderboard, deterministic portfolio]
    - temporal_ordering_required: true
    - task: causal_inference
    - task: anomaly_detection

# ——— SCHEMA CONTRACT ———
requires:
  capabilities: [sklearn, pandas, matplotlib, optuna, skore]

# ——— DAG TEMPLATE ———
dag:
  - id: load_prepared_data
    intent: >
      Load the modelling table and bind `X_train`, `y_train`, `X_test`, `y_test`.
      PREFER a chained upstream transform run: if `upstream/transformed_train.parquet` and
      `upstream/transformed_test.parquet` exist, load those as-is — they were already split
      and fitted train-only by the upstream run, so re-splitting or re-fitting any
      transformation would undo that leakage guard. Otherwise load the workspace dataset and
      make a stratified 80/20 split with random_state=42. Exclude the target from the
      features. Print the resulting shapes and bind `n_train_rows` and `n_features`.
      Do NOT encode or scale anything here — categorical columns are handled inside the
      modelling pipeline so the encoder is refitted per fold. Leave them as they are.
    worker_tier: T3
    produces: [X_train, y_train, X_test, y_test, n_train_rows, n_features]
    required_metrics: [n_train_rows, n_features]
    attached_skills: [large_dataset_handling]
    fallback_native: gads_load_prepared_split
    fallback_call: "_r = gads_load_prepared_split(target_column=globals().get('target_column'), sample_rows=globals().get('sample_rows')); X_train = _r['X_train']; y_train = _r['y_train']; X_test = _r['X_test']; y_test = _r['y_test']; n_train_rows = _r['n_train_rows']; n_features = _r['n_features']"
    postconditions:
      - "len(X_train) == len(y_train) and len(X_test) == len(y_test)"
      - "list(X_train.columns) == list(X_test.columns)"

  - id: characterize_task
    intent: >
      Derive the facts that drive model choice and store them in `dataset_facts` (a dict):
      n_rows, n_features, rows_per_feature, n_numeric, n_categorical, max_cat_cardinality,
      missing_rate, has_missing, n_classes, minority_class_rate, class_balance. Do NOT
      profile the whole dataset — this is not EDA, it is the short list of quantities the
      selection rules branch on. Bind `n_classes` and `minority_class_rate` as top-level
      scalars. Print the facts as a readable summary.
    depends_on: [load_prepared_data]
    worker_tier: T3
    produces: [dataset_facts, n_classes, minority_class_rate]
    required_metrics: [n_classes, minority_class_rate]
    attached_skills: [tabular_profiling, model_selection_tabular]
    fallback_native: gads_dataset_facts
    fallback_call: "dataset_facts = gads_dataset_facts(X_train, y_train); n_classes = dataset_facts['n_classes']; minority_class_rate = dataset_facts['minority_class_rate']"
    postconditions:
      - "dataset_facts.get('n_rows') == len(X_train)"

  - id: shortlist_candidates
    intent: >
      THE REASONING STEP — this is the node the recipe exists for. Using `dataset_facts` and
      the model-selection decision table, nominate 2 to 4 candidate estimators and justify
      them. Store `candidates` as a list of {"name": str, "params": dict} using the exact
      supported names, and `selection_rationale` as text containing one sentence per
      candidate AND at least one sentence naming a family you ruled out and why.
      Always include a regularized linear model as the interpretable baseline.
      Do NOT fit anything here, and do NOT set random_state / class_weight /
      scale_pos_weight / n_jobs in params — the natives apply those uniformly so the
      comparison is not confounded.
    depends_on: [characterize_task]
    worker_tier: T2
    produces: [candidates, selection_rationale, n_candidates]
    required_metrics: [n_candidates]
    attached_skills: [model_selection_tabular, supervised_modeling]
    rationale_required: true
    fallback_native: gads_default_shortlist
    fallback_call: "_s = gads_default_shortlist(dataset_facts); candidates = _s['candidates']; selection_rationale = _s['selection_rationale']; n_candidates = _s['n_candidates']"
    postconditions:
      - "2 <= len(candidates) <= 5"
      - "len(selection_rationale) > 80"

  - id: bakeoff
    intent: >
      Score every shortlisted candidate under ONE protocol by calling the native:
        result = gads_candidate_bakeoff(X_train, y_train, candidates, cv=5, seed=42)
        bakeoff_table = result["bakeoff_table"]; best_candidate = result["best_candidate"]
        best_cv_score = result["best_cv_score"]
      Do NOT write the cross-validation loop yourself — identical folds, one seed and one
      metric across all candidates is the property that makes the comparison mean anything,
      and the native is what guarantees it. Print the table, then state whether the winner's
      margin over the runner-up exceeds the fold standard deviation. If it does not, say so:
      a lead inside noise is not a result.
    depends_on: [shortlist_candidates]
    worker_tier: T2
    produces: [bakeoff_table, best_candidate, best_cv_score]
    required_metrics: [best_cv_score]
    attached_skills: [supervised_modeling, model_selection_tabular]
    fallback_native: gads_candidate_bakeoff
    fallback_call: "_b = gads_candidate_bakeoff(X_train, y_train, candidates, cv=5, seed=42); bakeoff_table = _b['bakeoff_table']; best_candidate = _b['best_candidate']; best_cv_score = _b['best_cv_score']"
    postconditions:
      - "len(bakeoff_table) >= 2"

  - id: tune_best
    intent: >
      Define an Optuna search space for `best_candidate` — the width, the priors and which
      4 to 6 parameters matter are YOUR judgment — then hand it to the native:
        tuning_result = gads_tune_model(X_train, y_train, best_candidate,
                                        search_space=search_space, n_trials=40, timeout_s=240)
        tuned_model = tuning_result["tuned_model"]; best_params = tuning_result["best_params"]
        best_cv_score_tuned = tuning_result["best_cv_score_tuned"]
        n_trials_completed = tuning_result["n_trials_completed"]
      Do NOT write the study loop, and do NOT change n_trials or timeout_s upward — the
      budget is enforced inside the native because nothing upstream can see it. Use
      "log": true for parameters spanning orders of magnitude. Report whether tuning
      actually beat the untuned baseline (`baseline_cv_score`); if it did not, say so.
    depends_on: [bakeoff]
    worker_tier: T2
    produces: [tuned_model, best_params, tuning_result, n_trials_completed, best_cv_score_tuned]
    required_metrics: [n_trials_completed, best_cv_score_tuned]
    attached_skills: [supervised_modeling, model_selection_tabular]
    fallback_native: gads_tune_model
    fallback_call: "_t = gads_tune_model(X_train, y_train, best_candidate, n_trials=30, timeout_s=200); tuned_model = _t['tuned_model']; best_params = _t['best_params']; best_cv_score_tuned = _t['best_cv_score_tuned']; n_trials_completed = _t['n_trials_completed']; tuning_result = _t"
    postconditions:
      - "n_trials_completed >= 1"

  - id: selection_audit
    intent: >
      Run BOTH gates on what was chosen and fitted, and do not re-implement either:
        selection_audit = gads_audit_model_choice(best_candidate, dataset_facts,
                                                  bakeoff_table, tuning_result)
        audit = gads_audit_model(tuned_model, X_train=X_train, y_train=y_train,
                                 X_test=X_test, y_test=y_test)
      The first adjudicates the model CHOICE against the selection rules; the second is the
      skore methodological audit of the fitted estimator. Bind `n_selection_issues` and
      `selection_issues` (the combined findings list). Summarize every finding as an insight
      and address it in plain language — these findings are material for the report, not
      errors to suppress.
    depends_on: [tune_best]
    worker_tier: T2
    produces: [selection_audit, selection_issues, n_selection_issues]
    required_metrics: [n_selection_issues]
    attached_skills: [model_audit, model_selection_tabular]
    fallback_native: gads_audit_model_choice
    fallback_call: "selection_audit = gads_audit_model_choice(best_candidate, dataset_facts, globals().get('bakeoff_table'), globals().get('tuning_result')); selection_issues = list(selection_audit.get('warnings', [])) + list(selection_audit.get('issues', [])); n_selection_issues = selection_audit['n_selection_issues']"
    postconditions:
      - "'n_selection_issues' in dir()"

  - id: holdout_evaluation
    intent: >
      Refit the tuned model on the full training partition and evaluate it ONCE on the
      untouched test partition. Bind `macro_f1`, `roc_auc` and `log_loss` as top-level
      scalars under exactly those names, plus `y_pred` and `y_prob`. Report the
      majority-class baseline next to every headline metric — a metric without a baseline is
      not evidence. For a BINARY target, calibrate the decision threshold with
      gads_calibrate_threshold(y_test, y_prob) before computing any label-based metric. For
      3 or more classes use argmax and do NOT slice predict_proba to one column.
      Alias the sklearn import if it would shadow the `log_loss` variable name.
    depends_on: [tune_best]
    worker_tier: T2
    produces: [y_pred, y_prob, evaluation, macro_f1, roc_auc, log_loss]
    required_metrics: [macro_f1, roc_auc, log_loss]
    attached_skills: [supervised_modeling]
    fallback_native: gads_evaluate_holdout
    fallback_call: "_e = gads_evaluate_holdout(tuned_model, X_train, y_train, X_test, y_test); y_pred = _e['y_pred']; y_prob = _e['y_prob']; macro_f1 = _e['macro_f1']; roc_auc = _e['roc_auc']; log_loss = _e['log_loss']; evaluation = _e"
    postconditions:
      - "macro_f1 >= 0.0 and macro_f1 <= 1.0"

  - id: feature_importance
    intent: >
      Measure what actually drives the predictions:
        importance = gads_feature_importance(tuned_model, X_test, y_test, n_repeats=5)
        importance_table = importance["importance_table"]
        n_features_reported = importance["n_features_reported"]
      Permutation importance on HELD-OUT data — do not use `.feature_importances_` as the
      headline, it is impurity-based and biased toward high-cardinality features. Plot the
      top 20 as a horizontal bar chart with the most important at the top, save the figure,
      and read the top 5 in plain language: what does each one mean for this problem?
    depends_on: [holdout_evaluation]
    worker_tier: T2
    produces: [importance_table, importance, n_features_reported]
    required_metrics: [n_features_reported]
    attached_skills: [supervised_modeling, tabular_visualization, visualization_best_practices]
    fallback_native: gads_feature_importance
    fallback_call: "importance = gads_feature_importance(tuned_model, X_test, y_test, n_repeats=5); importance_table = importance['importance_table']; n_features_reported = importance['n_features_reported']"
    postconditions:
      - "len(importance_table) >= 1"

  - id: performance_report
    intent: >
      Write the model card to `model_card.md` and bind `model_card_text`. Cover, in order:
      what was chosen and WHY (quote the rationale), what it beat in the bakeoff and by how
      much relative to fold noise, the tuned hyperparameters and whether tuning helped, the
      held-out metrics against the majority-class baseline, the audit findings from both
      gates, and the top drivers. Close with honest limitations — a truncated search, a
      model that does not beat its baseline, or a wide-uncertainty estimate on small data
      all belong there. Do not overstate: this report's value is that it can be defended.
    depends_on: [selection_audit, feature_importance]
    worker_tier: T2
    produces: [model_card_text]
    attached_skills: []
    fallback_native: gads_model_card
    fallback_call: "_c = gads_model_card(chosen=globals().get('best_candidate'), dataset_facts=globals().get('dataset_facts'), bakeoff_table=globals().get('bakeoff_table'), tuning_result=globals().get('tuning_result'), evaluation=globals().get('evaluation'), importance=globals().get('importance'), selection_rationale=globals().get('selection_rationale'), selection_audit=globals().get('selection_audit')); model_card_text = _c['model_card_text']"
    postconditions:
      - "len(model_card_text) > 200"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "HOLDOUT DISCIPLINE: the test partition is read exactly once, in holdout_evaluation. No node before it may touch X_test or y_test; feature_importance may, because it runs after evaluation is final."
  - "ONE PROTOCOL: candidates are compared only via gads_candidate_bakeoff — identical folds, one seed (42), one metric. Never hand-write the comparison loop; a ranking produced on differently-shuffled folds reflects the split, not the model."
  - "PREPROCESSING INSIDE THE PIPELINE: the bakeoff, tuning and evaluation natives wrap every candidate in a Pipeline whose first step imputes and encodes, so preprocessing is refitted on each training fold. Never encode the feature matrix yourself before these calls — an encoder fitted once across all folds leaks fold statistics into the score. A consequence: `tuned_model` is a sklearn Pipeline, not a bare estimator, and permutation importance is reported per ORIGINAL column rather than per one-hot dummy."
  - "TUNING IS TRAIN-ONLY: hyperparameter search runs strictly inside the training partition, via gads_tune_model. Its n_trials/timeout_s budget must not be raised — RuntimeOracle cannot see a study inside a native, so the wall-clock cap in the native is the only thing enforcing it."
  - "PERMUTATION, NOT IMPURITY: feature importance is permutation importance on held-out data. `.feature_importances_` is impurity-based and biased toward high-cardinality and continuous features; it may be shown for contrast but never as the headline."
  - "BASELINE ALWAYS: every headline metric is reported next to the trivial baseline (majority class for classification). A model that does not beat it is a finding to report, not a failure to hide."
  - "CLASS WEIGHTS: the bakeoff and tuning natives apply class_weight='balanced' (or scale_pos_weight) uniformly. Do not set these in candidate params — doing so confounds the comparison."
  - "STRING LABELS: class labels may be strings ('<=50K'/'>50K'), not 0/1. A thresholded probability yields 0/1, so comparing it against a string y_test raises \"Labels in y_true and y_pred should be of the same type\". Map thresholded predictions back to the original label dtype before ANY metric, confusion matrix or report — e.g. `classes = sorted(pd.Series(y_test).unique()); y_pred = np.where(y_prob >= threshold, classes[-1], classes[0])`. Never cast the labels themselves with int()."
  - "NATIVE RETURN KEYS: read the exact keys a native documents. gads_calibrate_threshold returns `best_threshold`, gads_feature_importance returns `importance_table`, gads_candidate_bakeoff returns `bakeoff_table`/`best_candidate`. Short aliases (`threshold`, `importance`, `table`, `best`) are also provided, but print `sorted(result.keys())` if unsure rather than guessing a third name."
  - "THRESHOLD CALIBRATION: when n_classes == 2, calibrate the decision threshold via gads_calibrate_threshold before computing any label-based metric. For 3+ classes a single threshold is meaningless under argmax — use argmax, and never slice predict_proba to one column."
  - "REASONED CHOICE: shortlist_candidates must state why each candidate was nominated AND name at least one family ruled out. A choice without a defence is a failed node even when the code runs."
  - "RANDOM SEEDS: random_state=42 everywhere, so the run is reproducible."
---

# Reasoned Model Selection for Tabular Classification

## Rationale

`tabular_automl.autogluon.*` already produces a better model than this recipe will. It
produces a leaderboard, not an argument. This recipe exists for the case where the
deliverable is a **defended** choice: which model, why that one, what it beat, by how much,
and what the choice costs.

That difference is also what makes it the interesting recipe to measure. AutoGluon's
accuracy is the harness's; the reasoning in `shortlist_candidates` is the model's, and it
is the thing under test (approach_docs/022 §1).

The DAG follows the 019 rule on every node. Deciding *which* candidates to try, *how wide*
a search space to give the winner, and *how* to narrate the result are genuinely variable
work and stay model-generated. Comparing candidates on identical folds, confining tuning to
the training partition, measuring importance on held-out data by permutation, and enforcing
a wall-clock budget are single-right-answer operations that hand-written code reproduces
incorrectly more often than not — those are native.

The gate deserves its own note. `gads_audit_model_choice` checks the choice against the
same rules the `model_selection_tabular` skill states in prose, but it **adjudicates rather
than decides**: the model still picks, and the gate reports afterwards. A native that
simply returned the right estimator would be more accurate and would measure nothing. This
is also the extension point — new constraints ("prefer RF under 1000 rows") are added as
rules to the gate and prose to the skill, never as a native that takes the decision away.

## When to use

A tabular classification target where the user wants to know *which* model and *why* —
"compare models", "which algorithm", "tune the hyperparameters", "what drives the
predictions". For maximum accuracy with no interpretability or justification requirement,
use `tabular_automl.autogluon.standard`. For a single interpretable baseline with readable
coefficients and no comparison, use `binary_classification.tabular.standard`.

## Chaining

Designed to run downstream of `tabular_eda.descriptive.standard` →
`tabular_transform.apply_manifest`, launched with `artifacts_from: <transform run uuid>`,
in which case node 1 picks up `upstream/transformed_{train,test}.parquet` and does not
re-split. It also runs standalone against a clean dataset, which is what makes a benchmark
sweep one launch per dataset instead of three.

## v1.1 — evidence-driven hardening

The local A/B on `qwen3.8-27b` (2026-08-18, runs `ea7593b6` / `8fa512a0`) produced the first
real evidence for the hardening ladder, and every v1.1 change is a promotion justified by it
rather than by speculation:

- **String labels: prose → invariant.** Thresholded predictions are `0/1` while `y_test`
  holds `'<=50K'`/`'>50K'`; the resulting dtype clash caused 3 of 14 failures.
  `binary_classification.tabular.standard` already carried this as an invariant — it should
  have been carried across from the start.
- **Native return keys: prose → invariant, plus short aliases in the natives.**
  `result['threshold']` (canonical: `best_threshold`) killed `holdout_evaluation` in BOTH
  runs — 5 failures, and the repeated-reason stop that ended run A. `result['importance']`
  (canonical: `importance_table`) cost `feature_importance` an attempt. The model reaches
  for the short natural name, so the natives now answer to both.

Nodes 1–6 passed on the model's own code in both runs, including the reasoning node, the
bakeoff and authoring a valid Optuna search space — so no selection rule (MS001–MS011) has
earned promotion above `warn` yet. That distribution is still being collected.

## Key constraints

- Binary and multiclass are both in scope. The threshold-calibration invariant applies only
  when `n_classes == 2`; multiclass decisions are argmax.
- The tuning budget (`n_trials=40`, `timeout_s=240`) is deliberate and must not be raised:
  the native fallback path runs under a 360s executor timeout, and the sandbox body timeout
  is 600s local / 300s cloud.
- Nine nodes is longer than any other recipe here. Resume-from-failed-node keeps a mid-DAG
  failure cheap, and every judgment node carries a fallback — but node count is this
  recipe's main local-model risk and the first thing to measure.
