---
id: survival_analysis.ml
version: 1.1.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [survival_ml, survival_prediction, survival_analysis, time_to_event, survival]
  data_modality: [tabular]
  signals:
    - objective_contains: [predict survival, risk score, survival curve, random survival forest, individual risk, time-to-event prediction, rank patients by risk, who will churn first, predict time until]
    - has_duration_and_event_columns: true
  anti_signals:
    - objective_contains: [hazard ratio interpretation, explain the effect of, which factors drive]

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [pandas, scikit-learn, scikit-survival]

# ——— DAG TEMPLATE ———
dag:
  - id: prepare_survival_data
    intent: "Load the dataset into `df`. Identify the DURATION column (time observed, float >= 0) and the EVENT column (1/True = event occurred, 0/False = right-censored) from the objective/hints; store their names as `time_col` and `event_col`. NEVER drop censored rows — censoring carries information. Exclude any leakage/immortal-time features (anything recorded at or after the event). Print the number of events vs censored, the censoring rate, and the median observed time."
    worker_tier: T2
    produces: [df, time_col, event_col]
    attached_skills: [survival_analysis]
    postconditions:
      - "time_col is not None and event_col is not None"
      - "df[event_col].nunique() == 2"

  - id: build_target_and_features
    intent: "Build the scikit-survival structured target with the native node: `y = gads_make_surv_target(df, time_col=time_col, event_col=event_col)` — this returns the required `(event: bool, time: float)` structured array with the boolean cast handled and censored rows kept. Build the feature matrix `X = df.drop(columns=[time_col, event_col])` and encode all categorical columns to numeric (sksurv OneHotEncoder or pandas.get_dummies with drop_first=True) so X is fully numeric. Print X.shape and confirm no non-numeric columns remain."
    depends_on: [prepare_survival_data]
    worker_tier: T2
    produces: [X, y]
    attached_skills: [survival_scikit_survival]
    postconditions:
      - "len(X) == len(y)"

  - id: stratified_survival_split
    intent: "Split BOTH the feature matrix and the structured target: `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)`. The structured array y must be split alongside X (do not split only X). Print the number of events in train and test to confirm both splits contain events."
    depends_on: [build_target_and_features]
    worker_tier: T3
    produces: [X_train, X_test, y_train, y_test]
    attached_skills: [survival_scikit_survival]
    postconditions:
      - "len(X_train) == len(y_train)"
      - "len(X_test) == len(y_test)"

  - id: train_survival_model
    intent: "Train a Random Survival Forest as the strong default: `from sksurv.ensemble import RandomSurvivalForest; surv_model = RandomSurvivalForest(n_estimators=200, min_samples_leaf=15, n_jobs=-1, random_state=42); surv_model.fit(X_train, y_train)`. (GradientBoostingSurvivalAnalysis or CoxPHSurvivalAnalysis are acceptable alternatives.) `surv_model.predict(X)` returns a risk score; larger = higher risk. Report the training completed and the model type."
    depends_on: [stratified_survival_split]
    worker_tier: T2
    produces: [surv_model]
    attached_skills: [survival_scikit_survival]
    postconditions:
      - "hasattr(surv_model, 'predict')"

  - id: evaluate_survival
    intent: "Evaluate with the native node — do NOT hand-roll the metrics (they raise when evaluation times fall outside the follow-up range, and IBS needs a 2-D survival-probability matrix): `surv_metrics = gads_evaluate_survival(surv_model, X_train, y_train, X_test, y_test)`. It writes survival_metrics.json and emits insights. Report the IPCW C-index (preferred under censoring) and Harrell's C-index (0.5=random, >0.7=good), the mean time-dependent AUC, and the Integrated Brier Score (lower is better; 0.25=uninformative)."
    depends_on: [train_survival_model]
    worker_tier: T2
    produces: [surv_metrics]
    attached_skills: [survival_scikit_survival]
    postconditions:
      - "'harrell_cindex' in surv_metrics or 'ipcw_cindex' in surv_metrics"

  - id: risk_profiles_report
    intent: "Make the model's predictions concrete with the native node — do NOT hand-roll the survival-curve plotting (undefined time grids and plotting boilerplate reliably break here). Write EXACTLY this one statement and NOTHING else (the native selects low/median/high-risk subjects, plots their predicted survival curves, saves the figure, and emits an insight):\n\nprofiles = gads_plot_survival_curves(surv_model, X_test, n_profiles=3)\n\nThat is the entire task. Do not add any print statements or other code after it."
    depends_on: [evaluate_survival]
    worker_tier: T2
    attached_skills: [survival_scikit_survival, visualization_best_practices]
    postconditions:
      - "surv_metrics is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "NEVER drop right-censored rows or convert the problem to yes/no classification — censoring carries information and both destroy it."
  - "STRUCTURED TARGET: the scikit-survival target is a structured array (event: bool, time: float), built via gads_make_surv_target — never a 1-D vector of times or events. Split it alongside X."
  - "Random seeds must be fixed for reproducibility (random_state=42)."
  - "CENSORING-AWARE EVALUATION: never report plain accuracy/RMSE on the duration. Use the native gads_evaluate_survival (IPCW C-index, time-dependent AUC, Integrated Brier Score) — plain metrics ignore censoring and are invalid here."
  - "LEAKAGE / IMMORTAL TIME: exclude any covariate measured at or after the event, or that guarantees survival up to some time."
  - "Keep the SAME encoded feature columns for train and test (fit any encoder on train, apply to test)."
---

# Machine-Learning Survival Prediction (scikit-survival)

## Rationale
The workflow for **predicting** individual time-to-event risk with right-censored data,
when discrimination matters more than a defensible coefficient story. Random Survival
Forests and gradient-boosted survival models capture non-linearities and interactions a
Cox model cannot, at the cost of interpretability. Two things make or break correctness and
are exactly where hand-written code fails: (1) the target is scikit-survival's `(event,
time)` **structured array**, not a 1-D vector — built here by the native
`gads_make_surv_target`; and (2) evaluation must be **censoring-aware** (IPCW C-index,
time-dependent AUC, Integrated Brier Score) at follow-up times inside the train/test
support — handled by the native `gads_evaluate_survival`, which routinely-failing manual
code gets wrong.

## When to use
A duration column plus an event/censoring indicator, when the deliverable is *scoring or
ranking individuals by risk, or predicting their survival curves* (churn-risk scoring,
predictive maintenance, patient risk stratification) and interpretation is secondary. When
the deliverable is instead *explaining and defending the effect of each covariate* (hazard
ratios), use `survival_analysis.cox_regression` (lifelines).

## Key Constraints
- The duration and event columns must be identifiable; the event indicator must be binary.
- X must be fully numeric (encode categoricals); use the same columns for train and test.
- `surv_model.predict` returns a risk score (higher = higher risk), not a predicted time.
