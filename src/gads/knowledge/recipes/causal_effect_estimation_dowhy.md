---
id: causal_effect.observational.dowhy
version: 1.1.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [causal_inference]
  data_modality: [tabular]
  signals:
    - treatment_outcome_pair: true
  anti_signals:
    - task: heterogeneous_treatment_effects

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [dowhy, statsmodels, sklearn, pandas]

# ——— DAG TEMPLATE ———
dag:
  - id: define_causal_question
    intent: >
      From the dataset schema and objective, identify:
      (1) TREATMENT: the column representing the intervention (if continuous, note it will
          be binarised at its median in the next step);
      (2) OUTCOME: the target column;
      (3) CONFOUNDERS: ALL numeric columns that are neither treatment, outcome, nor
          temporal/ID columns (common temporal/ID patterns: 'time', 'date', 'timestamp',
          'id', 'index'). If more than 10 confounders exist, retain the 10 with the
          highest absolute correlation with the outcome for tractability;
      (4) CLASS BALANCE: compute the minority class fraction of the outcome. Print it —
          this drives estimator selection.
      Print a variable-role summary table.
    worker_tier: T2
    produces: [treatment_col, outcome_col, confounder_cols, minority_class_frac]
    postconditions:
      - "isinstance(treatment_col, str)"
      - "isinstance(outcome_col, str)"
      - "len(confounder_cols) <= 10"

  - id: engineer_and_build_graph
    intent: >
      (1) TREATMENT ENGINEERING: if treatment_col is continuous, create a binary column
          `high_{treatment_col}` = 1 where value > median(treatment_col), else 0.
          Update treatment_col to the new binary column name.
      (2) GML CONSTRUCTION: build the GML string programmatically — loop over
          confounder_cols to generate edges. Do NOT hardcode node IDs.
          Pattern:
            nodes = [treatment_col, outcome_col] + confounder_cols
            edges = [(c, treatment_col) for c in confounder_cols] +
                    [(c, outcome_col) for c in confounder_cols] +
                    [(treatment_col, outcome_col)]
            Then format as GML with directed=1.
      (3) Instantiate CausalModel(data=df, treatment=treatment_col,
          outcome=outcome_col, graph=gml_string).
    depends_on: [define_causal_question]
    worker_tier: T2
    produces: [causal_model, treatment_col]
    postconditions:
      - "causal_model is not None"

  - id: identify_estimand
    intent: "Call causal_model.identify_effect(proceed_when_unidentifiable=True). Print the estimand type and adjustment set."
    depends_on: [engineer_and_build_graph]
    worker_tier: T2
    produces: [identified_estimand]
    postconditions:
      - "identified_estimand is not None"

  - id: estimate_effect
    intent: >
      Select estimator based on minority_class_frac:
      - If minority_class_frac < 0.05 (rare outcome): use backdoor.propensity_score_matching
        (propensity_score_weighting produces NaN on severely imbalanced outcomes).
      - If minority_class_frac >= 0.05: use backdoor.linear_regression (robust default).
      Call causal_model.estimate_effect(identified_estimand, method_name=chosen_method,
      target_units="ate"). Store float(causal_estimate.value) as `ate`. Print the ATE.
    depends_on: [identify_estimand]
    worker_tier: T2
    produces: [causal_estimate, ate]
    postconditions:
      - "causal_estimate is not None"
    required_metrics: [ate]

  - id: refute_estimate
    intent: >
      Run BOTH refuters using the existing causal_model and causal_estimate — do NOT
      rebuild the model:
      (1) placebo_treatment_refuter (placebo_type="permute", random_seed=42);
      (2) data_subset_refuter (subset_fraction=0.8, random_seed=42).
      Store float(refute_placebo.new_effect) as `placebo_new_effect` and
      float(refute_subset.new_effect) as `subset_new_effect`.
      Print both values and store in refutation_results dict.
    depends_on: [estimate_effect]
    worker_tier: T2
    produces: [refutation_results, placebo_new_effect, subset_new_effect]
    postconditions:
      - "refutation_results is not None"
    required_metrics: [placebo_new_effect, subset_new_effect]

# ——— GLOBAL INVARIANTS ———
invariants:
  - "USE DOWHY LIBRARY: always use CausalModel.estimate_effect() — NEVER implement propensity scoring, DR/AIPW, or causal estimation manually from scratch. DoWhy handles all of this internally."
  - "LARGE DATASET PERFORMANCE: for datasets >50K rows, subsample to 20K rows before fitting any models. Use df_sample = df.sample(20000, random_state=42)."
  - "CONFOUNDERS: infer from schema as numeric columns that are not treatment, outcome, or temporal/ID. Never ask the user to list them."
  - "TREATMENT ENGINEERING: if treatment is continuous, always binarise at its median. Name the new column high_{original_name}."
  - "GML CONSTRUCTION: always build the GML string programmatically from the confounder list. Never hardcode node IDs in the source code."
  - "ESTIMATOR SELECTION: check minority_class_frac. Use propensity_score_matching for rare outcomes (<5%), linear_regression otherwise. Never use propensity_score_weighting — it produces NaN on imbalanced data."
  - "CLASS IMBALANCE: never use imblearn/SMOTE (not installed). Use class_weight='balanced' in sklearn propensity models if needed."
  - "REFUTATION: always reuse the existing causal_model object — never rebuild it. Both refuters are mandatory. Store placebo_new_effect and subset_new_effect as plain Python floats."
  - "POSTCONDITION CONTRACTS: required_columns must contain only string column names from the actual dataset schema, never integer indices."
  - "Random state must be 42 everywhere."
  - "Store all metric values (ate, placebo_new_effect, subset_new_effect) as plain Python floats, not numpy scalars."
---

# Causal Effect Estimation with DoWhy (Observational Data)

## Rationale
This recipe implements the DoWhy four-step workflow: **Model → Identify → Estimate → Refute**. It encodes the methodology decisions that should not need to appear in a spec: confounder identification from the schema, treatment engineering from continuous variables, estimator selection based on outcome class balance, and programmatic GML construction.

## When to use
Use when the objective is to estimate *how much* a treatment causally affects an outcome in observational tabular data. The spec needs only to identify the treatment column, outcome column, and any columns that are purely temporal or identifiers (to exclude from confounders).

## Key Constraints
- Treatment and outcome must be identifiable from the schema and objective.
- Temporal/ID columns must be excluded from the confounder set — the recipe infers this automatically.
- Refutation is mandatory before reporting an ATE.
