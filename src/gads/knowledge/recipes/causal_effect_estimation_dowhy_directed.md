---
id: causal_effect.observational.dowhy.directed
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
    - routing: "never match this recipe — delegation-dial instrumentation variant, selected only by an explicit spec pin"

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
      Identify the variable roles from the OBJECTIVE and the live schema of `df`.
      The objective is authoritative — if it names columns for any role, use exactly
      those names (verify each exists in df.columns).
      (1) TREATMENT: the column whose effect is being estimated. Store in
          `treatment_col`.
      (2) OUTCOME: the column being affected (a spec-hinted target column is the
          outcome). Store in `outcome_col`.
      (3) CONFOUNDERS: pre-treatment variables that plausibly influence both treatment
          and outcome. If the objective lists confounders, use exactly that list. If it
          identifies POST-TREATMENT variables (mediators or consequences of the
          treatment), EXCLUDE them — adjusting for them biases the estimate. Only when
          the objective does not settle the adjustment set, select confounders
          programmatically: numeric, non-temporal, non-identifier columns plausibly
          measured pre-treatment, capped at 10. Store in `confounder_cols`
          (at most 10).
      (4) Compute `minority_class_frac` for a categorical outcome (≤20 distinct
          values); use 0.5 for continuous outcomes.
      Print all four variables with a one-line justification of each role assignment.
    worker_tier: T2
    attached_skills: []          # D3: no curated skill (approach_docs/014)
    produces: [treatment_col, outcome_col, confounder_cols, minority_class_frac]
    postconditions:
      - "isinstance(treatment_col, str)"
      - "isinstance(outcome_col, str)"
      - "len(confounder_cols) <= 10"

  - id: estimate_and_refute
    intent: >
      Estimate the average treatment effect with the DoWhy four-step workflow,
      realized as code in this task:
      (1) MODEL — if `treatment_col` is continuous, binarize it at its median first.
          If df exceeds 20000 rows, subsample to 20000 with random_state=42. Build a
          dowhy.CausalModel over the estimation dataframe with treatment=treatment_col,
          outcome=outcome_col, common_causes=confounder_cols.
      (2) IDENTIFY — identify the estimand with the backdoor criterion.
      (3) ESTIMATE — estimate the ATE with the method given by the estimator-policy
          invariant, target_units="ate".
      (4) REFUTE — run BOTH refuters: placebo_treatment_refuter and
          data_subset_refuter, each with random_seed=42.
      Unpack the results into plain floats `ate`, `placebo_new_effect`,
      `subset_new_effect`, keep `df_sample` (the estimation dataframe) and the
      (possibly binarized) `treatment_col`, collect the two refuter values in
      `refutation_results`, print all three numbers, and emit a causal_effect insight
      interpreting them (placebo should be ~0; subset should be ~ATE).
    depends_on: [define_causal_question]
    worker_tier: T2
    attached_skills: []          # D3: no curated skill (approach_docs/014)
    produces: [ate, placebo_new_effect, subset_new_effect, treatment_col, df_sample, refutation_results]
    postconditions:
      - "isinstance(ate, float)"
      - "isinstance(placebo_new_effect, float)"
      - "isinstance(subset_new_effect, float)"
    required_metrics: [ate, placebo_new_effect, subset_new_effect]

  - id: visualize_results
    intent: >
      Visualize the causal result using the variables already in the kernel (`ate`,
      `placebo_new_effect`, `subset_new_effect`, `treatment_col`, `outcome_col`,
      `confounder_cols`, `df_sample`) — never recompute the effect and never hardcode
      column names. Produce a summary figure showing (a) the ATE against both
      refutation checks with a zero reference line and (b) the relevance of the
      adjustment set. Save as causal_effect_summary.png and print a confirmation.
    depends_on: [estimate_and_refute]
    worker_tier: T2
    attached_skills: []          # D3: no curated skill (approach_docs/014)
    produces: [causal_effect_summary_path]
    postconditions:
      - "ate is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "FOUR-STEP DOWHY IS MANDATORY: model → identify (backdoor) → estimate → refute with the dowhy library. NEVER implement propensity scoring / DML / AIPW from scratch, and NEVER report an ATE without both refutation checks (placebo_treatment_refuter, data_subset_refuter)."
  - "ROLES COME FROM THE OBJECTIVE: column roles named in the objective are authoritative. Never invent column names not present in the schema; never ignore an exclusion the objective states."
  - "NO POST-TREATMENT ADJUSTMENT: never include mediators or downstream consequences of the treatment in the confounder set — conditioning on them biases the effect."
  - "CONFOUNDER FALLBACK: only when the objective does not settle the adjustment set, infer confounders programmatically from the schema (numeric, non-temporal/ID, cap 10) and state that this heuristic assumes all selected columns are pre-treatment."
  - "ESTIMATOR POLICY: backdoor.linear_regression when the outcome is continuous or its minority class share is >= 5%; backdoor.propensity_score_matching only when a categorical outcome's minority class share is < 5%."
  - "METRICS AS FLOATS: store ate, placebo_new_effect, and subset_new_effect as plain Python floats, not numpy scalars."
  - "REUSE KERNEL VARIABLES: the visualization step must reuse the estimates already in the kernel — never recompute the effect."
  - "Random state must be 42 everywhere (refuters, subsampling, matching)."
---

# Causal Effect Estimation with DoWhy — Directed variant (dial rung D3)

## Rationale
Delegation-dial instrumentation variant of `causal_effect.observational.dowhy`
(approach_docs/013). It fixes the SAME methodology decisions as the parent recipe —
DoWhy four-step, backdoor identification, the estimator policy, both refuters,
seed 42, the confounder-selection rules — but at **decision level only**: no skills
are attached (no canonical code patterns) and the mechanical core is NOT delegated to
the native kernel function. The model realizes every step as code itself. Every task
therefore sits at rung D3 (directed), making the whole project a clean D3 measurement
point: methodology fixed ex ante, realization free.

## When to use
Only for delegation-dial measurement runs, pinned explicitly by a spec
(`recipe_id: causal_effect.observational.dowhy.directed`). For production causal
effect estimation always use `causal_effect.observational.dowhy` — the native node
deterministically handles the steps where LLM-written code reliably fails.

## Key Constraints
- Same methodology contract as the parent recipe; only the realization channel differs.
- The Router must never select this variant on its own (see anti_signals) — a routed
  match would contaminate the dial measurement.
