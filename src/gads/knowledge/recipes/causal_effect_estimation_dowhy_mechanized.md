---
id: causal_effect.observational.dowhy.mechanized
version: 1.0.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [causal_inference]
  data_modality: [tabular]
  signals:
    - treatment_outcome_pair: true
  pin_only: true          # research instrument — reachable only by an explicit spec pin
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
  - id: estimate_causal_effect
    intent: >
      (1) Identify the variable roles from the OBJECTIVE and the live schema of `df`.
      The objective is authoritative — if it names columns for any role, use exactly
      those names (verify each exists in df.columns); if it identifies POST-TREATMENT
      variables (mediators or consequences of the treatment), EXCLUDE them from the
      confounders. Only when the objective does not settle the adjustment set, select
      confounders programmatically (numeric, non-temporal/ID, cap 10) and state that
      this heuristic assumes all selected columns are pre-treatment. Store
      `treatment_col`, `outcome_col`, `confounder_cols`, each with a one-line
      justification printed.
      (2) Estimate the average treatment effect AND run both refutation tests in a
      single call to the pre-defined native kernel function
      `gads_causal_estimate_ate(df, treatment_col, outcome_col, confounder_cols)`
      (exact call pattern in the attached skill). It handles subsampling,
      continuous-treatment binarization, graph construction, estimand identification,
      estimator selection, and both refuters internally — do NOT reimplement any
      of it.
      (3) Unpack the result into plain floats `ate`, `placebo_new_effect`,
      `subset_new_effect`, collect the two refuter values in `refutation_results`,
      print all three numbers, and emit a causal_effect insight interpreting them
      (placebo should be ~0; subset should be ~ATE).
    worker_tier: T2
    attached_skills: [causal_inference_dowhy]
    produces: [treatment_col, outcome_col, confounder_cols, ate, placebo_new_effect, subset_new_effect, refutation_results]
    postconditions:
      - "isinstance(treatment_col, str)"
      - "isinstance(outcome_col, str)"
      - "len(confounder_cols) <= 10"
      - "isinstance(ate, float)"
      - "isinstance(placebo_new_effect, float)"
      - "isinstance(subset_new_effect, float)"
    required_metrics: [ate, placebo_new_effect, subset_new_effect]

# ——— GLOBAL INVARIANTS ———
invariants:
  - "NATIVE NODE IS MANDATORY: estimate the effect with gads_causal_estimate_ate(df, treatment_col, outcome_col, confounder_cols). NEVER build a CausalModel by hand, NEVER implement propensity scoring / DML / AIPW from scratch, and NEVER skip refutation."
  - "ROLES COME FROM THE OBJECTIVE: column roles named in the objective are authoritative. Never invent column names not present in the schema; never ignore an exclusion the objective states."
  - "NO POST-TREATMENT ADJUSTMENT: never include mediators or downstream consequences of the treatment in the confounder set — conditioning on them biases the effect."
  - "CONFOUNDER FALLBACK: only when the objective does not settle the adjustment set, infer confounders programmatically from the schema (numeric, non-temporal/ID, cap 10) and state that this heuristic assumes all selected columns are pre-treatment."
  - "METRICS AS FLOATS: store ate, placebo_new_effect, and subset_new_effect as plain Python floats, not numpy scalars."
  - "Random state must be 42 everywhere (the native node already enforces this internally)."
---

# Causal Effect Estimation with DoWhy — Mechanized variant (dial rung D5)

## Rationale
Delegation-dial instrumentation variant of `causal_effect.observational.dowhy`
(approach_docs/013). The entire mechanical core — subsampling, binarization, graph
construction, identification, estimation, both refuters — is delegated to the audited
native kernel function `gads_causal_estimate_ate` in a SINGLE task, so the project's
minimum task rung is D5 (mechanized). The model contributes exactly what D5 leaves to
it: binding variable roles from the research question, and interpreting the estimate
against its refutations in the synthesis narrative. There is no visualization task —
any lower-rung task would drag the project rung down (project rung = min over tasks).

## When to use
Only for delegation-dial measurement runs, pinned explicitly by a spec
(`recipe_id: causal_effect.observational.dowhy.mechanized`). For production causal
effect estimation use `causal_effect.observational.dowhy`, which adds the guided
role-assignment task and the summary figure.

## Key Constraints
- Same estimation semantics as the parent recipe's native node (deterministic given
  the adjustment set, seed 42).
- Produces metrics.json (ate, placebo_new_effect, subset_new_effect) but NO
  causal_effect_summary.png — benchmark scorers must not require the figure for
  D5 runs.
- The Router must never select this variant on its own (see anti_signals).
