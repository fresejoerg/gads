---
id: causal_effect.observational.dowhy
version: 2.1.0
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
          the objective does not settle the adjustment set, fall back to the
          programmatic selection pattern in the attached skill. Store in
          `confounder_cols` (at most 10).
      (4) Compute `minority_class_frac` for a categorical outcome (≤20 distinct
          values); use 0.5 for continuous outcomes.
      Print all four variables with a one-line justification of each role assignment.
    worker_tier: T2
    produces: [treatment_col, outcome_col, confounder_cols, minority_class_frac]
    attached_skills: [causal_inference_dowhy]
    postconditions:
      - "isinstance(treatment_col, str)"
      - "isinstance(outcome_col, str)"
      - "len(confounder_cols) <= 10"

  - id: estimate_and_refute
    intent: >
      Estimate the average treatment effect AND run both refutation tests in a single
      call to the pre-defined native kernel function `gads_causal_estimate_ate` (exact
      call pattern in the attached skill). It handles subsampling, continuous-treatment
      binarization, graph construction, estimand identification, estimator selection,
      and both refuters internally — do NOT reimplement any of it. Unpack the result
      into plain floats `ate`, `placebo_new_effect`, `subset_new_effect`, keep
      `df_sample` and the (possibly binarized) `treatment_col`, collect the two
      refuter values in `refutation_results`, print all three numbers, and emit a
      causal_effect insight interpreting them (placebo should be ~0; subset should be
      ~ATE).
    depends_on: [define_causal_question]
    worker_tier: T2
    attached_skills: [causal_inference_dowhy]
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
      column names. Produce a summary figure (pattern in the attached skill) showing
      (a) the ATE against both refutation checks with a zero reference line and (b) the
      relevance of the adjustment set. Save as causal_effect_summary.png and print a
      confirmation.
    depends_on: [estimate_and_refute]
    worker_tier: T2
    attached_skills: [causal_inference_dowhy, visualization_best_practices]
    produces: [causal_effect_summary_path]
    postconditions:
      - "ate is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "NATIVE NODE IS MANDATORY: estimate the effect with gads_causal_estimate_ate(df, treatment_col, outcome_col, confounder_cols). NEVER build a CausalModel by hand, NEVER implement propensity scoring / DML / AIPW from scratch, and NEVER skip refutation."
  - "ROLES COME FROM THE OBJECTIVE: column roles named in the objective are authoritative. Never invent column names not present in the schema; never ignore an exclusion the objective states."
  - "NO POST-TREATMENT ADJUSTMENT: never include mediators or downstream consequences of the treatment in the confounder set — conditioning on them biases the effect."
  - "CONFOUNDER FALLBACK: only when the objective does not settle the adjustment set, infer confounders programmatically from the schema (numeric, non-temporal/ID, cap 10) and state that this heuristic assumes all selected columns are pre-treatment."
  - "METRICS AS FLOATS: store ate, placebo_new_effect, and subset_new_effect as plain Python floats, not numpy scalars."
  - "REUSE KERNEL VARIABLES: the visualization step must reuse the estimates already in the kernel — never recompute the effect."
  - "Random state must be 42 everywhere (the native node already enforces this internally)."
---

# Causal Effect Estimation with DoWhy (Observational Data)

## Rationale
Implements the DoWhy four-step workflow — **Model → Identify → Estimate → Refute** —
delegating the mechanical core to the audited native kernel function
`gads_causal_estimate_ate`, which deterministically handles the steps where LLM-written
code reliably fails (subsampling, treatment binarization, graph construction, estimator
selection, both refuters). The model contributes the two judgments that genuinely vary
per problem: assigning variable roles from the research question, and interpreting the
estimate against its refutations.

The critical methodological judgment is the **adjustment set**. A valid confounder is a
*pre-treatment* variable influencing both treatment and outcome; conditioning on
post-treatment variables (mediators, colliders) biases the estimate. No schema heuristic
can distinguish pre- from post-treatment — that knowledge lives in the research question,
which is why objective-named roles and exclusions are authoritative and the programmatic
selection is only a declared-assumption fallback.

## When to use
Use when the objective is to estimate *how much* a treatment causally affects an outcome
in observational tabular data, and a single average effect (ATE) is the deliverable. Use
`causal_effect.heterogeneous.cate` when effect *variation across units* is the question,
`causal_effect.iv_panel.linearmodels` when unobserved confounding demands an instrument
or panel structure, and `causal_effect.bayesian.pymc` when full posterior uncertainty is
required.

## Key Constraints
- Treatment and outcome must be identifiable from the objective plus schema.
- Refutation (placebo + subset) is mandatory and runs inside the native node before any
  ATE is reported.
- Specs for datasets containing post-treatment variables should name them explicitly in
  the objective so they are excluded from adjustment.
