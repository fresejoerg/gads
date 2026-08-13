---
id: causal_effect.observational.sklearn.directed
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
    - routing: "never match this recipe — delegation-dial API-surface arm (approach_docs/014), selected only by an explicit spec pin"

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [sklearn, pandas, numpy]

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
      Print all three variables with a one-line justification of each role assignment.
    worker_tier: T2
    attached_skills: []          # D3: no curated skill (approach_docs/014)
    produces: [treatment_col, outcome_col, confounder_cols]
    postconditions:
      - "isinstance(treatment_col, str)"
      - "isinstance(outcome_col, str)"
      - "len(confounder_cols) <= 10"

  - id: estimate_and_refute
    intent: >
      Estimate the average treatment effect by linear-regression adjustment with
      scikit-learn, realized as code in this task:
      (1) PREP — if `treatment_col` is continuous, binarize it at its median first.
          If df exceeds 20000 rows, subsample to 20000 with random_state=42. Keep the
          estimation dataframe as `df_sample`.
      (2) ESTIMATE — build X = df_sample[[treatment_col] + confounder_cols] and
          y = df_sample[outcome_col], fit sklearn.linear_model.LinearRegression, and
          read the ATE as the fitted coefficient belonging to `treatment_col` (the
          first column of X). Store it in `ate`.
      (3) PLACEBO REFUTE — copy `df_sample`, replace the treatment column with a
          random permutation of itself (numpy default_rng seed 42), refit the same
          regression, and store the treatment coefficient in `placebo_new_effect`
          (expected ~0).
      (4) SUBSET REFUTE — refit the same regression on
          df_sample.sample(frac=0.8, random_state=42) and store the treatment
          coefficient in `subset_new_effect` (expected ~ATE).
      Store all three as plain Python floats, print them, and emit a causal_effect
      insight interpreting them (placebo should be ~0; subset should be ~ATE).
    depends_on: [define_causal_question]
    worker_tier: T2
    attached_skills: []          # D3: no curated skill (approach_docs/014)
    produces: [ate, placebo_new_effect, subset_new_effect, treatment_col, df_sample]
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
  - "LINEAR-REGRESSION ADJUSTMENT IS MANDATORY: the ATE is the treatment coefficient of a LinearRegression fitted on [treatment] + confounders with scikit-learn. Use ONLY scikit-learn + pandas + numpy. NEVER import dowhy, econml, statsmodels, or any causal-inference package — this recipe is a delegation-dial API-surface arm (approach_docs/014) and the library choice is the experimental variable."
  - "REFUTERS ARE MANDATORY: never report an ATE without both checks — the permutation placebo (seed 42) and the 80% subset refit (random_state=42)."
  - "ROLES COME FROM THE OBJECTIVE: column roles named in the objective are authoritative. Never invent column names not present in the schema; never ignore an exclusion the objective states."
  - "NO POST-TREATMENT ADJUSTMENT: never include mediators or downstream consequences of the treatment in the confounder set — conditioning on them biases the effect."
  - "CONFOUNDER FALLBACK: only when the objective does not settle the adjustment set, infer confounders programmatically from the schema (numeric, non-temporal/ID, cap 10) and state that this heuristic assumes all selected columns are pre-treatment."
  - "CONTINUOUS OUTCOME ONLY: this variant assumes a continuous outcome (linear adjustment). All delegation-dial benchmarks satisfy this."
  - "METRICS AS FLOATS: store ate, placebo_new_effect, and subset_new_effect as plain Python floats, not numpy scalars."
  - "REUSE KERNEL VARIABLES: the visualization step must reuse the estimates already in the kernel — never recompute the effect."
  - "Random state must be 42 everywhere (permutation, subsampling)."
---

# Causal Effect Estimation with scikit-learn — Directed variant (dial rung D3)

## Rationale
Phase A arm of the API-familiarity agenda (approach_docs/014). Methodologically
equivalent to `causal_effect.observational.dowhy.directed` — backdoor
linear-regression adjustment IS a linear fit on treatment + covariates, and the
refuters mirror DoWhy's placebo_treatment_refuter and data_subset_refuter — but
realized through the MOST ubiquitous API surface in the training distribution
(sklearn LinearRegression + pandas). No skills attached, no native steps: every
task rungs at D3. Together with the statsmodels arm this brackets the familiarity
gradient: dowhy (niche) → statsmodels (common) → sklearn (ubiquitous), holding
task, benchmarks, gold ATEs, and rung constant (hypothesis H1).

## When to use
Only for delegation-dial measurement runs, pinned explicitly by a spec
(`recipe_id: causal_effect.observational.sklearn.directed`). Never for production —
use `causal_effect.observational.dowhy`.

## Key Constraints
- The refuter values are semantically comparable to DoWhy's (placebo ~0, subset
  ~ATE) but NOT numerically identical to the dowhy-recipe reference runs — scorers
  must check semantics/tolerance against the gold ATE, not bitwise refuter equality.
- The Router must never select this variant on its own (see anti_signals).
