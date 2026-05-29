---
id: causal_effect.observational.dowhy
version: 1.0.0
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
    intent: "Identify the treatment variable, outcome variable, and candidate observed confounders from the dataset schema. Print a summary of variable roles."
    worker_tier: T2
    produces: [treatment_col, outcome_col, confounder_cols]
    postconditions:
      - "isinstance(treatment_col, str)"
      - "isinstance(outcome_col, str)"

  - id: build_causal_graph
    intent: "Construct the causal DAG as a GML string encoding the assumed causal structure among treatment, outcome, confounders, and any mediators or instruments. Instantiate a DoWhy CausalModel."
    depends_on: [define_causal_question]
    worker_tier: T1
    produces: [causal_model]
    postconditions:
      - "causal_model is not None"

  - id: identify_estimand
    intent: "Call causal_model.identify_effect() to obtain the identified estimand. Print the estimand type (backdoor, frontdoor, or IV) and the adjustment set."
    depends_on: [build_causal_graph]
    worker_tier: T1
    produces: [identified_estimand]
    postconditions:
      - "identified_estimand is not None"

  - id: estimate_effect
    intent: "Estimate the Average Treatment Effect (ATE) using the identified estimand. Use propensity-score weighting (econml.dml.LinearDML or DoWhy's propensity_score_weighting method). Store the numeric ATE in a variable named exactly `ate`."
    depends_on: [identify_estimand]
    worker_tier: T2
    produces: [causal_estimate, ate]
    postconditions:
      - "causal_estimate is not None"
    required_metrics: [ate]

  - id: refute_estimate
    intent: "Run at minimum two refutation tests: (1) random_common_cause placebo and (2) data_subset_refuter. Print each refutation result and whether the estimate passes. Save a summary artifact."
    depends_on: [estimate_effect]
    worker_tier: T2
    produces: [refutation_results]
    postconditions:
      - "refutation_results is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "The treatment and outcome columns must never appear in the confounder adjustment set simultaneously."
  - "Random seeds must be fixed for reproducibility (random_state=42)."
  - "Refutation is mandatory — never report an ATE without running at least one refuter."
  - "Specify the causal graph as a GML string passed to CausalModel(graph=...). Never use eval()."
---

# Causal Effect Estimation with DoWhy (Observational Data)

## Rationale
This recipe implements the industry-standard DoWhy four-step causal inference workflow: **Model → Identify → Estimate → Refute**. It is designed for observational datasets where a treatment and outcome are measured alongside potential confounders, but no randomised experiment was conducted. The mandatory refutation step protects against spurious causal claims.

## When to use
Use when the objective is to estimate *how much* a treatment (binary or continuous) causally affects an outcome, given observational tabular data. Examples: effect of a medication on recovery, impact of a price change on demand, influence of education on earnings.

## Key Constraints
- A treatment column and outcome column must be identifiable from the schema.
- The causal graph must encode domain assumptions — instruct the user to review the assumed edges.
- Always run refutation before reporting the ATE to the Synthesizer.
