---
id: causal_effect.iv_panel.linearmodels
version: 1.0.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [causal_inference]
  data_modality: [tabular]
  signals:
    - instrument_or_panel: true
  anti_signals:
    - task: causal_discovery

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [linearmodels, statsmodels, pandas, numpy]

# ——— DAG TEMPLATE ———
dag:
  - id: specify_instrument_or_panel
    intent: "Determine the estimation strategy from the objective: (A) Instrumental Variables — identify the instrument(s) Z that affect treatment T but have no direct path to outcome Y; or (B) Panel/DiD — identify entity and time columns. Print the chosen strategy and variable mapping."
    worker_tier: T1
    produces: [strategy, treatment_col, outcome_col]
    postconditions:
      - "strategy in ['iv', 'panel', 'did']"

  - id: check_instrument_strength
    intent: "For IV: run the first-stage OLS regression (T ~ Z + controls) and report the F-statistic. Flag weak instruments if F < 10. For panel/DiD: verify panel balance and print a pre-trend summary."
    depends_on: [specify_instrument_or_panel]
    worker_tier: T2
    produces: [first_stage_f]
    postconditions:
      - "isinstance(first_stage_f, (int, float))"
    skippable_if: "strategy == 'did'"

  - id: estimate_model
    intent: "Estimate the causal effect. IV: use linearmodels.iv.model.IV2SLS. Panel/FE: use linearmodels.panel.model.PanelOLS with entity_effects=True. DiD: use a two-way FE regression or statsmodels OLS with interaction term. Store the numeric effect estimate in a variable named exactly `effect_estimate`."
    depends_on: [check_instrument_strength]
    worker_tier: T2
    produces: [model_result, effect_estimate]
    postconditions:
      - "model_result is not None"
    required_metrics: [effect_estimate]

  - id: robustness_checks
    intent: "Run at least one robustness check: for IV, test over-identification (Sargan/Hansen J-test if multiple instruments); for DiD, plot pre-trends and run a placebo test on a pre-period; for panel, try clustered standard errors. Print and save the robustness results as a text artifact."
    depends_on: [estimate_model]
    worker_tier: T2
    postconditions:
      - "model_result is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "Always report first-stage F-statistic before interpreting IV results. Do not proceed if F < 10."
  - "Cluster standard errors at the entity level for panel data."
  - "For DiD, verify parallel pre-trends before interpreting the DiD coefficient."
  - "Never use pickle — use joblib for any serialization."
---

# Instrumental Variables & Panel Econometrics

## Rationale
When unobserved confounding makes standard regression or DoWhy-style backdoor adjustment infeasible, **Instrumental Variables (IV)** and **panel/difference-in-differences (DiD)** methods can still identify causal effects. IV exploits a natural experiment (an exogenous shock to treatment); panel methods use within-entity variation over time to difference out unobserved fixed confounders.

## When to use
Use when: (A) a natural instrument is available (lottery assignment, geographic discontinuity, random eligibility); (B) repeated observations per entity allow fixed-effects to absorb confounders; or (C) a policy change creates a treatment/control group split over time (DiD). Examples: IV for returns to education (proximity to college as instrument), FE panel for firm-level investment, DiD for minimum wage effects.

## Key Constraints
- IV requires instrument relevance (strong first stage, F > 10) AND exclusion restriction (instrument affects outcome only through treatment).
- DiD requires parallel pre-trends — this must be tested, not assumed.
- linearmodels requires a MultiIndex DataFrame (entity, time) for panel estimation.
