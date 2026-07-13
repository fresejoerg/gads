---
id: causal_effect.heterogeneous.cate
version: 1.1.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [causal_inference]
  data_modality: [tabular]
  signals:
    - heterogeneous_effects: true
  anti_signals:
    - task: causal_discovery

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [econml, causalml, sklearn, pandas, shap]

# ——— DAG TEMPLATE ———
dag:
  - id: prepare_treatment_outcome_features
    intent: "Split the dataset into: Y (outcome array), T (treatment array, binary or continuous), X (effect modifiers — variables that may change treatment effect magnitude), W (baseline controls that affect outcome but not effect heterogeneity). Roles named in the objective are authoritative; exclude any post-treatment variables it identifies. Print shapes and a one-line justification per role."
    worker_tier: T2
    attached_skills: [causal_ml_econml]
    produces: [Y, T, X, W]
    postconditions:
      - "len(Y) == len(T)"
      - "len(X) == len(Y)"

  - id: fit_nuisance_models
    intent: "Fit propensity model (P(T|X,W)) and outcome model (E[Y|T,X,W]) using cross-fitting (cross_val_predict). Use sklearn HistGradientBoostingClassifier/Regressor or RandomForest. These are nuisance models only — do not interpret their coefficients."
    depends_on: [prepare_treatment_outcome_features]
    worker_tier: T2
    attached_skills: [causal_ml_econml, supervised_modeling]
    produces: [propensity_model, outcome_model]
    postconditions:
      - "propensity_model is not None"
      - "outcome_model is not None"

  - id: estimate_cate
    intent: "Estimate Conditional Average Treatment Effects (CATE) using CausalForestDML from econml. Fit the model on (Y, T, X, W). Compute cate_estimates = cate_model.effect(X). Also compute the overall ATE. Store ATE in a variable named exactly `ate`."
    depends_on: [fit_nuisance_models]
    worker_tier: T2
    attached_skills: [causal_ml_econml]
    produces: [cate_model, cate_estimates, ate]
    postconditions:
      - "cate_estimates is not None"
    required_metrics: [ate]

  - id: validate_cate
    intent: "Compute SHAP values on the CATE model to identify which features drive effect heterogeneity. Save a SHAP beeswarm or bar chart as Figure 1. Print the top 5 effect-driving features."
    depends_on: [estimate_cate]
    worker_tier: T2
    attached_skills: [causal_ml_econml, visualization_best_practices]
    postconditions:
      - "isinstance(cate_estimates, object)"

  - id: segment_who_benefits
    intent: "Rank units by their predicted CATE. Segment into high/medium/low effect groups (top/bottom terciles). Plot the CATE distribution as Figure 2. If causalml is available, also compute an uplift curve."
    depends_on: [validate_cate]
    worker_tier: T2
    attached_skills: [causal_ml_econml, visualization_best_practices]
    postconditions:
      - "isinstance(cate_estimates, object)"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "Nuisance models must be fit with cross-fitting (use cross_val_predict or CausalForestDML's built-in CV) to avoid overfitting bias."
  - "Always report the overall ATE alongside CATE for a sanity check."
  - "Random seeds must be fixed (random_state=42)."
---

# Heterogeneous Treatment Effects (CATE / Uplift Modeling)

## Rationale
Rather than estimating a single Average Treatment Effect, this recipe uses **Causal Machine Learning** to estimate the Conditional Average Treatment Effect (CATE) — i.e., who benefits most or least from an intervention. It uses EconML's CausalForestDML (Double ML + Random Forest) with mandatory cross-fitting to avoid overfitting, and SHAP for interpretability of effect drivers.

## When to use
Use when the objective goes beyond a single ATE to understand *heterogeneity*: which subgroups respond most to a treatment, which customers to target with an intervention, or how effect size varies with individual characteristics. Examples: personalised pricing, targeted treatment in clinical trials, marketing uplift modelling.

## Key Constraints
- Requires identifying effect modifiers (X) separately from baseline controls (W).
- CausalForestDML requires binary or continuous treatment; encode categorical treatment appropriately.
- Nuisance models must be cross-fit; never fit on the same fold used for CATE estimation.
