---
id: causal_effect.bayesian.pymc
version: 1.2.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [causal_inference]
  data_modality: [tabular]
  signals:
    - bayesian_inference: true
  anti_signals:
    - task: causal_discovery

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [pymc, bambi, arviz, pandas, numpy, matplotlib]

# ——— DAG TEMPLATE ———
dag:
  - id: prepare_bayesian_inputs
    intent: >
      Prepare the modeling inputs from the OBJECTIVE and the schema (code patterns in
      the attached skill). The objective is authoritative for variable roles.
      (1) TREATMENT (`treatment_col`): if continuous, compute its global median BEFORE
          any subsampling and binarize into high_{name} at that median.
      (2) OUTCOME (`outcome_col`): a spec-hinted target column is the outcome.
      (3) CONFOUNDERS (`confounder_cols`): objective-named confounders/exclusions are
          authoritative — never adjust for post-treatment variables the objective
          identifies. Otherwise fall back to the programmatic selection in the skill
          (numeric, non-temporal/ID, cap 10).
      (4) SUBSAMPLE for MCMC into `df_model` (≤5,000 rows; stratified when the outcome
          minority class is under 10% — keep ALL minority rows).
      (5) Standardize continuous confounders on df_model.
      (6) Bind and print the scalar evidence: `n_rows`, `minority_class_frac`,
          `naive_outcome_rate`.
      Print final shapes, class balance, and a one-line justification per role.
    worker_tier: T2
    produces: [df_model, treatment_col, outcome_col, confounder_cols, minority_class_frac]
    attached_skills: [bayesian_causal_pymc]
    postconditions:
      - "len(df_model) <= 5000"
      - "isinstance(treatment_col, str)"
    required_metrics: [naive_outcome_rate, minority_class_frac, n_rows]

  - id: specify_bayesian_model
    intent: >
      Build a Bambi model on df_model. Construct the formula string programmatically
      from the role variables (never hardcode column names); choose the family from the
      outcome type: bernoulli for a binary 0/1 outcome, gaussian otherwise. Store in
      `bayesian_model` and print the model summary.
    depends_on: [prepare_bayesian_inputs]
    worker_tier: T2
    produces: [bayesian_model]
    attached_skills: [bayesian_causal_pymc]
    postconditions:
      - "bayesian_model is not None"

  - id: fit_posterior
    intent: >
      Sample the posterior with MCMC under the sandbox sampler settings mandated in the
      attached skill (single chain, fixed seed, no progress bar), store the result in
      `idata`, persist it with joblib as bayesian_idata.joblib, and print the
      InferenceData groups to confirm the posterior was sampled.
    depends_on: [specify_bayesian_model]
    worker_tier: T2
    produces: [idata]
    attached_skills: [bayesian_causal_pymc]
    postconditions:
      - "idata is not None"

  - id: extract_ate_and_uncertainty
    intent: >
      Summarize the treatment-coefficient posterior: point estimate `ate` (posterior
      mean, plain float), 94% HDI `ate_hdi`, `p_positive` = P(effect > 0), and
      convergence diagnostic `max_rhat` (extraction pattern in the skill — R-hat is
      undefined for a single chain; guard with try/except). Print a summary table.
    depends_on: [fit_posterior]
    worker_tier: T2
    produces: [ate, ate_hdi, p_positive, max_rhat]
    attached_skills: [bayesian_causal_pymc]
    postconditions:
      - "isinstance(ate, float)"
    required_metrics: [ate]

  - id: visualize_posterior
    intent: >
      Plot the posterior distribution of the treatment coefficient: histogram with the
      posterior mean, a zero-effect reference line, and the 94% HDI shaded; annotate
      P(effect > 0). Save as posterior_ate.png. Also compute and print the naive
      (unadjusted) association for comparison against the adjusted estimate.
    depends_on: [extract_ate_and_uncertainty]
    worker_tier: T2
    attached_skills: [bayesian_causal_pymc, visualization_best_practices]
    postconditions:
      - "ate is not None"

  - id: posterior_predictive_check
    intent: >
      Run a posterior predictive check (bayesian_model.predict(idata, kind='pps')),
      plot observed vs predicted outcome distributions, save as a second figure, and
      comment on model fit quality.
    depends_on: [visualize_posterior]
    worker_tier: T2
    attached_skills: [bayesian_causal_pymc, visualization_best_practices]
    postconditions:
      - "idata is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "ROLES COME FROM THE OBJECTIVE: column roles named in the objective are authoritative. Never invent column names not in the schema; never adjust for post-treatment variables the objective identifies."
  - "SUBSAMPLING: cap at 5,000 rows for MCMC. For imbalanced outcomes (<10% minority), stratify — ALL minority rows first, then fill with majority."
  - "TREATMENT ENGINEERING: binarize a continuous treatment at its GLOBAL median, computed before subsampling."
  - "FORMULA BUILDING: always build the Bambi formula string programmatically from the role variables."
  - "REPORT UNCERTAINTY: the 94% HDI and P(effect > 0) are primary deliverables alongside the point estimate — never report the ATE alone."
  - "CONVERGENCE: report max R-hat when defined; flag the single-chain limitation otherwise."
  - "SAVE IDATA: persist the InferenceData with joblib for downstream use."
  - "Random seed must be 42 everywhere."
---

# Bayesian Causal Effect Estimation (PyMC / Bambi)

## Rationale
Bayesian causal inference yields a full posterior over the treatment effect rather than
a point estimate: principled uncertainty quantification, natural handling of imbalanced
outcomes, and room for domain priors. The recipe encodes the methodology decisions —
role assignment discipline (objective-authoritative, no post-treatment adjustment),
subsampling strategy, treatment engineering, uncertainty reporting — while the sampler
mechanics and API patterns live in the `bayesian_causal_pymc` skill.

## When to use
When uncertainty quantification is the point: clinical/medical questions, rare-event
outcomes, small samples, or when "probability the effect is positive" is more useful
than a p-value. Particularly robust for severely imbalanced outcomes where propensity
weighting fails. Use `causal_effect.observational.dowhy` when a fast point estimate with
refutation tests suffices.

## Key Constraints
- Spec must identify treatment and outcome (or describe them unambiguously).
- Specs for datasets containing post-treatment variables should name them in the
  objective so they are excluded from adjustment.
- Full-dataset MCMC belongs in a handover bundle — the recipe enforces the 5K sandbox cap.
