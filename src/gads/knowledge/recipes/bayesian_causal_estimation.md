---
id: causal_effect.bayesian.pymc
version: 1.1.0
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
      From the dataset schema and objective:
      (1) TREATMENT: identify treatment column. If continuous, compute its global
          median (before any subsampling) and store it. Note that binarisation
          at that median happens in this same step.
      (2) OUTCOME: identify outcome column. Check minority class fraction — print it.
      (3) CONFOUNDERS: numeric columns that are not treatment, outcome, or temporal/ID
          columns (patterns: 'time', 'date', 'timestamp', 'id', 'index'). Cap at 10
          by highest absolute correlation with outcome.
      (4) SUBSAMPLING: if len(df) > 5000:
            - If minority_frac < 0.10: stratified — take ALL minority rows, then
              sample majority rows to reach 5000 total.
            - Else: random sample to 5000 rows.
          Store as df_model.
      (5) TREATMENT ENGINEERING: create binary column high_{treatment_col} = 1
          where value > global_median, else 0, on df_model.
      (6) STANDARDISE continuous confounders (zero mean, unit variance) on df_model.
      Print final shapes, class balance, and variable roles.
    worker_tier: T2
    produces: [df_model, treatment_col, outcome_col, confounder_cols, minority_class_frac]
    postconditions:
      - "len(df_model) <= 5000"
      - "isinstance(treatment_col, str)"

  - id: specify_bayesian_model
    intent: >
      Build a Bambi model using the prepared df_model:
        formula = f"{outcome_col} ~ {treatment_col} + {' + '.join(confounder_cols)}"
        family = "bernoulli" if outcome is binary (0/1), else "gaussian"
      Do NOT hardcode variable names — build the formula string programmatically.
      Print the model summary.
    depends_on: [prepare_bayesian_inputs]
    worker_tier: T2
    produces: [bayesian_model]
    postconditions:
      - "bayesian_model is not None"

  - id: fit_posterior
    intent: >
      Fit using MCMC (≤5K rows is fast enough — ~7s per 1K rows in the sandbox):
        idata = bayesian_model.fit(draws=500, tune=300, chains=1, cores=1,
                                   target_accept=0.9, random_seed=42, progressbar=False)
      Save idata to disk: joblib.dump(idata, "bayesian_idata.joblib")
      Print the InferenceData groups to confirm posterior was sampled.
    depends_on: [specify_bayesian_model]
    worker_tier: T2
    produces: [idata]
    postconditions:
      - "idata is not None"

  - id: extract_ate_and_uncertainty
    intent: >
      Extract from idata.posterior[treatment_col]:
        ate = float(posterior_samples.mean())
        ate_hdi = az.hdi(posterior_samples, hdi_prob=0.94)
        p_positive = float((posterior_samples > 0).mean())
        try:
            max_rhat = float(az.rhat(idata).to_array().max())
        except Exception:
            max_rhat = float("nan")   # single chain — rhat undefined
      Print a summary table. Store ate as required metric.
    depends_on: [fit_posterior]
    worker_tier: T2
    produces: [ate, ate_hdi, p_positive, max_rhat]
    postconditions:
      - "isinstance(ate, float)"
    required_metrics: [ate]

  - id: visualize_posterior
    intent: >
      Plot the posterior distribution of the treatment coefficient:
      histogram with mean (red dashed), zero-effect reference (black dotted),
      and 94% HDI shaded. Annotate P(effect > 0). Save as Figure 1 (posterior_ate.png).
      Also compute and print the naive (unadjusted) association for comparison.
    depends_on: [extract_ate_and_uncertainty]
    worker_tier: T2
    postconditions:
      - "ate is not None"

  - id: posterior_predictive_check
    intent: "Run a posterior predictive check via bayesian_model.predict(idata, kind='pps'). Plot observed vs predicted distribution. Save as Figure 2. Comment on model fit."
    depends_on: [visualize_posterior]
    worker_tier: T2
    postconditions:
      - "idata is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "SUBSAMPLING: cap at 5,000 rows for MCMC. For imbalanced outcomes (<10% minority), use stratified sampling — ALL minority rows first, then fill with majority."
  - "TREATMENT ENGINEERING: binarise continuous treatment at its GLOBAL median (computed before subsampling, so the threshold is representative of the full dataset)."
  - "CONFOUNDER IDENTIFICATION: infer from schema — numeric columns that are not treatment, outcome, or temporal/ID. Never hardcode column names. Cap at 10."
  - "FORMULA BUILDING: always build the Bambi formula string programmatically. Never hardcode variable names."
  - "MCMC PARAMETERS: ALWAYS chains=1, cores=1. NEVER chains=2 or cores=2 — multiprocessing forks hang in Docker. draws=500, tune=300, progressbar=False, random_seed=42."
  - "R-HAT EXTRACTION: max_rhat = float(az.rhat(idata).to_array().max()) — rhat() returns a Dataset, not a scalar. Wrap in try/except for single-chain runs."
  - "SAVE IDATA: always joblib.dump the InferenceData for downstream use."
  - "Report 94% HDI and P(effect > 0) alongside the point estimate — these are the primary Bayesian deliverables."
---

# Bayesian Causal Effect Estimation (PyMC / Bambi)

## Rationale
Bayesian causal inference yields a full posterior distribution over the treatment effect rather than a point estimate, enabling principled uncertainty quantification, natural handling of imbalanced outcomes, and incorporation of domain priors. This recipe encodes all methodology decisions — subsampling strategy, treatment engineering, MCMC parameters, formula construction — so the spec needs only to identify variable roles and the research question.

## When to use
Use when uncertainty quantification matters: medical/clinical studies, rare-event outcomes, small samples, or when the posterior probability that an effect is positive is more interpretable than a p-value. Particularly robust for severely imbalanced outcomes where propensity weighting fails.

## Key Constraints
- Spec must identify treatment and outcome columns (or enough description to infer them).
- Spec should flag any purely temporal or ID columns so they are excluded from confounders.
- Full-dataset MCMC runs belong in the handover bundle — the recipe enforces the 5K sandbox limit.
