---
id: causal_effect.bayesian.pymc
version: 1.0.0
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
    intent: "Identify treatment, outcome, and confounder columns. Subsample to ≤10K rows if dataset exceeds that size (required for sandbox MCMC/ADVI performance). Standardise continuous confounders (zero mean, unit variance). Print final shapes and variable summary."
    worker_tier: T2
    produces: [df_model, treatment_col, outcome_col, confounder_cols]
    postconditions:
      - "len(df_model) <= 10000"
      - "isinstance(treatment_col, str)"

  - id: specify_bayesian_model
    intent: "Using Bambi, specify a Bayesian regression model: outcome ~ treatment + confounders. Use family='bernoulli' for binary outcomes, 'gaussian' for continuous. Define weakly informative priors (Normal(0, 0.5) for treatment, Normal(0, 1) for confounders). Print the model summary."
    depends_on: [prepare_bayesian_inputs]
    worker_tier: T2
    produces: [bayesian_model]
    postconditions:
      - "bayesian_model is not None"

  - id: fit_posterior
    intent: "Fit the model. For ≤5K rows: use MCMC with draws=500, tune=300, chains=1, cores=1, progressbar=False. For >5K rows: call model.build() then use PyMC ADVI via `with model.backend.model: approx=pm.fit(n=20000, method='advi', progressbar=False); idata=approx.sample(2000)`. Store the InferenceData in `idata`."
    depends_on: [specify_bayesian_model]
    worker_tier: T2
    produces: [idata]
    postconditions:
      - "idata is not None"

  - id: extract_ate_and_uncertainty
    intent: "Extract the posterior distribution of the treatment coefficient. Compute: mean ATE, standard deviation, and 94% Highest Density Interval (HDI) using arviz. Store the mean ATE in a variable named exactly `ate`. Print the full uncertainty summary."
    depends_on: [fit_posterior]
    worker_tier: T2
    produces: [ate, ate_hdi, ate_posterior]
    postconditions:
      - "isinstance(ate, float)"
    required_metrics: [ate]

  - id: visualize_posterior
    intent: "Plot the posterior distribution of the treatment effect as a histogram with the mean (red dashed line), zero-effect reference (black dotted), and 94% HDI shaded region. Save as Figure 1. Also save an ArviZ forest plot of all coefficients as Figure 2."
    depends_on: [extract_ate_and_uncertainty]
    worker_tier: T2
    postconditions:
      - "ate is not None"

  - id: posterior_predictive_check
    intent: "Run a posterior predictive check: sample from the posterior predictive distribution and compare to the observed outcome distribution. Save the PPC plot as Figure 3. Comment on model fit quality."
    depends_on: [visualize_posterior]
    worker_tier: T2
    postconditions:
      - "idata is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "Always subsample to ≤10K rows before fitting — MCMC/ADVI on larger datasets will timeout in the sandbox."
  - "Use ADVI (method='advi') not NUTS for sandbox runs. NUTS is reserved for the handover bundle."
  - "Report the 94% HDI alongside the point estimate — the interval is the primary Bayesian deliverable."
  - "Random seed must be 42 throughout for reproducibility."
  - "Use progressbar=False in pm.fit() to keep stdout clean."
---

# Bayesian Causal Effect Estimation (PyMC / Bambi)

## Rationale
Unlike frequentist methods that produce a single point estimate, **Bayesian causal inference** yields a full **posterior distribution** over the treatment effect. This enables: (1) principled uncertainty quantification (94% HDI instead of p-values), (2) incorporation of prior domain knowledge, (3) natural handling of small or imbalanced samples, and (4) counterfactual simulation by sampling from the posterior. This recipe uses Bambi (on top of PyMC) for standard regression models and ArviZ for posterior visualisation.

## When to use
Use when uncertainty quantification matters more than raw speed: medical/clinical studies, small-N experiments, hierarchical data (users nested in markets), or when you want to encode prior knowledge about plausible effect sizes. Also use when the outcome is rare (like fraud) and you need the posterior probability that the effect is positive, rather than a NaN-prone IPS weight.

## Key Constraints
- Subsample to ≤10K rows for sandbox fits. Full-dataset runs belong in the handover bundle.
- ADVI is the required fitting method in the sandbox; NUTS is too slow.
- HDI, not confidence intervals — Bayesian credible intervals have a direct probability interpretation.
