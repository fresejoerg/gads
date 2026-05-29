---
id: bayesian_causal_pymc
description: "PyMC/Bambi Bayesian causal inference: posterior ATEs with full uncertainty, Bayesian IV, ADVI for large datasets. Sandbox performance rules."
triggers: ["pymc", "bambi", "bayesian", "posterior", "MCMC", "prior", "credible interval", "uncertainty quantification", "probabilistic", "hierarchical", "arviz", "trace", "variational inference", "ADVI", "Bayesian causal", "Bayesian treatment effect"]
---
# Bayesian Causal Inference with PyMC and Bambi

## When to use PyMC vs DoWhy
| Situation | Use |
|---|---|
| Need a point estimate + fast result | DoWhy / EconML |
| Need full uncertainty quantification | **PyMC / Bambi** |
| Hierarchical / grouped data (users in markets) | **PyMC** |
| Want to incorporate domain priors | **PyMC** |
| Dataset > 50K rows | Use ADVI (see below) or sample 10K rows |

## ⚠️ SANDBOX PERFORMANCE RULES (mandatory)
- **MCMC (NUTS) on >10K rows will timeout.** For large datasets, use one of:
  1. Subsample to ≤10K rows before fitting
  2. Use variational inference: `pm.fit(method="advi")` (10–100× faster)
- Set `draws=500, tune=500` for sandbox runs (not the default 1000/1000)
- Always set `random_seed=42` for reproducibility

---

## Option A — Bambi (recommended for standard regression models)

```python
import bambi as bmb
import arviz as az
import pandas as pd
import numpy as np

# Bambi uses R-style formulas — much simpler than raw PyMC
# treatment and confounders on the right-hand side
model = bmb.Model(
    "outcome ~ treatment + confounder1 + confounder2",
    data=df,
    family="gaussian"   # or "bernoulli" for binary outcome
)

# For sandbox: use ADVI (variational) instead of MCMC
idata = model.fit(
    method="advi",       # variational inference — fast on large N
    num_samples=2000,    # posterior samples to draw from approximation
    random_seed=42
)

# For small datasets only: full MCMC
# idata = model.fit(draws=500, tune=500, random_seed=42, target_accept=0.9)

# Extract treatment effect posterior
treatment_posterior = az.extract(idata, var_names=["treatment"])["treatment"].values
ate_mean = float(treatment_posterior.mean())
ate_hdi = az.hdi(treatment_posterior, hdi_prob=0.94)
print(f"Bayesian ATE: {ate_mean:.4f}  94% HDI: [{ate_hdi[0]:.4f}, {ate_hdi[1]:.4f}]")
```

## Option B — Binary outcome (logistic regression)

```python
model = bmb.Model(
    "fraud ~ high_amount + V1 + V2 + V3 + V4 + V5",
    data=df.sample(5000, random_state=42),   # subsample for sandbox
    family="bernoulli",
    link="logit"
)
idata = model.fit(method="advi", num_samples=2000, random_seed=42)

# Posterior probability of treatment effect
treat_coef = az.extract(idata, var_names=["high_amount"])["high_amount"].values
print(f"P(treatment increases outcome) = {(treat_coef > 0).mean():.3f}")
```

## Option C — Raw PyMC for custom structural causal models

```python
import pymc as pm
import numpy as np

# Standardise inputs for better sampling
X = df[["V1", "V2", "V3"]].values          # confounders
T = df["treatment"].values                   # treatment (0/1)
Y = df["outcome"].values                     # outcome

with pm.Model() as causal_model:
    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=1)            # intercept
    beta_t = pm.Normal("beta_t", mu=0, sigma=0.5)        # treatment effect ← ATE
    beta_x = pm.Normal("beta_x", mu=0, sigma=1, shape=X.shape[1])
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Likelihood
    mu = alpha + beta_t * T + pm.math.dot(X, beta_x)
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

    # Variational inference (use for sandbox; NUTS is too slow on large N)
    approx = pm.fit(n=10000, method="advi", random_seed=42,
                    progressbar=False)
    idata = approx.sample(2000, random_seed=42)

ate_samples = idata.posterior["beta_t"].values.flatten()
print(f"Bayesian ATE: {ate_samples.mean():.4f} ± {ate_samples.std():.4f}")
print(f"94% HDI: {az.hdi(ate_samples, hdi_prob=0.94)}")
```

## Posterior Visualisation

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(ate_samples, bins=50, density=True, alpha=0.7, color="steelblue")
hdi = az.hdi(ate_samples, hdi_prob=0.94)
ax.axvline(ate_samples.mean(), color="red", linestyle="--", label=f"Mean={ate_samples.mean():.4f}")
ax.axvline(0, color="black", linestyle=":", alpha=0.5, label="Zero effect")
ax.axvspan(hdi[0], hdi[1], alpha=0.2, color="steelblue", label=f"94% HDI [{hdi[0]:.4f}, {hdi[1]:.4f}]")
ax.set_xlabel("Treatment Effect (ATE)")
ax.set_title("Posterior Distribution of Causal Treatment Effect")
ax.legend()
plt.tight_layout()
plt.savefig("posterior_ate.png", dpi=150, bbox_inches="tight")
plt.close()
```

## Bayesian Instrumental Variables

```python
# Two-stage Bayesian IV using PyMC
with pm.Model() as iv_model:
    # Stage 1: instrument Z → treatment T
    gamma_z = pm.Normal("gamma_z", mu=0, sigma=1)
    sigma_t = pm.HalfNormal("sigma_t", sigma=1)
    T_hat = pm.Normal("T_hat", mu=gamma_z * Z, sigma=sigma_t, observed=T)

    # Stage 2: predicted treatment → outcome
    beta_iv = pm.Normal("beta_iv", mu=0, sigma=0.5)   # LATE estimate
    sigma_y = pm.HalfNormal("sigma_y", sigma=1)
    Y_obs = pm.Normal("Y_obs", mu=beta_iv * T_hat, sigma=sigma_y, observed=Y)

    approx = pm.fit(n=10000, method="advi", progressbar=False, random_seed=42)
    idata_iv = approx.sample(2000, random_seed=42)

late = idata_iv.posterior["beta_iv"].values.flatten()
print(f"Bayesian LATE: {late.mean():.4f}  94% HDI: {az.hdi(late, hdi_prob=0.94)}")
```

## Sandbox Rules
- `pymc`, `arviz`, `bambi` are available.
- Always subsample or use ADVI for datasets >10K rows.
- `draws=500, tune=500` for NUTS in sandbox; default 1000/1000 will timeout.
- Use `progressbar=False` in `pm.fit()` to avoid cluttering stdout.
- `pm.set_data()` is not needed for simple models — just pass `observed=Y`.
