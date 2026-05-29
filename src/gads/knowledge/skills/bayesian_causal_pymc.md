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
- **Bambi MCMC benchmarks in sandbox:** ~7s / 1K rows, ~60s / 8K rows (draws=500, tune=300, chains=1)
- **Safe rule:** subsample to ≤5K rows for MCMC; ≤20K rows for ADVI
- Set `draws=500, tune=300, chains=1, cores=1` for sandbox MCMC (not the defaults)
- Always set `random_seed=42` and `progressbar=False`

---

## Option A — Bambi MCMC (≤5K rows, most accurate)

```python
import bambi as bmb
import arviz as az

# Subsample if needed
df_model = df.sample(min(len(df), 5000), random_state=42)

model = bmb.Model(
    "outcome ~ treatment + confounder1 + confounder2",
    data=df_model,
    family="gaussian"   # or "bernoulli" for binary outcome
)

idata = model.fit(
    draws=500, tune=300,
    chains=1, cores=1,
    target_accept=0.9,
    random_seed=42,
    progressbar=False
)

# Extract treatment effect posterior
treat_post = idata.posterior["treatment"].values.flatten()
ate = float(treat_post.mean())
ate_hdi = az.hdi(treat_post, hdi_prob=0.94)
print(f"Bayesian ATE: {ate:.4f}  94% HDI: [{float(ate_hdi[0]):.4f}, {float(ate_hdi[1]):.4f}]")
print(f"P(effect > 0): {(treat_post > 0).mean():.3f}")
```

## Option B — Bambi + PyMC ADVI (>5K rows, faster but approximate)

```python
import bambi as bmb
import pymc as pm
import arviz as az

# ADVI via the underlying PyMC model
model = bmb.Model(
    "outcome ~ treatment + confounder1 + confounder2",
    data=df,
    family="gaussian"
)
model.build()   # required before accessing backend

with model.backend.model:
    approx = pm.fit(n=20000, method="advi", progressbar=False, random_seed=42)
    idata = approx.sample(2000, random_seed=42)

# Variable names match the Bambi formula terms
treat_post = idata.posterior["treatment"].values.flatten()
ate = float(treat_post.mean())
ate_hdi = az.hdi(treat_post, hdi_prob=0.94)
print(f"Bayesian ATE (ADVI): {ate:.4f}  94% HDI: [{float(ate_hdi[0]):.4f}, {float(ate_hdi[1]):.4f}]")
# NOTE: ADVI underestimates uncertainty for strongly correlated parameters.
# If HDI is implausibly narrow, switch to Option A (MCMC on a subsample).
```

## Option C — Binary outcome (logistic)

```python
model = bmb.Model(
    "fraud ~ high_amount + V1 + V2 + V3 + V4 + V5",
    data=df.sample(3000, random_state=42),
    family="bernoulli",
    link="logit"
)
idata = model.fit(draws=500, tune=300, chains=1, cores=1,
                  random_seed=42, progressbar=False)

treat_coef = idata.posterior["high_amount"].values.flatten()
print(f"Posterior mean log-odds: {treat_coef.mean():.4f}")
print(f"P(effect > 0): {(treat_coef > 0).mean():.3f}")
```

## Option D — Raw PyMC for custom structural causal models

```python
import pymc as pm
import numpy as np

X = df[["V1", "V2", "V3"]].values    # confounders (standardised)
T = df["treatment"].values            # treatment (0/1)
Y = df["outcome"].values              # outcome

with pm.Model() as causal_model:
    alpha  = pm.Normal("alpha",  mu=0, sigma=1)
    beta_t = pm.Normal("beta_t", mu=0, sigma=0.5)   # treatment ATE
    beta_x = pm.Normal("beta_x", mu=0, sigma=1, shape=X.shape[1])
    sigma  = pm.HalfNormal("sigma", sigma=1)

    mu    = alpha + beta_t * T + pm.math.dot(X, beta_x)
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

    # For sandbox: ADVI. For full analysis: NUTS with draws=500, tune=300
    approx = pm.fit(n=20000, method="advi", progressbar=False, random_seed=42)
    idata  = approx.sample(2000, random_seed=42)

ate_samples = idata.posterior["beta_t"].values.flatten()
print(f"ATE: {ate_samples.mean():.4f} ± {ate_samples.std():.4f}")
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
