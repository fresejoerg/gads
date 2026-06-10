---
id: causal_impact_timeseries
description: "causalimpact: Bayesian structural time-series causal analysis for intervention effects. For pre/post event studies on time-series data. (pip: pycausalimpact, import: causalimpact)"
triggers: ["causal impact", "intervention effect", "time series causal", "pre-post", "event study", "pycausalimpact", "causalimpact", "campaign effect", "policy effect", "interrupted time series", "before after"]
---
# Time-Series Causal Impact Analysis with pycausalimpact

## When to use
Use when you have a **time series metric** and a **one-time intervention** (product launch,
policy change, marketing campaign) and want to estimate what would have happened *without* the
intervention (the counterfactual), and how much the intervention changed the outcome.

Requires: a pre-period (before intervention) and a post-period (after intervention).
Optionally: one or more **control time series** (unaffected by the intervention) that serve as
covariates to build the counterfactual.

## Basic usage

```python
from causalimpact import CausalImpact
import pandas as pd

# data: DataFrame with DatetimeIndex
# Column 0: outcome metric (the series affected by intervention)
# Columns 1+: control series (optional but recommended)
# pre_period: list of [start_date, last_date_before_intervention]
# post_period: list [first_date_after_intervention, end_date]

pre_period  = ["2023-01-01", "2023-06-30"]
post_period = ["2023-07-01", "2023-12-31"]

ci = CausalImpact(data, pre_period, post_period)

# Print textual summary
print(ci.summary())

# Access numeric results
results = ci.summary_data
abs_effect   = float(results.loc["Average", "abs_effect"])
rel_effect   = float(results.loc["Average", "rel_effect"])
p_value      = float(results.loc["Average", "p_value"])
print(f"Absolute effect: {abs_effect:.4f}")
print(f"Relative effect: {rel_effect*100:.2f}%")
print(f"P(effect > 0): {1 - p_value:.3f}")
```

## With control series (recommended)

```python
# data layout: first column = outcome, remaining = controls
data = pd.DataFrame({
    "outcome":   outcome_series,
    "control_1": control_series_1,
    "control_2": control_series_2,
}, index=date_index)

ci = CausalImpact(data, pre_period, post_period,
                  model_args={"nseasons": 7})  # weekly seasonality
print(ci.summary())
```

## Plot and save the impact chart

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = ci.plot(figsize=(12, 8))
plt.tight_layout()
plt.savefig("causal_impact_plot.png", dpi=150, bbox_inches="tight")
plt.close()
```

## Output variables to store as metrics

```python
summary = ci.summary_data
# Store key scalars for the orchestrator
abs_effect_mean  = float(summary.loc["Average", "abs_effect"])
rel_effect_mean  = float(summary.loc["Average", "rel_effect"])
cum_effect       = float(summary.loc["Cumulative", "abs_effect"])
p_value          = float(summary.loc["Average", "p_value"])
print(f"abs_effect_mean={abs_effect_mean}")
print(f"rel_effect_mean={rel_effect_mean}")
print(f"cum_effect={cum_effect}")
```

## Sandbox Rules
- Package is `pycausalimpact` but **import as `from causalimpact import CausalImpact`** — never `import pycausalimpact`.
- Data must have a DatetimeIndex or integer index in chronological order.
- If no control series is available, still works but counterfactual is less precise.
- `matplotlib.use("Agg")` before plotting to avoid display errors in headless sandbox.
