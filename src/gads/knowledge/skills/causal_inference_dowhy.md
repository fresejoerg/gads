---
id: causal_inference_dowhy
description: "DoWhy 4-step API: CausalModel construction (GML string), identify_effect, estimate_effect (propensity/DML), refute_estimate. Sandbox-safe patterns."
triggers: ["causal", "dowhy", "treatment effect", "ATE", "average treatment effect", "confounder", "backdoor", "frontdoor", "counterfactual", "refute", "propensity score", "causal model", "causal effect", "identify effect", "estimate effect"]
---
# Causal Inference with DoWhy

## The 4-Step Workflow

```python
import dowhy
from dowhy import CausalModel

# Step 1 — Build CausalModel using a GML string (NEVER use eval())
gml_string = """
graph [
  directed 1
  node [ id "age"    label "age" ]
  node [ id "income" label "income" ]
  node [ id "treatment" label "treatment" ]
  node [ id "outcome"   label "outcome" ]
  edge [ source "age"    target "treatment" ]
  edge [ source "age"    target "outcome"   ]
  edge [ source "income" target "treatment" ]
  edge [ source "income" target "outcome"   ]
  edge [ source "treatment" target "outcome" ]
]
"""
model = CausalModel(
    data=df,
    treatment="treatment",
    outcome="outcome",
    graph=gml_string
)

# Step 2 — Identify the estimand
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

# Step 3 — Estimate the ATE
# Option A: Propensity score weighting (no extra libs)
causal_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_weighting",
    target_units="ate",
    method_params={"weighting_scheme": "ips_weight"}
)
ate = causal_estimate.value
print(f"ATE = {ate:.4f}")

# Option B: Linear DML (requires econml)
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
causal_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.LinearDML",
    target_units="ate",
    confidence_intervals=False,
    method_params={
        "init_params": {
            "model_y": RandomForestRegressor(n_estimators=50, random_state=42),
            "model_t": RandomForestClassifier(n_estimators=50, random_state=42),
            "random_state": 42
        },
        "fit_params": {}
    }
)
ate = causal_estimate.value
```

## Refutation (MANDATORY before reporting)

```python
# Placebo: replace treatment with random noise — ATE should collapse to ~0
refute_placebo = model.refute_estimate(
    identified_estimand, causal_estimate,
    method_name="placebo_treatment_refuter",
    placebo_type="permute", random_seed=42
)
print(refute_placebo)

# Data subset: estimate on 80% subsample — should be stable
refute_subset = model.refute_estimate(
    identified_estimand, causal_estimate,
    method_name="data_subset_refuter",
    subset_fraction=0.8, random_seed=42
)
print(refute_subset)

refutation_results = {
    "placebo": str(refute_placebo),
    "subset": str(refute_subset)
}
```

## Available Estimand Methods
| method_name | When to use |
|---|---|
| `backdoor.propensity_score_weighting` | Binary treatment, observed confounders |
| `backdoor.propensity_score_matching` | Small-medium datasets |
| `backdoor.linear_regression` | Continuous treatment, linear assumption |
| `backdoor.econml.dml.LinearDML` | High-dimensional controls, needs econml |
| `iv.instrumental_variable` | Instrument available |
| `frontdoor.two_stage_regression` | Measured mediator, frontdoor criterion |

## Sandbox Rules
- **Graph specification**: Always use GML strings. Never use `eval()` or `exec()`.
- **Serialization**: Use `joblib.dump(model, 'model.joblib')` — pickle is blocked.
- `dowhy`, `econml`, `statsmodels` are available. `graphviz` binary is available but prefer `networkx` + `matplotlib` for portability.
