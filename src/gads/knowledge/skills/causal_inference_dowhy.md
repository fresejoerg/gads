---
id: causal_inference_dowhy
description: "DoWhy 0.14 API: CausalModel (GML string), identify_effect, estimate_effect, refute_estimate. Correct patterns for DoWhy 0.14 — includes blocked old APIs."
triggers: ["causal", "dowhy", "treatment effect", "ATE", "average treatment effect", "confounder", "backdoor", "frontdoor", "counterfactual", "refute", "propensity score", "causal model", "causal effect", "identify effect", "estimate effect"]
---
# Causal Inference with DoWhy (version 0.14)

## ❌ WRONG IMPORTS — DO NOT USE THESE (they do not exist in DoWhy 0.14)

```python
# ALL OF THESE WILL FAIL — never write them:
from dowhy.utils import Graph          # ❌ removed in 0.14
import causalgraphicalmodels           # ❌ different unrelated package
from dowhy.causal_graph import CausalGraph  # ❌ not the right import path
import networkx as nx; G = nx.DiGraph()    # ❌ do NOT pass nx graph to CausalModel
```

## ✅ The 4-Step Workflow (DoWhy 0.14)

```python
from dowhy import CausalModel

# Step 1 — Build CausalModel with a GML string (the ONLY correct way in 0.14)
gml_string = """graph [
  directed 1
  node [ id "age"       label "age" ]
  node [ id "income"    label "income" ]
  node [ id "treatment" label "treatment" ]
  node [ id "outcome"   label "outcome" ]
  edge [ source "age"       target "treatment" ]
  edge [ source "age"       target "outcome"   ]
  edge [ source "income"    target "treatment" ]
  edge [ source "income"    target "outcome"   ]
  edge [ source "treatment" target "outcome"   ]
]"""

causal_model = CausalModel(
    data=df,
    treatment="treatment",
    outcome="outcome",
    graph=gml_string
)

# Step 2 — Identify the estimand (uses causal_model from Step 1)
identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

# Step 3 — Estimate the ATE
# IMPORTANT: for imbalanced outcomes (rare events <5%), use linear_regression or
# propensity_score_matching — NOT propensity_score_weighting (produces NaN).
# Option A: linear regression (robust, works on imbalanced data)
causal_estimate = causal_model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
    target_units="ate"
)
ate = float(causal_estimate.value)
print(f"ATE = {ate:.6f}")

# Option B: propensity score matching (good for moderate imbalance)
causal_estimate = causal_model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching",
    target_units="ate"
)
ate = float(causal_estimate.value)

# Option C: propensity score weighting — ONLY for balanced outcomes (>10% minority class)
# WARNING: produces NaN for rare events (e.g. <1% fraud rate). Use A or B instead.
# causal_estimate = causal_model.estimate_effect(
#     identified_estimand,
#     method_name="backdoor.propensity_score_weighting",
#     target_units="ate",
#     method_params={"weighting_scheme": "ips_weight"}
# )
```

## ✅ Refutation — REUSE causal_model FROM KERNEL, DO NOT REBUILD

The `causal_model`, `identified_estimand`, and `causal_estimate` variables are
already in the kernel from the previous tasks. **Do NOT import Graph or rebuild
the model**. Use the existing variables directly:

```python
# causal_model, identified_estimand, causal_estimate are already in the kernel

# Placebo test: permute treatment — ATE should collapse to ~0
refute_placebo = causal_model.refute_estimate(
    identified_estimand, causal_estimate,
    method_name="placebo_treatment_refuter",
    placebo_type="permute", random_seed=42
)
print("PLACEBO REFUTATION:")
print(refute_placebo)

# Data subset test: estimate on 80% subsample — should be numerically stable
refute_subset = causal_model.refute_estimate(
    identified_estimand, causal_estimate,
    method_name="data_subset_refuter",
    subset_fraction=0.8, random_seed=42
)
print("SUBSET REFUTATION:")
print(refute_subset)

refutation_results = {
    "placebo_new_effect": float(refute_placebo.new_effect),
    "placebo_p_value": float(refute_placebo.refutation_result.get("p_value", 0.0)),
    "subset_new_effect": float(refute_subset.new_effect)
}
print("refutation_results:", refutation_results)
```

## Estimand Method Reference
| method_name | When to use |
|---|---|
| `backdoor.linear_regression` | **Default — works for all imbalance levels** |
| `backdoor.propensity_score_matching` | Moderate imbalance, small-medium N |
| `backdoor.propensity_score_weighting` | **Only** for balanced outcomes (>10% minority). NaN on rare events. |
| `backdoor.econml.dml.LinearDML` | High-dimensional, needs econml |
| `iv.instrumental_variable` | Instrument variable available |

## Sandbox Rules
- **Version**: DoWhy 0.14 is installed. Old APIs from ≤0.11 do not work.
- **Graph spec**: Always use a GML string passed to `CausalModel(graph=...)`.
- **Refutation**: reuse `causal_model` from the kernel — never rebuild it.
- **Serialization**: `joblib.dump(causal_model, 'model.joblib')` — pickle is blocked.
