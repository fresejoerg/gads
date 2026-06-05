---
id: causal_inference_dowhy
description: "DoWhy 0.14 API: use gads_causal_estimate_ate() native node (preferred) or manual 4-step. Correct GML format (integer id + string label). Includes refutation and blocked old APIs."
triggers: ["causal", "dowhy", "treatment effect", "ATE", "average treatment effect", "confounder", "backdoor", "frontdoor", "counterfactual", "refute", "propensity score", "causal model", "causal effect", "identify effect", "estimate effect"]
---
# Causal Inference with DoWhy (version 0.14)

## ✅ PREFERRED: Use the Native Node (one function call)

`gads_causal_estimate_ate` is pre-defined in the kernel. It handles subsampling,
treatment binarization, GML construction, CausalModel, identification, estimation,
and both refutation tests — all in one call. **Prefer this over writing DoWhy manually.**

```python
# PREFERRED PATTERN — call the native node:
result = gads_causal_estimate_ate(df, treatment_col, outcome_col, confounder_cols)

# Unpack required_metrics as top-level scalars (exact names required):
ate = result["ate"]
placebo_new_effect = result["placebo_new_effect"]
subset_new_effect = result["subset_new_effect"]

# Other return values available if needed:
# result["treatment_col"]       — actual col used (may be 'high_X' if binarized)
# result["identified_estimand"] — DoWhy estimand object
# result["causal_estimate"]     — DoWhy estimate object
# result["df_sample"]           — dataframe used for estimation

print(f"ATE={ate:.4f}  placebo={placebo_new_effect:.4f}  subset={subset_new_effect:.4f}")
gads_emit_insight("causal_effect", f"ATE={ate:.4f}, placebo_new_effect={placebo_new_effect:.4f}, subset_new_effect={subset_new_effect:.4f}")
```

## ❌ DO NOT IMPLEMENT FROM SCRATCH

**Never implement propensity score matching, doubly-robust estimation, AIPW, or any causal estimator manually.** DoWhy implements all of these internally. Use `causal_model.estimate_effect()` — it handles propensity scores, outcome models, and weighting automatically.

**For datasets >50K rows**: subsample to 20K before any propensity/outcome model fitting:
```python
df_sample = df.sample(20000, random_state=42) if len(df) > 20000 else df
```

## ❌ WRONG IMPORTS — DO NOT USE THESE (they do not exist in DoWhy 0.14)

```python
# ALL OF THESE WILL FAIL — never write them:
from dowhy.utils import Graph          # ❌ removed in 0.14
import causalgraphicalmodels           # ❌ different unrelated package
from dowhy.causal_graph import CausalGraph  # ❌ not the right import path
import networkx as nx; G = nx.DiGraph()    # ❌ do NOT pass nx graph to CausalModel
```

## Schema Analysis Patterns (run before building the model)

```python
import pandas as pd
import numpy as np

# 1. Identify confounders automatically from schema
TEMPORAL_ID_PATTERNS = {"time", "date", "timestamp", "id", "index"}
confounder_cols = [
    c for c in df.columns
    if c not in (treatment_col, outcome_col)
    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    and not any(p in c.lower() for p in TEMPORAL_ID_PATTERNS)
]
# Cap at 10 by absolute correlation with outcome
if len(confounder_cols) > 10:
    corrs = df[confounder_cols].corrwith(df[outcome_col]).abs()
    confounder_cols = corrs.nlargest(10).index.tolist()
print(f"Confounders ({len(confounder_cols)}): {confounder_cols}")

# 2. Engineer binary treatment from continuous column
global_median = df[treatment_col].median()
binary_treatment_col = f"high_{treatment_col}"
df[binary_treatment_col] = (df[treatment_col] > global_median).astype(int)
treatment_col = binary_treatment_col   # update variable

# 3. Check class balance → drives estimator selection
minority_frac = df[outcome_col].value_counts(normalize=True).min()
print(f"Minority class fraction: {minority_frac:.4f}")
estimator = (
    "backdoor.propensity_score_matching"
    if minority_frac < 0.05
    else "backdoor.linear_regression"
)
print(f"Selected estimator: {estimator}")
# NOTE: NEVER use propensity_score_weighting — it produces NaN for rare outcomes.
```

## Programmatic GML Construction (fallback — prefer gads_causal_estimate_ate above)

GML CRITICAL: nodes need **integer id** and **string label**. Edges use integer source/target.
String ids fail with NetworkXError — always use integers.

```python
# CORRECT GML: integer id + string label, edges by index
nodes = [treatment_col, outcome_col] + list(confounder_cols)
node_idx = {n: i for i, n in enumerate(nodes)}
node_str = "\n".join(f'  node [ id {i} label "{n}" ]' for i, n in enumerate(nodes))
edges = ([(c, treatment_col) for c in confounder_cols]
         + [(c, outcome_col) for c in confounder_cols]
         + [(treatment_col, outcome_col)])
edge_str = "\n".join(f'  edge [ source {node_idx[s]} target {node_idx[t]} ]' for s, t in edges)
gml_string = f"graph [ directed 1\n{node_str}\n{edge_str}\n]"
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

## EDA Scalar Patterns (store these for postconditions)

```python
# Always store these as plain Python scalars after loading the data:
naive_outcome_rate = float(df[outcome_col].mean())   # e.g. fraud rate
minority_class_frac = float(df[outcome_col].value_counts(normalize=True).min())
n_rows = int(len(df))
print(f"Outcome rate: {naive_outcome_rate:.4f}  Minority frac: {minority_class_frac:.4f}  N: {n_rows}")
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

# Store as plain scalars for postconditions and required_metrics:
placebo_new_effect = float(refute_placebo.new_effect)
subset_new_effect  = float(refute_subset.new_effect)
print(f"placebo_new_effect={placebo_new_effect:.6f}")
print(f"subset_new_effect={subset_new_effect:.6f}")

refutation_results = {
    "placebo_new_effect": placebo_new_effect,
    "subset_new_effect": subset_new_effect
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
