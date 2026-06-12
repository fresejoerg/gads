---
id: causal_inference_dowhy
description: "DoWhy ATE estimation via the gads_causal_estimate_ate native node (preferred, one call). Includes refutation. Lean — the native node replaces manual DoWhy boilerplate."
triggers: ["dowhy", "treatment effect", "average treatment effect", "ate", "backdoor", "refute", "propensity score", "causal model", "estimate effect", "identify effect"]
---
# Causal Inference with DoWhy

## ✅ Use the native node — one call does everything

`gads_causal_estimate_ate` is already defined in the kernel. It handles subsampling,
treatment binarization, GML construction, identification, estimator selection, and
**both** refutation tests internally. Prefer it over writing DoWhy by hand.

**Do NOT import it.** It is pre-defined — never write `import gads_utils`,
`from gads_helpers import ...`, or any `gads`-prefixed import. Just call the function.

```python
result = gads_causal_estimate_ate(df, treatment_col, outcome_col, confounder_cols)

# Required metrics — unpack as top-level floats with these exact names:
ate = float(result["ate"])
placebo_new_effect = float(result["placebo_new_effect"])   # ~0 if estimate is valid
subset_new_effect = float(result["subset_new_effect"])     # ~ate if estimate is stable
treatment_col = result["treatment_col"]   # may be 'high_<name>' if it was binarized
df_sample = result["df_sample"]           # rows actually used (after subsampling)

print(f"ATE={ate:.6f}  placebo={placebo_new_effect:.6f}  subset={subset_new_effect:.6f}")
gads_emit_insight("causal_effect",
                  f"ATE={ate:.4f}, placebo={placebo_new_effect:.4f}, subset={subset_new_effect:.4f}")
```

The function returns a dict; other keys available if needed: `identified_estimand`,
`causal_estimate`, `df_sample`.

## ❌ Do not implement from scratch

Never hand-write propensity score matching, doubly-robust/AIPW estimation, or manual
GML graphs. The native node does all of this. Do **not** import `dowhy.utils.Graph`,
`dowhy.causal_graph.CausalGraph`, or `causalgraphicalmodels` — they do not exist in
DoWhy 0.14. Do not pass a `networkx.DiGraph` to `CausalModel`.

## Manual fallback (only if the native node is unavailable)

If you must build the graph yourself, GML node ids are **integers**, labels are strings,
and edges reference the integer ids. String ids raise `NetworkXError`:

```python
nodes = [treatment_col, outcome_col] + list(confounder_cols)
idx = {n: i for i, n in enumerate(nodes)}
node_str = "\n".join(f'  node [ id {i} label "{n}" ]' for i, n in enumerate(nodes))
edges = ([(c, treatment_col) for c in confounder_cols]
         + [(c, outcome_col) for c in confounder_cols]
         + [(treatment_col, outcome_col)])
edge_str = "\n".join(f'  edge [ source {idx[s]} target {idx[t]} ]' for s, t in edges)
gml_string = f"graph [ directed 1\n{node_str}\n{edge_str}\n]"
```

Estimator choice for manual use: `backdoor.propensity_score_matching` for rare outcomes
(minority class < 5%), `backdoor.linear_regression` otherwise. **Never**
`propensity_score_weighting` — it returns NaN on imbalanced data.

## Sandbox rules
- DoWhy 0.14 — old (≤0.11) APIs do not work.
- Subsample over 20K rows to 20K before any manual fitting (the native node does this for you).
- Serialize with `joblib.dump(obj, 'model.joblib')` — pickle is blocked.
</content>
