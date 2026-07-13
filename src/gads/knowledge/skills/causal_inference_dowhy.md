---
id: causal_inference_dowhy
description: "DoWhy ATE estimation via the gads_causal_estimate_ate native node (preferred, one call). Includes refutation, variable-role identification (with post-treatment exclusion), and result visualization patterns."
triggers: ["dowhy", "treatment effect", "average treatment effect", "ate", "backdoor", "refute", "propensity score", "causal model", "estimate effect", "identify effect", "confounder"]
---
# Causal Inference with DoWhy

## Identifying variable roles (treatment / outcome / confounders)

**The objective is authoritative.** If it names the treatment, the outcome, specific
confounders, or variables to EXCLUDE, use exactly those — copy the column names from the
objective (verify each exists in `df.columns`).

Only when the objective does not enumerate confounders, select them programmatically —
and even then, **first remove any variable the objective describes as post-treatment**
(a mediator or downstream consequence of the treatment, e.g. "X happens after / because
of the treatment"). Adjusting for post-treatment variables biases the effect estimate,
and no schema heuristic can detect them — only the objective text can:

```python
import numpy as np
temporal_id_patterns = {'time', 'date', 'timestamp', 'id', 'index'}
post_treatment_cols = []   # fill from the objective, e.g. ['opened', 'agreement']
confounder_cols = [
    c for c in df.columns
    if c not in (treatment_col, outcome_col)
    and c not in post_treatment_cols
    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    and not any(p in c.lower() for p in temporal_id_patterns)
]
if len(confounder_cols) > 10:   # cap for estimator stability
    corrs = df[confounder_cols].corrwith(df[outcome_col]).abs()
    confounder_cols = corrs.nlargest(10).index.tolist()
print(f"treatment={treatment_col} outcome={outcome_col} confounders={confounder_cols}")
```

This fallback is a heuristic (it assumes every remaining numeric column is a
pre-treatment covariate) — state that assumption when reporting results.

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

## Visualizing the result

Reuse the kernel variables (`ate`, `placebo_new_effect`, `subset_new_effect`,
`treatment_col`, `outcome_col`, `confounder_cols`, `df_sample`) — never recompute the
effect. A two-panel matplotlib summary works for any outcome type:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dfv = df_sample if 'df_sample' in globals() else df
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Panel 1: the estimate against its refutation checks
ax = axes[0]
vals = [ate, placebo_new_effect, subset_new_effect]
bars = ax.bar(['ATE', 'Placebo (~0)', 'Subset (~ATE)'], vals, alpha=0.85, edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
ax.set_ylabel('Effect size'); ax.set_title('ATE vs refutation checks')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{v:.4f}', ha='center', va='bottom', fontsize=9)

# Panel 2: adjustment-set relevance (|corr with outcome|)
ax = axes[1]
conf_corrs = dfv[confounder_cols].corrwith(dfv[outcome_col]).abs().sort_values()
ax.barh(range(len(conf_corrs)), conf_corrs.values, alpha=0.7)
ax.set_yticks(range(len(conf_corrs))); ax.set_yticklabels(list(conf_corrs.index), fontsize=8)
ax.set_xlabel('|corr with outcome|'); ax.set_title('Confounder relevance')

plt.suptitle(f'Causal effect of {treatment_col} on {outcome_col}', fontweight='bold')
plt.tight_layout()
plt.savefig('causal_effect_summary.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved causal_effect_summary.png")
```

## Sandbox rules
- DoWhy 0.14 — old (≤0.11) APIs do not work.
- Subsample over 20K rows to 20K before any manual fitting (the native node does this for you).
- Serialize with `joblib.dump(obj, 'model.joblib')` — pickle is blocked.
</content>
