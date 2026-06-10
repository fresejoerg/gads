# Proposed Recipe Invariants

This file contains proposed additions to recipe invariants offline based on trace distillation.

## Recipe: [causal_effect_estimation_dowhy.md](file://src/gads/knowledge/recipes/causal_effect_estimation_dowhy.md) (ID: `causal_effect.observational.dowhy`)

```yaml
invariants:
  - "LARGE DATASET PERFORMANCE: for datasets >20K rows, ALWAYS subsample df to 20K before CausalModel construction to prevent timeouts. Use df_sample = df.sample(20000, random_state=42)."
```

