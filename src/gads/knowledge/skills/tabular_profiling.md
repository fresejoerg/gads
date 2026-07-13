---
id: tabular_profiling
description: "Canonical first-task dataset profile for tabular workflows: schema, nulls, target distribution, problem type, identifier-column detection, naive baseline."
triggers: ["profile the dataset", "naive baseline", "naive_baseline", "target distribution", "problem type", "profile"]
---
# Tabular Dataset Profile — Canonical First Task

Every tabular workflow starts by establishing ground truth about the data and a baseline
that any model must beat. Store results in plainly named global variables — downstream
tasks read them from the live kernel.

```python
import pandas as pd
import numpy as np

# 1. Schema ground truth
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

# 2. Target distribution (target_col comes from the objective / spec hints)
print(df[target_col].value_counts())

# 3. Problem type — numeric targets with few distinct values are coded class labels
n_unique = df[target_col].nunique()
if n_unique == 2:
    problem_type = 'binary'
elif df[target_col].dtype == object or n_unique <= 20:
    problem_type = 'multiclass'
else:
    problem_type = 'regression'
print("problem_type:", problem_type)

# 4. Identifier-like columns (leakage risk) — heuristic, so print what it caught
drop_cols = [c for c in df.columns if any(x in c.lower() for x in ['id', 'index', 'key', 'name'])]
print("drop_cols:", drop_cols)

# 5. Naive baseline — majority-class fraction (classification) or target std (regression)
if problem_type in ('binary', 'multiclass'):
    vc = df[target_col].value_counts(normalize=True)
    naive_baseline = float(vc.max())
else:
    naive_baseline = float(df[target_col].std())
print("naive_baseline:", naive_baseline)
```

Rules:
- Keep every variable at global scope (no functions) — the next task reads them from the kernel.
- Use plain `print()` for the scalar evidence lines exactly as shown.
- `pd.api.types.is_datetime64_ns` does not exist — never call it.
