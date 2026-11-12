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

# 4. Identifier columns (leakage risk). Match whole names or per-row uniqueness, NEVER substrings:
#    'consumer_price_index', 'valid' and 'keyword' are features, not identifiers.
import re
_ID_NAME = re.compile(r'^(id|uuid|guid|index|rownum|row_id|unnamed: 0)$|_(id|uuid|guid)$')
drop_cols = [c for c in df.columns if c != target_col and (
    _ID_NAME.search(str(c).lower())  # the name IS an identifier (id, customer_id, uuid, index) - never a substring
    or (not pd.api.types.is_float_dtype(df[c]) and len(df) > 20
        and df[c].nunique(dropna=True) == df[c].notna().sum()))]  # or it is unique on every row
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
