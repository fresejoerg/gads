---
id: supervised_modeling
description: "Training patterns for sklearn/LightGBM/XGBoost: numeric-only feature matrix, HistGradientBoosting quirks, multiclass metrics"
triggers: ["classifier", "classification", "regression model", "train a model", "fit a model", "predict", "f1", "macro-f1", "accuracy", "auc", "log loss", "xgboost", "lightgbm", "gradient boosting", "random forest", "feature importance"]
---
# Supervised Modeling in the Sandbox

## Numeric-only feature matrix

Before calling `.fit()` on ANY sklearn estimator, X must contain ONLY numeric columns — raw
text columns cause `ValueError: could not convert string to float`. Filter explicitly:

```python
X = df[feature_cols]  # explicit list of numeric column names
# or
X = df.select_dtypes(include='number').drop(columns=['id'], errors='ignore')
```

X_test MUST use the same columns as X_train for `.predict()` / `.predict_proba()`.

## Gradient boosting — all three options work

```python
from sklearn.ensemble import HistGradientBoostingClassifier        # Option A
model = HistGradientBoostingClassifier(max_iter=200, random_state=42)

import lightgbm as lgb                                             # Option B
model = lgb.LGBMClassifier(n_estimators=200, random_state=42)

import xgboost as xgb                                              # Option C
model = xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss')
```

HistGradientBoosting quirks:
- Use `max_iter=` (NOT `n_estimators=`) for the number of trees.
- `.feature_importances_` is patched into it by this sandbox (split-gain sum) — use it normally.

## Multiclass metrics (3+ classes)

NEVER slice predict_proba down to one column for multiclass targets:

```python
from sklearn.metrics import f1_score, log_loss as _log_loss_fn
y_pred = model.predict(X_test)                  # shape (n,)
y_proba = model.predict_proba(X_test)           # full (n, k) matrix — do NOT take [:, 1]
macro_f1 = f1_score(y_test, y_pred, average='macro', labels=model.classes_)
log_loss = _log_loss_fn(y_test, y_proba, labels=model.classes_)
```

Alias the `log_loss` import when the contract requires a variable named `log_loss`.
Do NOT wrap metric computation in try/except — errors must propagate so the task can retry.
Use `random_state=42` everywhere.
