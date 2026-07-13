---
id: autogluon_tabular
description: "AutoGluon TabularPredictor code patterns: problem-type detection, eval metrics, both fit policies (deterministic portfolio / time-budgeted), leaderboard evaluation, threshold calibration, feature importance. Tabular classification/regression only."
triggers: ["autogluon", "automl", "TabularPredictor", "auto ml", "automated machine learning", "weighted ensemble", "model portfolio"]
---
# AutoGluon TabularPredictor — Canonical Patterns

AutoGluon is installed as `autogluon.tabular`. One `.fit()` call trains and ensembles
gradient-boosted trees and forests, handling missing values, categorical encoding, and
feature-type inference internally. **Never build manual sklearn pipelines, encoders, or
imputers around it.**

## Problem type and eval metric

```python
n_unique = df[target_col].nunique()
if n_unique == 2:
    problem_type = 'binary'
elif df[target_col].dtype == object or n_unique <= 20:
    problem_type = 'multiclass'      # includes integer-coded class labels (0..k)
else:
    problem_type = 'regression'
eval_metric = 'roc_auc' if problem_type == 'binary' else ('f1_macro' if problem_type == 'multiclass' else 'rmse')
```

A numeric target with few distinct values is a coded multiclass label, not a regression
target — do not decide by dtype alone.

## Leakage guard: drop identifier-like columns before fitting

```python
drop_cols = [c for c in df.columns if any(x in c.lower() for x in ['id', 'index', 'key', 'name'])]
df_clean = df.drop(columns=drop_cols, errors='ignore')
print("drop_cols:", drop_cols)
```

This is a name heuristic — always print what it dropped so a false positive is visible.

## Split (seeded, stratified when possible)

```python
from sklearn.model_selection import train_test_split
try:
    df_train, df_test = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean[target_col])
except ValueError:   # continuous target — stratification impossible
    df_train, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)
```

## Fit — two policies (the RECIPE INVARIANTS block states which one applies)

Write the whole call as ONE logical line inside the parentheses — no backslash continuations.

**Deterministic fixed portfolio** (reproducible: identical data + seed ⇒ identical scores;
independent of machine load — required whenever results must be repeatable):

```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label=target_col, eval_metric=eval_metric, verbosity=0).fit(df_train, hyperparameters={'GBM': {}, 'XGB': {}, 'RF': {}}, num_bag_folds=0, fit_weighted_ensemble=True)
```

**Time-budgeted exploration** (searches a wider model space; scores vary run-to-run with
machine load — never use when reproducibility matters):

```python
predictor = TabularPredictor(label=target_col, eval_metric=eval_metric, verbosity=0).fit(df_train, presets='good_quality', time_limit=120, excluded_model_types=['NN_TORCH', 'FASTAI'])
```

Never mix the two policies. `verbosity=0` always (log spam otherwise). The sandbox has
no GPU — exclude neural nets under the time-budgeted policy.

## Evaluate

```python
leaderboard = predictor.leaderboard(df_test, silent=True)
test_score = float(leaderboard.iloc[0]['score_test'])
print(leaderboard.head(8))
```

## predict_proba returns a DataFrame

```python
y_prob = predictor.predict_proba(df_test)   # DataFrame; columns ARE the class labels
p1 = y_prob.iloc[:, 1]                      # use .iloc — numpy-style y_prob[:, 1] fails
```

## Threshold calibration (binary only) — label-dtype-safe

`gads_calibrate_threshold` is pre-defined in the kernel (do not import it). Class labels
may be strings, so map the thresholded boolean back to the actual labels — a plain
`.astype(int)` breaks `confusion_matrix` when y_true holds strings:

```python
cal = gads_calibrate_threshold(y_test, y_prob)
best_t = cal['best_threshold']
y_pred = (y_prob.iloc[:, 1] >= best_t).map({True: y_prob.columns[1], False: y_prob.columns[0]})
```

## Feature importance (permutation) — always subsample

Without subsampling this runs over the full test set and exceeds the sandbox timeout.
Build the subsample DataFrame FIRST (per class for imbalanced binary targets, so the
minority class is represented), then make exactly ONE feature_importance call with
exactly these keyword arguments — never add `subsample_size` (the subsampling is
already done):

```python
if problem_type == 'binary' and df_test[target_col].value_counts().min() < 100:
    fi_df = df_test.groupby(target_col, group_keys=False).apply(lambda x: x.sample(min(len(x), 500), random_state=42))
else:
    fi_df = df_test.sample(min(len(df_test), 1000), random_state=42)
fi = predictor.feature_importance(fi_df, num_shuffle_sets=1, silent=True)
```

`fi` is a DataFrame whose **index** holds the feature names (`fi.index`, `fi['importance']`);
`fi.columns` does NOT contain feature names.

## Persist

```python
import joblib
joblib.dump(predictor, 'model.joblib')   # pickle is sandbox-blocked
```

## Scale limits

Datasets over 500K rows: subsample to ≤200K before fitting (`df.sample(n, random_state=42)`)
to avoid OOM.

## Do NOT use AutoGluon for

- Causal effect estimation — use `gads_causal_estimate_ate()` (pre-defined native node)
- Time-series forecasting — `TimeSeriesPredictor` has its own skill
- Pure NLP — sentence-transformers + sklearn
