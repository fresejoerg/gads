---
id: model_audit
description: "Methodological-soundness audit for any fitted sklearn-compatible supervised model via the native gads_audit_model (skore EstimatorReport): detects leakage, over/under-fit, class imbalance, worse-than-baseline, low-value features. Writes model_checks.json + emits issue insights."
triggers: ["audit", "methodological", "soundness", "leakage", "data leakage", "overfitting", "underfitting", "worse than baseline", "sanity check model", "diagnostic checks", "skore", "estimatorreport", "model checks", "is this model any good"]
---
# Methodological Audit with the Native `gads_audit_model`

## Call the native function — do not re-implement the checks
After a supervised model is fitted, pass it through the native audit. It wraps skore's
`EstimatorReport` and runs skore's automated diagnostic checks. Re-implementing leakage /
overfitting / baseline comparisons by hand is exactly what this node exists to avoid.

```python
audit = gads_audit_model(
    baseline_model,                      # any fitted or unfitted sklearn-compatible estimator/Pipeline
    X_train=X_train, y_train=y_train,    # the SAME split you trained/tested on — required to judge
    X_test=X_test,   y_test=y_test)      #   overfitting and leakage honestly
n_issues = audit["n_issues"]             # int
for c in audit["issues"]:                # each: code, title, severity, explanation, documentation_url
    print(c["code"], c["title"], "->", c["explanation"])
```

If you only have a single `(X, y)` and no split, the function will split internally
(stratified for classification): `gads_audit_model(model, X=X, y=y)`. Prefer passing the
real train/test split when you have it — an internal re-split cannot see leakage that
happened in your own upstream preprocessing.

## What it produces (deterministic)
- **`model_checks.json`** — an artifact with `ml_task`, a flat `metrics` dict, and every
  check partitioned into `issues` / `tips` / `passed` / `not_applicable`. Each issue/tip
  carries `code`, `title`, `explanation`, and a `documentation_url` with the mitigation.
- **Insights** — one `gads_emit_insight` per `issue`-severity finding (automatic).
- **Return dict** — `{issues, tips, passed, not_applicable, n_issues, n_tips, metrics,
  ml_task, checks_path}`.

## What the checks catch (skore SKD00x)
Potential over/under-fitting, high class imbalance, underrepresented classes, unscaled
coefficients, MDI bias for high-cardinality tree features, highly correlated inputs,
**model worse than a HistGradientBoosting baseline**, golden (leaky) features, useless
features, **train/test overlap in time series**, hyperparameters at a search edge / worth
tuning, estimator left untuned.

## Reporting
State the issue count and summarize each issue in an insight — a model that trips
`SKD009` (worse than baseline) or `SKD011` (golden/leaky feature) is not yet defensible,
regardless of headline accuracy. The audit is a diagnostic: it never raises, so a clean
run with `n_issues == 0` is itself a positive signal worth reporting.

## Pitfalls
- Passing an internally re-split `(X, y)` when your preprocessing already touched the full
  dataset hides leakage — pass the genuine held-out split.
- The audit reflects the estimator you hand it; audit the model you actually report, not a
  throwaway.
