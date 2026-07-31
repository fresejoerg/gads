---
id: survival_scikit_survival
description: "scikit-survival for survival PREDICTION: the (event, time) structured target array (build it with native gads_make_surv_target), RandomSurvivalForest / GradientBoostingSurvivalAnalysis, categorical encoding, and censoring-aware evaluation via native gads_evaluate_survival (IPCW C-index, time-dependent AUC, Integrated Brier Score)."
triggers: ["scikit-survival", "sksurv", "random survival forest", "randomsurvivalforest", "gradient boosting survival", "coxphsurvivalanalysis", "surv.from_arrays", "surv.from_dataframe", "structured array", "concordance_index_ipcw", "integrated brier score", "cumulative_dynamic_auc", "predict_survival_function", "survival prediction", "risk score"]
---
# Survival Prediction with scikit-survival (sksurv)

`scikit-survival` is sklearn-compatible but the **target `y` is a structured array**, not a
1-D vector: fields `(event: bool, time: float)`. Getting this array right is the #1 failure
point — always build it with the native node.

## 1. Build the target with the native `gads_make_surv_target`
```python
y = gads_make_surv_target(df, time_col="time", event_col="event")
# y is a numpy structured array, dtype [('event', '?'), ('time', '<f8')]; censored rows kept,
# event cast to bool. y["event"] is boolean, y["time"] is float.
X = df.drop(columns=["time", "event"])
```
Do not hand-roll `Surv.from_arrays` unless you must — the native handles the boolean cast,
0/1 and string event encodings, and validation. If you do it manually:
```python
from sksurv.util import Surv
y = Surv.from_arrays(event=df["event"].astype(bool), time=df["time"].astype(float))
```

## 2. Encode categoricals — X must be fully numeric
sksurv estimators need a numeric matrix. Use its one-hot encoder (or pandas):
```python
from sksurv.preprocessing import OneHotEncoder
X = OneHotEncoder().fit_transform(X)        # expects pandas 'category'/object columns
# or: X = pd.get_dummies(X, drop_first=True)
```

## 3. Train/test split — split the structured array too
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

## 4. Fit a model
```python
from sksurv.ensemble import RandomSurvivalForest        # strong default
model = RandomSurvivalForest(n_estimators=200, min_samples_leaf=15,
                             n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# alternatives:
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis     # linear, interpretable coefs
```
`model.predict(X)` → a **risk score** (higher = higher risk / earlier event).
`model.predict_survival_function(X)` → per-row step functions (`fn(t)` = P(survive past t)).

To visualize representative risk profiles, PREFER the native (it picks low/median/high-risk
subjects, plots their survival curves, saves the figure + `risk_profiles.json`, and emits an
insight — no manual time grid needed):
```python
profiles = gads_plot_survival_curves(surv_model, X_test, n_profiles=3)
```

## 5. Evaluate with the native `gads_evaluate_survival` — DO NOT hand-roll metrics
```python
metrics = gads_evaluate_survival(model, X_train, y_train, X_test, y_test)
metrics["ipcw_cindex"]              # IPCW-corrected concordance (preferred under censoring)
metrics["harrell_cindex"]          # Harrell's C-index
metrics["mean_auc"]                # mean time-dependent AUC
metrics["integrated_brier_score"]  # lower is better; 0.25 = uninformative
```
It writes `survival_metrics.json`, picks safe evaluation times inside the train/test
follow-up overlap, and emits insights. Doing this by hand routinely fails because:
- evaluation times outside the follow-up range make `concordance_index_ipcw` /
  `cumulative_dynamic_auc` / `integrated_brier_score` **raise**;
- `integrated_brier_score` needs a **2-D array of predicted survival probabilities**
  `(n_samples × n_times)` obtained by evaluating each `predict_survival_function` step
  function at each time — not the risk score.

If you must compute a metric directly, the signatures are:
```python
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                            cumulative_dynamic_auc, integrated_brier_score)
concordance_index_censored(y_test["event"], y_test["time"], risk)[0]   # -> c-index (float)
concordance_index_ipcw(y_train, y_test, risk)[0]                        # -> c-index (float)
cumulative_dynamic_auc(y_train, y_test, risk, times)                    # -> (auc_array, mean_auc)
integrated_brier_score(y_train, y_test, surv_prob_2d, times)            # -> float
```

## Hyperparameter tuning with the right scorer
Plain `GridSearchCV` scoring fails because `y` is structured. Wrap the estimator:
```python
from sksurv.metrics import as_concordance_index_ipcw_scorer
scored = as_concordance_index_ipcw_scorer(RandomSurvivalForest(random_state=42))
# then GridSearchCV(scored, param_grid={"estimator__min_samples_leaf": [10, 20]}, cv=3)
```

## Pitfalls
- Passing a 1-D `y` (just the time, or just the event) → cryptic dtype errors. Always the
  structured `(event, time)` array from `gads_make_surv_target`.
- Do not drop censored rows to "clean" the data — sksurv needs them.
- `RandomSurvivalForest.predict` returns a risk score, not a time; larger = higher risk.
- Keep the SAME encoded columns for train and test (fit the encoder on train).
