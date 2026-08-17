---
id: model_selection_tabular
description: "Decision rules for choosing a tabular supervised model and defending the choice: when Random Forest beats XGBoost, when linear wins, how dataset shape (n rows, p features, cardinality, missingness, imbalance, budget) determines the shortlist. Pairs with the native gads_candidate_bakeoff / gads_tune_model / gads_audit_model_choice."
triggers: ["model selection", "choose a model", "which model", "compare models", "model comparison", "best algorithm", "random forest or xgboost", "rf vs xgboost", "shortlist", "candidate models", "hyperparameter tuning", "tune the model", "search space", "optuna", "justify the model", "reasoned choice", "why this model"]
---
# Choosing a Tabular Model — and Defending the Choice

Your job on a selection node is to **nominate 2–4 candidates and say why**, not to fit
anything. The fitting, the fold protocol and the tuning are handled by native functions.
A choice with no stated reason is a failed node even if the code runs.

## The decision table

Read the dataset facts, then apply these in order. They are conditions, not verdicts —
several can fire at once, and the shortlist should reflect the tension.

| Condition | Lean toward | Why |
|---|---|---|
| `n_rows` < ~1,000 | regularized linear, Random Forest | Boosted ensembles overfit at this scale without a tuning budget you do not have. |
| `n_features` > `n_rows` (p ≫ n) | L1/L2 linear, Random Forest | Deep boosted models memorize wide data. Regularized linear handles p ≫ n far more reliably. |
| `max_cat_cardinality` > ~50 | CatBoost, or target/frequency encoding first | XGBoost and sklearn RF have no native categorical handling; naive one-hot explodes the matrix and dilutes every split. |
| `has_missing` is true | LightGBM, XGBoost, HistGradientBoosting | These consume NaN natively. sklearn's RandomForest and LogisticRegression **cannot** — they raise. |
| `minority_class_rate` < ~0.05 | any, with class weighting + threshold calibration | The model choice matters less than the weighting and the threshold. Report PR-AUC next to ROC-AUC. |
| `interpretability_required` | linear, shallow decision tree | A complex model must clear the interpretable baseline by a material margin to justify itself. |
| `n_rows` > ~10,000, mixed types, tuning budget available | LightGBM first | The usual winner on mixed tabular data at this scale. |
| CPU-only, tight budget | Random Forest | Parallelizes trivially and needs almost no tuning. **An untuned GBM frequently loses to a default RF** — this is the single most common selection mistake. |
| Noisy labels / weak signal | Random Forest | More robust out of the box; boosting chases label noise. |
| Regression needing extrapolation beyond the training range | linear model, GAM | Trees **cannot** extrapolate at all — they predict a constant outside the training range. |

### Random Forest vs XGBoost, specifically

The honest short version:

- **RF wins** on small data, wide data, noisy labels, and whenever the tuning budget is
  small — because its defaults are already near its ceiling.
- **XGBoost/LightGBM win** on larger data with real tuning, because their ceiling is
  higher *and only reachable through tuning*. Untuned, they are frequently worse than RF.
- So the deciding question is rarely "which is better" — it is **"is there a tuning budget
  here?"** With `n_trials` in the tens and thousands of rows, take the GBM. Without,
  take the forest.
- LightGBM over XGBoost when there are high-cardinality categoricals or missing values;
  XGBoost when you want the most predictable, well-documented behavior.

## Always include a baseline

Every shortlist must contain a **regularized linear model** (`logistic_regression` for
classification, `ridge` for regression), even when you expect it to lose. It is the
reference the whole report is read against: a complex model that only matches the linear
baseline has not earned its complexity, and that is a finding worth reporting.

## Writing the shortlist

Emit a list of `{"name": ..., "params": {...}}` using these exact names:

```
logistic_regression  ridge  linear_regression  elastic_net  decision_tree
random_forest  extra_trees  hist_gradient_boosting  gradient_boosting
xgboost  lightgbm  catboost  knn  svm  naive_bayes
```

Give a one-sentence justification per candidate, **and name at least one family you ruled
out and why**. The ruled-out reasoning is what makes the choice defensible rather than
arbitrary.

Do NOT set `random_state`, `class_weight`, `scale_pos_weight` or `n_jobs` in `params` —
the natives apply those uniformly so the comparison is not confounded. Use `params` only
for genuine structural choices (e.g. `{"max_depth": 5}` on a tree you want kept readable).

## Writing a search space

`gads_tune_model` takes a declarative space, so you never write the study loop:

```python
search_space = {
    "n_estimators":     {"type": "int",   "low": 100, "high": 700, "step": 100},
    "learning_rate":    {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "max_depth":        {"type": "int",   "low": 3,   "high": 10},
    "subsample":        {"type": "float", "low": 0.6, "high": 1.0},
    "max_features":     {"type": "categorical", "choices": ["sqrt", "log2", None]},
}
result = gads_tune_model(X_train, y_train, "lightgbm", search_space=search_space,
                         n_trials=40, timeout_s=240)
tuned_model = result["tuned_model"]
best_params = result["best_params"]
best_cv_score_tuned = result["best_cv_score_tuned"]
n_trials_completed = result["n_trials_completed"]
```

`tuned_model` comes back as a **sklearn Pipeline**, not a bare estimator: the natives wrap
every candidate in `Pipeline([("prep", ColumnTransformer(...)), ("est", estimator)])` so
imputation and categorical encoding are refitted on each training fold. Two consequences
worth knowing: you never need to encode the feature matrix yourself (and must not — an
encoder fitted outside the folds leaks), and `tuned_model.feature_importances_` does not
exist. Use `gads_feature_importance`, which reports per original column.

Space-design rules:
- Use `"log": True` for anything spanning orders of magnitude (`learning_rate`, `C`,
  `alpha`, `reg_lambda`). A linear scale wastes almost every trial at the large end.
- Keep it to **4–6 parameters**. A wider space with a fixed trial budget searches nothing
  well — trials are the scarce resource, not parameters.
- Never put `random_state` or `n_jobs` in the space.
- Parameter names must be the estimator's real constructor arguments:
  `HistGradientBoosting` uses `max_iter` and `max_leaf_nodes` (**not** `n_estimators` /
  `max_depth`); CatBoost uses `iterations` and `depth`.

## Reading the bakeoff honestly

`gads_candidate_bakeoff` returns a table with `mean_score` and `std_score` across identical
folds. Before declaring a winner:

- **Compare the gap to the fold standard deviation.** A 0.002 lead with a 0.02 fold std is
  noise, and picking that "winner" is not a result. When candidates are within noise,
  prefer the simpler or cheaper one and say so.
- If your pre-registered choice lost, either take the winner or state explicitly why the
  margin is worth trading away (interpretability, inference cost, robustness). Both are
  legitimate; silently keeping the loser is not.
- `gads_audit_model_choice` will flag exactly these cases (`MS007`, `MS010`), so address
  them in your narrative rather than letting the gate say it for you.

## What the gate checks

`gads_audit_model_choice(chosen, dataset_facts, bakeoff_table, tuning_result)` adjudicates
the choice against the rules above and writes `model_choice_checks.json`. Codes: `MS001`
small-data boosting, `MS002` complex-on-wide, `MS003` high-cardinality unhandled, `MS004`
missing values with an estimator that cannot take them, `MS005` severe imbalance, `MS006`
interpretability unmet, `MS007` chosen model lost the bakeoff, `MS008` tuning did not help,
`MS009` tree asked to extrapolate, `MS010` candidates indistinguishable, `MS011` search
ended on budget.

At v1.0 every finding is a **warning** — it does not fail your node. Treat the findings as
material for the report, not as errors to suppress.
