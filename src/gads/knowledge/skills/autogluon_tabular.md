---
id: autogluon_tabular
description: "AutoGluon TabularPredictor and TimeSeriesPredictor: single-call AutoML for classification, regression, and time-series forecasting. Handles all preprocessing internally."
triggers: ["autogluon", "automl", "TabularPredictor", "TimeSeriesPredictor", "automated machine learning", "model ensemble", "auto ml", "forecasting", "time series forecast", "predict", "classification", "regression", "leaderboard"]
---
# AutoGluon in GADS Sandbox

AutoGluon is installed as `autogluon.tabular` and `autogluon.timeseries`. It trains a weighted ensemble of LightGBM, CatBoost, XGBoost, Random Forest, Extra Trees, and neural nets in a single `.fit()` call, handling all preprocessing internally. **Use it instead of building manual sklearn pipelines.**

## ⚠️ Critical Sandbox Rules

- **Always set `time_limit`** — default is no limit, which will exhaust the 600s sandbox budget. Use `time_limit=120` for quick runs, `time_limit=240` for better quality.
- **Always set `verbosity=0`** — suppresses hundreds of log lines that pollute stdout.
- **Predictor persists across tasks** — after `.fit()`, the predictor object is in kernel memory. Downstream tasks call `.predict()` or `.leaderboard()` directly; do NOT re-fit.
- **Save with joblib, not pickle** — `joblib.dump(predictor, 'model.joblib')` (pickle is sandbox-blocked).
- **`presets='good_quality'`** is the default for sandbox — fast enough, meaningfully better than a single model.

---

## TabularPredictor — Classification & Regression

```python
from autogluon.tabular import TabularPredictor
import joblib

# Fit (handles missing values, encoding, feature engineering, ensembling internally)
predictor = TabularPredictor(
    label=target_col,           # name of the target column
    eval_metric='roc_auc',      # for binary classification; use 'accuracy', 'rmse', 'r2' etc.
    verbosity=0
).fit(
    df_train,
    presets='good_quality',     # 'medium_quality' (faster), 'high_quality' (slower, more accurate)
    time_limit=120,             # seconds; increase to 240 for better accuracy
    excluded_model_types=['NN_TORCH', 'FASTAI']  # drop neural nets in CPU-only sandbox if time is tight
)

# Evaluate
results = predictor.evaluate(df_test, silent=True)
print(f"Test {predictor.eval_metric}: {results[predictor.eval_metric]:.4f}")

# Leaderboard (optional — shows all model scores)
lb = predictor.leaderboard(df_test, silent=True)
print(lb[['model', 'score_test', 'fit_time']].head(8).to_string(index=False))

# Predict
y_pred = predictor.predict(df_test)
y_prob = predictor.predict_proba(df_test)  # for classification

# Feature importance
fi = predictor.feature_importance(df_test, silent=True)
print(fi.head(10).to_string())

# Save
joblib.dump(predictor, 'model.joblib')
```

### Preset guide

| Preset | Sandbox fit time (100K rows) | When to use |
|---|---|---|
| `'medium_quality'` | ~30s | Prototyping, quick EDA |
| `'good_quality'` | ~2 min | **Default** — good balance |
| `'high_quality'` | ~5 min | When accuracy matters, within time_limit=240 |
| `'best_quality'` | Hours | Not suitable for sandbox |

### Task type detection
AutoGluon auto-detects binary classification, multiclass, or regression from the target column. You can override:
```python
TabularPredictor(label=target_col, problem_type='binary')  # 'binary', 'multiclass', 'regression'
```

### Metrics reference
| Task | Good default metric |
|---|---|
| Binary classification | `'roc_auc'` |
| Multiclass | `'accuracy'` |
| Regression | `'rmse'` |

---

## TimeSeriesPredictor — Forecasting

```python
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# Data must be in long format: one row per (item_id, timestamp)
# Required columns: item_id (series identifier), timestamp, target value
# Convert from wide format if needed:
ts_df = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column='item_id',        # column identifying each series (use a constant if single series)
    timestamp_column='date'     # datetime column
)

predictor_ts = TimeSeriesPredictor(
    prediction_length=12,       # steps ahead to forecast
    target='value',             # column to predict
    eval_metric='MASE',         # 'MASE', 'MAPE', 'WQL'
    verbosity=0
).fit(
    ts_df,
    presets='fast_training',    # 'fast_training', 'medium_quality', 'best_quality'
    time_limit=120
)

forecasts = predictor_ts.predict(ts_df)
print(forecasts.head(20))  # columns: mean, 0.1, 0.25, 0.5, 0.75, 0.9 quantiles

# Leaderboard
lb_ts = predictor_ts.leaderboard(ts_df, silent=True)
```

### Single time series (no item_id)
```python
df['item_id'] = 'series_1'  # add constant item_id
ts_df = TimeSeriesDataFrame.from_data_frame(df, id_column='item_id', timestamp_column='date')
```

---

## ❌ Do NOT use AutoGluon for

- **Causal inference** — use `gads_causal_estimate_ate()` from the causal native node
- **NLP / embeddings** — use sentence-transformers + sklearn (already in sandbox)
- **Datasets > 500K rows without subsampling** — risk of OOM; sample first: `df.sample(100000, random_state=42)`
