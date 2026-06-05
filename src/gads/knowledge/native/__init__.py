"""
GADS Native Node Registry

Pre-written, audited functions injected into the sandbox preamble for high-stakes
recipe steps. These replace stochastic LLM-generated code for operations where the
local model reliably fails (wrong API calls, missing subsampling, incorrect parameters).

Usage in executor.py telemetry_preamble:
    from gads.knowledge.native import NATIVE_PREAMBLE
    wrapped_code = NATIVE_PREAMBLE + "\n" + current_code

Each function prints progress lines that appear in task stdout.
"""

from typing import Callable, Dict

# Import all native modules — functions are registered at module level
from .ml import gads_automl_fit, gads_automl_predict, gads_timeseries_fit, gads_timeseries_predict, gads_calibrate_threshold

NATIVE_REGISTRY: Dict[str, Callable] = {
    "gads_automl_fit": gads_automl_fit,
    "gads_automl_predict": gads_automl_predict,
    "gads_timeseries_fit": gads_timeseries_fit,
    "gads_timeseries_predict": gads_timeseries_predict,
    "gads_calibrate_threshold": gads_calibrate_threshold,
}

# Preamble injected into every sandbox execution when AutoGluon recipes are active.
# Defines the native functions directly in the kernel so the Coder can call them.
AUTOGLUON_PREAMBLE = '''
import warnings
warnings.filterwarnings("ignore")

def gads_calibrate_threshold(y_true, y_prob, metric="f1"):
    """
    Finds the optimal decision threshold for binary classification.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    
    # Extract positive class probabilities if y_prob is a DataFrame or 2D array
    if hasattr(y_prob, "ndim") and y_prob.ndim == 2:
        if hasattr(y_prob, "iloc"):
            y_prob = y_prob.iloc[:, 1]
        else:
            y_prob = y_prob[:, 1]
    elif isinstance(y_prob, pd.DataFrame):
        y_prob = y_prob.iloc[:, 1]
    elif isinstance(y_prob, list):
        y_prob = np.array(y_prob)
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
            
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_score = -1.0
    
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        if metric == "f1":
            score = float(f1_score(y_true, preds, zero_division=0))
        elif metric == "precision":
            score = float(precision_score(y_true, preds, zero_division=0))
        elif metric == "recall":
            score = float(recall_score(y_true, preds, zero_division=0))
        elif metric == "accuracy":
            score = float(accuracy_score(y_true, preds))
        else:
            score = float(f1_score(y_true, preds, zero_division=0))
            
        if score > best_score:
            best_score = score
            best_threshold = float(t)
            
    print(f"[gads_calibrate_threshold] Best threshold: {best_threshold:.4f} with {metric} score: {best_score:.4f}")
    return {"best_threshold": best_threshold, "best_score": best_score}

def gads_automl_fit(df, target_col, time_limit=120, presets="good_quality", eval_metric=None, problem_type=None):
    """
    Train an AutoGluon TabularPredictor on df. Handles all preprocessing internally.

    Args:
        df: pandas DataFrame including the target column.
        target_col: name of the column to predict.
        time_limit: max training seconds (default 120).
        presets: 'medium_quality', 'good_quality', 'high_quality'.
        eval_metric: auto-selected if None ('roc_auc' binary, 'accuracy' multiclass, 'rmse' regression).
        problem_type: auto-detected if None ('binary', 'multiclass', 'regression').

    Returns:
        dict with keys: predictor, test_score, eval_metric, leaderboard, feature_importance
    """
    import pandas as pd
    import joblib
    from autogluon.tabular import TabularPredictor
    from sklearn.model_selection import train_test_split

    # Drop obvious ID columns
    id_patterns = ['id', 'index', 'key', 'name', 'uuid', 'rownum']
    drop_cols = [c for c in df.columns if c != target_col and
                 any(p == c.lower() or c.lower().endswith('_' + p) or c.lower().startswith(p + '_')
                     for p in id_patterns)]
    if drop_cols:
        print(f"[gads_automl_fit] Dropping ID-like columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Subsample large datasets
    if len(df) > 200_000:
        print(f"[gads_automl_fit] Subsampling from {len(df)} to 200,000 rows")
        df = df.sample(200_000, random_state=42).reset_index(drop=True)

    # Stratified split for classification, random for regression
    target = df[target_col]
    is_cls = (problem_type in ('binary', 'multiclass')) or (target.nunique() <= 20 and target.dtype == object) or (target.nunique() == 2)
    try:
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42,
                                             stratify=target if is_cls else None)
    except Exception:
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    print(f"[gads_automl_fit] Train: {len(df_train)} rows  Test: {len(df_test)} rows")

    predictor = TabularPredictor(
        label=target_col,
        eval_metric=eval_metric,
        problem_type=problem_type,
        verbosity=0
    ).fit(
        df_train,
        presets=presets,
        time_limit=time_limit,
        excluded_model_types=["NN_TORCH", "FASTAI"]
    )

    results = predictor.evaluate(df_test, silent=True)
    metric = predictor.eval_metric
    test_score = float(results[metric])
    print(f"[gads_automl_fit] Test {metric}: {test_score:.4f}")

    lb = predictor.leaderboard(df_test, silent=True)
    print("[gads_automl_fit] Leaderboard:")
    print(lb[["model", "score_test", "fit_time"]].head(8).to_string(index=False))

    fi = predictor.feature_importance(df_test, silent=True)
    print(f"[gads_automl_fit] Top features: {fi.index[:5].tolist()}")

    joblib.dump(predictor, "model.joblib")
    print("[gads_automl_fit] Saved predictor to model.joblib")

    return {
        "predictor": predictor,
        "df_train": df_train,
        "df_test": df_test,
        "test_score": test_score,
        "eval_metric": metric,
        "leaderboard": lb,
        "feature_importance": fi,
    }


def gads_automl_predict(predictor, df):
    """Run inference with a fitted TabularPredictor."""
    y_pred = predictor.predict(df)
    try:
        y_prob = predictor.predict_proba(df)
        return {"predictions": y_pred, "probabilities": y_prob}
    except Exception:
        return {"predictions": y_pred}


def gads_timeseries_fit(df, target_col, timestamp_col, item_id_col=None,
                        prediction_length=None, time_limit=120, presets="fast_training"):
    """
    Train an AutoGluon TimeSeriesPredictor.

    Args:
        df: pandas DataFrame with timestamp and target columns.
        target_col: column to forecast.
        timestamp_col: datetime column name.
        item_id_col: series grouping column; if None, treats whole df as one series.
        prediction_length: steps ahead to forecast; defaults to 10% of median series length.
        time_limit: max training seconds (default 120).
        presets: 'fast_training', 'medium_quality'.

    Returns:
        dict with keys: predictor_ts, prediction_length, best_model_mase
    """
    import pandas as pd
    import joblib
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Add constant item_id if not provided
    if item_id_col is None or item_id_col not in df.columns:
        df["item_id"] = "series_1"
        item_id_col = "item_id"

    # Keep only needed columns
    keep_cols = [item_id_col, timestamp_col, target_col]
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df[keep_cols], id_column=item_id_col, timestamp_column=timestamp_col
    )
    print(f"[gads_timeseries_fit] TimeSeriesDataFrame: {ts_df.num_items} series, {len(ts_df)} rows")

    # Derive prediction_length from data
    if prediction_length is None:
        lengths = ts_df.groupby(level=0).size()
        median_len = int(lengths.median())
        prediction_length = max(1, int(median_len * 0.1))
        print(f"[gads_timeseries_fit] Auto prediction_length={prediction_length} (10% of median series length {median_len})")

    predictor_ts = TimeSeriesPredictor(
        prediction_length=prediction_length,
        target=target_col,
        eval_metric="MASE",
        verbosity=0
    ).fit(ts_df, presets=presets, time_limit=time_limit)

    lb = predictor_ts.leaderboard(ts_df, silent=True)
    best_model_mase = float(lb["score_val"].iloc[0]) if len(lb) > 0 else float("nan")
    print(f"[gads_timeseries_fit] Best model MASE: {best_model_mase:.4f}  (<1.0 beats naive baseline)")
    print(lb[["model", "score_val", "fit_time"]].head(6).to_string(index=False))

    joblib.dump(predictor_ts, "model_timeseries.joblib")
    print("[gads_timeseries_fit] Saved to model_timeseries.joblib")

    return {
        "predictor_ts": predictor_ts,
        "ts_df": ts_df,
        "prediction_length": prediction_length,
        "best_model_mase": best_model_mase,
        "leaderboard_ts": lb,
    }


def gads_timeseries_predict(predictor_ts, ts_df):
    """Generate forecasts from a fitted TimeSeriesPredictor."""
    forecasts = predictor_ts.predict(ts_df)
    print(f"[gads_timeseries_predict] Forecasts shape: {forecasts.shape}")
    return forecasts
'''
