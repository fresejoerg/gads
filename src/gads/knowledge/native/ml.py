"""
AutoGluon native node wrappers — imported by __init__.py for the NATIVE_REGISTRY.

These are thin stubs that import from __init__.AUTOGLUON_PREAMBLE at runtime
(the actual implementations live there as string literals for sandbox injection).
For type-checking and IDE support, we re-export the same signatures here.
"""

from typing import Any, Dict, Optional
import pandas as pd


def gads_automl_fit(
    df: pd.DataFrame,
    target_col: str,
    time_limit: int = 120,
    presets: str = "good_quality",
    eval_metric: Optional[str] = None,
    problem_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Train AutoGluon TabularPredictor. See AUTOGLUON_PREAMBLE for full implementation."""
    raise NotImplementedError("Call this via the sandbox preamble injection, not directly.")


def gads_automl_predict(predictor: Any, df: pd.DataFrame) -> Dict[str, Any]:
    """Run inference with a fitted TabularPredictor."""
    raise NotImplementedError("Call this via the sandbox preamble injection, not directly.")


def gads_timeseries_fit(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str,
    item_id_col: Optional[str] = None,
    prediction_length: Optional[int] = None,
    time_limit: int = 120,
    presets: str = "fast_training",
) -> Dict[str, Any]:
    """Train AutoGluon TimeSeriesPredictor. See AUTOGLUON_PREAMBLE for full implementation."""
    raise NotImplementedError("Call this via the sandbox preamble injection, not directly.")


def gads_timeseries_predict(predictor_ts: Any, ts_df: Any) -> Any:
    """Generate forecasts from a fitted TimeSeriesPredictor."""
    raise NotImplementedError("Call this via the sandbox preamble injection, not directly.")


# NOTE: unlike the AutoGluon wrappers above, this is the REAL implementation, not a stub.
# AUTOGLUON_PREAMBLE splices it in via inspect.getsource, so there is exactly one copy and
# NATIVE_SOURCE["gads_calibrate_threshold"] exports working code for the fallback path
# (the stub pattern above would inject a function that raises). Annotation-free with
# imports inside so the source injects verbatim into the sandbox kernel.
def gads_calibrate_threshold(y_true, y_prob, metric="f1"):
    """
    Calibrates probabilistic scores into decisions.

    BINARY (2 classes): sweeps the decision threshold and returns the one maximizing
    `metric`. Return shape is unchanged from the original binary-only implementation —
    {"best_threshold": float, "best_score": float} — with additive keys only, so existing
    callers (binary_classification.tabular.standard, tabular_automl.autogluon.*) are
    unaffected.

    MULTICLASS (3+ classes): a single scalar threshold is meaningless under argmax, so
    this instead fits per-class multiplicative weights by coordinate ascent — the
    cost-sensitive argmax `classes[argmax(proba * weights)]`. Equal weights reproduce
    plain argmax, so the result can never be worse than the uncalibrated baseline.
    Returns "best_threshold": None and a "class_weights" vector to apply.

    Returns dict with keys: best_threshold, best_score, mode, n_classes and — multiclass
    only — class_weights, classes, baseline_score. `threshold` and `score` are provided
    as aliases of best_threshold/best_score: models reach for the short natural name.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    y_true_arr = np.asarray(y_true)
    if hasattr(y_true, "values"):
        y_true_arr = y_true.values
    y_true_arr = np.asarray(y_true_arr).ravel()

    # Normalize y_prob to a numpy array, keeping 2D shape for the multiclass decision.
    if isinstance(y_prob, pd.DataFrame):
        prob_classes = list(y_prob.columns)
        prob_arr = y_prob.values
    else:
        prob_classes = None
        prob_arr = np.asarray(y_prob)
        if prob_arr.dtype == object:
            prob_arr = np.asarray(prob_arr.tolist())

    n_true_classes = len(np.unique(y_true_arr))
    n_prob_cols = prob_arr.shape[1] if prob_arr.ndim == 2 else 1
    is_multiclass = n_prob_cols > 2 or (prob_arr.ndim == 2 and n_true_classes > 2)

    # ------------------------------------------------------------------ multiclass
    if is_multiclass:
        classes = prob_classes
        if classes is None:
            classes = sorted(np.unique(y_true_arr).tolist())
        if len(classes) != prob_arr.shape[1]:
            classes = list(range(prob_arr.shape[1]))

        classes_arr = np.asarray(classes)

        def _score_with(weights):
            idx = np.argmax(prob_arr * weights, axis=1)
            preds = classes_arr[idx]
            if metric == "accuracy":
                return float(accuracy_score(y_true_arr, preds))
            return float(f1_score(y_true_arr, preds, average="macro", zero_division=0))

        k = prob_arr.shape[1]
        weights = np.ones(k, dtype=float)
        baseline_score = _score_with(weights)
        best_score = baseline_score

        # Coordinate ascent: one class at a time, multiplicative grid. Two passes is
        # enough in practice and keeps this bounded — it runs inside a retry loop.
        grid = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]
        for _ in range(2):
            improved = False
            for c in range(k):
                current = weights[c]
                for g in grid:
                    weights[c] = current * g
                    s = _score_with(weights)
                    if s > best_score + 1e-12:
                        best_score = s
                        current = weights[c]
                        improved = True
                weights[c] = current
            if not improved:
                break

        print(f"[gads_calibrate_threshold] Multiclass ({k} classes): macro-F1 "
              f"{baseline_score:.4f} (argmax) -> {best_score:.4f} (weighted argmax)")
        print(f"[gads_calibrate_threshold] Apply with: "
              f"preds = np.asarray(classes)[np.argmax(y_prob * class_weights, axis=1)]")
        # "threshold"/"score" are ALIASES: local models reach for the short natural key
        # name and a KeyError there is fatal to the node (022 v1.1 fix 1 — `threshold`
        # killed holdout_evaluation 5 times across the local A/B).
        return {"best_threshold": None, "threshold": None,
                "best_score": best_score, "score": best_score, "mode": "multiclass",
                "n_classes": k, "class_weights": weights.tolist(),
                "classes": [c.item() if hasattr(c, "item") else c for c in classes],
                "baseline_score": baseline_score}

    # ------------------------------------------------------------------ binary
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

    # Binarize non-{0,1} targets (e.g. string labels '<=50K'/'>50K'): the positive
    # class is the lexicographically last unique value, matching the ordering of
    # AutoGluon's predict_proba columns from which callers take iloc[:, 1].
    uniq = sorted(np.unique(y_true).tolist())
    if len(uniq) == 2 and uniq != [0, 1] and uniq != [False, True]:
        y_true = (y_true == uniq[-1]).astype(int)

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
    return {"best_threshold": best_threshold, "threshold": best_threshold,
            "best_score": best_score, "score": best_score,
            "mode": "binary", "n_classes": 2}
