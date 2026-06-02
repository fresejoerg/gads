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
