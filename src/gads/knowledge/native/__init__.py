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

import inspect as _inspect
from typing import Callable, Dict

# Import all native modules — functions are registered at module level
from . import ml as _ml_mod
from .ml import gads_automl_fit, gads_automl_predict, gads_timeseries_fit, gads_timeseries_predict, gads_calibrate_threshold
from .causal import gads_causal_estimate_ate, gads_causal_bayesian_ate
from .recommendation import (gads_build_interaction_matrix, gads_temporal_loo_split,
                             gads_fit_and_recommend, gads_evaluate_topn, gads_recommend_and_evaluate,
                             gads_dense_core_sample, gads_characterize_recommendations)
from .model_audit import gads_audit_model
from .survival import (gads_make_surv_target, gads_evaluate_survival, gads_cox_ph_report,
                       gads_kaplan_meier, gads_plot_survival_curves)
from .eda import (gads_profile_dataframe, gads_assess_quality, gads_recommend_split,
                  gads_recommend_transformations, gads_write_transformation_manifest,
                  gads_apply_transformations, gads_eda_summary)
from .model_selection import (gads_load_prepared_split, gads_dataset_facts,
                              gads_default_shortlist, gads_candidate_bakeoff,
                              gads_tune_model, gads_evaluate_holdout,
                              gads_feature_importance, gads_audit_model_choice,
                              gads_model_card)
from .diagnostics import gads_plot_classification_curves
from .knowledge_graph import (gads_load_text_corpus, gads_resolve_entities,
                              gads_build_lpg, gads_audit_graph, gads_build_ontology,
                              gads_extract_entities, gads_extract_triplets)

NATIVE_REGISTRY: Dict[str, Callable] = {
    "gads_automl_fit": gads_automl_fit,
    "gads_automl_predict": gads_automl_predict,
    "gads_timeseries_fit": gads_timeseries_fit,
    "gads_timeseries_predict": gads_timeseries_predict,
    "gads_calibrate_threshold": gads_calibrate_threshold,
    "gads_causal_estimate_ate": gads_causal_estimate_ate,
    "gads_causal_bayesian_ate": gads_causal_bayesian_ate,
    "gads_build_interaction_matrix": gads_build_interaction_matrix,
    "gads_dense_core_sample": gads_dense_core_sample,
    # Fallback-only (issue #30): deliberately NOT in RECOMMENDATION_PREAMBLE.
    "gads_characterize_recommendations": gads_characterize_recommendations,
    "gads_temporal_loo_split": gads_temporal_loo_split,
    "gads_fit_and_recommend": gads_fit_and_recommend,
    "gads_evaluate_topn": gads_evaluate_topn,
    "gads_recommend_and_evaluate": gads_recommend_and_evaluate,
    "gads_audit_model": gads_audit_model,
    "gads_make_surv_target": gads_make_surv_target,
    "gads_evaluate_survival": gads_evaluate_survival,
    "gads_cox_ph_report": gads_cox_ph_report,
    "gads_kaplan_meier": gads_kaplan_meier,
    "gads_plot_survival_curves": gads_plot_survival_curves,
    # EDA (approach_docs/021). Only gads_apply_transformations is auto-injected via the
    # preamble — applying a manifest is a correctness operation whose ORDERING is a
    # leakage guard. The profiling / quality / recommendation / summary natives are
    # registered for the opt-in fallback path only, so the recipe's judgment nodes stay
    # model-generated and their capability stays measured.
    "gads_profile_dataframe": gads_profile_dataframe,
    "gads_assess_quality": gads_assess_quality,
    "gads_recommend_split": gads_recommend_split,
    "gads_recommend_transformations": gads_recommend_transformations,
    "gads_write_transformation_manifest": gads_write_transformation_manifest,
    "gads_apply_transformations": gads_apply_transformations,
    "gads_eda_summary": gads_eda_summary,
    # Model selection (approach_docs/022). Split along the 019 line: the four PROTOCOL
    # natives below are auto-injected via MODEL_SELECTION_PREAMBLE because the recipe's
    # intents tell the Coder to call them by name (identical folds, budgeted HPO, held-out
    # permutation importance, the choice gate — all single-right-answer operations). The
    # five JUDGMENT natives (load, facts, shortlist, holdout evaluation, model card) are
    # registered for the opt-in fallback path ONLY and deliberately absent from the
    # preamble, so those nodes stay model-generated and their capability stays measured.
    "gads_candidate_bakeoff": gads_candidate_bakeoff,
    "gads_tune_model": gads_tune_model,
    "gads_feature_importance": gads_feature_importance,
    "gads_audit_model_choice": gads_audit_model_choice,
    "gads_load_prepared_split": gads_load_prepared_split,
    "gads_dataset_facts": gads_dataset_facts,
    "gads_default_shortlist": gads_default_shortlist,
    "gads_evaluate_holdout": gads_evaluate_holdout,
    "gads_model_card": gads_model_card,
    # Cross-recipe: any classifier that yields probabilities. Auto-injected via
    # DIAGNOSTICS_PREAMBLE and also declarable as a node's fallback_native.
    "gads_plot_classification_curves": gads_plot_classification_curves,
    # Knowledge graph (approach_docs/030). The deterministic half only — corpus
    # chunking, entity resolution, LPG materialisation and the audit gate. The
    # extraction natives (gads_extract_entities / gads_extract_triplets) are pending
    # sandbox infra: a local-model-scoped LiteLLM key, and GLiNER/REBEL pre-cached.
    "gads_load_text_corpus": gads_load_text_corpus,
    "gads_resolve_entities": gads_resolve_entities,
    "gads_build_lpg": gads_build_lpg,
    "gads_audit_graph": gads_audit_graph,
    "gads_build_ontology": gads_build_ontology,
    # LLM-backed extraction. Reach the local model ONLY, via the sandbox-scoped LiteLLM
    # key; they raise rather than degrade silently if that env is missing.
    "gads_extract_entities": gads_extract_entities,
    "gads_extract_triplets": gads_extract_triplets,
}

# Preamble injected into every sandbox execution when AutoGluon recipes are active.
# Defines the native functions directly in the kernel so the Coder can call them.
#
# NOTE: gads_calibrate_threshold is NOT a literal here — it is spliced in from ml.py via
# inspect.getsource (see the composition below). It used to be duplicated as a literal,
# which meant the kernel got a working copy while NATIVE_SOURCE exported the raising stub
# from ml.py. One source of truth now; the AutoGluon wrappers below stay literal because
# their ml.py counterparts are deliberately stubs.
_AUTOGLUON_REST = '''
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

AUTOGLUON_PREAMBLE = (
    'import warnings\nwarnings.filterwarnings("ignore")\n\n'
    + _inspect.getsource(_ml_mod.gads_calibrate_threshold)
    + "\n"
    + _AUTOGLUON_REST
)

# Preamble injected when causal keywords are detected in task code.
# Defines gads_causal_estimate_ate and gads_causal_bayesian_ate in the kernel.
CAUSAL_PREAMBLE = '''
import warnings as _warnings_causal
_warnings_causal.filterwarnings("ignore")

def gads_causal_estimate_ate(df, treatment_col, outcome_col, confounder_cols,
                              method="auto", max_rows=20000):
    """Full DoWhy ATE estimation + placebo/subset refutation in one call.

    Returns dict: {ate, placebo_new_effect, subset_new_effect, treatment_col,
                   identified_estimand, causal_estimate, df_sample}
    """
    import warnings; warnings.filterwarnings("ignore")
    from dowhy import CausalModel

    if len(df) > max_rows:
        df_sample = df.sample(max_rows, random_state=42).reset_index(drop=True)
        print(f"[gads_causal_estimate_ate] Subsampled {len(df):,} → {max_rows:,} rows")
    else:
        df_sample = df.copy()

    actual_treatment_col = treatment_col
    if df_sample[treatment_col].nunique() > 2:
        global_median = float(df[treatment_col].median())
        bin_col = f"high_{treatment_col}"
        df_sample[bin_col] = (df_sample[treatment_col] > global_median).astype(int)
        actual_treatment_col = bin_col
        print(f"[gads_causal_estimate_ate] Binarized \'{treatment_col}\' at median={global_median:.4f} → \'{bin_col}\'")

    nodes = [actual_treatment_col, outcome_col] + list(confounder_cols)
    node_idx = {n: i for i, n in enumerate(nodes)}
    node_str = "\\n".join(f\'  node [ id {i} label "{n}" ]\' for i, n in enumerate(nodes))
    edges = ([(c, actual_treatment_col) for c in confounder_cols]
             + [(c, outcome_col) for c in confounder_cols]
             + [(actual_treatment_col, outcome_col)])
    edge_str = "\\n".join(f\'  edge [ source {node_idx[s]} target {node_idx[t]} ]\' for s, t in edges)
    gml_string = f"graph [ directed 1\\n{node_str}\\n{edge_str}\\n]"

    model = CausalModel(data=df_sample, treatment=actual_treatment_col,
                        outcome=outcome_col, graph=gml_string)
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(f"[gads_causal_estimate_ate] Backdoor vars: {identified_estimand.get_backdoor_variables()}")

    if method == "auto":
        n_unique = df_sample[outcome_col].nunique()
        minority_frac = float(df_sample[outcome_col].value_counts(normalize=True).min()) if n_unique <= 10 else 0.5
        chosen_method = "backdoor.propensity_score_matching" if minority_frac < 0.05 else "backdoor.linear_regression"
    else:
        chosen_method = method
    print(f"[gads_causal_estimate_ate] Method: {chosen_method}")

    causal_estimate = model.estimate_effect(identified_estimand, method_name=chosen_method, target_units="ate")
    ate = float(causal_estimate.value)
    print(f"[gads_causal_estimate_ate] ATE={ate:.4f}")

    ref_placebo = model.refute_estimate(identified_estimand, causal_estimate,
                                         method_name="placebo_treatment_refuter",
                                         placebo_type="permute", random_seed=42)
    placebo_new_effect = float(ref_placebo.new_effect)
    print(f"[gads_causal_estimate_ate] placebo_new_effect={placebo_new_effect:.4f}")

    ref_subset = model.refute_estimate(identified_estimand, causal_estimate,
                                        method_name="data_subset_refuter",
                                        subset_fraction=0.8, random_seed=42)
    subset_new_effect = float(ref_subset.new_effect)
    print(f"[gads_causal_estimate_ate] subset_new_effect={subset_new_effect:.4f}")

    return {"ate": ate, "placebo_new_effect": placebo_new_effect,
            "subset_new_effect": subset_new_effect, "treatment_col": actual_treatment_col,
            "identified_estimand": identified_estimand, "causal_estimate": causal_estimate,
            "df_sample": df_sample}


def gads_causal_bayesian_ate(df, treatment_col, outcome_col, confounder_cols, max_rows=5000):
    """Bambi MCMC causal effect estimation.

    Returns dict: {ate, hdi_lower, hdi_upper, p_positive, idata, treatment_col}
    Note: ate is the log-odds coefficient for binary outcomes, linear coeff for continuous.
    """
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import bambi as bmb
    import arviz as az
    import joblib
    from sklearn.preprocessing import StandardScaler

    df = df.copy()
    actual_treatment_col = treatment_col
    if df[treatment_col].nunique() > 2:
        global_median = float(df[treatment_col].median())
        bin_col = f"high_{treatment_col}"
        df[bin_col] = (df[treatment_col] > global_median).astype(int)
        actual_treatment_col = bin_col

    if len(df) > max_rows:
        if df[outcome_col].nunique() <= 2:
            minority_frac = float(df[outcome_col].value_counts(normalize=True).min())
            if minority_frac < 0.10:
                minority_val = df[outcome_col].value_counts().idxmin()
                df_min = df[df[outcome_col] == minority_val]
                df_maj = df[df[outcome_col] != minority_val].sample(max_rows - len(df_min), random_state=42)
                df = pd.concat([df_min, df_maj]).sample(frac=1, random_state=42).reset_index(drop=True)
            else:
                df = df.sample(max_rows, random_state=42).reset_index(drop=True)
        else:
            df = df.sample(max_rows, random_state=42).reset_index(drop=True)
        print(f"[gads_causal_bayesian_ate] Using {len(df):,} rows after subsampling")

    df_model = df.copy()
    for col in confounder_cols:
        if col in df_model.columns and df_model[col].dtype.kind in "fiu":
            from sklearn.preprocessing import StandardScaler as _SS
            df_model[col] = _SS().fit_transform(df_model[[col]]).flatten()

    family = "bernoulli" if df_model[outcome_col].nunique() <= 2 else "gaussian"
    formula = f"{outcome_col} ~ {actual_treatment_col} + " + " + ".join(confounder_cols)
    print(f"[gads_causal_bayesian_ate] Formula: {formula}  Family: {family}")

    model = bmb.Model(formula, df_model, family=family)
    idata = model.fit(draws=500, tune=300, chains=1, cores=1,
                      target_accept=0.9, random_seed=42, progressbar=False)
    joblib.dump(idata, "bayesian_idata.joblib")

    coef = idata.posterior[actual_treatment_col].values.flatten()
    ate = float(coef.mean())
    hdi_vals = az.hdi(coef, hdi_prob=0.94)
    hdi_lower = float(hdi_vals[0])
    hdi_upper = float(hdi_vals[1])
    p_positive = float((coef > 0).mean())
    print(f"[gads_causal_bayesian_ate] ATE={ate:.4f}  HDI=[{hdi_lower:.4f}, {hdi_upper:.4f}]  P(>0)={p_positive:.4f}")

    return {"ate": ate, "hdi_lower": hdi_lower, "hdi_upper": hdi_upper,
            "p_positive": p_positive, "idata": idata, "treatment_col": actual_treatment_col}
'''


# Preamble injected when recommendation / collaborative-filtering keywords are detected.
# Built from the recommendation module source (single source of truth — no duplicated copy)
# so the injected functions can never drift from the importable/testable definitions.
from . import recommendation as _rec_mod

RECOMMENDATION_PREAMBLE = (
    "import warnings as _w_rec\n_w_rec.filterwarnings('ignore')\n\n"
    + "\n\n".join(_inspect.getsource(_fn) for _fn in (
        _rec_mod.gads_dense_core_sample,
        _rec_mod.gads_build_interaction_matrix,
        _rec_mod.gads_temporal_loo_split,
        _rec_mod.gads_fit_and_recommend,
        _rec_mod.gads_evaluate_topn,
        _rec_mod.gads_recommend_and_evaluate,
    ))
)


# Preamble injected when model-audit keywords are detected. Built from the model_audit
# module source (single source of truth) so the injected function tracks the tested one.
from . import model_audit as _audit_mod

MODEL_AUDIT_PREAMBLE = (
    "import warnings as _w_audit\n_w_audit.filterwarnings('ignore')\n\n"
    + _inspect.getsource(_audit_mod.gads_audit_model)
)


# Preamble injected when survival-analysis keywords are detected. Built from the survival
# module source (single source of truth) so the injected functions track the tested ones.
# NOTE: only the "correctness" natives are auto-injected here — the structured-target build,
# the censoring-aware evaluation, and the Cox PH-assumption gate, i.e. operations with one
# right answer. The PLOTTING natives (gads_kaplan_meier, gads_plot_survival_curves) are
# deliberately NOT in the always-on preamble: those nodes are model-generated (capability
# stays measured) and the natives serve only as an opt-in fallback (see NATIVE_REGISTRY and
# the local-fallback design). They remain importable/registered for that fallback path.
from . import survival as _surv_mod

SURVIVAL_PREAMBLE = (
    "import warnings as _w_surv\n_w_surv.filterwarnings('ignore')\n\n"
    + "\n\n".join(_inspect.getsource(_fn) for _fn in (
        _surv_mod.gads_make_surv_target,
        _surv_mod.gads_evaluate_survival,
        _surv_mod.gads_cox_ph_report,
    ))
)


# Preamble injected when transformation-manifest keywords are detected. Deliberately only
# the APPLIER plus the vocabularies it validates against: applying a manifest has one right
# answer and its ordering (split -> fit on train -> apply) is the leakage guard, so it must
# not depend on generated code. The recommendation/profiling natives are fallback-only.
from . import eda as _eda_mod

EDA_PREAMBLE = (
    "import warnings as _w_eda\n_w_eda.filterwarnings('ignore')\n\n"
    + "\n\n".join(_inspect.getsource(_fn) for _fn in (
        _eda_mod.gads_write_transformation_manifest,
        _eda_mod.gads_apply_transformations,
    ))
)

# Preamble injected when model-selection keywords are detected. Only the PROTOCOL natives
# (approach_docs/022 §3): comparing candidates on identical folds, running HPO inside the
# training partition under a budget the RuntimeOracle provably cannot see (§7), measuring
# importance on held-out data by permutation rather than impurity, and adjudicating the
# choice. The judgment natives are fallback-only — see the NATIVE_REGISTRY note above.
from . import model_selection as _ms_mod

MODEL_SELECTION_PREAMBLE = (
    "import warnings as _w_ms\n_w_ms.filterwarnings('ignore')\n\n"
    + "\n\n".join(_inspect.getsource(_fn) for _fn in (
        _ms_mod.gads_candidate_bakeoff,
        _ms_mod.gads_tune_model,
        _ms_mod.gads_feature_importance,
        _ms_mod.gads_audit_model_choice,
    ))
)


# Diagnostic curves are recipe-agnostic — the model-selection, binary-classification,
# AutoGluon and NLP-classification recipes all reach the same point of having probabilities
# and labels — so they get their own small preamble rather than living inside any one of the
# task-specific ones.
from . import diagnostics as _diag_mod

DIAGNOSTICS_PREAMBLE = (
    "import warnings as _w_diag\n_w_diag.filterwarnings('ignore')\n\n"
    + _inspect.getsource(_diag_mod.gads_plot_classification_curves)
)


# Per-native source, for the opt-in local-fallback path (invoke ONE native on demand rather
# than injecting the always-on preamble). Includes the demoted plotting natives.
from . import knowledge_graph as _kg_mod

# All four are auto-injected together: they form one pipeline (load -> resolve -> build
# -> audit) and generated code that calls any of them almost always calls the next.
KNOWLEDGE_GRAPH_PREAMBLE = (
    "import warnings as _w_kg\n_w_kg.filterwarnings('ignore')\n\n"
    + "\n\n".join(_inspect.getsource(_fn) for _fn in (
        _kg_mod.gads_load_text_corpus,
        _kg_mod.gads_resolve_entities,
        _kg_mod.gads_build_lpg,
        _kg_mod.gads_audit_graph,
        _kg_mod.gads_build_ontology,
        _kg_mod.gads_extract_entities,
        _kg_mod.gads_extract_triplets,
    ))
)


NATIVE_SOURCE = {name: _inspect.getsource(fn) for name, fn in NATIVE_REGISTRY.items()}


# Keyword → preamble routing table. Single source of truth for "which native definitions
# does this code need in the kernel", used by the executor before running generated code AND
# by kernel rehydration when replaying a prior run's code into a fresh session (the replayed
# code calls the same natives, so it needs the same definitions).
_PREAMBLE_ROUTES = (
    ("AutoGluon", ("autogluon", "TabularPredictor", "TimeSeriesPredictor",
                   "gads_automl_fit", "gads_timeseries_fit", "gads_calibrate_threshold"),
     lambda: AUTOGLUON_PREAMBLE),
    ("causal", ("CausalModel", "dowhy", "causal_estimate", "gads_causal_estimate_ate",
                "gads_causal_bayesian_ate", "bambi", "bmb.Model"),
     lambda: CAUSAL_PREAMBLE),
    ("recommendation", ("gads_recommend_and_evaluate", "gads_build_interaction_matrix",
                        "gads_fit_and_recommend", "gads_evaluate_topn",
                        "gads_temporal_loo_split", "AlternatingLeastSquares", "implicit.als"),
     lambda: RECOMMENDATION_PREAMBLE),
    ("model-audit", ("gads_audit_model", "EstimatorReport", "skore"),
     lambda: MODEL_AUDIT_PREAMBLE),
    ("eda", ("gads_apply_transformations", "gads_write_transformation_manifest",
             "eda_transformations.meta.json", "transformation_provenance"),
     lambda: EDA_PREAMBLE),
    ("model-selection", ("gads_candidate_bakeoff", "gads_tune_model",
                         "gads_audit_model_choice", "gads_feature_importance",
                         "model_choice_checks"),
     lambda: MODEL_SELECTION_PREAMBLE),
    ("diagnostics", ("gads_plot_classification_curves", "roc_curve", "precision_recall_curve",
                     "figure_roc_curve", "figure_precision_recall"),
     lambda: DIAGNOSTICS_PREAMBLE),
    ("survival", ("gads_make_surv_target", "gads_evaluate_survival", "gads_cox_ph_report",
                  "CoxPHFitter", "CoxPHSurvivalAnalysis", "RandomSurvivalForest",
                  "lifelines", "sksurv", "Surv.from_", "KaplanMeierFitter"),
     lambda: SURVIVAL_PREAMBLE),
    ("knowledge-graph", ("gads_load_text_corpus", "gads_resolve_entities",
                         "gads_build_lpg", "gads_audit_graph", "MultiDiGraph",
                         "knowledge_graph_nodes", "graph_checks",
                         "gads_build_ontology", "gads_extract_entities",
                         "gads_extract_triplets"),
     lambda: KNOWLEDGE_GRAPH_PREAMBLE),
)


def preamble_for_code(code: str):
    """Return (preamble_text, [names]) for the native groups this code references.

    Best-effort per group: a preamble that fails to build is skipped rather than breaking
    execution. Order is stable (definition order above) so injected source is deterministic.
    """
    parts, names = [], []
    for name, keywords, get_preamble in _PREAMBLE_ROUTES:
        if any(kw in code for kw in keywords):
            try:
                parts.append(get_preamble() + "\n")
                names.append(name)
            except Exception as e:  # pragma: no cover - defensive
                print(f"    [Native] Warning: could not load {name} preamble: {e}", flush=True)
    return "".join(parts), names
