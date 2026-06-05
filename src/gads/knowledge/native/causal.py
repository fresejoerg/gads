"""
GADS Native Causal Nodes

Pre-written, audited functions for causal estimation injected into the sandbox
preamble when causal keywords are detected. These replace stochastic LLM-generated
DoWhy boilerplate with one deterministic call that handles:
  - Large-dataset subsampling (20K rows for DoWhy, 5K for Bambi)
  - Continuous treatment binarization at the global median
  - Programmatic GML construction (no hardcoded node IDs)
  - Estimator selection based on outcome balance
  - Mandatory refutation (both placebo and data-subset)
"""

from typing import Any, Dict, List


def gads_causal_estimate_ate(
    df,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: List[str],
    method: str = "auto",
    max_rows: int = 20000,
) -> Dict[str, Any]:
    """
    Full DoWhy 4-step causal estimation with refutation.

    Handles subsampling, continuous treatment binarization, GML graph construction,
    estimand identification, ATE estimation, and two refutation tests — all in one
    call so local models do not need to reproduce the DoWhy boilerplate.

    Returns:
        dict with keys:
          ate (float)               — average treatment effect
          placebo_new_effect (float) — near-zero if estimate is valid
          subset_new_effect (float)  — near ate if estimate is stable
          treatment_col (str)        — actual column used (may be 'high_X' if binarized)
          identified_estimand        — DoWhy estimand object
          causal_estimate            — DoWhy estimate object
          df_sample                  — the dataframe used (after subsampling)
    """
    import warnings
    warnings.filterwarnings("ignore")
    from dowhy import CausalModel

    # 1. Subsample for performance
    if len(df) > max_rows:
        df_sample = df.sample(max_rows, random_state=42).reset_index(drop=True)
        print(f"[gads_causal_estimate_ate] Subsampled {len(df):,} → {max_rows:,} rows")
    else:
        df_sample = df.copy()

    # 2. Binarize continuous treatment at global median
    actual_treatment_col = treatment_col
    if df_sample[treatment_col].nunique() > 2:
        global_median = float(df[treatment_col].median())
        bin_col = f"high_{treatment_col}"
        df_sample[bin_col] = (df_sample[treatment_col] > global_median).astype(int)
        actual_treatment_col = bin_col
        print(f"[gads_causal_estimate_ate] Binarized '{treatment_col}' at median={global_median:.4f} → '{bin_col}'")

    # 3. Build GML: nodes with integer id + string label; edges by index
    nodes = [actual_treatment_col, outcome_col] + list(confounder_cols)
    node_idx = {n: i for i, n in enumerate(nodes)}
    node_str = "\n".join(f'  node [ id {i} label "{n}" ]' for i, n in enumerate(nodes))
    edges = (
        [(c, actual_treatment_col) for c in confounder_cols]
        + [(c, outcome_col) for c in confounder_cols]
        + [(actual_treatment_col, outcome_col)]
    )
    edge_str = "\n".join(
        f'  edge [ source {node_idx[s]} target {node_idx[t]} ]' for s, t in edges
    )
    gml_string = f"graph [ directed 1\n{node_str}\n{edge_str}\n]"

    # 4. Build CausalModel
    model = CausalModel(
        data=df_sample,
        treatment=actual_treatment_col,
        outcome=outcome_col,
        graph=gml_string,
    )

    # 5. Identify estimand
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(f"[gads_causal_estimate_ate] Backdoor vars: {identified_estimand.get_backdoor_variables()}")

    # 6. Select estimator based on outcome balance
    if method == "auto":
        n_unique = df_sample[outcome_col].nunique()
        if n_unique <= 10:
            minority_frac = float(df_sample[outcome_col].value_counts(normalize=True).min())
        else:
            minority_frac = 0.5
        chosen_method = (
            "backdoor.propensity_score_matching"
            if minority_frac < 0.05
            else "backdoor.linear_regression"
        )
    else:
        chosen_method = method
    print(f"[gads_causal_estimate_ate] Method: {chosen_method}")

    # 7. Estimate ATE
    causal_estimate = model.estimate_effect(
        identified_estimand, method_name=chosen_method, target_units="ate"
    )
    ate = float(causal_estimate.value)
    print(f"[gads_causal_estimate_ate] ATE={ate:.4f}")

    # 8. Refute — reuse existing model (do NOT rebuild)
    ref_placebo = model.refute_estimate(
        identified_estimand,
        causal_estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
        random_seed=42,
    )
    placebo_new_effect = float(ref_placebo.new_effect)
    print(f"[gads_causal_estimate_ate] placebo_new_effect={placebo_new_effect:.4f} (should be ~0)")

    ref_subset = model.refute_estimate(
        identified_estimand,
        causal_estimate,
        method_name="data_subset_refuter",
        subset_fraction=0.8,
        random_seed=42,
    )
    subset_new_effect = float(ref_subset.new_effect)
    print(f"[gads_causal_estimate_ate] subset_new_effect={subset_new_effect:.4f} (should be ~{ate:.2f})")

    return {
        "ate": ate,
        "placebo_new_effect": placebo_new_effect,
        "subset_new_effect": subset_new_effect,
        "treatment_col": actual_treatment_col,
        "identified_estimand": identified_estimand,
        "causal_estimate": causal_estimate,
        "df_sample": df_sample,
    }


def gads_causal_bayesian_ate(
    df,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: List[str],
    max_rows: int = 5000,
) -> Dict[str, Any]:
    """
    Bambi MCMC causal effect estimation (Bayesian).

    Handles subsampling (stratified for rare binary outcomes), continuous treatment
    binarization, confounder standardization, and model fitting with chains=1.

    Returns:
        dict with keys:
          ate (float)        — posterior mean of treatment coefficient
          hdi_lower (float)  — 94% HDI lower bound
          hdi_upper (float)  — 94% HDI upper bound
          p_positive (float) — P(effect > 0)
          idata              — ArviZ InferenceData object
          treatment_col (str) — actual column used
    """
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import bambi as bmb
    import arviz as az
    import joblib
    from sklearn.preprocessing import StandardScaler

    df = df.copy()

    # 1. Binarize continuous treatment at global median
    actual_treatment_col = treatment_col
    if df[treatment_col].nunique() > 2:
        global_median = float(df[treatment_col].median())
        bin_col = f"high_{treatment_col}"
        df[bin_col] = (df[treatment_col] > global_median).astype(int)
        actual_treatment_col = bin_col
        print(f"[gads_causal_bayesian_ate] Binarized '{treatment_col}' at median={global_median:.4f} → '{bin_col}'")

    # 2. Subsample — stratified for rare binary outcomes
    if len(df) > max_rows:
        if df[outcome_col].nunique() <= 2:
            minority_frac = float(df[outcome_col].value_counts(normalize=True).min())
            if minority_frac < 0.10:
                minority_val = df[outcome_col].value_counts().idxmin()
                df_minority = df[df[outcome_col] == minority_val]
                n_majority = max_rows - len(df_minority)
                df_majority = df[df[outcome_col] != minority_val].sample(n_majority, random_state=42)
                df = pd.concat([df_minority, df_majority]).sample(frac=1, random_state=42).reset_index(drop=True)
            else:
                df = df.sample(max_rows, random_state=42).reset_index(drop=True)
        else:
            df = df.sample(max_rows, random_state=42).reset_index(drop=True)
        print(f"[gads_causal_bayesian_ate] Using {len(df):,} rows after subsampling")

    # 3. Standardize continuous confounders
    df_model = df.copy()
    for col in confounder_cols:
        if col in df_model.columns and df_model[col].dtype.kind in "fiu":
            scaler = StandardScaler()
            df_model[col] = scaler.fit_transform(df_model[[col]]).flatten()

    # 4. Build Bambi model
    family = "bernoulli" if df_model[outcome_col].nunique() <= 2 else "gaussian"
    formula = f"{outcome_col} ~ {actual_treatment_col} + " + " + ".join(confounder_cols)
    print(f"[gads_causal_bayesian_ate] Formula: {formula}  Family: {family}")
    model = bmb.Model(formula, df_model, family=family)

    # 5. Fit MCMC
    idata = model.fit(
        draws=500, tune=300, chains=1, cores=1,
        target_accept=0.9, random_seed=42, progressbar=False,
    )
    joblib.dump(idata, "bayesian_idata.joblib")
    print("[gads_causal_bayesian_ate] Saved idata to bayesian_idata.joblib")

    # 6. Extract posterior treatment coefficient
    coef = idata.posterior[actual_treatment_col].values.flatten()
    ate = float(coef.mean())
    hdi_vals = az.hdi(coef, hdi_prob=0.94)
    hdi_lower = float(hdi_vals[0])
    hdi_upper = float(hdi_vals[1])
    p_positive = float((coef > 0).mean())

    print(f"[gads_causal_bayesian_ate] ATE={ate:.4f}  HDI=[{hdi_lower:.4f}, {hdi_upper:.4f}]  P(>0)={p_positive:.4f}")

    return {
        "ate": ate,
        "hdi_lower": hdi_lower,
        "hdi_upper": hdi_upper,
        "p_positive": p_positive,
        "idata": idata,
        "treatment_col": actual_treatment_col,
    }
