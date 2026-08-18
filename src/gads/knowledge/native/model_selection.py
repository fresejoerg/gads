"""
GADS Native Model-Selection Nodes (approach_docs/022)

The protocol half of the "reasoned model choice" recipe. The split follows 019: the
*judgment* (which candidates to shortlist, how wide a search space to give them, how to
narrate the result) stays model-generated so its capability stays measured; the
*protocol* (identical folds, tuning confined to the training partition, permutation
importance on held-out data, a budget that is actually enforced) is nativized because it
has one right answer and hand-written code reproduces it incorrectly more often than not.

The centrepiece is `gads_audit_model_choice`, which adjudicates a model choice against the
decision rules WITHOUT making it — the same shape as `gads_audit_model` (skore) does for a
fitted estimator. That is what lets new constraints ("prefer RF under 1000 rows") be
encoded without taking the decision away from the model.

IMPORTANT — every function here is declared as a `fallback_native` on some recipe node, and
the executor's fallback path injects exactly ONE function's source
(`executor.py:_run_native_fallback`). So each function must be **fully self-contained**:
no calls to sibling functions in this module, no module-level helpers, imports inside.
Nested closures are fine (they travel with the source). This is why label encoding and
estimator construction are repeated inside several functions rather than factored out.

All functions are annotation-free so `inspect.getsource` output execs cleanly in the kernel.
"""


def gads_load_prepared_split(source=None, target_column=None, test_size=0.2, seed=42,
                             sample_rows=None):
    """Load the modelling table and return a train/test split.

    Prefers a chained upstream transform run: `upstream/transformed_train.parquet` +
    `transformed_test.parquet` (written by gads_apply_transformations). Falls back to
    `upstream/transformed_<x>.parquet`, then to any CSV/Parquet in the workspace, splitting
    it stratified. Never re-fits a transformation — if the upstream partitions exist they
    are used as-is, because refitting on the combined data would undo the leakage guard the
    upstream run established.

    Returns dict: {X_train, y_train, X_test, y_test, n_train_rows, n_features,
                   target_column, source_kind, class_names}
    """
    import os
    import glob
    import numpy as np
    import pandas as pd

    def _read(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    train_df = None
    test_df = None
    full_df = None
    source_kind = None

    up_train = "upstream/transformed_train.parquet"
    up_test = "upstream/transformed_test.parquet"
    if os.path.exists(up_train) and os.path.exists(up_test):
        train_df, test_df = _read(up_train), _read(up_test)
        source_kind = "upstream_split"
        print(f"[gads_load_prepared_split] Using upstream partitions: "
              f"{len(train_df)} train / {len(test_df)} test rows")
    else:
        cands = []
        if source:
            cands = [source, os.path.join("upstream", source)]
        else:
            cands = (sorted(glob.glob("upstream/transformed_*.parquet"))
                     + sorted(glob.glob("*.parquet"))
                     + sorted(glob.glob("upstream/*.parquet"))
                     + sorted(glob.glob("*.csv")))
            cands = [c for c in cands if "provenance" not in c and "meta" not in c]
        found = [c for c in cands if c and os.path.exists(c)]
        if not found:
            raise FileNotFoundError(
                "No dataset found. Looked for upstream/transformed_{train,test}.parquet "
                f"then {cands[:4]}")
        full_df = _read(found[0])
        source_kind = f"single_file:{found[0]}"
        print(f"[gads_load_prepared_split] Loaded {found[0]}: {full_df.shape}")

    # Resolve the target column.
    probe = train_df if train_df is not None else full_df
    if target_column is None or target_column not in probe.columns:
        for guess in ("target", "label", "class", "y", "outcome"):
            if guess in probe.columns:
                target_column = guess
                break
        else:
            target_column = probe.columns[-1]
        print(f"[gads_load_prepared_split] Target column inferred: '{target_column}'")

    if full_df is not None:
        from sklearn.model_selection import train_test_split
        if sample_rows and len(full_df) > sample_rows:
            full_df = full_df.sample(int(sample_rows), random_state=seed).reset_index(drop=True)
            print(f"[gads_load_prepared_split] Capped to {len(full_df)} rows (sample_rows)")
        y_all = full_df[target_column]
        X_all = full_df.drop(columns=[target_column])
        strat = y_all if y_all.nunique() <= max(20, int(len(y_all) * 0.05)) else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=test_size, random_state=seed, stratify=strat)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=test_size, random_state=seed)
    else:
        y_train = train_df[target_column]
        X_train = train_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        if sample_rows and len(X_train) > sample_rows:
            idx = X_train.sample(int(sample_rows), random_state=seed).index
            X_train, y_train = X_train.loc[idx], y_train.loc[idx]
            print(f"[gads_load_prepared_split] Capped TRAIN to {len(X_train)} rows "
                  "(test partition left whole)")
        # Align columns defensively — a mismatch here is silent death downstream.
        common = [c for c in X_train.columns if c in X_test.columns]
        if len(common) != X_train.shape[1] or len(common) != X_test.shape[1]:
            print(f"[gads_load_prepared_split] WARNING: column mismatch, using "
                  f"{len(common)} shared columns")
            X_train, X_test = X_train[common], X_test[common]

    class_names = sorted(pd.Series(y_train).unique().tolist())
    print(f"[gads_load_prepared_split] X_train={X_train.shape} X_test={X_test.shape} "
          f"target='{target_column}' classes={len(class_names)}")
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
            "n_train_rows": int(len(X_train)), "n_features": int(X_train.shape[1]),
            "target_column": target_column, "source_kind": source_kind,
            "class_names": class_names}


def gads_dataset_facts(X_train, y_train, task_kind=None, interpretability_required=False):
    """Derive the modelling-relevant facts that drive model choice.

    Deliberately NOT a full profile (that is the EDA recipe's job) — only the handful of
    quantities the selection rules in `model_selection_tabular` actually branch on, so the
    audit gate and the shortlist speak the same vocabulary.

    Returns dict: {n_rows, n_features, rows_per_feature, n_numeric, n_categorical,
                   max_cat_cardinality, missing_rate, has_missing, task_kind, n_classes,
                   minority_class_rate, class_balance, interpretability_required}
    """
    import numpy as np
    import pandas as pd

    X = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    y = pd.Series(y_train).reset_index(drop=True)

    n_rows, n_features = int(X.shape[0]), int(X.shape[1])
    num_cols = list(X.select_dtypes(include="number").columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    max_card = int(max([X[c].nunique() for c in cat_cols], default=0))
    total_cells = max(1, n_rows * n_features)
    missing_rate = float(X.isna().sum().sum()) / total_cells

    if task_kind is None:
        if y.dtype.kind in "OUSb" or y.nunique() <= max(2, min(20, int(n_rows * 0.05))):
            task_kind = "classification"
        else:
            task_kind = "regression"

    facts = {"n_rows": n_rows, "n_features": n_features,
             "rows_per_feature": float(n_rows) / max(1, n_features),
             "n_numeric": len(num_cols), "n_categorical": len(cat_cols),
             "max_cat_cardinality": max_card,
             "missing_rate": missing_rate, "has_missing": bool(missing_rate > 0),
             "task_kind": task_kind, "n_classes": None, "minority_class_rate": None,
             "class_balance": None,
             "interpretability_required": bool(interpretability_required)}

    if task_kind == "classification":
        vc = y.value_counts(normalize=True)
        facts["n_classes"] = int(y.nunique())
        facts["minority_class_rate"] = float(vc.min())
        facts["class_balance"] = {str(k): float(v) for k, v in vc.items()}

    print(f"[gads_dataset_facts] {task_kind}: {n_rows} rows x {n_features} features "
          f"({len(num_cols)} num / {len(cat_cols)} cat, max cardinality {max_card})")
    print(f"[gads_dataset_facts] rows/feature={facts['rows_per_feature']:.1f}  "
          f"missing={missing_rate:.4f}  classes={facts['n_classes']}  "
          f"minority={facts['minority_class_rate']}")
    return facts


def gads_default_shortlist(dataset_facts):
    """Conservative rule-based candidate shortlist — the FALLBACK for the reasoning node.

    This exists so a weak model never blocks the run, not to make the choice: node 3 is
    model-generated by design (019 — the reasoning is what is being measured) and this
    only fires after the model has exhausted its retries. The rules are the same ones the
    `model_selection_tabular` skill states in prose; keeping them conservative means the
    fallback is defensible rather than optimal.

    Returns dict: {candidates, selection_rationale, n_candidates, source}
    """
    f = dataset_facts or {}
    n_rows = int(f.get("n_rows") or 0)
    n_features = int(f.get("n_features") or 1)
    task = f.get("task_kind") or "classification"
    has_missing = bool(f.get("has_missing"))
    max_card = int(f.get("max_cat_cardinality") or 0)
    minority = f.get("minority_class_rate")
    interp = bool(f.get("interpretability_required"))
    is_clf = task == "classification"

    cands = []
    why = []

    linear = "logistic_regression" if is_clf else "ridge"
    cands.append({"name": linear, "params": {}})
    why.append(f"{linear}: mandatory interpretable baseline — every other candidate must "
               "beat it to justify its complexity.")

    if interp:
        cands.append({"name": "decision_tree", "params": {"max_depth": 5}})
        why.append("decision_tree (depth 5): interpretability was requested, so the "
                   "shortlist stays readable.")

    if n_rows < 1000:
        cands.append({"name": "random_forest", "params": {}})
        why.append(f"random_forest: only {n_rows} rows — boosted ensembles overfit at this "
                   "scale without a tuning budget, RF is the robust default.")
    elif n_features > n_rows:
        cands.append({"name": "random_forest", "params": {}})
        why.append(f"random_forest: wide data ({n_features} features vs {n_rows} rows) — "
                   "avoid deep boosted models.")
    else:
        if has_missing or max_card > 50:
            cands.append({"name": "hist_gradient_boosting", "params": {}})
            why.append("hist_gradient_boosting: handles NaN natively and copes with "
                       "high-cardinality categoricals without one-hot explosion.")
        else:
            cands.append({"name": "random_forest", "params": {}})
            why.append("random_forest: strong untuned baseline that parallelises well.")
        cands.append({"name": "lightgbm", "params": {}})
        why.append(f"lightgbm: {n_rows} rows with a tuning budget available — the usual "
                   "winner on mixed-type tabular data at this scale.")

    if is_clf and minority is not None and minority < 0.05:
        why.append(f"Minority class is {minority:.1%}: class weighting is applied to every "
                   "candidate and the decision threshold is calibrated afterwards.")

    seen, uniq = set(), []
    for c in cands:
        if c["name"] not in seen:
            seen.add(c["name"])
            uniq.append(c)

    rationale = "\n".join("- " + w for w in why)
    print(f"[gads_default_shortlist] {len(uniq)} candidates: "
          f"{[c['name'] for c in uniq]}")
    print(rationale)
    return {"candidates": uniq, "selection_rationale": rationale,
            "n_candidates": len(uniq), "source": "native_default_rules"}


def gads_candidate_bakeoff(X_train, y_train, candidates, cv=5, seed=42, scoring=None,
                           task_kind=None, max_seconds=300):
    """Compare candidate estimators under ONE protocol: identical folds, one seed, one metric.

    This is the invariant this recipe most depends on. Comparing models on differently-shuffled
    folds, or on different metrics, produces a ranking that reflects the split rather than the
    model — and it is the single most common way hand-written comparison code goes wrong.

    `candidates` is a list of {"name": str, "params": dict} (or bare strings). Labels are
    label-encoded internally (XGBoost requires 0..k-1 and roc_auc scorers need it) without
    ever mutating the caller's y. A candidate that fails is recorded and skipped rather than
    killing the node.

    Returns dict: {bakeoff_table, best_candidate, best_cv_score, scoring, n_candidates,
                   failures, folds}
    """
    import time
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder

    t0 = time.time()
    X = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    y_raw = pd.Series(y_train).reset_index(drop=True)

    if task_kind is None:
        task_kind = ("classification"
                     if (y_raw.dtype.kind in "OUSb" or y_raw.nunique() <= 20)
                     else "regression")
    is_clf = task_kind == "classification"

    if is_clf:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        n_classes = len(le.classes_)
    else:
        y = y_raw.values.astype(float)
        n_classes = 0

    if scoring is None:
        if not is_clf:
            scoring = "neg_root_mean_squared_error"
        elif n_classes == 2:
            scoring = "roc_auc"
        else:
            scoring = "f1_macro"

    folds = (StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed) if is_clf
             else KFold(n_splits=cv, shuffle=True, random_state=seed))

    # Class weighting is an invariant of this recipe, applied uniformly so the comparison
    # is not confounded by some candidates handling imbalance and others not.
    pos_weight = 1.0
    if is_clf and n_classes == 2:
        n_pos = float((y == 1).sum())
        n_neg = float((y == 0).sum())
        pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    def _mk(name, params):
        p = dict(params or {})
        nm = str(name).lower().replace("-", "_").replace(" ", "_")
        if nm in ("logistic_regression", "logisticregression", "logreg"):
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=2000, class_weight="balanced",
                                      random_state=seed, **p)
        if nm in ("ridge", "ridge_classifier"):
            if is_clf:
                from sklearn.linear_model import RidgeClassifier
                return RidgeClassifier(class_weight="balanced", random_state=seed, **p)
            from sklearn.linear_model import Ridge
            return Ridge(random_state=seed, **p)
        if nm in ("linear_regression", "ols"):
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**p)
        if nm in ("elastic_net", "elasticnet"):
            from sklearn.linear_model import ElasticNet
            return ElasticNet(random_state=seed, **p)
        if nm in ("decision_tree", "decisiontree", "tree"):
            if is_clf:
                from sklearn.tree import DecisionTreeClassifier
                return DecisionTreeClassifier(class_weight="balanced",
                                              random_state=seed, **p)
            from sklearn.tree import DecisionTreeRegressor
            return DecisionTreeRegressor(random_state=seed, **p)
        if nm in ("random_forest", "randomforest", "rf"):
            if is_clf:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_jobs=-1, class_weight="balanced",
                                              random_state=seed, **p)
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_jobs=-1, random_state=seed, **p)
        if nm in ("extra_trees", "extratrees"):
            if is_clf:
                from sklearn.ensemble import ExtraTreesClassifier
                return ExtraTreesClassifier(n_jobs=-1, class_weight="balanced",
                                            random_state=seed, **p)
            from sklearn.ensemble import ExtraTreesRegressor
            return ExtraTreesRegressor(n_jobs=-1, random_state=seed, **p)
        if nm in ("hist_gradient_boosting", "histgradientboosting", "hgb", "hist_gbm"):
            if is_clf:
                from sklearn.ensemble import HistGradientBoostingClassifier
                return HistGradientBoostingClassifier(random_state=seed, **p)
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(random_state=seed, **p)
        if nm in ("gradient_boosting", "gbm"):
            if is_clf:
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(random_state=seed, **p)
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(random_state=seed, **p)
        if nm in ("xgboost", "xgb", "xgbclassifier"):
            import xgboost as xgb
            if is_clf:
                kw = dict(random_state=seed, eval_metric="logloss", n_jobs=-1,
                          tree_method="hist")
                if n_classes == 2:
                    kw["scale_pos_weight"] = pos_weight
                kw.update(p)
                return xgb.XGBClassifier(**kw)
            return xgb.XGBRegressor(random_state=seed, n_jobs=-1, tree_method="hist", **p)
        if nm in ("lightgbm", "lgbm", "lgb"):
            import lightgbm as lgb
            if is_clf:
                return lgb.LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1,
                                          class_weight="balanced", **p)
            return lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **p)
        if nm in ("catboost", "cat"):
            if is_clf:
                from catboost import CatBoostClassifier
                return CatBoostClassifier(random_seed=seed, verbose=0,
                                          allow_writing_files=False,
                                          auto_class_weights="Balanced", **p)
            from catboost import CatBoostRegressor
            return CatBoostRegressor(random_seed=seed, verbose=0,
                                     allow_writing_files=False, **p)
        if nm in ("knn", "kneighbors"):
            if is_clf:
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier(n_jobs=-1, **p)
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor(n_jobs=-1, **p)
        if nm in ("svm", "svc", "svr"):
            if is_clf:
                from sklearn.svm import SVC
                return SVC(probability=True, class_weight="balanced",
                           random_state=seed, **p)
            from sklearn.svm import SVR
            return SVR(**p)
        if nm in ("naive_bayes", "gaussian_nb", "nb"):
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB(**p)
        raise ValueError(f"Unknown candidate '{name}'. Supported: logistic_regression, "
                         "ridge, linear_regression, elastic_net, decision_tree, "
                         "random_forest, extra_trees, hist_gradient_boosting, "
                         "gradient_boosting, xgboost, lightgbm, catboost, knn, svm, "
                         "naive_bayes")

    # Preprocessing lives INSIDE the pipeline so it is refitted on each training fold.
    # Encoding once outside the folds leaks fold statistics into the CV score — the exact
    # class of leakage this recipe's invariants exist to prevent. It also means permutation
    # importance later reports per ORIGINAL column rather than per one-hot dummy.
    def _is_tree(name):
        t = str(name).lower().replace("-", "_").replace(" ", "_")
        return t in ("decision_tree", "decisiontree", "tree", "random_forest",
                     "randomforest", "rf", "extra_trees", "extratrees",
                     "hist_gradient_boosting", "histgradientboosting", "hgb", "hist_gbm",
                     "gradient_boosting", "gbm", "xgboost", "xgb", "xgbclassifier",
                     "lightgbm", "lgbm", "lgb", "catboost", "cat")

    def _prep(tree_based):
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline as _Pipe
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
        num = list(X.select_dtypes(include="number").columns)
        cat = [c for c in X.columns if c not in num]
        parts = []
        if num:
            if tree_based:
                parts.append(("num", SimpleImputer(strategy="median"), num))
            else:
                parts.append(("num", _Pipe([("i", SimpleImputer(strategy="median")),
                                            ("s", StandardScaler())]), num))
        if cat:
            if tree_based:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                                     unknown_value=-1, encoded_missing_value=-2)
            else:
                enc = OneHotEncoder(handle_unknown="infrequent_if_exist",
                                    min_frequency=0.01, sparse_output=False)
            parts.append(("cat", _Pipe([("i", SimpleImputer(strategy="most_frequent")),
                                        ("e", enc)]), cat))
        if not parts:
            return None
        return ColumnTransformer(parts, remainder="drop")

    def _wrap(est, name):
        prep = _prep(_is_tree(name))
        if prep is None:
            return est
        from sklearn.pipeline import Pipeline as _Pipe
        return _Pipe([("prep", prep), ("est", est)])

    norm = []
    for c in (candidates or []):
        if isinstance(c, str):
            norm.append({"name": c, "params": {}})
        elif isinstance(c, dict):
            norm.append({"name": c.get("name") or c.get("estimator") or c.get("model"),
                         "params": c.get("params") or c.get("hyperparameters") or {}})
    if not norm:
        raise ValueError("No candidates supplied to gads_candidate_bakeoff")

    print(f"[gads_candidate_bakeoff] {len(norm)} candidates, {cv}-fold "
          f"{'stratified ' if is_clf else ''}CV, seed={seed}, scoring={scoring}")

    rows, failures = [], []
    for c in norm:
        if time.time() - t0 > max_seconds:
            failures.append({"name": c["name"], "error": "budget exhausted before fit"})
            print(f"[gads_candidate_bakeoff] Budget exhausted — skipping {c['name']}")
            continue
        t1 = time.time()
        try:
            est = _wrap(_mk(c["name"], c["params"]), c["name"])
            scores = cross_val_score(est, X, y, cv=folds, scoring=scoring, n_jobs=1,
                                     error_score="raise")
            rows.append({"candidate": c["name"], "mean_score": float(np.mean(scores)),
                         "std_score": float(np.std(scores)),
                         "min_score": float(np.min(scores)),
                         "max_score": float(np.max(scores)),
                         "fit_seconds": round(time.time() - t1, 2),
                         "params": c["params"]})
            print(f"[gads_candidate_bakeoff]   {c['name']:<26s} "
                  f"{np.mean(scores):.4f} +/- {np.std(scores):.4f}  "
                  f"({time.time() - t1:.1f}s)")
        except Exception as e:
            failures.append({"name": c["name"], "error": f"{type(e).__name__}: {e}"})
            print(f"[gads_candidate_bakeoff]   {c['name']:<26s} FAILED "
                  f"({type(e).__name__}: {str(e)[:100]})")

    if not rows:
        raise RuntimeError(f"All candidates failed in the bakeoff: {failures}")

    table = pd.DataFrame(rows).sort_values("mean_score", ascending=False).reset_index(drop=True)
    best = str(table.loc[0, "candidate"])
    best_score = float(table.loc[0, "mean_score"])
    print(f"[gads_candidate_bakeoff] Winner: {best} ({scoring}={best_score:.4f})")
    if len(table) > 1:
        margin = best_score - float(table.loc[1, "mean_score"])
        print(f"[gads_candidate_bakeoff] Margin over runner-up "
              f"({table.loc[1, 'candidate']}): {margin:.4f} "
              f"(fold std {table.loc[0, 'std_score']:.4f})")

    # Short-name aliases alongside the canonical keys (022 v1.1 fix 1).
    return {"bakeoff_table": table, "table": table,
            "best_candidate": best, "best": best,
            "best_cv_score": best_score, "score": best_score,
            "scoring": scoring, "n_candidates": len(rows), "failures": failures,
            "folds": int(cv)}


def gads_tune_model(X_train, y_train, estimator_name, search_space=None, n_trials=40,
                    timeout_s=240, cv=5, seed=42, scoring=None, task_kind=None):
    """Optuna hyperparameter search, confined to the training partition, under a HARD
    wall-clock budget.

    The budget lives here rather than in the prompt because nothing upstream can enforce it:
    RuntimeOracle scans the generated code before native preambles are injected
    (executor.py:767 vs :806), so a study inside this function is invisible to it — and even
    a model-written study would be under-estimated, since the oracle only multiplies by
    `n_estimators`/`cv` when they are literal constants and has no `n_trials` term at all
    (approach_docs/022 §7). `timeout_s` must stay well under the executor's 360s native
    fallback timeout.

    `search_space` is a declarative dict — the model's judgment, expressed without writing
    the study loop:
        {"n_estimators": {"type": "int", "low": 100, "high": 800},
         "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": true},
         "criterion": {"type": "categorical", "choices": ["gini", "entropy"]}}
    When None, a conservative built-in space for the estimator family is used — required,
    because this function is node 5's fallback and fires exactly when the model failed to
    produce a space.

    Returns dict: {tuned_model, best_params, best_cv_score_tuned, n_trials_completed,
                   timed_out, baseline_cv_score, search_space_used, scoring, study}
    """
    import time
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder

    t0 = time.time()
    X = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    y_raw = pd.Series(y_train).reset_index(drop=True)

    if task_kind is None:
        task_kind = ("classification"
                     if (y_raw.dtype.kind in "OUSb" or y_raw.nunique() <= 20)
                     else "regression")
    is_clf = task_kind == "classification"

    if is_clf:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        n_classes = len(le.classes_)
    else:
        y = y_raw.values.astype(float)
        n_classes = 0

    if scoring is None:
        if not is_clf:
            scoring = "neg_root_mean_squared_error"
        elif n_classes == 2:
            scoring = "roc_auc"
        else:
            scoring = "f1_macro"

    folds = (StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed) if is_clf
             else KFold(n_splits=cv, shuffle=True, random_state=seed))

    pos_weight = 1.0
    if is_clf and n_classes == 2:
        n_pos = float((y == 1).sum())
        n_neg = float((y == 0).sum())
        pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    nm = str(estimator_name).lower().replace("-", "_").replace(" ", "_")

    def _mk(params):
        p = dict(params or {})
        if nm in ("logistic_regression", "logisticregression", "logreg"):
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=2000, class_weight="balanced",
                                      random_state=seed, **p)
        if nm in ("ridge", "ridge_classifier"):
            if is_clf:
                from sklearn.linear_model import RidgeClassifier
                return RidgeClassifier(class_weight="balanced", random_state=seed, **p)
            from sklearn.linear_model import Ridge
            return Ridge(random_state=seed, **p)
        if nm in ("linear_regression", "ols"):
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**p)
        if nm in ("elastic_net", "elasticnet"):
            from sklearn.linear_model import ElasticNet
            return ElasticNet(random_state=seed, **p)
        if nm in ("decision_tree", "decisiontree", "tree"):
            if is_clf:
                from sklearn.tree import DecisionTreeClassifier
                return DecisionTreeClassifier(class_weight="balanced", random_state=seed, **p)
            from sklearn.tree import DecisionTreeRegressor
            return DecisionTreeRegressor(random_state=seed, **p)
        if nm in ("random_forest", "randomforest", "rf"):
            if is_clf:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_jobs=-1, class_weight="balanced",
                                              random_state=seed, **p)
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_jobs=-1, random_state=seed, **p)
        if nm in ("extra_trees", "extratrees"):
            if is_clf:
                from sklearn.ensemble import ExtraTreesClassifier
                return ExtraTreesClassifier(n_jobs=-1, class_weight="balanced",
                                            random_state=seed, **p)
            from sklearn.ensemble import ExtraTreesRegressor
            return ExtraTreesRegressor(n_jobs=-1, random_state=seed, **p)
        if nm in ("hist_gradient_boosting", "histgradientboosting", "hgb", "hist_gbm"):
            if is_clf:
                from sklearn.ensemble import HistGradientBoostingClassifier
                return HistGradientBoostingClassifier(random_state=seed, **p)
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(random_state=seed, **p)
        if nm in ("gradient_boosting", "gbm"):
            if is_clf:
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(random_state=seed, **p)
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(random_state=seed, **p)
        if nm in ("xgboost", "xgb", "xgbclassifier"):
            import xgboost as xgb
            if is_clf:
                kw = dict(random_state=seed, eval_metric="logloss", n_jobs=-1,
                          tree_method="hist")
                if n_classes == 2:
                    kw["scale_pos_weight"] = pos_weight
                kw.update(p)
                return xgb.XGBClassifier(**kw)
            return xgb.XGBRegressor(random_state=seed, n_jobs=-1, tree_method="hist", **p)
        if nm in ("lightgbm", "lgbm", "lgb"):
            import lightgbm as lgb
            if is_clf:
                return lgb.LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1,
                                          class_weight="balanced", **p)
            return lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **p)
        if nm in ("catboost", "cat"):
            if is_clf:
                from catboost import CatBoostClassifier
                return CatBoostClassifier(random_seed=seed, verbose=0,
                                          allow_writing_files=False,
                                          auto_class_weights="Balanced", **p)
            from catboost import CatBoostRegressor
            return CatBoostRegressor(random_seed=seed, verbose=0,
                                     allow_writing_files=False, **p)
        if nm in ("knn", "kneighbors"):
            if is_clf:
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier(n_jobs=-1, **p)
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor(n_jobs=-1, **p)
        if nm in ("svm", "svc", "svr"):
            if is_clf:
                from sklearn.svm import SVC
                return SVC(probability=True, class_weight="balanced", random_state=seed, **p)
            from sklearn.svm import SVR
            return SVR(**p)
        if nm in ("naive_bayes", "gaussian_nb", "nb"):
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB(**p)
        raise ValueError(f"Unknown estimator '{estimator_name}' for tuning")

    # Preprocessing inside the pipeline, refitted per fold — same rationale as the bakeoff:
    # encoding fitted once outside the folds leaks fold statistics into every trial's score,
    # which would make the whole search optimize against a contaminated objective.
    _tree_based = nm in ("decision_tree", "decisiontree", "tree", "random_forest",
                         "randomforest", "rf", "extra_trees", "extratrees",
                         "hist_gradient_boosting", "histgradientboosting", "hgb",
                         "hist_gbm", "gradient_boosting", "gbm", "xgboost", "xgb",
                         "xgbclassifier", "lightgbm", "lgbm", "lgb", "catboost", "cat")

    def _wrap(est):
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline as _Pipe
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
        num = list(X.select_dtypes(include="number").columns)
        cat = [c for c in X.columns if c not in num]
        parts = []
        if num:
            if _tree_based:
                parts.append(("num", SimpleImputer(strategy="median"), num))
            else:
                parts.append(("num", _Pipe([("i", SimpleImputer(strategy="median")),
                                            ("s", StandardScaler())]), num))
        if cat:
            if _tree_based:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                                     unknown_value=-1, encoded_missing_value=-2)
            else:
                enc = OneHotEncoder(handle_unknown="infrequent_if_exist",
                                    min_frequency=0.01, sparse_output=False)
            parts.append(("cat", _Pipe([("i", SimpleImputer(strategy="most_frequent")),
                                        ("e", enc)]), cat))
        if not parts:
            return est
        return _Pipe([("prep", ColumnTransformer(parts, remainder="drop")),
                      ("est", est)])

    # Built-in conservative spaces — REQUIRED for the fallback path, which fires precisely
    # when the model failed to define a space of its own.
    if not search_space:
        if nm in ("random_forest", "randomforest", "rf", "extra_trees", "extratrees"):
            search_space = {
                "n_estimators": {"type": "int", "low": 200, "high": 800, "step": 100},
                "max_depth": {"type": "int", "low": 4, "high": 24},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]}}
        elif nm in ("xgboost", "xgb", "xgbclassifier", "lightgbm", "lgbm", "lgb"):
            search_space = {
                "n_estimators": {"type": "int", "low": 100, "high": 700, "step": 100},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True}}
        elif nm in ("hist_gradient_boosting", "histgradientboosting", "hgb", "hist_gbm",
                    "gradient_boosting", "gbm"):
            search_space = {
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "max_iter" if nm.startswith("hist") else "n_estimators":
                    {"type": "int", "low": 100, "high": 600, "step": 100},
                "max_leaf_nodes" if nm.startswith("hist") else "max_depth":
                    {"type": "int", "low": 15, "high": 63} if nm.startswith("hist")
                    else {"type": "int", "low": 2, "high": 8}}
        elif nm in ("catboost", "cat"):
            search_space = {
                "iterations": {"type": "int", "low": 200, "high": 800, "step": 200},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "depth": {"type": "int", "low": 4, "high": 10}}
        elif nm in ("logistic_regression", "logisticregression", "logreg"):
            search_space = {"C": {"type": "float", "low": 1e-3, "high": 100.0, "log": True}}
        elif nm in ("ridge", "ridge_classifier"):
            search_space = {"alpha": {"type": "float", "low": 1e-3, "high": 100.0,
                                      "log": True}}
        elif nm in ("decision_tree", "decisiontree", "tree"):
            search_space = {"max_depth": {"type": "int", "low": 2, "high": 20},
                            "min_samples_leaf": {"type": "int", "low": 1, "high": 40}}
        elif nm in ("knn", "kneighbors"):
            search_space = {"n_neighbors": {"type": "int", "low": 3, "high": 40},
                            "weights": {"type": "categorical",
                                        "choices": ["uniform", "distance"]}}
        else:
            search_space = {}
        print(f"[gads_tune_model] No search space supplied — using the built-in "
              f"conservative space for '{nm}' ({len(search_space)} params)")

    # Untuned reference, so "did tuning help?" is answerable — and, as a side effect, a
    # measurement of what one trial costs, which is what makes the budget guard possible.
    _t_base = time.time()
    try:
        baseline_cv_score = float(np.mean(cross_val_score(
            _wrap(_mk({})), X, y, cv=folds, scoring=scoring, n_jobs=1)))
        per_trial_estimate = max(0.1, time.time() - _t_base)
    except Exception as e:
        baseline_cv_score = float("nan")
        per_trial_estimate = max(0.1, time.time() - _t_base)
        print(f"[gads_tune_model] Baseline CV failed: {type(e).__name__}: {e}")

    if not search_space:
        print("[gads_tune_model] Empty search space — returning the untuned estimator")
        model = _wrap(_mk({}))
        model.fit(X, y)
        return {"tuned_model": model, "model": model, "best_params": {}, "params": {},
                "best_cv_score_tuned": baseline_cv_score, "n_trials_completed": 0,
                "timed_out": False, "baseline_cv_score": baseline_cv_score,
                "search_space_used": {}, "scoring": scoring, "study": None}

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _objective(trial):
        params = {}
        for pname, spec in search_space.items():
            if not isinstance(spec, dict):
                params[pname] = spec
                continue
            ptype = str(spec.get("type", "float")).lower()
            if ptype == "int":
                params[pname] = trial.suggest_int(
                    pname, int(spec["low"]), int(spec["high"]),
                    step=int(spec.get("step", 1)))
            elif ptype == "categorical":
                params[pname] = trial.suggest_categorical(pname, spec["choices"])
            else:
                params[pname] = trial.suggest_float(
                    pname, float(spec["low"]), float(spec["high"]),
                    log=bool(spec.get("log", False)))
        est = _wrap(_mk(params))
        return float(np.mean(cross_val_score(est, X, y, cv=folds, scoring=scoring,
                                             n_jobs=1, error_score="raise")))

    # Optuna checks its timeout only BETWEEN trials, so a trial starting one second before
    # the deadline still runs to completion. Measured overshoot on a random forest was
    # +84%: the baseline fit uses default n_estimators=100 while tuned trials draw 400-800,
    # so no fixed allowance derived from the baseline can predict trial cost. That matters
    # because the executor's native-fallback path kills the whole call at 360s.
    #
    # Instead, stop ADAPTIVELY: after each trial, if starting another one would likely
    # breach the budget (judged by the slowest trial observed so far), stop the study. The
    # timeout handed to Optuna stays as a backstop.
    _trial_times = [per_trial_estimate]

    def _budget_callback(study_, trial_):
        try:
            if trial_.datetime_start and trial_.datetime_complete:
                _trial_times.append(
                    (trial_.datetime_complete - trial_.datetime_start).total_seconds())
        except Exception:
            pass
        projected = (time.time() - t0) + max(_trial_times) * 1.1
        if projected > float(timeout_s):
            study_.stop()

    print(f"[gads_tune_model] Tuning '{nm}': up to {n_trials} trials / {timeout_s}s "
          f"wall-clock (adaptive stop; one default-params CV took "
          f"{per_trial_estimate:.1f}s), {cv}-fold CV, scoring={scoring}")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_objective, n_trials=int(n_trials), timeout=float(timeout_s),
                   catch=(Exception,), callbacks=[_budget_callback],
                   show_progress_bar=False)

    complete = [t for t in study.trials
                if str(t.state) == "TrialState.COMPLETE" or t.value is not None]
    n_done = len(complete)
    elapsed = time.time() - t0
    timed_out = bool(n_done < int(n_trials))
    if elapsed > float(timeout_s) * 1.25:
        print(f"[gads_tune_model] NOTE: total {elapsed:.0f}s exceeded the {timeout_s}s "
              f"budget — a single trial ran longer than the reserved allowance")

    if n_done == 0:
        print("[gads_tune_model] WARNING: no trial completed — falling back to defaults")
        model = _wrap(_mk({}))
        model.fit(X, y)
        return {"tuned_model": model, "model": model, "best_params": {}, "params": {},
                "best_cv_score_tuned": baseline_cv_score, "n_trials_completed": 0,
                "timed_out": timed_out, "baseline_cv_score": baseline_cv_score,
                "search_space_used": search_space, "scoring": scoring, "study": study}

    best_params = dict(study.best_params)
    best_cv = float(study.best_value)
    model = _wrap(_mk(best_params))
    model.fit(X, y)

    print(f"[gads_tune_model] {n_done}/{n_trials} trials in {elapsed:.0f}s"
          f"{' (TIMED OUT — budget, not convergence, ended the search)' if timed_out else ''}")
    print(f"[gads_tune_model] {scoring}: {baseline_cv_score:.4f} (defaults) -> "
          f"{best_cv:.4f} (tuned)")
    print(f"[gads_tune_model] Best params: {best_params}")
    return {"tuned_model": model, "model": model,
            "best_params": best_params, "params": best_params,
            "best_cv_score_tuned": best_cv, "n_trials_completed": int(n_done),
            "timed_out": timed_out, "baseline_cv_score": baseline_cv_score,
            "search_space_used": search_space, "scoring": scoring, "study": study}


def gads_evaluate_holdout(model, X_train, y_train, X_test, y_test, task_kind=None,
                          calibrate=True):
    """Fit on the full training partition, evaluate ONCE on the untouched holdout.

    Binds the recipe's headline metrics under their exact contract names and always
    reports the trivial baseline next to them, because a metric without a baseline is not
    evidence. Threshold calibration is applied for binary targets only; for 3+ classes a
    single threshold is meaningless under argmax, so per-class weights are used instead
    (see gads_calibrate_threshold).

    Returns dict: {macro_f1, roc_auc, log_loss, accuracy, baseline_macro_f1,
                   beats_baseline, y_pred, y_prob, best_threshold, class_names, task_kind}
    """
    import numpy as np
    import pandas as pd

    Xtr = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    Xte = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
    ytr_raw = pd.Series(y_train).reset_index(drop=True)
    yte_raw = pd.Series(y_test).reset_index(drop=True)

    if task_kind is None:
        task_kind = ("classification"
                     if (ytr_raw.dtype.kind in "OUSb" or ytr_raw.nunique() <= 20)
                     else "regression")

    # gads_tune_model returns a Pipeline whose first step already preprocesses; a bare
    # estimator handed in directly does not, and would die on any object column. Wrap only
    # in the latter case so a tuned model is never double-preprocessed.
    _already_wrapped = hasattr(model, "steps") or hasattr(model, "named_steps")
    _non_numeric = [c for c in Xtr.columns if Xtr[c].dtype == object]
    if not _already_wrapped and (_non_numeric or Xtr.isna().any().any()):
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline as _Pipe
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OrdinalEncoder, StandardScaler
        _num = list(Xtr.select_dtypes(include="number").columns)
        _cat = [c for c in Xtr.columns if c not in _num]
        _parts = []
        if _num:
            _parts.append(("num", _Pipe([("i", SimpleImputer(strategy="median")),
                                         ("s", StandardScaler())]), _num))
        if _cat:
            _parts.append(("cat", _Pipe([
                ("i", SimpleImputer(strategy="most_frequent")),
                ("e", OrdinalEncoder(handle_unknown="use_encoded_value",
                                     unknown_value=-1, encoded_missing_value=-2))]), _cat))
        if _parts:
            model = _Pipe([("prep", ColumnTransformer(_parts, remainder="drop")),
                           ("est", model)])
            print(f"[gads_evaluate_holdout] Wrapped a bare estimator in a preprocessing "
                  f"pipeline ({len(_num)} numeric, {len(_cat)} categorical columns)")

    if task_kind == "regression":
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        model.fit(Xtr, ytr_raw.values.astype(float))
        y_pred = model.predict(Xte)
        yte = yte_raw.values.astype(float)
        rmse = float(np.sqrt(mean_squared_error(yte, y_pred)))
        mae = float(mean_absolute_error(yte, y_pred))
        r2 = float(r2_score(yte, y_pred))
        base_pred = np.full_like(yte, float(np.mean(ytr_raw.values.astype(float))))
        baseline_rmse = float(np.sqrt(mean_squared_error(yte, base_pred)))
        print(f"[gads_evaluate_holdout] RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
        print(f"[gads_evaluate_holdout] Mean-predictor baseline RMSE={baseline_rmse:.4f} "
              f"-> {'BEATS' if rmse < baseline_rmse else 'DOES NOT BEAT'} baseline")
        return {"rmse": rmse, "mae": mae, "r2": r2, "baseline_rmse": baseline_rmse,
                "beats_baseline": bool(rmse < baseline_rmse), "y_pred": y_pred,
                "task_kind": "regression"}

    from sklearn.metrics import (f1_score, roc_auc_score, log_loss as _sk_log_loss,
                                 accuracy_score, classification_report)
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder().fit(pd.concat([ytr_raw, yte_raw]).astype(str))
    ytr = le.transform(ytr_raw.astype(str))
    yte = le.transform(yte_raw.astype(str))
    class_names = [str(c) for c in le.classes_]
    n_classes = len(class_names)

    model.fit(Xtr, ytr)

    # Not every valid candidate exposes predict_proba (RidgeClassifier, LinearSVC, ...).
    # Fall back to decision_function, softmax-normalised, so log_loss and ROC-AUC stay
    # computable rather than failing the node over a scoring-API detail.
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(Xte)
    elif hasattr(model, "decision_function"):
        d = np.asarray(model.decision_function(Xte), dtype=float)
        if d.ndim == 1:
            d = np.column_stack([-d, d])
        d = d - d.max(axis=1, keepdims=True)
        e = np.exp(d)
        y_prob = e / e.sum(axis=1, keepdims=True)
        print("[gads_evaluate_holdout] No predict_proba; using softmax(decision_function). "
              "Probabilities are ordinally valid but not calibrated.")
    else:
        pred_enc = np.asarray(model.predict(Xte)).astype(int)
        y_prob = np.zeros((len(pred_enc), n_classes), dtype=float)
        y_prob[np.arange(len(pred_enc)), pred_enc] = 1.0
        print("[gads_evaluate_holdout] Estimator exposes neither predict_proba nor "
              "decision_function; log_loss will be degenerate.")

    best_threshold = None

    if n_classes == 2:
        pos = y_prob[:, 1]
        if calibrate:
            ths = np.linspace(0.01, 0.99, 99)
            scores = [f1_score(yte, (pos >= t).astype(int), zero_division=0) for t in ths]
            best_threshold = float(ths[int(np.argmax(scores))])
        else:
            best_threshold = 0.5
        y_pred_enc = (pos >= best_threshold).astype(int)
        roc_auc = float(roc_auc_score(yte, pos))
    else:
        y_pred_enc = np.argmax(y_prob, axis=1)
        try:
            roc_auc = float(roc_auc_score(yte, y_prob, multi_class="ovr",
                                          average="macro"))
        except Exception:
            roc_auc = float("nan")

    macro_f1 = float(f1_score(yte, y_pred_enc, average="macro", zero_division=0))
    accuracy = float(accuracy_score(yte, y_pred_enc))
    log_loss = float(_sk_log_loss(yte, y_prob, labels=list(range(n_classes))))

    # Majority-class baseline — the number every headline metric must be read against.
    majority = int(pd.Series(ytr).value_counts().idxmax())
    base_pred = np.full_like(yte, majority)
    baseline_macro_f1 = float(f1_score(yte, base_pred, average="macro", zero_division=0))
    baseline_accuracy = float(accuracy_score(yte, base_pred))

    y_pred = np.asarray(class_names, dtype=object)[y_pred_enc]

    print(f"[gads_evaluate_holdout] n_classes={n_classes}  holdout n={len(yte)}")
    if best_threshold is not None:
        print(f"[gads_evaluate_holdout] Calibrated threshold: {best_threshold:.4f}")
    print(f"[gads_evaluate_holdout] macro_f1={macro_f1:.4f}  roc_auc={roc_auc:.4f}  "
          f"log_loss={log_loss:.4f}  accuracy={accuracy:.4f}")
    print(f"[gads_evaluate_holdout] Majority baseline: macro_f1={baseline_macro_f1:.4f} "
          f"accuracy={baseline_accuracy:.4f} -> "
          f"{'BEATS' if macro_f1 > baseline_macro_f1 else 'DOES NOT BEAT'} baseline")
    print(classification_report(yte, y_pred_enc, target_names=class_names,
                                zero_division=0))

    # `predictions`/`probabilities`/`threshold` are aliases (022 v1.1 fix 1).
    return {"macro_f1": macro_f1, "roc_auc": roc_auc, "log_loss": log_loss,
            "predictions": y_pred, "probabilities": y_prob,
            "threshold": best_threshold,
            "accuracy": accuracy, "baseline_macro_f1": baseline_macro_f1,
            "baseline_accuracy": baseline_accuracy,
            "beats_baseline": bool(macro_f1 > baseline_macro_f1),
            "y_pred": y_pred, "y_prob": y_prob, "best_threshold": best_threshold,
            "class_names": class_names, "task_kind": "classification"}


def gads_feature_importance(model, X_test, y_test, n_repeats=5, seed=42, top_k=20,
                            use_shap=False, write_path="feature_importance.csv"):
    """Permutation importance on HELD-OUT data.

    Two invariants, both of which hand-written code gets wrong routinely:
    (1) importance is measured on the holdout, not on training data — training importance
        rewards memorisation; (2) permutation, not impurity — sklearn's
        `.feature_importances_` is biased toward high-cardinality and continuous features,
        which is a known-wrong default rather than a matter of taste. Impurity values are
        reported alongside for comparison but never as the headline.

    Returns dict: {importance_table, top_features, n_features_reported, method,
                   shap_available}
    """
    import numpy as np
    import pandas as pd
    from sklearn.inspection import permutation_importance

    X = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
    feat_names = [str(c) for c in X.columns]
    y = pd.Series(y_test).reset_index(drop=True)

    # Match the label space the estimator was fitted on — and ONLY when it actually
    # differs. Encoding y when the model already speaks the caller's labels makes every
    # permutation score 0.0, which yields an all-ties table that still looks like a
    # ranking. Silent, plausible, and wrong; hence the explicit subset test.
    y_eval = y
    try:
        classes = getattr(model, "classes_", None)
        if classes is not None:
            cls_str = set(str(c) for c in classes)
            y_str = set(str(v) for v in pd.Series(y).unique())
            if not y_str.issubset(cls_str):
                # Model was fitted on encoded labels (e.g. by gads_evaluate_holdout);
                # reproduce that encoding rather than passing raw labels.
                from sklearn.preprocessing import LabelEncoder
                y_eval = pd.Series(
                    LabelEncoder().fit(pd.Series(y).astype(str)).transform(
                        pd.Series(y).astype(str)))
                print("[gads_feature_importance] Model uses an encoded label space; "
                      "encoding y_test to match")
    except Exception:
        y_eval = y

    print(f"[gads_feature_importance] Permutation importance on {len(X)} held-out rows, "
          f"{len(feat_names)} features, {n_repeats} repeats")
    r = permutation_importance(model, X, y_eval, n_repeats=int(n_repeats),
                               random_state=seed, n_jobs=-1)

    table = pd.DataFrame({
        "feature": feat_names,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    # Impurity importance for contrast only — explicitly labelled as the biased measure.
    try:
        imp = getattr(model, "feature_importances_", None)
        if imp is not None and len(imp) == len(feat_names):
            table["impurity_importance_biased"] = [
                float(dict(zip(feat_names, imp))[f]) for f in table["feature"]]
    except Exception:
        pass

    table["rank"] = table.index + 1
    n_report = int(min(top_k, len(table)))
    top = table.head(n_report)

    try:
        table.to_csv(write_path, index=False)
        print(f"[gads_feature_importance] Wrote {write_path}")
    except Exception as e:
        print(f"[gads_feature_importance] Could not write {write_path}: {e}")

    print(f"[gads_feature_importance] Top {min(10, n_report)} features:")
    for _, row in top.head(10).iterrows():
        print(f"    {int(row['rank']):>2d}. {row['feature']:<32s} "
              f"{row['importance_mean']:+.5f} +/- {row['importance_std']:.5f}")

    n_useless = int((table["importance_mean"] <= 0).sum())
    if n_useless:
        print(f"[gads_feature_importance] {n_useless} feature(s) had zero or negative "
              "importance — permuting them did not hurt the model")

    shap_available = False
    if use_shap:
        try:
            import shap
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(X.head(min(500, len(X))))
            shap_available = True
            print(f"[gads_feature_importance] SHAP values computed on "
                  f"{min(500, len(X))} rows")
        except Exception as e:
            print(f"[gads_feature_importance] SHAP unavailable ({type(e).__name__}: "
                  f"{str(e)[:80]}); permutation importance stands on its own")

    emit = globals().get("gads_emit_insight")
    if callable(emit):
        try:
            names = ", ".join(top.head(5)["feature"].tolist())
            emit("feature_importance",
                 f"Top held-out permutation-importance features: {names}.",
                 f"{n_report} features ranked; {n_useless} at or below zero importance.")
        except Exception:
            pass

    # `importance` is an ALIAS of importance_table — models reach for the short natural
    # key name (022 v1.1 fix 1; result['importance'] cost feature_importance an attempt
    # in the local A/B).
    return {"importance_table": table, "importance": table,
            "top_features": top["feature"].tolist(),
            "n_features_reported": n_report, "method": "permutation_holdout",
            "shap_available": shap_available}


def gads_audit_model_choice(chosen, dataset_facts, bakeoff_table=None, tuning_result=None,
                            write_path="model_choice_checks.json", emit_insights=True,
                            blocking_rules=None):
    """Adjudicate a model CHOICE against the selection rules — without making it.

    The counterpart to `gads_audit_model` (which audits a fitted estimator via skore). This
    audits the *decision*: the model shortlists and picks (019 — that judgment is what is
    being measured), and this gate checks the pick against the same rules the
    `model_selection_tabular` skill states in prose, after the fact.

    Severity is "warn" for every rule at v1.0 by design (approach_docs/022 §5.2): the point
    of the first version is to collect the violation distribution, not to enforce it. A rule
    graduates to blocking by being named in `blocking_rules`, which is how the hardening
    ladder advances without touching this function.

    Fail-open like gads_audit_model: any internal error returns a dict with an `error` key
    rather than raising, so a diagnostic failure never fails the task.

    Returns dict: {issues, warnings, passed, n_selection_issues, n_blocking, chosen,
                   checks_path}
    """
    import json

    result = {"issues": [], "warnings": [], "passed": [], "n_selection_issues": 0,
              "n_blocking": 0, "chosen": str(chosen), "checks_path": write_path}
    try:
        f = dataset_facts or {}
        blocking = set(blocking_rules or [])
        name = str(chosen).lower().replace("-", "_").replace(" ", "_")

        boosted = name in ("xgboost", "xgb", "lightgbm", "lgbm", "lgb", "catboost", "cat",
                           "gradient_boosting", "gbm", "hist_gradient_boosting",
                           "histgradientboosting", "hgb", "hist_gbm")
        tree_ens = boosted or name in ("random_forest", "randomforest", "rf",
                                       "extra_trees", "extratrees")
        linear = name in ("logistic_regression", "logisticregression", "logreg", "ridge",
                          "ridge_classifier", "linear_regression", "ols", "elastic_net",
                          "elasticnet")
        interpretable = linear or name in ("decision_tree", "decisiontree", "tree")
        nan_native = name in ("xgboost", "xgb", "lightgbm", "lgbm", "lgb", "catboost",
                              "cat", "hist_gradient_boosting", "histgradientboosting",
                              "hgb", "hist_gbm")

        n_rows = int(f.get("n_rows") or 0)
        n_features = int(f.get("n_features") or 0)
        max_card = int(f.get("max_cat_cardinality") or 0)
        minority = f.get("minority_class_rate")
        has_missing = bool(f.get("has_missing"))
        interp_req = bool(f.get("interpretability_required"))
        task = f.get("task_kind") or "classification"

        checks = []

        def _add(code, title, triggered, message, hint=""):
            checks.append({"code": code, "title": title, "triggered": bool(triggered),
                           "message": message, "hint": hint,
                           "severity": ("block" if code in blocking else "warn")})

        _add("MS001", "Boosted ensemble on small data",
             boosted and 0 < n_rows < 1000,
             f"{chosen} was chosen on only {n_rows} rows. Boosted ensembles overfit at "
             "this scale without a careful tuning budget.",
             "A regularized linear model or a random forest is the safer default below "
             "~1000 rows.")

        _add("MS002", "Complex model on wide data",
             tree_ens and n_features > n_rows > 0,
             f"{chosen} was chosen with {n_features} features against only {n_rows} rows "
             "(p > n).",
             "L1/L2-regularized linear models handle p > n far more reliably.")

        _add("MS003", "High-cardinality categoricals",
             max_card > 50 and name in ("xgboost", "xgb", "random_forest", "randomforest",
                                        "rf", "extra_trees", "extratrees"),
             f"Max categorical cardinality is {max_card} and {chosen} has no native "
             "categorical handling, so encoding choice dominates the result.",
             "CatBoost handles this natively; otherwise use target or frequency encoding "
             "rather than one-hot.")

        _add("MS004", "Missingness imputed away for an estimator that cannot use it",
             has_missing and not nan_native,
             f"The data has missing values and {chosen} cannot consume NaN, so the "
             "pipeline imputes them. Imputation discards the missingness pattern, which "
             "is often predictive in its own right.",
             "LightGBM / XGBoost / HistGradientBoosting learn a split direction for NaN "
             "and can use missingness as signal rather than filling it in.")

        _add("MS005", "Severe class imbalance",
             task == "classification" and minority is not None and minority < 0.05,
             f"The minority class is {float(minority):.2%} of the training data.",
             "Class weighting is applied by the bakeoff natives; the decision threshold "
             "must also be calibrated, and PR-AUC reported alongside ROC-AUC.")

        _add("MS006", "Interpretability requirement not met",
             interp_req and not interpretable,
             f"Interpretability was requested but {chosen} is not directly interpretable.",
             "Lead with a linear model or a shallow tree; a complex model needs a material "
             "margin over that baseline to justify itself.")

        _add("MS009", "Tree ensemble asked to extrapolate",
             task == "regression" and tree_ens,
             f"{chosen} is a tree ensemble, and trees cannot extrapolate beyond the range "
             "seen in training.",
             "If targets outside the training range must be predicted, use a linear model "
             "or a GAM.")

        # --- Evidence-based checks: the choice against what the bakeoff measured ---
        if bakeoff_table is not None:
            try:
                tbl = bakeoff_table
                if hasattr(tbl, "to_dict"):
                    recs = tbl.to_dict(orient="records")
                else:
                    recs = list(tbl)
                recs = [r for r in recs if r.get("mean_score") is not None]
                recs.sort(key=lambda r: r["mean_score"], reverse=True)
                if recs:
                    winner = str(recs[0].get("candidate"))
                    chosen_rec = next((r for r in recs
                                       if str(r.get("candidate")).lower() == name), None)
                    if chosen_rec is not None and str(winner).lower() != name:
                        gap = float(recs[0]["mean_score"]) - float(chosen_rec["mean_score"])
                        noise = float(recs[0].get("std_score") or 0.0)
                        _add("MS007", "Chosen model lost the bakeoff",
                             gap > noise,
                             f"{chosen} scored {float(chosen_rec['mean_score']):.4f} but "
                             f"{winner} scored {float(recs[0]['mean_score']):.4f} "
                             f"(gap {gap:.4f} vs fold std {noise:.4f}).",
                             "Either take the winner, or state explicitly why the margin "
                             "is worth trading away.")
                    if len(recs) > 1:
                        spread = float(recs[0]["mean_score"]) - float(recs[-1]["mean_score"])
                        _add("MS010", "Candidates indistinguishable",
                             spread < 0.005,
                             f"All candidates scored within {spread:.4f} of each other — "
                             "the choice is not supported by the evidence.",
                             "Prefer the simplest or cheapest candidate when the spread is "
                             "within noise.")
            except Exception as be:
                print(f"[gads_audit_model_choice] bakeoff comparison skipped: {be}")

        if tuning_result is not None:
            try:
                tuned = tuning_result.get("best_cv_score_tuned")
                base = tuning_result.get("baseline_cv_score")
                if tuned is not None and base is not None and base == base:
                    _add("MS008", "Tuning did not help",
                         float(tuned) <= float(base) + 1e-9,
                         f"Tuned score {float(tuned):.4f} is no better than the untuned "
                         f"{float(base):.4f}.",
                         "Report the untuned model — the tuning budget bought nothing.")
                if tuning_result.get("timed_out"):
                    _add("MS011", "Search ended on budget, not convergence",
                         True,
                         f"The study completed {tuning_result.get('n_trials_completed')} "
                         "trials and stopped on the wall-clock budget.",
                         "Treat the tuned hyperparameters as a budget-limited result.")
            except Exception as te:
                print(f"[gads_audit_model_choice] tuning comparison skipped: {te}")

        triggered = [c for c in checks if c["triggered"]]
        result["issues"] = [c for c in triggered if c["severity"] == "block"]
        result["warnings"] = [c for c in triggered if c["severity"] != "block"]
        result["passed"] = [c["code"] for c in checks if not c["triggered"]]
        result["n_selection_issues"] = len(triggered)
        result["n_blocking"] = len(result["issues"])

        try:
            with open(write_path, "w") as fh:
                json.dump({"chosen": str(chosen), "dataset_facts": f,
                           "checks": checks,
                           "n_selection_issues": result["n_selection_issues"],
                           "n_blocking": result["n_blocking"]}, fh, indent=2, default=str)
        except Exception as we:
            print(f"[gads_audit_model_choice] could not write {write_path}: {we}")

        print(f"[gads_audit_model_choice] '{chosen}' — {result['n_selection_issues']} "
              f"finding(s) ({result['n_blocking']} blocking), "
              f"{len(result['passed'])} rule(s) passed")
        for c in triggered:
            tag = "BLOCK" if c["severity"] == "block" else "warn"
            print(f"  [{tag}] {c['code']} {c['title']}: {c['message']}")
            if c["hint"]:
                print(f"          -> {c['hint']}")
        if not triggered:
            print("  No selection rule was violated.")

        if emit_insights:
            emit = globals().get("gads_emit_insight")
            if callable(emit):
                for c in triggered:
                    try:
                        emit(f"model_choice_{c['code']}",
                             f"Selection check {c['code']} ({c['title']}): {c['message']}",
                             c["hint"])
                    except Exception:
                        pass
        return result
    except Exception as e:
        print(f"[gads_audit_model_choice] audit failed ({type(e).__name__}: {e}); "
              "continuing without it")
        result["error"] = f"{type(e).__name__}: {e}"
        return result


def gads_model_card(chosen=None, dataset_facts=None, bakeoff_table=None,
                    tuning_result=None, evaluation=None, importance=None,
                    selection_rationale=None, audit=None, selection_audit=None,
                    write_path="model_card.md"):
    """Assemble the performance report — the FALLBACK for the narrative node.

    Reporting nodes are a repeatedly-measured local-model failure in this project, and the
    standing remedy is a minimal intent plus a fallback native rather than more prompt. This
    writes a defensible model card from whatever evidence is actually present; every
    argument is optional and missing sections are simply omitted rather than invented.

    Returns dict: {model_card_text, card_path, sections}
    """
    import json

    L = []
    sections = []

    def _sec(title):
        sections.append(title)
        L.append(f"\n## {title}\n")

    L.append("# Model Card\n")
    if chosen:
        L.append(f"**Selected model:** `{chosen}`\n")

    f = dataset_facts or {}
    if f:
        _sec("Data")
        L.append(f"- Rows (train): {f.get('n_rows')}, features: {f.get('n_features')} "
                 f"({f.get('rows_per_feature', 0):.1f} rows per feature)")
        L.append(f"- {f.get('n_numeric')} numeric / {f.get('n_categorical')} categorical; "
                 f"max categorical cardinality {f.get('max_cat_cardinality')}")
        L.append(f"- Missing cell rate: {f.get('missing_rate', 0):.4f}")
        if f.get("n_classes"):
            L.append(f"- Task: {f.get('task_kind')} with {f.get('n_classes')} classes; "
                     f"minority class {float(f.get('minority_class_rate') or 0):.2%}")

    if selection_rationale:
        _sec("Why this model")
        L.append(str(selection_rationale))

    if bakeoff_table is not None:
        _sec("Candidate comparison")
        L.append("All candidates were scored on identical cross-validation folds with the "
                 "same seed and metric.\n")
        try:
            if hasattr(bakeoff_table, "to_markdown"):
                cols = [c for c in ["candidate", "mean_score", "std_score", "fit_seconds"]
                        if c in bakeoff_table.columns]
                L.append(bakeoff_table[cols].to_markdown(index=False))
            else:
                L.append("```\n" + str(bakeoff_table) + "\n```")
        except Exception:
            L.append("```\n" + str(bakeoff_table) + "\n```")

    if tuning_result:
        _sec("Hyperparameter tuning")
        n_done = tuning_result.get("n_trials_completed")
        base = tuning_result.get("baseline_cv_score")
        tuned = tuning_result.get("best_cv_score_tuned")
        L.append(f"- Trials completed: {n_done}"
                 + (" (search ended on the wall-clock budget, not convergence)"
                    if tuning_result.get("timed_out") else ""))
        if base is not None and tuned is not None:
            L.append(f"- CV {tuning_result.get('scoring', 'score')}: {base:.4f} (defaults) "
                     f"-> {tuned:.4f} (tuned)")
        if tuning_result.get("best_params"):
            L.append("- Best hyperparameters:\n\n```json\n"
                     + json.dumps(tuning_result["best_params"], indent=2, default=str)
                     + "\n```")

    ev = evaluation or {}
    if ev:
        _sec("Held-out performance")
        if ev.get("task_kind") == "regression":
            L.append(f"- RMSE: {ev.get('rmse'):.4f} (mean-predictor baseline: "
                     f"{ev.get('baseline_rmse'):.4f})")
            L.append(f"- MAE: {ev.get('mae'):.4f}, R²: {ev.get('r2'):.4f}")
        else:
            for k in ("macro_f1", "roc_auc", "log_loss", "accuracy"):
                if ev.get(k) is not None:
                    L.append(f"- {k}: {float(ev[k]):.4f}")
            if ev.get("baseline_macro_f1") is not None:
                L.append(f"- Majority-class baseline macro-F1: "
                         f"{float(ev['baseline_macro_f1']):.4f}")
            if ev.get("best_threshold") is not None:
                L.append(f"- Calibrated decision threshold: "
                         f"{float(ev['best_threshold']):.4f}")
        if ev.get("beats_baseline") is not None:
            L.append(f"- **Beats the trivial baseline: "
                     f"{'yes' if ev['beats_baseline'] else 'NO'}**")

    imp = importance or {}
    if imp:
        _sec("What drives the predictions")
        L.append("Permutation importance measured on held-out data (not impurity "
                 "importance, which is biased toward high-cardinality features).\n")
        for i, feat in enumerate(imp.get("top_features", [])[:10], 1):
            L.append(f"{i}. `{feat}`")

    findings = []
    for src in (selection_audit, audit):
        if not src:
            continue
        for c in (src.get("issues") or []):
            findings.append(("BLOCKING", c.get("code", ""), c.get("title", ""),
                             c.get("message") or c.get("explanation") or ""))
        for c in (src.get("warnings") or []):
            findings.append(("warning", c.get("code", ""), c.get("title", ""),
                             c.get("message") or c.get("explanation") or ""))
        for c in (src.get("tips") or []):
            findings.append(("tip", c.get("code", ""), c.get("title", ""),
                             c.get("explanation") or ""))
    if findings:
        _sec("Methodological findings")
        for sev, code, title, msg in findings:
            L.append(f"- **[{sev}] {code} {title}** — {msg}")

    _sec("Limitations")
    lims = ["The held-out partition was evaluated once; no further selection was made "
            "against it."]
    if tuning_result and tuning_result.get("timed_out"):
        lims.append("The hyperparameter search was truncated by its wall-clock budget, so "
                    "the reported configuration is budget-limited rather than optimal.")
    if f.get("n_rows") and int(f.get("n_rows")) < 1000:
        lims.append(f"With only {f.get('n_rows')} training rows, cross-validation "
                    "estimates carry wide uncertainty.")
    if ev and ev.get("beats_baseline") is False:
        lims.append("The model does not beat the trivial baseline — it should not be "
                    "deployed on this evidence.")
    for l in lims:
        L.append(f"- {l}")

    text = "\n".join(L) + "\n"
    try:
        with open(write_path, "w") as fh:
            fh.write(text)
        print(f"[gads_model_card] Wrote {write_path} ({len(sections)} sections)")
    except Exception as e:
        print(f"[gads_model_card] Could not write {write_path}: {e}")

    print(text[:1500])
    return {"model_card_text": text, "text": text, "card_path": write_path,
            "sections": sections}
