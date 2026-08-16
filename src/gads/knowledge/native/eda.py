"""
GADS Native EDA Nodes

Deterministic primitives for the `tabular_eda.descriptive.standard` recipe
(approach_docs/021). Three of these are opt-in *fallbacks* for nodes whose work is
genuine judgment (profiling, quality assessment, transformation recommendation) — the
model writes those normally and its capability stays measured. One,
`gads_apply_transformations`, is a correctness primitive that is always available:
applying a transformation manifest has exactly one right answer, and the ORDER of
operations (split -> fit on train -> apply to all partitions) is a leakage guard that
must not depend on what a model happened to generate.

Design per approach_docs/019: nativize invariant/correctness operations, keep genuinely
variable work model-generated. Annotation-free and self-contained (imports inside) so the
source injects verbatim into the sandbox kernel via the preamble.
"""


def gads_profile_dataframe(df, top_k=10):
    """Per-column profile of a DataFrame: dtype, nulls, cardinality, numeric stats.

    Full-data (never sampled) — this is the difference between the recipe and the
    DataAnalyzer's 5000-row planner probe. Returns a dict with `columns` (per-column
    stats) plus dataset-level `n_rows`, `n_cols`, `missing_cell_rate`.
    """
    import numpy as np
    import pandas as pd

    n_rows = int(len(df))
    n_cols = int(df.shape[1])
    columns = {}

    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        entry = {
            "dtype": str(s.dtype),
            "count": int(n_rows - n_missing),
            "n_missing": n_missing,
            "missing_rate": round(n_missing / n_rows, 6) if n_rows else 0.0,
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            clean = s.dropna()
            if len(clean):
                entry.update({
                    "min": float(clean.min()), "max": float(clean.max()),
                    "mean": float(clean.mean()), "std": float(clean.std()),
                    "q25": float(clean.quantile(0.25)),
                    "median": float(clean.median()),
                    "q75": float(clean.quantile(0.75)),
                    "skew": float(clean.skew()) if len(clean) > 2 else 0.0,
                })
        else:
            vc = s.value_counts(dropna=True).head(top_k)
            entry["top_values"] = {str(k): int(v) for k, v in vc.items()}
        columns[str(col)] = entry

    total_cells = n_rows * n_cols
    missing_cells = int(df.isna().sum().sum())
    profile = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_cell_rate": round(missing_cells / total_cells, 6) if total_cells else 0.0,
        "columns": columns,
    }
    print(f"  [EDA] Profiled {n_cols} columns over {n_rows:,} rows "
          f"(missing cell rate {profile['missing_cell_rate']:.4f})")
    return profile


def gads_assess_quality(df, profile=None, near_constant_threshold=0.99,
                        high_missing_threshold=0.6, high_cardinality_threshold=50):
    """Data-quality assessment: duplicates, constants, id-like and high-missing columns,
    outlier rates, and repeated-entity candidates (for grouped splitting).

    Returns {duplicate_rows, columns: {col: {flags, outlier_rate, ...}},
    n_flagged_columns, group_candidates}.
    """
    import pandas as pd

    if profile is None:
        profile = gads_profile_dataframe(df)
    n_rows = int(len(df))
    columns = {}
    group_candidates = []

    for col in df.columns:
        s = df[col]
        pstat = profile["columns"].get(str(col), {})
        n_unique = int(pstat.get("n_unique", s.nunique(dropna=True)))
        missing_rate = float(pstat.get("missing_rate", 0.0))
        flags = []

        is_numeric = (pd.api.types.is_numeric_dtype(s)
                      and not pd.api.types.is_bool_dtype(s))
        is_float = pd.api.types.is_float_dtype(s)
        is_datetime = pd.api.types.is_datetime64_any_dtype(s)
        lowered = str(col).lower()
        name_hints_id = any(t in lowered for t in
                            ("id", "uuid", "guid", "key", "index", "_no", "number", "code"))
        # An identifier is nearly unique AND not a continuous measurement. Continuous
        # floats are nearly always fully unique, so uniqueness alone would condemn the
        # most informative features in the dataset — require a non-float dtype, and for
        # integers additionally require the NAME to look like an identifier (an integer
        # measurement such as a timestamp or a count can also be unique per row).
        nearly_unique = bool(n_rows) and n_unique / max(n_rows, 1) >= near_constant_threshold

        if n_unique <= 1:
            flags.append("constant")
        elif is_datetime:
            # A timestamp is unique per row by nature but is never an identifier — and it
            # is very often the column a time-ordered split depends on.
            pass
        elif nearly_unique and not is_float and (not is_numeric or name_hints_id):
            flags.append("id_like")
        else:
            top_share = 0.0
            vc = s.value_counts(dropna=True, normalize=True)
            if len(vc):
                top_share = float(vc.iloc[0])
            if top_share >= near_constant_threshold:
                flags.append("near_constant")

        if missing_rate >= high_missing_threshold:
            flags.append("high_missing")

        outlier_rate = None
        if is_numeric:
            clean = s.dropna()
            if len(clean) > 3:
                q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    outlier_rate = round(float(((clean < lo) | (clean > hi)).mean()), 6)
                else:
                    outlier_rate = 0.0
            skew = float(pstat.get("skew", 0.0) or 0.0)
            if abs(skew) > 2:
                flags.append("heavy_tailed")
        else:
            if n_unique > high_cardinality_threshold and "id_like" not in flags:
                flags.append("high_cardinality")
            # A repeated entity id: many distinct values but each appearing several times.
            if n_rows and 1 < n_unique < n_rows * 0.5 and n_rows / max(n_unique, 1) >= 2:
                if n_unique >= 10:
                    group_candidates.append(str(col))

        if "constant" in flags or "id_like" in flags or "high_missing" in flags:
            flags.append("drop_suggested")

        columns[str(col)] = {"flags": flags, "outlier_rate": outlier_rate,
                             "n_unique": n_unique, "missing_rate": missing_rate}

    duplicate_rows = int(df.duplicated().sum())
    n_flagged = sum(1 for c in columns.values() if c["flags"])
    print(f"  [EDA] Quality: {duplicate_rows:,} duplicate rows, "
          f"{n_flagged} flagged column(s), {len(group_candidates)} group candidate(s)")
    return {"duplicate_rows": duplicate_rows, "columns": columns,
            "n_flagged_columns": int(n_flagged), "group_candidates": group_candidates}


def gads_recommend_split(df, profile=None, quality=None, target_col=None,
                         ratios=None, random_state=42):
    """Recommend HOW to split a single dataset into train/val/test.

    The default random split is wrong more often than people assume; everything needed to
    know that has already been measured. Precedence is deliberate and documented in
    approach_docs/021 §9.3:

        time_ordered > grouped > stratified > random

    A time-ordered split cannot be traded for stratification without reintroducing
    look-ahead, so it wins outright. Returns a manifest `split` block.
    """
    import pandas as pd

    if profile is None:
        profile = gads_profile_dataframe(df)
    if quality is None:
        quality = gads_assess_quality(df, profile)
    if ratios is None:
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

    block = {"applies_to": None, "method": "random", "ratios": ratios,
             "stratify_by": None, "group_by": None, "time_column": None,
             "random_state": random_state, "rationale": ""}

    # 1. Temporal — a datetime column, or a monotonically non-decreasing numeric one.
    time_col = None
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            time_col = str(col)
            break
    if time_col is None:
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                clean = s.dropna()
                lowered = str(col).lower()
                looks_temporal = any(t in lowered for t in
                                     ("date", "time", "year", "month", "day", "ts", "timestamp"))
                if looks_temporal and len(clean) > 2 and clean.is_monotonic_increasing:
                    time_col = str(col)
                    break
    if time_col is not None:
        block["method"] = "time_ordered"
        block["time_column"] = time_col
        block["rationale"] = (
            f"`{time_col}` orders the rows in time — a random split would train on the "
            f"future and test on the past, inflating the score.")
        if quality.get("group_candidates"):
            block["rationale"] += (
                f" NOTE: repeated-entity column(s) {quality['group_candidates'][:3]} also "
                f"suggest a grouped split; temporal takes precedence — review if entity "
                f"leakage matters more than look-ahead here.")
        print(f"  [EDA] Split: time_ordered on `{time_col}`")
        return block

    # 2. Grouped — a repeated entity would otherwise appear in several partitions.
    if quality.get("group_candidates"):
        group_col = quality["group_candidates"][0]
        block["method"] = "grouped"
        block["group_by"] = group_col
        block["rationale"] = (
            f"`{group_col}` repeats across rows — splitting randomly would place the same "
            f"entity in train and test, leaking identity.")
        print(f"  [EDA] Split: grouped on `{group_col}`")
        return block

    # 3. Stratified — a low-cardinality (especially imbalanced) classification target.
    if target_col is not None and str(target_col) in df.columns:
        s = df[str(target_col)].dropna()
        n_unique = int(s.nunique())
        if 1 < n_unique <= 20:
            block["method"] = "stratified"
            block["stratify_by"] = str(target_col)
            minority = float(s.value_counts(normalize=True).min())
            block["rationale"] = (
                f"`{target_col}` is a {n_unique}-class target with a {minority:.1%} minority "
                f"class — stratify so every partition keeps it represented.")
            print(f"  [EDA] Split: stratified on `{target_col}`")
            return block

    block["rationale"] = "No temporal ordering, repeated entity, or discrete target detected."
    print("  [EDA] Split: random")
    return block


def gads_recommend_transformations(df, profile=None, quality=None, target_col=None,
                                   ml_intent=False, source_file="dataset",
                                   write_path="eda_transformations.meta.json",
                                   split=None):
    """Deterministic transformation recommendations — the conservative fallback for the
    recipe's judgment node, so a weak model can never block the manifest.

    Heuristics (approach_docs/021 §4.2): missingness drives imputation, skew and outlier
    rate drive scaling, cardinality drives encoding, and structural flags drive dropping.
    Writes the manifest and returns it.

    `.meta.json` is deliberate: any other `.json` written to the workspace is
    auto-registered as an interactive Plotly artifact.
    """
    import json
    import pandas as pd

    if profile is None:
        profile = gads_profile_dataframe(df)
    if quality is None:
        quality = gads_assess_quality(df, profile)

    columns = {}
    for col in df.columns:
        name = str(col)
        s = df[col]
        pstat = profile["columns"].get(name, {})
        qstat = quality["columns"].get(name, {})
        flags = list(qstat.get("flags", []))
        missing_rate = float(pstat.get("missing_rate", 0.0))
        n_unique = int(pstat.get("n_unique", 0))
        is_numeric = (pd.api.types.is_numeric_dtype(s)
                      and not pd.api.types.is_bool_dtype(s))
        is_datetime = pd.api.types.is_datetime64_any_dtype(s)

        impute = None
        scale = None
        encode = None
        reasons = []

        if "drop_suggested" in flags:
            reasons.append("structurally unusable (" + ", ".join(
                f for f in flags if f in ("constant", "id_like", "high_missing")) + ")")
            columns[name] = {
                "dtype": str(s.dtype), "missing_rate": missing_rate, "n_unique": n_unique,
                "outlier_rate": qstat.get("outlier_rate"),
                "skew": pstat.get("skew"),
                "recommended_impute": "drop_column", "recommended_scale": None,
                "recommended_encode": None, "flags": flags,
                "rationale": "; ".join(reasons),
            }
            continue

        if is_datetime:
            # Never encode a timestamp. Frequency- or one-hot-encoding a datetime turns an
            # ordered quantity into meaningless noise (every value is unique, so the codes
            # carry no signal and do not transfer to unseen data). Leave it untouched and
            # tell the analyst to derive calendar features deliberately.
            if "datetime" not in flags:
                flags.append("datetime")
            if missing_rate > 0:
                impute = "forward_fill"
                reasons.append(f"{missing_rate:.1%} missing -> forward fill (time-ordered)")
            reasons.append("datetime left untransformed — derive calendar features "
                           "(year/month/dayofweek) explicitly if needed")
            columns[name] = {
                "dtype": str(s.dtype), "missing_rate": missing_rate, "n_unique": n_unique,
                "outlier_rate": qstat.get("outlier_rate"), "skew": pstat.get("skew"),
                "recommended_impute": impute, "recommended_scale": None,
                "recommended_encode": None, "flags": flags,
                "rationale": "; ".join(reasons),
            }
            continue

        if missing_rate > 0:
            impute = "median" if is_numeric else "mode"
            reasons.append(f"{missing_rate:.1%} missing -> {impute}")

        if is_numeric:
            skew = float(pstat.get("skew", 0.0) or 0.0)
            outlier_rate = float(qstat.get("outlier_rate") or 0.0)
            col_min = pstat.get("min", 0.0)
            if abs(skew) > 2 and col_min is not None and col_min >= 0:
                scale = "log1p"
                reasons.append(f"skew {skew:.2f} with non-negative values -> log1p")
            elif outlier_rate > 0.05:
                scale = "robust"
                reasons.append(f"{outlier_rate:.1%} outliers -> robust scaling")
            else:
                scale = "standard"
                reasons.append("well-behaved -> standard scaling")
        else:
            if n_unique <= 10:
                encode = "onehot"
                reasons.append(f"{n_unique} categories -> one-hot")
            elif n_unique <= 50:
                encode = "frequency"
                reasons.append(f"{n_unique} categories -> frequency encoding")
            else:
                encode = "target" if target_col else "frequency"
                if "high_cardinality" not in flags:
                    flags.append("high_cardinality")
                reasons.append(f"{n_unique} categories -> {encode} encoding")

        if target_col is not None and name == str(target_col):
            impute = impute if missing_rate > 0 else None
            scale = None
            encode = None
            reasons = ["declared target column — left untransformed"]

        columns[name] = {
            "dtype": str(s.dtype), "missing_rate": missing_rate, "n_unique": n_unique,
            "outlier_rate": qstat.get("outlier_rate"), "skew": pstat.get("skew"),
            "recommended_impute": impute, "recommended_scale": scale,
            "recommended_encode": encode, "flags": flags,
            "rationale": "; ".join(reasons) if reasons else "no transformation needed",
        }

    if ml_intent and split is None:
        split = gads_recommend_split(df, profile, quality, target_col)
    if split is not None and split.get("applies_to") is None:
        split["applies_to"] = source_file

    manifest = {
        "schema_version": "1.0",
        "generated_by": "gads_recommend_transformations",
        "target_column": str(target_col) if target_col is not None else None,
        "split": split,
        "files": {
            source_file: {
                "n_rows": profile["n_rows"], "n_cols": profile["n_cols"],
                "missing_cell_rate": profile["missing_cell_rate"],
                "columns": columns,
            }
        },
    }

    with open(write_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    n_drop = sum(1 for c in columns.values() if c["recommended_impute"] == "drop_column")
    print(f"  [EDA] Wrote {write_path}: {len(columns)} columns, {n_drop} marked for drop, "
          f"split={(split or {}).get('method', 'none')}")
    return manifest


def gads_write_transformation_manifest(decisions, df=None, target_column=None, split=None,
                                       source_file="dataset", profile=None, quality=None,
                                       write_path="eda_transformations.meta.json"):
    """Serialize per-column transformation DECISIONS into the canonical manifest.

    The split of labour that matters: choosing a strategy per column is judgment and stays
    with the model; producing the exact on-disk schema is invariant and belongs here. A
    cloud model asked to emit the schema by hand invented its own key names (`impute`
    instead of `recommended_impute`, `strategy`/`train_fraction` instead of
    `method`/`ratios`) and then never wrote the file at all — which is precisely the class
    of failure a native removes.

    `decisions` maps column name -> dict with any of impute / scale / encode / rationale /
    flags (the `recommended_` prefix is optional). Unknown vocabulary values raise, loudly,
    rather than producing a manifest the applier will reject later. Measured statistics are
    filled in from `profile` / `quality` / `df` when available.
    """
    import json

    # Defined INSIDE the function on purpose: natives are injected by
    # `inspect.getsource`, and the opt-in fallback path injects the function ALONE
    # (NATIVE_SOURCE[name]), without any module-level constants. Referencing a module
    # global here worked under the always-on preamble and NameError'd under the
    # fallback — i.e. it broke in exactly the state the fallback exists for.
    _GADS_IMPUTE_VALUES = ("median", "mean", "mode", "constant", "forward_fill",
                           "drop_rows", "drop_column", None)
    _GADS_SCALE_VALUES = ("standard", "minmax", "robust", "log1p", "quantile_normal", None)
    _GADS_ENCODE_VALUES = ("onehot", "ordinal", "target", "frequency", None)

    # The profiling helpers are fallback-only, so they may not be defined in the kernel.
    # Their absence only costs the enrichment statistics — never the manifest itself.
    try:
        if df is not None and profile is None:
            profile = gads_profile_dataframe(df)
        if df is not None and quality is None:
            quality = gads_assess_quality(df, profile)
    except NameError:
        pass
    pcols = (profile or {}).get("columns", {})
    qcols = (quality or {}).get("columns", {})

    def _pick(d, *names):
        for n in names:
            if n in d and d[n] is not None:
                return d[n]
        return None

    columns = {}
    for name, raw in (decisions or {}).items():
        name = str(name)
        if not isinstance(raw, dict):
            raise ValueError(f"decisions['{name}'] must be a dict, got {type(raw).__name__}")
        impute = _pick(raw, "recommended_impute", "impute")
        scale = _pick(raw, "recommended_scale", "scale")
        encode = _pick(raw, "recommended_encode", "encode")
        for label, val, allowed in (("impute", impute, _GADS_IMPUTE_VALUES),
                                    ("scale", scale, _GADS_SCALE_VALUES),
                                    ("encode", encode, _GADS_ENCODE_VALUES)):
            if val not in allowed:
                raise ValueError(
                    f"column '{name}': {label}={val!r} is not one of "
                    f"{[a for a in allowed if a is not None]} (or null)")
        p = pcols.get(name, {})
        q = qcols.get(name, {})
        flags = _pick(raw, "flags", "quality_flags") or q.get("flags") or []
        columns[name] = {
            "dtype": str(_pick(raw, "dtype", "type") or p.get("dtype", "unknown")),
            "missing_rate": p.get("missing_rate"),
            "n_unique": p.get("n_unique"),
            "outlier_rate": q.get("outlier_rate"),
            "skew": p.get("skew"),
            "recommended_impute": impute,
            "recommended_scale": scale,
            "recommended_encode": encode,
            "flags": list(flags),
            "rationale": str(raw.get("rationale") or ""),
        }

    # Accept the obvious aliases for the split block rather than failing on wording.
    norm_split = None
    if split:
        method = _pick(split, "method", "strategy") or "random"
        ratios = split.get("ratios")
        if not ratios:
            ratios = {
                "train": float(_pick(split, "train_fraction", "train") or 0.7),
                "val": float(_pick(split, "validation_fraction", "val", "valid_fraction") or 0.15),
                "test": float(_pick(split, "test_fraction", "test") or 0.15),
            }
        norm_split = {
            "applies_to": split.get("applies_to") or source_file,
            "method": method,
            "ratios": ratios,
            "stratify_by": _pick(split, "stratify_by", "stratify_on"),
            "group_by": _pick(split, "group_by", "group_on"),
            "time_column": _pick(split, "time_column", "time_col"),
            "random_state": int(split.get("random_state", 42)),
            "rationale": str(split.get("rationale") or ""),
        }
        if method == "stratified" and not norm_split["stratify_by"]:
            norm_split["stratify_by"] = target_column
        valid = ("time_ordered", "grouped", "stratified", "random")
        if method not in valid:
            raise ValueError(f"split method {method!r} is not one of {list(valid)}")

    manifest = {
        "schema_version": "1.0",
        "generated_by": "gads_write_transformation_manifest",
        "target_column": str(target_column) if target_column else None,
        "split": norm_split,
        "files": {
            source_file: {
                "n_rows": (profile or {}).get("n_rows"),
                "n_cols": (profile or {}).get("n_cols"),
                "missing_cell_rate": (profile or {}).get("missing_cell_rate"),
                "columns": columns,
            }
        },
    }
    with open(write_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  [EDA] Wrote {write_path}: {len(columns)} columns, "
          f"split={(norm_split or {}).get('method', 'none')}")
    return manifest


def gads_apply_transformations(df, manifest, source_file=None, apply_flags=True,
                               reuse_params_from=None,
                               provenance_path="transformation_provenance.meta.json",
                               out_prefix="transformed"):
    """Apply a transformation manifest deterministically.

    THE ORDER IS THE POINT. When the manifest carries a `split` block, this splits FIRST,
    fits every imputer/scaler/encoder on the TRAINING partition only, then applies those
    fitted parameters to train, val and test. Fitting before splitting would leak held-out
    statistics (the median, the category map, the scaler centre would all have seen the
    test rows) — which is exactly the mistake this native exists to make impossible.

    Fitted values are written to `provenance_path`, so a later call can pass
    `reuse_params_from=<that file>` to transform another file with identical parameters
    instead of refitting.

    `apply_flags=True` (default) drops columns flagged `drop_suggested`; every drop is
    recorded with its causing flag. Returns a dict of counts and the written paths.
    """
    import json
    import numpy as np
    import pandas as pd

    # Defined INSIDE the function on purpose: natives are injected by
    # `inspect.getsource`, and the opt-in fallback path injects the function ALONE
    # (NATIVE_SOURCE[name]), without any module-level constants. Referencing a module
    # global here worked under the always-on preamble and NameError'd under the
    # fallback — i.e. it broke in exactly the state the fallback exists for.
    _GADS_IMPUTE_VALUES = ("median", "mean", "mode", "constant", "forward_fill",
                           "drop_rows", "drop_column", None)
    _GADS_SCALE_VALUES = ("standard", "minmax", "robust", "log1p", "quantile_normal", None)
    _GADS_ENCODE_VALUES = ("onehot", "ordinal", "target", "frequency", None)

    if isinstance(manifest, str):
        with open(manifest) as f:
            manifest = json.load(f)

    files = manifest.get("files") or {}
    if source_file is None:
        source_file = next(iter(files), None)
    spec = files.get(source_file) or {}
    col_specs = spec.get("columns") or {}
    if not col_specs:
        raise ValueError(f"manifest has no column spec for '{source_file}' "
                         f"(available: {list(files)})")

    for name, cs in col_specs.items():
        if cs.get("recommended_impute") not in _GADS_IMPUTE_VALUES:
            raise ValueError(f"{name}: bad recommended_impute {cs.get('recommended_impute')!r}")
        if cs.get("recommended_scale") not in _GADS_SCALE_VALUES:
            raise ValueError(f"{name}: bad recommended_scale {cs.get('recommended_scale')!r}")
        if cs.get("recommended_encode") not in _GADS_ENCODE_VALUES:
            raise ValueError(f"{name}: bad recommended_encode {cs.get('recommended_encode')!r}")

    params = {}
    if reuse_params_from:
        with open(reuse_params_from) as f:
            params = (json.load(f) or {}).get("fitted_params", {})
        print(f"  [EDA] Reusing fitted parameters from {reuse_params_from}")

    # Columns the split depends on must survive the drop pass — dropping the time column
    # of a time-ordered split would silently degrade it to "first N rows in whatever order
    # they happen to be", which looks like it worked and is not reproducible.
    _split = manifest.get("split") or {}
    split_columns = {_split.get(k) for k in ("time_column", "group_by", "stratify_by")}
    split_columns.discard(None)

    work = df.copy()
    dropped = []
    if apply_flags:
        for name, cs in col_specs.items():
            if name not in work.columns or name in split_columns:
                continue
            flags = cs.get("flags") or []
            if "drop_suggested" in flags or cs.get("recommended_impute") == "drop_column":
                cause = ", ".join(f for f in flags if f != "drop_suggested") or "drop_column"
                dropped.append({"column": name, "cause": cause})
                work = work.drop(columns=[name])
        for d in dropped:
            print(f"  [EDA] Dropped `{d['column']}` ({d['cause']})")

    # ---- split FIRST (leakage guard) -------------------------------------------------
    split = manifest.get("split")
    partitions = {}
    if split:
        method = split.get("method", "random")
        ratios = split.get("ratios") or {"train": 0.7, "val": 0.15, "test": 0.15}
        rs = int(split.get("random_state", 42))
        n = len(work)
        if method == "time_ordered":
            tcol = split.get("time_column")
            if tcol not in work.columns:
                raise ValueError(
                    f"time_ordered split needs `{tcol}`, which is not in the data. "
                    f"Refusing to fall back to row order — that would silently produce a "
                    f"non-chronological split that still looks correct.")
            ordered = work.sort_values(tcol)
            i_tr = int(n * ratios.get("train", 0.7))
            i_va = i_tr + int(n * ratios.get("val", 0.0))
            partitions = {"train": ordered.iloc[:i_tr], "val": ordered.iloc[i_tr:i_va],
                          "test": ordered.iloc[i_va:]}
        elif method == "grouped":
            gcol = split.get("group_by")
            if gcol not in work.columns:
                raise ValueError(
                    f"grouped split needs `{gcol}`, which is not in the data. Refusing to "
                    f"fall back to a random split — the same entity would land in both "
                    f"train and test.")
            groups = pd.Series(work[gcol].unique())
            groups = groups.sample(frac=1.0, random_state=rs)
            g_tr = int(len(groups) * ratios.get("train", 0.7))
            g_va = g_tr + int(len(groups) * ratios.get("val", 0.0))
            s_tr, s_va = set(groups[:g_tr]), set(groups[g_tr:g_va])
            partitions = {
                "train": work[work[gcol].isin(s_tr)],
                "val": work[work[gcol].isin(s_va)],
                "test": work[~work[gcol].isin(s_tr | s_va)],
            }
        elif method == "stratified":
            scol = split.get("stratify_by")
            if scol not in work.columns:
                raise ValueError(
                    f"stratified split needs `{scol}`, which is not in the data. Refusing "
                    f"to fall back to a random split — a minority class could be starved "
                    f"out of a partition without any warning.")
            shuffled = work.sample(frac=1.0, random_state=rs)
            tr_parts, va_parts, te_parts = [], [], []
            for _, grp in shuffled.groupby(scol, dropna=False):
                m = len(grp)
                i_tr = int(m * ratios.get("train", 0.7))
                i_va = i_tr + int(m * ratios.get("val", 0.0))
                tr_parts.append(grp.iloc[:i_tr])
                va_parts.append(grp.iloc[i_tr:i_va])
                te_parts.append(grp.iloc[i_va:])
            partitions = {"train": pd.concat(tr_parts), "val": pd.concat(va_parts),
                          "test": pd.concat(te_parts)}
        else:
            shuffled = work.sample(frac=1.0, random_state=rs)
            i_tr = int(n * ratios.get("train", 0.7))
            i_va = i_tr + int(n * ratios.get("val", 0.0))
            partitions = {"train": shuffled.iloc[:i_tr], "val": shuffled.iloc[i_tr:i_va],
                          "test": shuffled.iloc[i_va:]}
        partitions = {k: v for k, v in partitions.items() if len(v)}
        print("  [EDA] Split (" + str(split.get("method")) + "): "
              + ", ".join(f"{k}={len(v):,}" for k, v in partitions.items()))
    else:
        partitions = {"all": work}

    fit_key = "train" if "train" in partitions else next(iter(partitions))
    fit_df = partitions[fit_key]

    # ---- fit on the training partition only ------------------------------------------
    if not params:
        params = {}
        for name, cs in col_specs.items():
            if name not in fit_df.columns:
                continue
            s = fit_df[name]
            p = {}
            imp = cs.get("recommended_impute")
            if imp == "median":
                p["impute_value"] = float(s.median()) if s.notna().any() else 0.0
            elif imp == "mean":
                p["impute_value"] = float(s.mean()) if s.notna().any() else 0.0
            elif imp == "mode":
                m = s.mode(dropna=True)
                p["impute_value"] = (m.iloc[0] if len(m) else "")
            elif imp == "constant":
                p["impute_value"] = 0
            sc = cs.get("recommended_scale")
            clean = s.dropna()
            if sc == "standard" and len(clean):
                p["center"], p["spread"] = float(clean.mean()), float(clean.std() or 1.0)
            elif sc == "minmax" and len(clean):
                p["min"], p["max"] = float(clean.min()), float(clean.max())
            elif sc == "robust" and len(clean):
                q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
                p["center"], p["spread"] = float(clean.median()), float((q3 - q1) or 1.0)
            enc = cs.get("recommended_encode")
            if enc == "frequency" and len(clean):
                p["freq_map"] = {str(k): float(v) for k, v in
                                 clean.value_counts(normalize=True).items()}
            elif enc == "ordinal" and len(clean):
                p["ordinal_map"] = {str(v): i for i, v in enumerate(sorted(clean.unique().tolist()))}
            elif enc == "onehot" and len(clean):
                p["categories"] = [str(v) for v in sorted(clean.unique().tolist())]
            if p:
                params[name] = p

    # ---- apply the SAME fitted parameters to every partition -------------------------
    def _apply(frame):
        out = frame.copy()
        for name, cs in col_specs.items():
            if name not in out.columns:
                continue
            p = params.get(name, {})
            imp = cs.get("recommended_impute")
            if imp == "drop_rows":
                out = out[out[name].notna()]
            elif imp == "forward_fill":
                out[name] = out[name].ffill()
            elif imp in ("median", "mean", "mode", "constant") and "impute_value" in p:
                out[name] = out[name].fillna(p["impute_value"])
            sc = cs.get("recommended_scale")
            if sc == "log1p":
                out[name] = np.log1p(out[name].clip(lower=0))
            elif sc in ("standard", "robust") and "center" in p:
                out[name] = (out[name] - p["center"]) / (p["spread"] or 1.0)
            elif sc == "minmax" and "min" in p:
                rng = (p["max"] - p["min"]) or 1.0
                out[name] = (out[name] - p["min"]) / rng
            enc = cs.get("recommended_encode")
            if enc == "frequency" and "freq_map" in p:
                out[name] = out[name].astype(str).map(p["freq_map"]).fillna(0.0)
            elif enc == "ordinal" and "ordinal_map" in p:
                out[name] = out[name].astype(str).map(p["ordinal_map"]).fillna(-1).astype(int)
            elif enc == "onehot" and "categories" in p:
                for cat in p["categories"]:
                    out[f"{name}__{cat}"] = (out[name].astype(str) == cat).astype(int)
                out = out.drop(columns=[name])
        return out

    written = {}
    n_rows_out = 0
    n_cols_out = 0
    for part, frame in partitions.items():
        transformed = _apply(frame)
        n_cols_out = int(transformed.shape[1])   # post-transform (one-hot expands columns)
        suffix = "" if part == "all" else f"_{part}"
        path = f"{out_prefix}{suffix}.parquet"
        try:
            transformed.to_parquet(path, index=False)
        except Exception:
            path = f"{out_prefix}{suffix}.csv"
            transformed.to_csv(path, index=False)
        written[part] = path
        n_rows_out += len(transformed)
        print(f"  [EDA] Wrote {path} ({len(transformed):,} rows x {transformed.shape[1]} cols)")

    provenance = {
        "schema_version": "1.0",
        "source_file": source_file,
        "manifest_schema_version": manifest.get("schema_version"),
        "apply_flags": bool(apply_flags),
        "columns_dropped": dropped,
        "split": split,
        "fit_partition": fit_key,
        "fitted_params": params,
        "outputs": written,
    }
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    print(f"  [EDA] Wrote {provenance_path} "
          f"(fitted on '{fit_key}', {len(params)} column parameter set(s))")

    return {
        "n_rows_in": int(len(df)),
        "n_rows_out": int(n_rows_out),
        "n_cols_in": int(df.shape[1]),
        "n_cols_out": n_cols_out,
        "columns_dropped": [d["column"] for d in dropped],
        "columns_transformed": len([c for c in col_specs
                                    if c not in [d["column"] for d in dropped]]),
        "outputs": written,
        "provenance_path": provenance_path,
        "n_train_rows": int(len(partitions.get("train", []))) if split else None,
        "n_test_rows": int(len(partitions.get("test", []))) if split else None,
    }


def gads_eda_summary(profile=None, quality=None, manifest=None,
                     write_path="eda_summary.md"):
    """Deterministic EDA narrative — the fallback for the reporting node, which is a
    documented local-model failure mode. Writes Markdown and returns the text."""
    lines = []
    if profile:
        lines.append(f"# Exploratory Data Analysis\n")
        lines.append(f"The dataset has **{profile['n_rows']:,} rows** and "
                     f"**{profile['n_cols']} columns**, with an overall missing-cell rate of "
                     f"**{profile['missing_cell_rate']:.2%}**.\n")
    if quality:
        lines.append(f"## Data quality\n")
        lines.append(f"- Duplicate rows: **{quality['duplicate_rows']:,}**")
        lines.append(f"- Columns carrying a quality flag: **{quality['n_flagged_columns']}**")
        by_flag = {}
        for col, q in (quality.get("columns") or {}).items():
            for f in q.get("flags", []):
                by_flag.setdefault(f, []).append(col)
        for flag, cols in sorted(by_flag.items()):
            lines.append(f"- `{flag}`: {', '.join('`%s`' % c for c in cols[:8])}"
                         + (f" (+{len(cols) - 8} more)" if len(cols) > 8 else ""))
        lines.append("")
    if manifest:
        split = manifest.get("split")
        if split:
            lines.append("## Recommended split\n")
            lines.append(f"**{split.get('method')}** — {split.get('rationale', '')}\n")
        lines.append("## Recommended transformations\n")
        lines.append("| column | impute | scale | encode | flags |")
        lines.append("|---|---|---|---|---|")
        for fname, fspec in (manifest.get("files") or {}).items():
            for col, cs in (fspec.get("columns") or {}).items():
                lines.append(
                    f"| `{col}` | {cs.get('recommended_impute') or '—'} | "
                    f"{cs.get('recommended_scale') or '—'} | "
                    f"{cs.get('recommended_encode') or '—'} | "
                    f"{', '.join(cs.get('flags') or []) or '—'} |")
        lines.append("")
    text = "\n".join(lines)
    with open(write_path, "w") as f:
        f.write(text)
    print(f"  [EDA] Wrote {write_path} ({len(text)} chars)")
    return text
