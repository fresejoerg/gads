"""
GADS Native Recommendation Nodes

Reusable, general-purpose collaborative-filtering primitives injected into the sandbox
preamble. They mechanize the parts LLMs re-author (and get wrong) run-to-run — sparse
matrix construction, temporal leave-one-out, ALS fit, top-N recommend, and top-N
evaluation with CORRECT index alignment (the recommend-output ↔ holdout ↔ item-index
bookkeeping that drove 0.129-vs-0.035 variance).

Design: general primitives parameterized by column names / method / K, plus one
orchestrator. Nothing here is dataset-specific or a patch for a single error case.

Functions are annotation-free and self-contained (imports inside) so their source can be
injected verbatim into the sandbox kernel via the preamble.
"""


def gads_dense_core_sample(df, user_col, item_col, max_rows=200000, min_interactions=5):
    """Down-sample an interaction log while PRESERVING density.

    Random row sampling is wrong for collaborative filtering and quietly destroys the data:
    interaction logs are long-tailed (most users touch 1-2 items), so a random subset shares
    almost no users or items, and the k-core filter that follows then collapses the matrix to
    near-nothing (observed: an 800k-review Amazon log reduced to a 10x13 matrix). Instead,
    keep the densest core: retain the most active users and the items they actually share.

    Deterministic (ranking by interaction count, ties broken by key). Returns the reduced
    frame; a no-op when the data already fits.
    """
    d = df.dropna(subset=[user_col, item_col]).drop_duplicates([user_col, item_col], keep="last")
    if max_rows is None or len(d) <= max_rows:
        return d

    n_before = len(d)
    # Spend the row budget ONLY on users who could survive the k-core: a user with fewer
    # than `min_interactions` rows is dropped by the filter below anyway, so including them
    # consumes budget and contributes nothing. (Skipping this step is what still produced an
    # empty core on a heavy-tailed log — the budget filled up with one-off users.)
    counts_all = d[user_col].value_counts()
    eligible = counts_all[counts_all >= min_interactions].index
    if len(eligible) > 0:
        d = d[d[user_col].isin(set(eligible))]

    # Rank users by activity, then take the smallest prefix of users whose interactions fit
    # the budget. Sorting by (-count, key) keeps it stable across runs.
    counts = d[user_col].value_counts()
    order = sorted(counts.index, key=lambda u: (-int(counts[u]), str(u)))
    cum, keep_users = 0, []
    for u in order:
        c = int(counts[u])
        if cum + c > max_rows and keep_users:
            break
        keep_users.append(u)
        cum += c
    d = d[d[user_col].isin(set(keep_users))]

    # Re-apply k-core: dropping users orphans items (and vice versa), so mutual support has
    # to be re-established or the matrix is dense in rows but ragged in columns.
    while True:
        n0 = len(d)
        uc = d[user_col].value_counts()
        d = d[d[user_col].isin(uc[uc >= min_interactions].index)]
        ic = d[item_col].value_counts()
        d = d[d[item_col].isin(ic[ic >= min_interactions].index)]
        if len(d) == n0 or len(d) == 0:
            break
    print(f"[gads_dense_core_sample] {n_before:,} -> {len(d):,} interactions "
          f"({d[user_col].nunique():,} users x {d[item_col].nunique():,} items) "
          f"— densest core kept, NOT a random sample")
    return d


def gads_build_interaction_matrix(df, user_col, item_col, rating_col=None, min_interactions=5,
                                  max_rows=None, min_interactions_floor=2):
    """Build a sparse user x item CSR from an interaction log.

    Iterative k-core filtering to `min_interactions` on both users and items, contiguous
    index maps, binary implicit signal. Returns a `bundle` dict the other native rec
    functions extend. General: works for any user-item interaction table.

    `max_rows` applies DENSE-CORE down-sampling (see gads_dense_core_sample) instead of the
    random row cap used elsewhere in GADS — random sampling of a long-tailed interaction log
    destroys the co-occurrence structure collaborative filtering depends on.
    """
    import numpy as np
    from scipy.sparse import csr_matrix

    d0 = df.dropna(subset=[user_col, item_col]).copy()
    d0 = d0.drop_duplicates([user_col, item_col], keep="last")

    def _kcore(frame, k):
        """Iterative k-core: users and items each need >= k interactions, mutually."""
        f = frame
        while True:
            n0 = len(f)
            uc = f[user_col].value_counts()
            f = f[f[user_col].isin(uc[uc >= k].index)]
            ic = f[item_col].value_counts()
            f = f[f[item_col].isin(ic[ic >= k].index)]
            if len(f) == n0 or len(f) == 0:
                return f

    # ADAPTIVE CORE: the requested k is often infeasible on real interaction logs. Amazon
    # Fashion averages 1.10 interactions per user — its 3-core is empty, while its 2-core is a
    # usable 4,819 x 3,905 matrix. Rather than failing on a legitimate (if sparse) dataset,
    # step k down to `min_interactions_floor` and use the strictest core that actually
    # survives, reporting which one was used. Deterministic; a no-op when k works as asked.
    k_used, d = None, None
    for k in range(int(min_interactions), int(min_interactions_floor) - 1, -1):
        cand = _kcore(d0, k)
        if len(cand) > 0:
            k_used, d = k, cand
            break
    if d is None or len(d) == 0:
        d, k_used = d0.iloc[0:0], int(min_interactions_floor)
    elif k_used != int(min_interactions):
        print(f"[gads_build_interaction_matrix] {min_interactions}-core is empty for this data; "
              f"relaxed to a {k_used}-core (mean {len(d0)/max(1, d0[user_col].nunique()):.2f} "
              f"interactions/user — a sparse log, so treat absolute metrics with care)")

    # Size reduction happens AFTER the feasible core is known, so the budget is spent on
    # interactions that can actually survive; then re-core, since sampling changes degrees.
    if max_rows is not None and len(d) > max_rows:
        d = gads_dense_core_sample(d, user_col, item_col, max_rows=max_rows,
                                   min_interactions=k_used)
        d = _kcore(d, k_used)
    if len(d) == 0:
        raise ValueError(
            f"[gads_build_interaction_matrix] no usable core: even a "
            f"{min_interactions_floor}-core is empty (tried {min_interactions} down to "
            f"{min_interactions_floor}). Two causes, in order of likelihood: (1) the frame was "
            f"RANDOMLY down-sampled upstream (df.sample(...)), which destroys the co-occurrence "
            f"structure of a long-tailed interaction log — pass the FULL frame and use max_rows= "
            f"here instead; (2) the data genuinely has no repeat interactions, in which case "
            f"collaborative filtering is not applicable and a content-based approach is needed."
        )
    users = {u: i for i, u in enumerate(d[user_col].unique())}
    items = {it: j for j, it in enumerate(d[item_col].unique())}
    d = d.assign(_u=d[user_col].map(users), _i=d[item_col].map(items))
    matrix = csr_matrix((np.ones(len(d)), (d["_u"], d["_i"])), shape=(len(users), len(items)))
    print(f"[gads_build_interaction_matrix] {len(d):,} interactions, {len(users):,} users x "
          f"{len(items):,} items, density {len(d)/(len(users)*len(items)):.4%}")
    return {"matrix": matrix, "user_index": users, "item_index": items, "df": d,
            "user_col": user_col, "item_col": item_col, "rating_col": rating_col,
            "n_users": len(users), "n_items": len(items), "min_interactions_used": k_used}


def gads_temporal_loo_split(bundle, time_col=None, random_state=42):
    """Temporal leave-one-out: hold out each user's most recent interaction (by
    `time_col`; random if absent) for users with >= 2 interactions. Adds `train_matrix`
    (binary) and `holdout` {user_idx: item_idx} to the bundle."""
    import numpy as np
    from scipy.sparse import csr_matrix

    d = bundle["df"]
    d = d.sort_values(time_col) if (time_col and time_col in d.columns) else d.sample(frac=1, random_state=random_state)
    cnt = d.groupby("_u").size()
    evaluable = cnt[cnt >= 2].index
    last = d[d["_u"].isin(evaluable)].groupby("_u").tail(1)
    holdout = {int(u): int(i) for u, i in zip(last["_u"], last["_i"])}
    train = d.drop(last.index)
    train_matrix = csr_matrix((np.ones(len(train)), (train["_u"], train["_i"])),
                              shape=(bundle["n_users"], bundle["n_items"]))
    print(f"[gads_temporal_loo_split] {len(holdout):,} evaluable users; train nnz={train_matrix.nnz:,}")
    out = dict(bundle); out["train_matrix"] = train_matrix; out["holdout"] = holdout
    return out


def gads_fit_and_recommend(bundle, method="als", factors=64, iterations=15,
                           regularization=0.05, alpha=40, N=20, random_state=42):
    """Fit an implicit recommender and generate top-N per holdout user, excluding seen
    items. `method='als'` uses implicit ALS (falls back to item-item cosine on
    ImportError); `method='cosine'` forces cosine. Recommendations are index-aligned by
    construction: recommend is called for np.arange(n_users) against the full train
    matrix, so row u is user u's list. Adds `recommendations` + `model_name`."""
    import numpy as np

    train = bundle["train_matrix"]
    n_users = bundle["n_users"]
    holdout_users = list(bundle["holdout"].keys())
    topn = None
    model_name = None

    if method == "als":
        try:
            from implicit.als import AlternatingLeastSquares
            model = AlternatingLeastSquares(factors=factors, regularization=regularization,
                                            iterations=iterations, random_state=random_state)
            model.fit((train * alpha).astype("float32"), show_progress=False)  # confidence-weighted
            ids, _ = model.recommend(np.arange(n_users), train, N=N, filter_already_liked_items=True)
            topn = {u: [int(x) for x in ids[u]] for u in holdout_users}
            model_name = "implicit-ALS"
        except ImportError:
            print("[gads_fit_and_recommend] `implicit` not installed — falling back to item-item cosine")
            method = "cosine"

    if topn is None:
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(train.T, dense_output=False)     # item x item
        scores = (train @ sim).toarray()                         # user x item
        scores[train.nonzero()] = -np.inf                        # exclude seen
        order = np.argsort(-scores, axis=1)[:, :N]
        topn = {u: [int(x) for x in order[u]] for u in holdout_users}
        model_name = "item-item-cosine"

    print(f"[gads_fit_and_recommend] model={model_name}, top-{N} for {len(topn):,} users")
    out = dict(bundle); out["recommendations"] = topn; out["model_name"] = model_name
    return out


def gads_evaluate_topn(bundle, k_values=(10, 20), write_metrics=True):
    """Recall@K / NDCG@K / HitRate@K over the held-out items vs a most-popular baseline.
    Correct alignment: recommendations and holdout are both in the item-index space.
    Writes metrics.json and returns a flat metrics dict (recall_at_10, ndcg_at_10, ...)."""
    import numpy as np, json

    holdout = bundle["holdout"]
    topn = bundle["recommendations"]
    train = bundle["train_matrix"]
    n = len(holdout)
    pop = np.asarray(train.sum(axis=0)).ravel().argsort()[::-1]  # most-popular item indices

    metrics = {}
    for k in k_values:
        hits = 0
        ndcg = 0.0
        for u, true_i in holdout.items():
            rec = topn.get(u, [])[:k]
            if true_i in rec:
                hits += 1
                ndcg += 1.0 / np.log2(rec.index(true_i) + 2)
        pop_k = set(int(x) for x in pop[:k])
        metrics[f"recall_at_{k}"] = hits / n if n else 0.0
        metrics[f"ndcg_at_{k}"] = ndcg / n if n else 0.0
        metrics[f"hit_rate_at_{k}"] = hits / n if n else 0.0
        metrics[f"popularity_recall_at_{k}"] = float(np.mean([i in pop_k for i in holdout.values()])) if n else 0.0

    r10, p10 = metrics.get("recall_at_10"), metrics.get("popularity_recall_at_10")
    if r10 is not None and p10 is not None:
        metrics["lift_over_popularity"] = (r10 / p10) if p10 > 0 else float("inf")
    metrics["n_users_evaluated"] = n

    if write_metrics:
        with open("metrics.json", "w") as f:
            json.dump({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, f, indent=2)
    print("[gads_evaluate_topn] " + "  ".join(
        f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)))
    return metrics


def gads_recommend_and_evaluate(df, user_col, item_col, rating_col=None, time_col=None,
                                method="als", N=20, k_values=(10, 20), min_interactions=5):
    """One-call CF pipeline: build -> temporal LOO -> fit+recommend -> evaluate. Returns
    {metrics, recommendations, bundle}. The mechanical core (esp. index alignment) is
    deterministic, so the result no longer varies with how the surrounding code is written."""
    b = gads_build_interaction_matrix(df, user_col, item_col, rating_col=rating_col,
                                      min_interactions=min_interactions)
    b = gads_temporal_loo_split(b, time_col=time_col)
    b = gads_fit_and_recommend(b, method=method, N=N)
    metrics = gads_evaluate_topn(b, k_values=k_values)
    return {"metrics": metrics, "recommendations": b["recommendations"], "bundle": b}


def gads_characterize_recommendations(bundle, metrics=None, n_examples=3,
                                      write_path="recommendation_profile.json",
                                      emit_insights=True):
    """Describe what the recommender actually recommends: example users, coverage, honesty.

    FALLBACK-ONLY native (issue #30). Characterization is legitimately variable work, so the
    node stays model-generated by default and this exists to rescue it when the local model
    exhausts its retries — the same demote-to-fallback treatment the survival plotting nodes
    got (approach_docs/019). It is therefore NOT part of RECOMMENDATION_PREAMBLE.

    Reports, for `n_examples` users spanning the recommendation set: their training history
    and their top recommendations (as original ids), plus catalog coverage — the fraction of
    the item catalog ever recommended, which is how you detect a model that just pushes the
    same popular handful at everyone. If `metrics` carries `lift_over_popularity` < 1 it says
    plainly that the model does not beat a most-popular baseline. Fail-open.
    """
    import json

    result = {"examples": [], "catalog_coverage": None, "n_users_with_recs": 0,
              "beats_popularity": None, "profile_path": write_path}
    try:
        recs = bundle.get("recommendations") or {}
        n_items = int(bundle.get("n_items") or 0)
        rev_user = {v: k for k, v in (bundle.get("user_index") or {}).items()}
        rev_item = {v: k for k, v in (bundle.get("item_index") or {}).items()}
        result["n_users_with_recs"] = len(recs)

        # Catalog coverage: distinct recommended items / catalog size.
        distinct = set()
        for lst in recs.values():
            distinct.update(int(i) for i in lst)
        if n_items:
            result["catalog_coverage"] = round(len(distinct) / n_items, 4)

        # Training history per user, read from the train matrix when present.
        train = bundle.get("train_matrix")
        picks = sorted(recs.keys())[:: max(1, len(recs) // max(1, n_examples))][:n_examples] if recs else []
        for u in picks:
            hist = []
            try:
                if train is not None:
                    row = train[int(u)]
                    hist = [rev_item.get(int(j), int(j)) for j in row.indices[:10]]
            except Exception:
                pass
            result["examples"].append({
                "user": rev_user.get(int(u), int(u)),
                "history": [str(h) for h in hist],
                "recommended": [str(rev_item.get(int(i), int(i))) for i in list(recs[u])[:10]],
            })

        lift = None
        if isinstance(metrics, dict):
            lift = metrics.get("lift_over_popularity")
        if lift is not None:
            result["beats_popularity"] = bool(lift >= 1.0)

        with open(write_path, "w") as f:
            json.dump({k: v for k, v in result.items() if k != "profile_path"}, f, indent=2)

        cov = result["catalog_coverage"]
        print(f"[gads_characterize_recommendations] {result['n_users_with_recs']:,} users with "
              f"recommendations; catalog coverage "
              f"{'n/a' if cov is None else f'{cov:.2%}'} of {n_items:,} items")
        for ex in result["examples"]:
            print(f"  user {ex['user']}: history={ex['history'][:3]} -> top recs={ex['recommended'][:3]}")

        if emit_insights:
            emit = globals().get("gads_emit_insight")
            if callable(emit):
                try:
                    emit("recommendation_profile.json",
                         f"The recommender covers {'n/a' if cov is None else f'{cov:.1%}'} of the "
                         f"item catalog across {result['n_users_with_recs']:,} users; low coverage "
                         f"means it concentrates on a popular few rather than personalizing.",
                         evidence=json.dumps(result["examples"][:1])[:300])
                except Exception:
                    pass
                if result["beats_popularity"] is False:
                    try:
                        emit("metrics.json",
                             f"The model does NOT beat a most-popular baseline "
                             f"(lift_over_popularity={lift:.2f} < 1.0), so it adds no "
                             f"personalization value on this data.", evidence=f"lift={lift}")
                    except Exception:
                        pass
        return result
    except Exception as e:
        print(f"[gads_characterize_recommendations] failed ({type(e).__name__}: {e}); continuing")
        result["error"] = f"{type(e).__name__}: {e}"
        return result
