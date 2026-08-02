---
id: recommendation.implicit.collaborative_filtering
version: 2.0.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [recommendation, collaborative_filtering, top_n_recommendation]
  data_modality: [tabular]
  signals:
    - user_item_interactions: true
    - implicit_feedback: true
    - objective_contains: [recommend, recommendation, suggest items, top-n, users who, personalize, what to show]
  anti_signals:
    - task: classification
    - task: learning_to_rank
    - single_target_column: true

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [scipy, sklearn, pandas, numpy, implicit]

# ——— EXECUTION DAG (mechanized: calls gads_* native nodes, D5) ———
dag:
  - id: build_interaction_matrix
    intent: "Identify the user-id, item-id, (optional) rating, and (optional) timestamp columns in `df`. Call the native `bundle = gads_build_interaction_matrix(df, user_col=<user col>, item_col=<item col>, rating_col=<rating col or None>, min_interactions=5, max_rows=200000)` — pass the ACTUAL column names. It k-core-filters, builds the sparse user × item CSR, and returns a `bundle` dict with the matrix and index maps (it prints shape/sparsity). Do NOT re-implement the matrix construction — just call it with the right columns. CRITICAL: pass the FULL dataframe and let `max_rows` handle any size reduction — NEVER call `df.sample(...)` or otherwise randomly subset the interactions first. Interaction logs are long-tailed, so a random subset shares almost no users or items and the k-core filter then collapses the matrix to near-nothing; `max_rows` does density-preserving dense-core sampling instead."
    worker_tier: T2
    attached_skills: [implicit_cf_recommender]
    produces: [bundle]
    postconditions:
      - "bundle is not None"

  - id: temporal_leave_one_out_split
    intent: "Call `bundle = gads_temporal_loo_split(bundle, time_col=<timestamp col or None>)`. It holds out each user's most recent interaction (temporal leave-one-out) and adds `train_matrix` and `holdout` {user_idx: item_idx} to the bundle. Do not re-implement the split."
    worker_tier: T2
    depends_on: [build_interaction_matrix]
    attached_skills: [implicit_cf_recommender]
    produces: [bundle]
    postconditions:
      - "'holdout' in bundle"

  - id: fit_and_recommend
    intent: "Call `bundle = gads_fit_and_recommend(bundle, method='als', N=20)`. It fits implicit ALS (the installed `implicit` library; cosine fallback only on ImportError) and generates index-aligned top-20 recommendations per holdout user, excluding seen items. It prints which model ran. Do not re-implement the fit or the recommend/index bookkeeping."
    worker_tier: T2
    depends_on: [temporal_leave_one_out_split]
    attached_skills: [implicit_cf_recommender]
    produces: [bundle]
    postconditions:
      - "'recommendations' in bundle"

  - id: evaluate_topn
    intent: "Call `metrics = gads_evaluate_topn(bundle, k_values=(10, 20))`. It computes Recall@K / NDCG@K / HitRate@K over the held-out items against a most-popular baseline (correct index alignment) and writes metrics.json, returning a flat dict. Then bind the contract-required TOP-LEVEL scalars by exactly these names: `recall_at_10 = metrics['recall_at_10']` and `ndcg_at_10 = metrics['ndcg_at_10']`. Emit an insight stating Recall@10 and the lift over popularity (`metrics['lift_over_popularity']`)."
    worker_tier: T2
    depends_on: [fit_and_recommend]
    attached_skills: [implicit_cf_recommender]
    required_metrics: [recall_at_10, ndcg_at_10]
    postconditions:
      - "'recall_at_10' in open('metrics.json').read()"

  - id: characterize_recommendations
    intent: "Using `bundle['recommendations']` and the index maps in `bundle`, inspect 2–3 example users: show their training history vs their top recommendations to sanity-check relevance. Report catalog coverage (fraction of items ever recommended). Emit an insight on recommendation quality and coverage."
    worker_tier: T2
    depends_on: [fit_and_recommend]
    postconditions:
      - "'recommendations' in bundle"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "MECHANIZED CORE (D5): the sparse-matrix build, temporal LOO split, ALS fit, top-N recommend, and Recall/NDCG evaluation are provided by the gads_* native functions (injected into the kernel). CALL them with the correct column names — do NOT re-implement the matrix, the recommend index bookkeeping, or the metric math. Re-implementing them is what caused run-to-run result variance."
  - "DEFAULT MODEL IS ALS: gads_fit_and_recommend uses implicit ALS by default; cosine is an ImportError-only fallback."
  - "IMPLICIT FEEDBACK: an interaction is a positive signal; absence is NOT a confirmed negative."
  - "TEMPORAL LEAVE-ONE-OUT + exclude already-seen items: handled inside the native functions; do not bypass them."
  - "ALWAYS report the lift over the most-popular baseline (gads_evaluate_topn computes it). A CF model that does not beat popularity has added no personalization value."
---

# Top-N Recommendation with Implicit-Feedback Collaborative Filtering

## Rationale
Most production recommenders learn from **implicit feedback** — clicks, views,
purchases, plays — not explicit ratings. The dominant, dependency-light baseline is
**collaborative filtering** over a sparse user × item matrix: either **item-item
similarity** (Amazon's classic "customers who bought…") or **matrix factorization**
(latent factors; ALS is the canonical implicit variant). These are the reference
methods on the standard recommendation benchmarks — **MovieLens (100K/1M/25M)**,
**Amazon Product Reviews** (He & McAuley), **RetailRocket**, **Yelp**, **Last.fm** —
and remain strong baselines any deep model must beat.

## When to use
The data is **user-item interactions** (one row per user touching an item, optionally
with a rating and timestamp) and the goal is to recommend items each user hasn't seen.
Not to be confused with:
- **learning-to-rank** — that needs explicit query groups + graded relevance labels;
- **classification** — a single supervised target per row.

## Key mechanics
- **Matrix.** CSR sparse user × item; implicit confidence weighting optional.
- **Split.** Temporal leave-one-out (most recent interaction per user held out).
- **Model.** ALS (`implicit`, preferred) — the standard implicit MF — with item-item
  cosine kNN / TruncatedSVD as the dependency-free fallback.
- **Metrics.** Recall@K, NDCG@K, HitRate@K vs a **most-popular baseline**.

## Alternatives
- **BPR** (also in the `implicit` library) is the pairwise-ranking implicit MF variant.
- **Two-tower / neural retrieval** is the modern deep approach at large scale.
- **Content-based / hybrid** helps the cold-start users this CF baseline filters out.
