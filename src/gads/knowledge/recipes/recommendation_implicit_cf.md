---
id: recommendation.implicit.collaborative_filtering
version: 1.0.0
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
  capabilities: [scipy, sklearn, pandas, numpy]

# ——— EXECUTION DAG ———
dag:
  - id: build_interaction_matrix
    intent: "Identify the user-id, item-id, and (optional) rating/interaction-strength and timestamp columns. Deduplicate user-item pairs (keep the most recent). Filter out cold users and items with fewer than 5 interactions to control sparsity. Map users and items to contiguous integer indices and build a scipy CSR sparse user × item matrix (implicit feedback: entry = 1 for an interaction, or a confidence 1 + alpha*rating if ratings exist). Print the matrix shape, #users, #items, and sparsity. Store the matrix and the index↔id maps."
    worker_tier: T2
    attached_skills: [implicit_cf_recommender, large_dataset_handling]
    produces: [interaction_matrix, user_index, item_index]
    postconditions:
      - "interaction_matrix is not None"

  - id: temporal_leave_one_out_split
    intent: "For every user with at least 2 interactions, hold out their MOST RECENT interaction (by timestamp; if no timestamp, a fixed-seed random one) as the test item, keeping the rest as training. Build the training matrix (test interactions removed) and a dict {user: held_out_item}. Print the number of evaluable users."
    worker_tier: T2
    depends_on: [build_interaction_matrix]
    attached_skills: [implicit_cf_recommender]
    produces: [train_matrix, test_holdout]
    postconditions:
      - "len(test_holdout) > 0"

  - id: fit_and_recommend
    intent: "Fit an implicit-feedback collaborative-filtering model on the training matrix. The DEFAULT and required model is implicit.als.AlternatingLeastSquares (factors=64, regularization=0.05, iterations=15, random_state=42) — the industry-standard implicit matrix factorization; the `implicit` library IS installed in this environment. Write `from implicit.als import AlternatingLeastSquares` inside a try/except and fall back to item-item cosine similarity (sklearn) ONLY if that import raises ImportError. Print a line stating which model actually ran ('implicit-ALS' or 'cosine-fallback'). Note the `implicit` API: fit takes a user×item CSR of confidence weights, and `model.recommend(user_ids, train_matrix, N=20, filter_already_liked_items=True)` already excludes seen items. Generate the top-20 recommended items for each evaluable user, EXCLUDING items already seen in training. Store recommendations {user: [item,...]}."
    worker_tier: T2
    depends_on: [temporal_leave_one_out_split]
    attached_skills: [implicit_cf_recommender]
    produces: [recommendations]
    postconditions:
      - "recommendations is not None"

  - id: evaluate_topn
    intent: "Evaluate the held-out items against the top-N lists: compute Recall@10, NDCG@10, and HitRate@10 (K=10; also report @20). Compute a POPULARITY baseline (recommend the globally most-popular unseen items) and report the lift of CF over it. IMPORTANT: bind `recall_at_10` and `ndcg_at_10` as TOP-LEVEL float variables of exactly those names (the execution contract probes the kernel for them by name — a dict entry or metrics.json alone does NOT satisfy it). Also write metrics.json with recall_at_10, ndcg_at_10, hit_rate_at_10, popularity_recall_at_10. Emit an insight stating Recall@10 and the lift over popularity."
    worker_tier: T2
    depends_on: [fit_and_recommend]
    attached_skills: [implicit_cf_recommender]
    required_metrics: [recall_at_10, ndcg_at_10]
    postconditions:
      - "'recall_at_10' in open('metrics.json').read()"

  - id: characterize_recommendations
    intent: "Inspect 2–3 example users: show their training history and their top recommendations to sanity-check relevance. Report catalog coverage (fraction of items ever recommended) as a diversity signal. Emit an insight on recommendation quality and coverage."
    worker_tier: T2
    depends_on: [fit_and_recommend]
    postconditions:
      - "recommendations is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "DEFAULT MODEL IS ALS: use implicit.als.AlternatingLeastSquares as the recommender; the `implicit` library is installed. Fall back to item-item cosine ONLY when `import implicit` raises ImportError, and print which model ran."
  - "IMPLICIT FEEDBACK: an interaction is a positive signal; absence is NOT a confirmed negative. Do not train as if unobserved user-item pairs are labelled negatives."
  - "TEMPORAL LEAVE-ONE-OUT: hold out each user's most recent interaction. Never let a user's future interaction leak into their training profile."
  - "Exclude already-seen (training) items from a user's recommendation list — re-recommending known items is not a hit."
  - "ALWAYS compare against a most-popular baseline. A collaborative filter that does not beat popularity has added no personalization value — report this honestly."
  - "Rank/quality metrics (Recall@K, NDCG@K, HitRate@K) are computed over held-out items, averaged across users. Fixed random seed for any sampling."
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
