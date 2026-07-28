---
id: learning_to_rank.tabular.lambdamart
version: 1.0.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [learning_to_rank, ranking, search_ranking]
  data_modality: [tabular]
  signals:
    - query_groups_present: true
    - graded_relevance: true
    - objective_contains: [rank, ranking, relevance, ndcg, search results, order results, top results]
  anti_signals:
    - task: classification
    - task: recommendation
    - single_row_per_entity: true

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [lightgbm, pandas, numpy, sklearn]

# ——— EXECUTION DAG ———
dag:
  - id: prepare_ltr_data
    intent: "Identify the query/group id column (qid), the graded relevance label (integer, higher = more relevant), and the numeric feature columns. Split into train/test BY QUERY (assign whole qids to one side — a query must never have documents in both train and test). Sort each split by qid so rows are grouped, and build the LightGBM `group` arrays (number of documents per qid, in order). Store X_train, y_train, group_train, X_test, y_test, group_test, qid_test. Print #queries and #documents per split and the relevance-label distribution."
    worker_tier: T2
    attached_skills: [learning_to_rank_lightgbm]
    produces: [X_train, y_train, group_train, X_test, y_test, group_test]
    postconditions:
      - "sum(group_train) == len(X_train)"
      - "len(y_train) == len(X_train)"

  - id: train_lambdamart
    intent: "Fit a LambdaMART ranker: lightgbm.LGBMRanker(objective='lambdarank', metric='ndcg', n_estimators=300, learning_rate=0.05, random_state=42). Pass group=group_train to .fit(), and eval_set/eval_group with eval_at=[5,10] for early monitoring. Store the fitted model in `ranker`."
    worker_tier: T2
    depends_on: [prepare_ltr_data]
    attached_skills: [learning_to_rank_lightgbm]
    produces: [ranker]
    required_metrics: [ndcg_at_10]
    postconditions:
      - "ranker is not None"

  - id: evaluate_ranking
    intent: "Score the test set and compute ranking metrics PER QUERY, then average across queries: NDCG@5, NDCG@10, MAP, and MRR. Also compute a baseline (documents in their original/random order, or ordered by the single strongest raw feature) and report the lift. Write metrics.json with keys ndcg_at_5, ndcg_at_10, map, mrr, baseline_ndcg_at_10. Emit an insight stating the NDCG@10 and how much it beats the baseline."
    worker_tier: T2
    depends_on: [train_lambdamart]
    attached_skills: [learning_to_rank_lightgbm]
    required_metrics: [ndcg_at_5, ndcg_at_10]
    postconditions:
      - "'ndcg_at_10' in open('metrics.json').read()"

  - id: inspect_ranking
    intent: "Report the top feature importances by gain, and for one example test query print its documents ordered by predicted score alongside their true relevance labels — so the ranking is legible, not just a number. Emit an insight on which features drive relevance."
    worker_tier: T2
    depends_on: [train_lambdamart]
    attached_skills: [visualization_best_practices]
    postconditions:
      - "ranker is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "GROUP-CONSISTENT SPLIT: split by query id, never shuffle documents across the train/test boundary of a query. A leaked query inflates NDCG and invalidates the result."
  - "The LightGBM `group` array must list the document count per query IN THE SAME ROW ORDER as the feature matrix, and must sum to the number of rows."
  - "NDCG@k is the primary metric and must be computed per query then averaged — never as a single global score over pooled documents."
  - "Graded relevance labels (e.g. 0–4) are expected; if labels are binary the pipeline still works but say so, and prefer MAP/MRR alongside NDCG."
  - "Fixed random seed (random_state=42)."
  - "Always report the lift over a trivial baseline — a ranker that does not beat feature-1 ordering has learned nothing."
---

# Learning to Rank with LambdaMART

## Rationale
**LambdaMART** — gradient-boosted regression trees trained with the LambdaRank
listwise objective — is the dominant production pattern for tabular/feature-based
ranking (web search, e-commerce result ordering, ad ranking). It directly optimizes a
smoothed NDCG surrogate, handles heterogeneous query-document features, and is robust
without heavy tuning. This is the method behind the classic LTR benchmarks
(**MSLR-WEB10K/30K**, **Yahoo! Learning to Rank Challenge**, **LETOR 4.0**,
**Istella**), all of which share the `qid` + graded-relevance + dense-feature schema
this recipe consumes.

## When to use
Documents are grouped under queries (a `qid`), each with a graded **relevance** label,
and the goal is to *order* documents within each query. This is distinct from:
- **classification/regression** — those score a row in isolation, with no query grouping;
- **recommendation** — no explicit query/relevance labels; use
  `recommendation.implicit.collaborative_filtering` for user-item interaction data.

## Key mechanics
- **Split by query.** Assign whole queries to train or test. Sort by `qid` and build the
  `group` array (docs per query). `sum(group) == len(X)`.
- **Model.** `LGBMRanker(objective='lambdarank', metric='ndcg', eval_at=[5,10])`.
- **Metrics.** NDCG@5/@10 (primary), MAP, MRR — computed per query, then averaged.
- **Baseline.** Compare to the documents' original order or single-best-feature ordering.

## Alternatives
- **XGBoost** `rank:ndcg` / `rank:pairwise` is an equivalent GBDT ranker.
- **RankNet / pairwise-logistic** are historical; LambdaMART supersedes them on tabular LTR.
- Neural listwise rankers (e.g. TF-Ranking style) matter at web scale but are overkill here.
