---
name: "Search Result Ranking (LambdaMART LTR)"
datasets:
  - synthetic_ltr_ranking.csv
recipe_id: learning_to_rank.tabular.lambdamart
domain: e-commerce search ranking
taxonomy:
  intent: predictive
  task: [ranking.learning_to_rank]
  modality: [tabular]
  domain: retail_ecommerce
  domain_detail: "query-document result ranking (MSLR-WEB10K-style LTR benchmark)"
  deliverable: [model_artifact]
  validation: [holdout_metric]
---
Rank search results by relevance.

Dataset: a learning-to-rank set in the standard LETOR / MSLR-WEB10K schema — one row per
query-document pair, grouped by `qid`, with a graded `relevance` label (0 = irrelevant …
4 = perfect) and 12 dense query-document features `f1`–`f12` (as would come from BM25,
TF-IDF, freshness, popularity, etc.). ~300 queries, ~5,500 documents.

Train a ranker that orders the documents within each query by relevance. Split by query
(no query may appear in both train and test), report **NDCG@10** (primary), NDCG@5, MAP
and MRR averaged across test queries, and show the lift over a trivial baseline. Report
which features drive relevance, and print one example query's ranked list against its
true labels so the ranking is legible.

(This file is a self-contained stand-in for the licensed public LTR benchmarks; drop in a
real MSLR-WEB10K / Yahoo LTR export with the same `qid` + `relevance` + feature columns to
run the benchmark proper.)
