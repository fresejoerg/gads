---
name: "MovieLens-100K Movie Recommendation (Implicit CF)"
datasets:
  - movielens_100k.csv
recipe_id: recommendation.implicit.collaborative_filtering
domain: movie recommendation
taxonomy:
  intent: prescriptive
  task: [recommendation.collaborative]
  modality: [tabular]
  domain: general
  domain_detail: "MovieLens-100K movie recommendation (GroupLens CF benchmark)"
  deliverable: [decision_policy]
  validation: [holdout_metric]
---
Recommend movies to users from their rating history.

Dataset: MovieLens-100K (GroupLens) — the standard collaborative-filtering benchmark.
100,000 ratings by 943 users on 1,682 movies; each user has rated at least 20 movies, so
the user × item matrix is dense enough for CF (unlike long-tail review logs). Columns:
`user_id`, `item_id`, `rating` (1–5), `timestamp`.

Treat a rating as an **implicit positive interaction** (the user engaged with the movie);
optionally weight confidence by the rating. Build a top-N recommender with implicit-feedback
collaborative filtering over the sparse user × item matrix — prefer ALS (the `implicit`
library) and fall back to item-item cosine if needed. Hold out each user's **most recent**
rating (temporal leave-one-out), recommend movies the user hasn't already rated, and report
**Recall@10** and **NDCG@10** (also @20) averaged over users, plus — critically — the lift
over a most-popular-movies baseline. Show a couple of example users' histories vs their
recommendations and the catalog coverage.
