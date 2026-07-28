---
name: "Amazon Fashion Product Recommendation (Implicit CF)"
datasets:
  - amazon-fashion-800k+-user-reviews-dataset.csv
recipe_id: recommendation.implicit.collaborative_filtering
domain: e-commerce product recommendation
sample_rows: 200000
taxonomy:
  intent: prescriptive
  task: [recommendation.collaborative]
  modality: [tabular]
  domain: retail_ecommerce
  domain_detail: "Amazon Fashion user–item recommendation (McAuley Amazon Reviews benchmark)"
  deliverable: [decision_policy]
  validation: [holdout_metric]
---
Recommend fashion products to users from their review history.

Dataset: the Amazon Fashion reviews (McAuley Amazon Product Reviews benchmark). Each row
is a user reviewing a product: `user_id`, `parent_asin` (the item), `rating` (1–5),
`timestamp`, plus review text and `helpful_vote`. Treat a review as an **implicit
positive interaction** (the user engaged with the item) — do not treat unobserved
user-item pairs as confirmed negatives.

Build a top-N recommender using implicit-feedback collaborative filtering over the
sparse user × item matrix. Filter cold users/items (fewer than 5 interactions), hold out
each user's **most recent** interaction (temporal leave-one-out), and recommend items the
user hasn't already seen. Report **Recall@10** and **NDCG@10** (also @20) averaged over
users, and — critically — the lift over a most-popular-items baseline. Show a couple of
example users' histories vs their recommendations and the catalog coverage.
