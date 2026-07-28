---
name: "Amazon Fashion Customer Segmentation (RFM + K-Means)"
datasets:
  - amazon-fashion-800k+-user-reviews-dataset.csv
recipe_id: clustering.segmentation.rfm_kmeans
domain: customer segmentation
sample_rows: 200000
taxonomy:
  intent: structure_discovery
  task: [clustering.partitional]
  modality: [tabular]
  domain: marketing
  domain_detail: "Amazon Fashion customer behavioral segmentation (RFM, UCI-Online-Retail-style)"
  deliverable: [segmentation, report_narrative]
---
Segment customers into behavioral groups.

Dataset: the Amazon Fashion reviews. Each row is a user's review of a product with a
`user_id`, `rating`, `timestamp`, and `helpful_vote`. There is no purchase price, so use
review activity as the behavioral signal.

Aggregate to one row per `user_id` and engineer RFM-style features: **Recency** (days
since the user's last review relative to the dataset's latest date), **Frequency**
(number of reviews), and a **Monetary/engagement** proxy (mean rating and total helpful
votes). Log-transform skewed features and standardize, choose the number of segments k by
silhouette/elbow evidence (not a hardcoded number), fit K-Means, and — the actual
deliverable — **profile and name every segment** (e.g. "loyal enthusiasts", "at-risk
lapsed", "new low-engagement") with its RFM profile, a PCA scatter, and a recommended
action per segment. Report the silhouette score.
