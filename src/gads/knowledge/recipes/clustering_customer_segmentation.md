---
id: clustering.segmentation.rfm_kmeans
version: 1.0.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [clustering, customer_segmentation, segmentation]
  data_modality: [tabular]
  signals:
    - no_labeled_target: true
    - objective_contains: [segment, segmentation, cluster, clusters, personas, group customers, customer groups, cohorts]
  anti_signals:
    - target_type: categorical_binary
    - task: classification

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [sklearn, pandas, numpy, matplotlib]

# ——— EXECUTION DAG ———
dag:
  - id: engineer_segmentation_features
    intent: "Aggregate the data to ONE ROW PER ENTITY (e.g. customer/user). Engineer interpretable segmentation features — RFM is the canonical set: Recency (days since the entity's last event relative to the dataset max date), Frequency (count of events), Monetary (sum or mean of transaction value; if no price column, use an engagement proxy such as total/av rating, helpful votes, or session count). Add any other behavioral features implied by the objective. Store the per-entity table in `customer_features` and print its shape and a describe() summary."
    worker_tier: T2
    attached_skills: [customer_segmentation_rfm]
    produces: [customer_features]
    postconditions:
      - "customer_features is not None"

  - id: scale_and_select_k
    intent: "Log1p-transform right-skewed features (frequency, monetary), then StandardScaler the feature matrix. Select the number of clusters k by evaluating silhouette score (and inertia/elbow) across k = 2..8; pick k and justify the choice in a printed line. Save the silhouette/elbow selection plot as Figure 1. Store the scaled matrix in `X_scaled` and the chosen `k`."
    worker_tier: T2
    depends_on: [engineer_segmentation_features]
    attached_skills: [customer_segmentation_rfm, visualization_best_practices]
    produces: [X_scaled, k]
    postconditions:
      - "X_scaled is not None"

  - id: fit_and_profile
    intent: "Fit KMeans(n_clusters=k, n_init=10, random_state=42) on X_scaled and assign a `segment` label to each entity. PROFILE every cluster: a table of mean (un-scaled) feature values per segment plus its size, and a 2D PCA scatter coloured by segment saved as Figure 2. Write metrics.json with silhouette_score, davies_bouldin_score, k, and cluster_sizes. Store the labelled table."
    worker_tier: T2
    depends_on: [scale_and_select_k]
    attached_skills: [customer_segmentation_rfm]
    required_metrics: [silhouette_score]
    postconditions:
      - "'silhouette_score' in open('metrics.json').read()"

  - id: interpret_segments
    intent: "Give every segment a business-meaningful name and description grounded in its profile (e.g. 'high-value loyal', 'at-risk lapsing', 'new low-engagement', 'bargain occasional'), and a recommended action per segment. Emit these as insights — this narrative IS the deliverable of a segmentation."
    worker_tier: T2
    depends_on: [fit_and_profile]
    postconditions:
      - "customer_features is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "UNSUPERVISED: there is no labeled target. If a ground-truth grouping exists, this is classification, not clustering — flag it instead of proceeding."
  - "ONE ROW PER ENTITY: cluster entities (customers), not raw transactions. Aggregate first."
  - "SCALE BEFORE KMEANS: KMeans is distance-based; standardize (and log-transform skewed monetary/frequency) or one feature's units will dominate the clusters."
  - "SELECT AND JUSTIFY k: choose k from silhouette/elbow evidence — never hardcode a number without support."
  - "PROFILE AND NAME EVERY CLUSTER: an unlabelled, uninterpreted segmentation is a failure. Each segment must be described by its feature profile and given an actionable name."
  - "Report silhouette_score (internal validity) and fix random_state=42."
---

# Customer Segmentation with RFM + K-Means

## Rationale
The dominant industry segmentation workflow is **RFM feature engineering + K-Means**:
reduce each customer to **Recency, Frequency, Monetary** (plus behavioral) features,
standardize, choose k by silhouette/elbow, cluster, then **profile and name** the
segments so the business can act on them. It is the textbook approach on the canonical
segmentation datasets — **UCI Online Retail**, **Instacart**, **Mall Customer
Segmentation** — because it is interpretable end to end: every segment is a point in
RFM space a marketer can reason about.

## When to use
The objective is to discover natural groups of entities (customers, users) with **no
labeled target**, for personas, targeting, or lifecycle analysis. If a target label
exists, use a supervised recipe instead.

## Key mechanics
- **Aggregate** to one row per customer; engineer RFM (+ behavioral) features.
- **Transform & scale** — log1p skewed features, then StandardScaler.
- **Choose k** — silhouette (primary) + elbow/inertia across k = 2..8.
- **Cluster & profile** — KMeans, then per-segment feature means, sizes, PCA scatter.
- **Name & act** — a business label and recommended action per segment.

## Alternatives
- **HDBSCAN** / **DBSCAN** — density-based; find non-spherical clusters and outliers, no fixed k.
- **Gaussian Mixture Models** — soft assignments, elliptical clusters.
- **Hierarchical/agglomerative** — a dendrogram when nested structure matters.
KMeans is the robust default; switch when clusters are non-convex or variable-density.
