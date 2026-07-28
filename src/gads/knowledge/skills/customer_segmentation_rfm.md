---
id: customer_segmentation_rfm
description: "Customer segmentation with RFM + K-Means: per-entity aggregation, log+scale, silhouette k-selection, KMeans, cluster profiling and naming"
triggers: ["segmentation", "customer segmentation", "segment customers", "clustering", "kmeans", "k-means", "rfm", "recency frequency monetary", "silhouette", "elbow method", "personas", "customer groups", "cohorts", "cluster profile"]
---
# Customer Segmentation: RFM + K-Means

## Aggregate to one row per entity, engineer RFM
Cluster customers, not transactions — aggregate first.
```python
import numpy as np, pandas as pd
snapshot = pd.to_datetime(df["timestamp"]).max()
g = df.groupby("customer_id")
rfm = pd.DataFrame({
    "recency":   (snapshot - pd.to_datetime(g["timestamp"].max())).dt.days,
    "frequency": g.size(),
    "monetary":  g["amount"].sum(),      # no price? use g["rating"].mean() / g["helpful_vote"].sum()
})
```

## Log-transform skew, then scale (KMeans is distance-based)
```python
from sklearn.preprocessing import StandardScaler
X = rfm.copy()
X[["frequency", "monetary"]] = np.log1p(X[["frequency", "monetary"]])   # tame right skew
X_scaled = StandardScaler().fit_transform(X)
```

## Select k by silhouette (+ elbow), don't hardcode
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sil = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    sil[k] = silhouette_score(X_scaled, km.labels_)
k = max(sil, key=sil.get)                      # justify with the silhouette curve (plot it)
```

## Fit, then PROFILE and NAME every segment (the deliverable)
```python
from sklearn.metrics import silhouette_score, davies_bouldin_score
km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
rfm["segment"] = km.labels_
profile = rfm.groupby("segment").agg(
    recency=("recency", "mean"), frequency=("frequency", "mean"),
    monetary=("monetary", "mean"), size=("segment", "size"))
print(profile)   # name each row: low recency + high freq/monetary => "loyal high-value", etc.

import json
json.dump({"silhouette_score": float(silhouette_score(X_scaled, km.labels_)),
           "davies_bouldin_score": float(davies_bouldin_score(X_scaled, km.labels_)),
           "k": int(k), "cluster_sizes": rfm["segment"].value_counts().to_dict()},
          open("metrics.json", "w"))
```
Then a 2-D PCA scatter coloured by `segment` (Figure 2), and one insight per segment
(who they are + a recommended action).

## Pitfalls
- Clustering raw transactions instead of aggregated customers.
- Skipping scaling → monetary magnitude dominates; skipping log → outliers dominate.
- Hardcoding k with no silhouette/elbow support.
- Leaving clusters as bare numbers — an un-named segmentation is not actionable.
- Alternatives when clusters are non-spherical/variable-density: HDBSCAN, GaussianMixture.
