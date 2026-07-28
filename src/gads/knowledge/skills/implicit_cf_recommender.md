---
id: implicit_cf_recommender
description: "Implicit-feedback collaborative filtering with scipy/sklearn: sparse user-item matrix, item-item cosine, temporal leave-one-out, Recall@K/NDCG@K vs popularity"
triggers: ["recommendation", "recommender", "collaborative filtering", "implicit feedback", "top-n", "top-k recommendation", "recall@k", "ndcg@k", "hit rate", "user-item matrix", "cosine similarity", "recommend items", "als", "matrix factorization"]
---
# Implicit-Feedback Collaborative Filtering (scipy / sklearn)

No `implicit`/`surprise`/`lightfm` in the sandbox — build the CF baseline from scipy
sparse + sklearn. Item-item cosine is the robust default; TruncatedSVD is the MF option.

## Build the sparse user × item matrix (with cold-start filtering)
```python
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix

d = df.dropna(subset=["user_id", "item_id"]).copy()
d = d.sort_values("timestamp")                        # keep most recent per pair
d = d.drop_duplicates(["user_id", "item_id"], keep="last")
for col, k in [("user_id", 5), ("item_id", 5)]:       # drop cold users/items
    keep = d[col].value_counts()[lambda s: s >= k].index
    d = d[d[col].isin(keep)]

users = {u: i for i, u in enumerate(d["user_id"].unique())}
items = {it: j for j, it in enumerate(d["item_id"].unique())}
d["u"], d["i"] = d["user_id"].map(users), d["item_id"].map(items)
M = csr_matrix((np.ones(len(d)), (d["u"], d["i"])), shape=(len(users), len(items)))
```

## Temporal leave-one-out split
Hold out each user's most recent interaction; keep the rest for training.
```python
last = d.sort_values("timestamp").groupby("u").tail(1)          # 1 held-out row per user
holdout = dict(zip(last["u"], last["i"]))
train = d[~d.index.isin(last.index)]
Mtr = csr_matrix((np.ones(len(train)), (train["u"], train["i"])), shape=M.shape)
```

## Item-item cosine → top-N (excluding seen)
```python
from sklearn.metrics.pairwise import cosine_similarity
S = cosine_similarity(Mtr.T, dense_output=False)               # item × item
scores = Mtr @ S                                               # user × item preference
scores = scores.toarray()
scores[Mtr.nonzero()] = -np.inf                                # exclude already-seen
topN = np.argsort(-scores, axis=1)[:, :10]                     # per-user top-10
# MF alternative: TruncatedSVD(n_components=64).fit_transform(Mtr) @ components_
```

## Evaluate vs a popularity baseline
```python
def recall_ndcg_at_k(topN, holdout, k=10):
    hits = ndcgs = 0
    for u, true_i in holdout.items():
        rec = topN[u][:k]
        if true_i in rec:
            hits += 1
            ndcgs += 1.0 / np.log2(list(rec).index(true_i) + 2)
    n = len(holdout)
    return hits / n, ndcgs / n                                 # recall@k == hitrate@k for 1 held-out

recall10, ndcg10 = recall_ndcg_at_k(topN, holdout, 10)
pop = np.argsort(-np.asarray(Mtr.sum(0)).ravel())             # most-popular items
pop_hits = np.mean([pop[:10].tolist().count(t) > 0 or t in pop[:10] for t in holdout.values()])
```
Report Recall@10, NDCG@10, and the lift over the popularity baseline. A CF model that
does not beat most-popular has added no personalization.

## Pitfalls
- Random (non-temporal) splits leak the future and inflate scores.
- Forgetting to mask seen items → "hits" that are just memorized history.
- Dense `user × item` blows up memory on large catalogues → cap items or use SVD factors.
