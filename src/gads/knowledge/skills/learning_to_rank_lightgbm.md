---
id: learning_to_rank_lightgbm
description: "LambdaMART learning-to-rank with LightGBM: query-consistent split, group arrays, NDCG/MAP/MRR evaluation"
triggers: ["learning to rank", "learning-to-rank", "ltr", "lambdamart", "lambdarank", "ndcg", "ranker", "rank documents", "query groups", "qid", "relevance label", "search ranking"]
---
# Learning to Rank with LightGBM (LambdaMART)

## Split by query, then build group arrays
A query's documents must all be on the same side of the split, and the `group` array
gives the document count per query IN ROW ORDER (it must sum to `len(X)`).

```python
import numpy as np, lightgbm as lgb

FEATURES = [c for c in df.columns if c not in ("qid", "relevance")]
qids = df["qid"].unique()
rng = np.random.default_rng(42)
rng.shuffle(qids)
cut = int(0.8 * len(qids))
train_qids, test_qids = set(qids[:cut]), set(qids[cut:])

def build(split_qids):
    d = df[df["qid"].isin(split_qids)].sort_values("qid")   # group rows by query
    group = d.groupby("qid", sort=True).size().to_numpy()   # docs per query, in order
    return d[FEATURES].to_numpy(), d["relevance"].to_numpy(), group, d["qid"].to_numpy()

X_train, y_train, group_train, _ = build(train_qids)
X_test,  y_test,  group_test,  qid_test = build(test_qids)
assert group_train.sum() == len(X_train)
```

## Train the ranker
```python
ranker = lgb.LGBMRanker(objective="lambdarank", metric="ndcg",
                        n_estimators=300, learning_rate=0.05, random_state=42)
ranker.fit(X_train, y_train, group=group_train,
           eval_set=[(X_test, y_test)], eval_group=[group_test], eval_at=[5, 10])
```

## Evaluate PER QUERY, then average
Use `sklearn.metrics.ndcg_score` per query (it expects 2-D `[[...]]` arrays). Never pool
all documents into one global NDCG.

```python
from sklearn.metrics import ndcg_score
scores = ranker.predict(X_test)

def per_query(y_true, y_score, groups, fn):
    out, s = [], 0
    for g in groups:
        yt, ys = y_true[s:s+g], y_score[s:s+g]
        if len(set(yt)) > 0 and g > 1:
            out.append(fn(yt, ys))
        s += g
    return float(np.mean(out)) if out else 0.0

ndcg10 = per_query(y_test, scores, group_test, lambda yt, ys: ndcg_score([yt], [ys], k=10))
```

MAP/MRR: within each query, sort docs by predicted score and score the positions of
relevant docs (relevance > 0). Compare NDCG@10 to a baseline that orders by the single
strongest raw feature — report the lift.

## Pitfalls
- `group` out of row order, or not summing to `len(X)` → silently wrong training.
- Shuffling rows across a query boundary → leakage, inflated NDCG.
- Global NDCG over pooled docs is meaningless — always per query then mean.
- XGBoost equivalent: `XGBRanker(objective="rank:ndcg")` with `qid`/group set similarly.
