#!/usr/bin/env python3
"""Generate a small, learnable learning-to-rank dataset in the MSLR-WEB10K / LETOR
schema (qid + graded relevance 0-4 + dense query-document features), as a self-contained
stand-in for the licensed public LTR benchmarks. Relevance is a monotone function of a
latent feature combination plus per-query effects and noise, so a LambdaMART ranker
recovers real signal (NDCG well above a trivial baseline) while nothing is memorizable.

Writes to $GADS_DATASETS_ROOT/synthetic_ltr_ranking.csv (default /home/joergf/datasets).
Swap in a real MSLR-WEB10K/Yahoo LTR export (same columns) for a benchmark run.
"""
import os
import numpy as np
import pandas as pd

OUT_ROOT = os.getenv("GADS_DATASETS_ROOT", "/home/joergf/datasets")
N_QUERIES = 300
N_FEATURES = 12
RNG = np.random.default_rng(42)

# A shared "relevance direction" — some features matter, some are noise (like a real LTR set).
w = RNG.normal(size=N_FEATURES) * np.concatenate([np.ones(6), np.zeros(6)])  # last 6 ~irrelevant

rows = []
for qid in range(1, N_QUERIES + 1):
    n_docs = int(RNG.integers(8, 30))                      # variable docs per query
    X = RNG.normal(size=(n_docs, N_FEATURES))
    query_effect = RNG.normal(scale=0.5)                   # per-query difficulty shift
    latent = X @ w + query_effect + RNG.normal(scale=1.0, size=n_docs)
    # graded relevance 0-4 by within-query quantile bins (each query spans the grade range)
    ranks = latent.argsort().argsort()                     # 0..n_docs-1
    grades = np.floor(ranks / n_docs * 5).astype(int).clip(0, 4)
    for i in range(n_docs):
        row = {"qid": qid, "relevance": int(grades[i])}
        row.update({f"f{j+1}": float(X[i, j]) for j in range(N_FEATURES)})
        rows.append(row)

df = pd.DataFrame(rows)
os.makedirs(OUT_ROOT, exist_ok=True)
out = os.path.join(OUT_ROOT, "synthetic_ltr_ranking.csv")
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} rows, {df['qid'].nunique()} queries, "
      f"relevance dist={df['relevance'].value_counts().sort_index().to_dict()}")
