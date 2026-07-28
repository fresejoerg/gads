#!/usr/bin/env python3
"""Produce a dense k-core subset of the Amazon Fashion reviews for collaborative
filtering. Iterative k-core filtering (drop users and items with < k interactions until
stable) is the standard preprocessing for the Amazon / MovieLens recommendation
benchmarks — raw interaction logs are too long-tailed for CF, so papers report on the
k-core. Output stays under the sandbox row budget so no random row-sampling is applied.

Writes $GADS_DATASETS_ROOT/amazon_fashion_5core.csv (default /home/joergf/datasets).
"""
import os
import pandas as pd

ROOT = os.getenv("GADS_DATASETS_ROOT", "/home/joergf/datasets")
SRC = os.path.join(ROOT, "amazon-fashion-800k+-user-reviews-dataset.csv")
K = 5
COLS = ["user_id", "parent_asin", "rating", "timestamp", "helpful_vote"]

df = pd.read_csv(SRC, usecols=COLS).dropna(subset=["user_id", "parent_asin"])
df = df.drop_duplicates(["user_id", "parent_asin"], keep="last")
print(f"raw interactions: {len(df):,}  users: {df.user_id.nunique():,}  items: {df.parent_asin.nunique():,}")

# Iterative k-core: repeat until no user/item falls below K.
while True:
    n0 = len(df)
    uc = df.user_id.value_counts()
    df = df[df.user_id.isin(uc[uc >= K].index)]
    ic = df.parent_asin.value_counts()
    df = df[df.parent_asin.isin(ic[ic >= K].index)]
    if len(df) == n0:
        break

print(f"{K}-core: {len(df):,} interactions  users: {df.user_id.nunique():,}  "
      f"items: {df.parent_asin.nunique():,}  density: {len(df)/(df.user_id.nunique()*df.parent_asin.nunique()):.4%}")

out = os.path.join(ROOT, "amazon_fashion_5core.csv")
df.to_csv(out, index=False)
print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
