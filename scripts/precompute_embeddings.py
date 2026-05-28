"""
Precompute sentence-transformer cosine similarities for the Chatbot Arena dataset.

Run with:
    uv run --with sentence-transformers --with pandas --with pyarrow --with tqdm \
        python scripts/precompute_embeddings.py

Output: /home/joergf/datasets/llm-classification-finetuning/semantic_similarity.parquet
Columns: id (int64), cosine_sim_st (float32)

GADS will symlink this file into the workspace alongside train.csv.
Add it to the spec's `datasets` list:
    datasets:
      - llm-classification-finetuning/train.csv
      - llm-classification-finetuning/semantic_similarity.parquet
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATASET_DIR = "/home/joergf/datasets/llm-classification-finetuning"
INPUT_CSV   = os.path.join(DATASET_DIR, "train.csv")
OUTPUT_FILE = os.path.join(DATASET_DIR, "semantic_similarity.parquet")
MODEL_NAME  = "all-MiniLM-L6-v2"
BATCH_SIZE  = 256  # fits comfortably in host RAM

def cosine_sim_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise cosine similarity between two (N, D) matrices."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (a_norm * b_norm).sum(axis=1).astype(np.float32)

def encode_in_batches(model, texts: list, batch_size: int, desc: str) -> np.ndarray:
    all_vecs = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch = texts[start : start + batch_size]
        vecs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_vecs.append(vecs)
    return np.vstack(all_vecs)

def main():
    print(f"Loading {INPUT_CSV} …")
    df = pd.read_csv(INPUT_CSV, usecols=["id", "response_a", "response_b"])
    df["response_a"] = df["response_a"].fillna("").astype(str)
    df["response_b"] = df["response_b"].fillna("").astype(str)
    print(f"  {len(df):,} rows")

    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    vecs_a = encode_in_batches(model, df["response_a"].tolist(), BATCH_SIZE, "Encoding response_a")
    vecs_b = encode_in_batches(model, df["response_b"].tolist(), BATCH_SIZE, "Encoding response_b")

    print("\nComputing cosine similarities …")
    sims = cosine_sim_row(vecs_a, vecs_b)

    out = pd.DataFrame({"id": df["id"].values, "cosine_sim_st": sims})
    print(f"  similarity range: {sims.min():.4f} – {sims.max():.4f}  mean: {sims.mean():.4f}")

    out.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved → {OUTPUT_FILE}")
    print(f"  shape: {out.shape}  size: {os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")

if __name__ == "__main__":
    main()
