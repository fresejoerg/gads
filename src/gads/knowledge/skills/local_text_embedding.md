---
id: local_text_embedding
description: "Local sentence-transformer embeddings using all-MiniLM-L6-v2 (cached, no download required)"
triggers: ["embed", "vectorize", "sentence_transformers", "semantic search", "cosine similarity"]
---
# Local Text Embedding

You can perform local text vectorization using the `sentence-transformers` library and the pre-cached `all-MiniLM-L6-v2` model. This is much faster and cheaper than using external APIs for large datasets.

## Implementation Pattern
Use the following code block to generate embeddings for a list of strings:

```python
from sentence_transformers import SentenceTransformer

# Load the pre-cached model (no download required)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings (returns a numpy array)
# Works on lists of strings, Series, or array-like
sentences = ["Sample text one", "Another example"]
embeddings = model.encode(sentences)

# embeddings.shape will be (len(sentences), 384)
```

## Practical Applications
- **Semantic Search**: Calculate cosine similarity between user queries and document embeddings.
- **Clustering**: Use sklearn's `KMeans` on the resulting embedding array to group reviews or documents.
- **Feature Engineering**: Use embeddings as high-dimensional features for classification or regression.

## Performance Note
The `all-MiniLM-L6-v2` model is extremely efficient. For datasets > 50,000 rows, consider batching or using a GPU if available (though CPU is usually sufficient for this model size).

## CRITICAL RULES
1. **NO EMULATION**: Do NOT attempt to generate or predict the embedding vectors yourself. You MUST write the Python code that loads the `SentenceTransformer` and calls `.encode()`.
2. **PERSISTENCE**: If you are embedding columns in a DataFrame, you MUST save the resulting DataFrame to disk as a Parquet file (e.g., `df.to_parquet('df_embedded.parquet')`) at the end of your task. Do NOT rely on keeping massive embedding matrices in the kernel memory for the next task to use, as this causes out-of-memory errors and state corruption. Downstream tasks should `pd.read_parquet('df_embedded.parquet')`.
