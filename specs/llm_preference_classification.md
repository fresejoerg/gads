---
name: "LLM Pairwise Preference Classification"
datasets:
  - llm-classification-finetuning/train.csv
  - llm-classification-finetuning/semantic_similarity.parquet
save_model: true
---
Predict which of two LLM responses (response_a or response_b) a human prefers, or whether
they tie. The dataset contains 57,477 human preference judgments from Chatbot Arena. Each row
has a prompt and two responses from different models, with binary labels winner_model_a,
winner_model_b, and winner_tie indicating the outcome.

Build a multi-class classifier (3 classes: model_a wins, model_b wins, tie) that predicts
human preference. Engineer features that capture the differences between the two responses:

1. Surface features: character/word length delta, Type-Token Ratio (vocabulary richness), Jaccard similarity.
2. Sentiment: NLTK VADER compound scores for each response, plus the delta.
3. Semantic similarity: the file `semantic_similarity.parquet` contains pre-computed
   sentence-transformer (all-MiniLM-L6-v2) cosine similarities for every row.
   Columns: `id` (join key), `cosine_sim_st` (float). Load and merge on `id` — do NOT
   recompute embeddings in-task, they are already available.

Train a Random Forest or Gradient Boosting classifier. Evaluate using macro-F1 and log-loss
(store results in variables named exactly `macro_f1` and `log_loss`).
Identify which response characteristics most strongly predict human preference via feature importance.
