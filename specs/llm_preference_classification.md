---
name: "LLM Pairwise Preference Classification"
datasets:
  - llm-classification-finetuning/train.csv
---
Predict which of two LLM responses (response_a or response_b) a human prefers, or whether
they tie. The dataset contains 57,477 human preference judgments from Chatbot Arena. Each row
has a prompt and two responses from different models, with binary labels winner_model_a,
winner_model_b, and winner_tie indicating the outcome.

Build a multi-class classifier (3 classes: model_a wins, model_b wins, tie) that predicts
human preference. Engineer features that capture the differences between the two responses —
such as length differences, vocabulary richness, sentiment contrast, and semantic similarity.
Evaluate using macro-F1 and log-loss (to match the competition's soft-probability submission format).
Identify which response characteristics most strongly predict human preference.
