---
id: nlp_classification.text.embedding_ensemble
version: 1.0.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [text_classification, nlp_classification, multiclass_classification, binary_classification]
  data_modality: [text, nlp, tabular_text]
  signals:
    - has_text_column: true
  anti_signals:
    - requires_generative_output: true
    - temporal_ordering_required: true

# ——— SCHEMA CONTRACT ———
requires:
  variables: []
  capabilities: [sklearn, pandas, sentence_transformers, matplotlib, numpy]

# ——— INVARIANTS ———
invariants:
  - "The target column must be excluded from all feature matrices (X)."
  - "Random seeds must be fixed (random_state=42) for reproducibility."
  - "Cross-validation must be used; never evaluate on the training split directly."
  - "A logistic regression baseline must be established before any ensemble method."
  - "All evaluation metrics must include both macro and weighted averages."

# ——— EXECUTION DAG ———
dag:
  - id: data_inspection
    intent: "Load the dataset, inspect dtypes and shape, identify the primary text column and label column, report class distribution and check for class imbalance."
    worker_tier: T3
    produces: [df, text_col, label_col, class_counts]
    postconditions:
      - "output_type == 'dataframe'"
      - "df.shape[0] > 0"
    attached_skills: []

  - id: text_feature_engineering
    intent: "Compute character-level and word-level features from the text column: character count, word count, sentence count, average word length, punctuation density, uppercase ratio, and unique word ratio. Append as new columns to df."
    worker_tier: T3
    depends_on: [data_inspection]
    produces: [df]
    postconditions:
      - "output_type == 'dataframe'"
      - "'word_count' in df.columns"
      - "'char_count' in df.columns"

  - id: text_cleaning
    intent: "Lowercase the text column, remove HTML tags and URLs, strip excessive whitespace, and handle null/empty texts by dropping or replacing with a placeholder. Produce a 'clean_text' column."
    worker_tier: T3
    depends_on: [text_feature_engineering]
    produces: [df]
    postconditions:
      - "output_type == 'dataframe'"
      - "'clean_text' in df.columns"
      - "df['clean_text'].isna().sum() == 0"

  - id: stratified_split
    intent: "Perform a stratified train/test split (80/20) on the processed dataset to preserve class ratios. Verify that the label distribution in train and test is within 5% of the overall distribution."
    worker_tier: T3
    depends_on: [text_cleaning]
    produces: [df_train, df_test]
    postconditions:
      - "output_type == 'dataframe'"
      - "abs(df_train[label_col].value_counts(normalize=True) - df_test[label_col].value_counts(normalize=True)).max() < 0.05"

  - id: baseline_tfidf
    intent: "Build a TF-IDF baseline: fit TfidfVectorizer(max_features=50000, ngram_range=(1,2)) on df_train['clean_text'], transform both splits. Train a LogisticRegression classifier with 5-fold cross-validation on the training set. Report macro-AUC, macro-F1, and weighted-F1. Save CV scores."
    worker_tier: T2
    depends_on: [stratified_split]
    produces: [tfidf_vec, X_train_tfidf, X_test_tfidf, baseline_cv_scores, baseline_model]
    postconditions:
      - "output_type == 'model'"
      - "baseline_cv_scores.mean() > 0.5"
    attached_skills: []

  - id: embedding_generation
    intent: "Generate sentence embeddings using a pretrained model. Prefer 'all-MiniLM-L6-v2' from sentence-transformers (fast, 384-dim). If sentence-transformers is unavailable, fall back to TruncatedSVD(n_components=100) on the TF-IDF matrix. Encode df_train['clean_text'] and df_test['clean_text'] into dense matrices."
    worker_tier: T2
    depends_on: [stratified_split]
    produces: [X_train_emb, X_test_emb]
    postconditions:
      - "output_type == 'ndarray'"
      - "X_train_emb.shape[0] == len(df_train)"
      - "X_train_emb.shape[1] >= 50"
    attached_skills: [local_text_embedding]

  - id: feature_fusion
    intent: "Concatenate the embedding features with the hand-crafted text features (word_count, char_count, sentence_count, etc.) into a single feature matrix X_train_combined and X_test_combined using np.hstack or scipy.sparse.hstack as appropriate."
    worker_tier: T3
    depends_on: [baseline_tfidf, embedding_generation]
    produces: [X_train_combined, X_test_combined, y_train, y_test]
    postconditions:
      - "output_type == 'ndarray'"
      - "X_train_combined.shape[0] == len(df_train)"

  - id: ensemble_classifier
    intent: "Train an ensemble on the combined features: (1) LightGBM or XGBoost classifier with early stopping if a validation split is available; fall back to RandomForestClassifier if neither is installed. Use 5-fold StratifiedKFold cross-validation. Report macro-AUC, macro-F1, weighted-F1, and per-class F1. Compare these metrics against the TF-IDF baseline."
    worker_tier: T2
    depends_on: [feature_fusion]
    produces: [ensemble_model, ensemble_cv_scores, y_pred, y_prob]
    postconditions:
      - "output_type == 'model'"
      - "ensemble_cv_scores.mean() >= baseline_cv_scores.mean()"
    attached_skills: []

  - id: evaluation_plots
    intent: "Generate and save: (1) a normalized confusion matrix heatmap (Figure 1). (2) ROC curves with AUC per class using OvR strategy, as a single figure (Figure 2). (3) A bar chart of per-class F1 scores comparing baseline vs ensemble (Figure 3). Use matplotlib; save each as a Plotly JSON artifact where possible."
    worker_tier: T2
    depends_on: [ensemble_classifier]
    attached_skills: [visualization_best_practices]
    postconditions:
      - "output_type == 'artifact'"

  - id: feature_importance
    intent: "If the ensemble model supports feature importances (LightGBM/XGBoost/RandomForest), extract the top-20 most important features. For TF-IDF features, show the top tokens; for embedding dimensions, note their index. For meta-features (word_count etc.), show their rank. Visualize as a horizontal bar chart (Figure 4)."
    worker_tier: T2
    depends_on: [ensemble_classifier, evaluation_plots]
    attached_skills: [visualization_best_practices]
    postconditions:
      - "output_type == 'artifact'"
    skippable_if: "not hasattr(ensemble_model, 'feature_importances_')"

  - id: error_analysis
    intent: "Identify the 10 most confidently misclassified examples (highest predicted probability for the wrong class). For each, print the text excerpt (first 200 chars), true label, predicted label, and confidence. Emit these as a gads_emit_insight with artifact='error_analysis' and include a brief hypothesis about why the model struggled."
    worker_tier: T2
    depends_on: [ensemble_classifier]
    postconditions:
      - "output_type == 'insight'"

---
# NLP Text Classification with Embedding Ensemble

This recipe implements a robust NLP classification pipeline combining TF-IDF baselines with
dense sentence embeddings, designed for tasks such as authorship attribution, fine-tuning
technique classification, sentiment analysis, or topic classification on essay/document data.

## Rationale

Pure TF-IDF misses semantic similarity between paraphrases. Pure embeddings miss precise lexical
signals (specific tokens, n-grams). The fusion approach captures both: (1) a TF-IDF baseline
establishes the minimum bar and makes the feature engineering auditable; (2) sentence embeddings
capture semantic similarity; (3) hand-crafted stylometric features (word count, punctuation
density) capture writing-style signals that embeddings encode only weakly.

LightGBM on the fused feature matrix typically outperforms fine-tuned transformers on tasks
where the signal is stylometric rather than semantic — and runs in seconds rather than minutes,
enabling cross-validation.

## Kaggle LLM Classification Context

This recipe was designed for tasks like the Kaggle LLM Fine-Tuning Classification competition
[https://www.kaggle.com/competitions/llm-classification-finetuning], where the task is to
classify student essays by the fine-tuning technique (if any) applied to the generating model.
Key signals in that task: sentence-level repetition patterns, unusual punctuation distributions,
and embedding-space clustering by generation style. The error analysis step is particularly
important — misclassified examples often reveal boundary conditions between fine-tuning regimes
that are useful for model iteration.

## Key Constraints

- **Class imbalance**: If any class has < 5% of total samples, consider class-weighted loss or
  SMOTE before the ensemble step.
- **Embedding fallback**: The sentence-transformer model (~90MB) requires network access on first
  use. In offline environments, TruncatedSVD on TF-IDF is an adequate substitute.
- **Memory**: For datasets > 500k rows, use batched encoding in the embedding step
  (`encode(texts, batch_size=64)`).
- **Leakage**: Never fit TF-IDF or compute embedding statistics on the test split.

## Evaluation Checklist

- [ ] Baseline AUC established before ensemble
- [ ] Cross-validation used (not train-set evaluation)
- [ ] Confusion matrix shows per-class behavior, not just overall accuracy
- [ ] Error analysis identifies systematic failure modes
- [ ] Feature importance is inspected for leakage (e.g. if a metadata column accidentally
      correlates with label)
