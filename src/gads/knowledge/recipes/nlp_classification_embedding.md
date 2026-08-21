---
id: nlp_classification.text.embedding_ensemble
version: 1.2.0
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
  - "A simple baseline (TF-IDF + logistic regression) must be established and reported BEFORE any ensemble method — report both, even if the ensemble does not win."
  - "All evaluation metrics must include both macro and weighted averages."
  - "Class imbalance is handled with class_weight='balanced' — never resampling libraries."

# ——— EXECUTION DAG ———
dag:
  - id: data_inspection
    intent: "Load the dataset, inspect dtypes and shape, identify the primary text column and label column, report the class distribution and flag class imbalance."
    worker_tier: T3
    produces: [df, text_col, label_col, class_counts]
    postconditions:
      - "df.shape[0] > 0"
    attached_skills: []

  - id: text_feature_engineering
    intent: "Compute stylometric features from the text column: character count, word count, sentence count, average word length, punctuation density, uppercase ratio, and unique-word ratio. Append as new columns to df."
    worker_tier: T3
    depends_on: [data_inspection]
    produces: [df]
    postconditions:
      - "'word_count' in df.columns"
      - "'char_count' in df.columns"

  - id: text_cleaning
    intent: "Lowercase the text column, remove HTML tags and URLs, strip excessive whitespace, and handle null/empty texts (drop or placeholder). Produce a 'clean_text' column with no nulls."
    worker_tier: T3
    depends_on: [text_feature_engineering]
    produces: [df]
    postconditions:
      - "'clean_text' in df.columns"
      - "df['clean_text'].isna().sum() == 0"

  - id: stratified_split
    intent: "Perform a stratified 80/20 train/test split (random_state=42) preserving class ratios. Verify the label distribution in train and test is within 5% of the overall distribution."
    worker_tier: T3
    depends_on: [text_cleaning]
    produces: [df_train, df_test]
    postconditions:
      - "abs(df_train[label_col].value_counts(normalize=True) - df_test[label_col].value_counts(normalize=True)).max() < 0.05"

  - id: baseline_tfidf
    intent: "Build the auditable baseline: fit TfidfVectorizer(max_features=50000, ngram_range=(1,2)) on df_train['clean_text'] only (never on test), transform both splits, and train a LogisticRegression (class_weight='balanced') with 5-fold cross-validation on the training set. Report macro-AUC, macro-F1, and weighted-F1; save the CV scores."
    worker_tier: T2
    depends_on: [stratified_split]
    produces: [tfidf_vec, X_train_tfidf, X_test_tfidf, baseline_cv_scores, baseline_model]
    attached_skills: [supervised_modeling]
    postconditions:
      - "baseline_cv_scores is not None"

  - id: embedding_generation
    intent: "Generate dense sentence embeddings for clean_text in both splits with the locally cached sentence-transformer (pattern in the attached skill). If sentence-transformers is unavailable, fall back to TruncatedSVD(n_components=100) on the TF-IDF matrix."
    worker_tier: T2
    depends_on: [stratified_split]
    produces: [X_train_emb, X_test_emb]
    postconditions:
      - "X_train_emb.shape[0] == len(df_train)"
    attached_skills: [local_text_embedding]

  - id: feature_fusion
    intent: "Concatenate the embeddings with the stylometric features into X_train_combined / X_test_combined (np.hstack or scipy.sparse.hstack as appropriate), and bind y_train / y_test."
    worker_tier: T3
    depends_on: [baseline_tfidf, embedding_generation]
    produces: [X_train_combined, X_test_combined, y_train, y_test]
    postconditions:
      - "X_train_combined.shape[0] == len(df_train)"

  - id: ensemble_classifier
    intent: "Train a gradient-boosted ensemble (LightGBM or XGBoost; RandomForest fallback) on the combined features with 5-fold StratifiedKFold cross-validation. Report macro-AUC, macro-F1, weighted-F1, and per-class F1, and compare each against the TF-IDF baseline — report the comparison honestly whichever way it goes."
    worker_tier: T2
    depends_on: [feature_fusion]
    produces: [ensemble_model, ensemble_cv_scores, y_pred, y_prob]
    attached_skills: [supervised_modeling]
    postconditions:
      - "ensemble_cv_scores is not None"

  - id: evaluation_plots
    intent: "Generate and save: (1) a normalized confusion-matrix heatmap (Figure 1); (2) a bar chart of per-class F1 comparing baseline vs ensemble (Figure 2). Prefer Plotly JSON artifacts. Do NOT plot ROC curves here — the diagnostic_curves node produces them from the native, and two hand-rolled implementations of the same curve is how they end up disagreeing."
    worker_tier: T2
    depends_on: [ensemble_classifier]
    attached_skills: [visualization_best_practices]
    postconditions:
      - "y_pred is not None"

  - id: feature_importance
    intent: "If the ensemble exposes feature importances, extract the top 20: name top TF-IDF tokens, note embedding-dimension indices, and rank the stylometric features. Visualize as a horizontal bar chart (Figure 4). Inspect the ranking for leakage — a metadata-like feature dominating is a red flag to report."
    worker_tier: T2
    depends_on: [ensemble_classifier, evaluation_plots]
    attached_skills: [visualization_best_practices]
    postconditions:
      - "ensemble_model is not None"
    skippable_if: "not hasattr(ensemble_model, 'feature_importances_')"

  - id: error_analysis
    intent: "Identify the 10 most confidently misclassified examples (highest predicted probability on the wrong class). For each, print a 200-char excerpt, true label, predicted label, and confidence. Emit a gads_emit_insight with artifact='error_analysis' including a brief hypothesis about the systematic failure mode."
    worker_tier: T2
    depends_on: [ensemble_classifier]
    postconditions:
      - "y_pred is not None"

  - id: diagnostic_curves
    report:
      title: ROC and Precision-Recall Curves
      summary: How the model trades false positives against false negatives, against both baselines.
    intent: >
      Plot the ROC and precision-recall curves for the ensemble by calling the native:
        curve_diagnostics = gads_plot_classification_curves(y_test, y_prob)
      Do NOT hand-roll the curves. The native handles the binary/multiclass split, the label
      dtype, the one-vs-rest expansion and the two baselines (the ROC chance diagonal and the
      PR no-skill line at the positive rate), and writes both figures as Plotly JSON the
      dashboard renders directly. Bind `average_precision` from the result. Then state, in one
      or two sentences, what the PR curve says that the ROC curve does not — under class
      imbalance the ROC baseline is fixed at 0.5 while the PR baseline moves with prevalence,
      which is the whole reason both are plotted.
    depends_on: [ensemble_classifier]
    worker_tier: T2
    produces: [curve_diagnostics, average_precision]
    required_metrics: [average_precision]
    attached_skills: [visualization_best_practices]
    fallback_native: gads_plot_classification_curves
    fallback_call: "curve_diagnostics = gads_plot_classification_curves(y_test, y_prob); average_precision = curve_diagnostics['average_precision']"
    postconditions:
      - "curve_diagnostics.get('roc_auc') is not None"

---
# NLP Text Classification with Embedding Ensemble

A general text-classification pipeline combining a TF-IDF baseline with dense sentence
embeddings and stylometric features — applicable to sentiment, topic, authorship,
intent, or any label-per-document task.

## Rationale

Pure TF-IDF misses semantic similarity between paraphrases; pure embeddings miss precise
lexical signals (specific tokens, n-grams); neither captures writing-style signals
(length, punctuation density) that matter when the label reflects *how* rather than
*what* was written. The fusion captures all three, and the mandatory baseline keeps the
gain auditable: (1) TF-IDF + logistic regression sets the floor and exposes the lexical
signal; (2) embeddings add semantics; (3) stylometric features add form. A
gradient-boosted model on the fused matrix trains in seconds — cheap enough for honest
cross-validation — and often rivals fine-tuned transformers when the signal is
stylometric rather than deeply semantic.

The error-analysis step is not optional garnish: confidently wrong examples are the
fastest route to discovering label noise, leakage, or class-boundary ambiguity.

## Key Constraints

- **Class imbalance**: use `class_weight='balanced'` in the classifiers. Resampling
  libraries are not available in the sandbox.
- **Embedding fallback**: if the cached sentence-transformer is unavailable,
  TruncatedSVD on TF-IDF is an adequate substitute.
- **Memory**: for very large corpora, encode in batches (`encode(texts, batch_size=64)`).
- **Leakage**: never fit TF-IDF, embedding statistics, or any scaler on the test split.

## Evaluation Checklist

- [ ] Baseline metrics established and reported before the ensemble
- [ ] Cross-validation used (never train-set evaluation)
- [ ] Confusion matrix shows per-class behavior, not just overall accuracy
- [ ] Error analysis identifies systematic failure modes
- [ ] Feature importances inspected for leakage
