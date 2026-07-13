---
id: thematic_analysis.unstructured.text
version: 1.2.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [thematic_analysis, topic_modeling, text_mining, sentiment_thematic]
  data_modality: [unstructured_text, text]
  signals:
    - contains_text_column: true

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [nltk, spacy, polars, duckdb, plotly, instructor]

# ——— DAG TEMPLATE ———
dag:
  - id: data_hygiene
    intent: "Clean and prepare the text corpus. Use efficient engines (DuckDB/Polars) if the dataset is large. Identify columns for text and metadata based on the user's objective. Remove nulls and normalize temporal/numerical fields."
    worker_tier: T3
    attached_skills: [large_dataset_handling]
    postconditions:
      - "len(df) > 0"

  - id: extract_global_themes
    intent: "Discover a comprehensive, human-meaningful set of themes from the corpus. Use sampling for discovery if needed, but ensure themes are representative of the full population. Return as a list of strings."
    depends_on: [data_hygiene]
    worker_tier: T1
    produces: [theme_set]

  - id: generate_theme_vectors
    intent: "Vectorize the entire corpus by computing the intensity/presence of each extracted theme for every row. Documents should be treated as distributions across themes. Avoid sampling; process all rows."
    depends_on: [extract_global_themes]
    worker_tier: T2
    produces: [df_vectorized]

  - id: metadata_correlation_analysis
    intent: "Analyze the statistical relationship between theme intensity and available metadata (e.g., ratings, categories, or success metrics) as defined in the user's objective."
    depends_on: [generate_theme_vectors]
    worker_tier: T2

  - id: temporal_trend_analysis
    intent: "Analyze the evolution of themes and associated metadata over time. Use appropriate smoothing and temporal aggregation to identify significant shifts."
    depends_on: [generate_theme_vectors]
    worker_tier: T2

  - id: research_synthesis_report
    intent: "Synthesize all findings into a final research report. Interleave narrative insights with interactive visualizations to identify the primary drivers of trends in the data."
    depends_on: [metadata_correlation_analysis, temporal_trend_analysis]
    worker_tier: T1
    attached_skills: [visualization_best_practices, tabular_visualization]

# ——— GLOBAL INVARIANTS ———
invariants:
  - "SCALABILITY: For large datasets, favor out-of-core operations (DuckDB/Polars) over in-memory Pandas."
  - "NON-EXCLUSIVITY: A single document can (and often should) belong to multiple themes with varying degrees of intensity."
  - "USER-DRIVEN: Use the specific column names and constraints provided in the user's research objective."
---

# Thematic Analysis of Unstructured Text

## Rationale
This SOP provides a structured path for transforming raw text into actionable insights by treating qualitative content as quantitative theme distributions. By decoupling the *steps* of the analysis from the *specifics* of the data, GADS can apply advanced LLM-assisted thematic discovery to any corpus while maintaining statistical rigor.

## When to use
Use this recipe when the goal is to understand "The Why" behind metadata trends. It is ideal for customer feedback, support tickets, or any document collection where qualitative themes drive quantitative outcomes.

## Key Constraints
- The user's input instruction must clarify which text column and metadata fields (time, scores) are to be analyzed.
- Large-scale vectorization should prioritize efficient batch processing or vectorized string matching.
