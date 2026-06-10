---
id: causal_effect.observational.dowhy
version: 1.1.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [causal_inference]
  data_modality: [tabular]
  signals:
    - treatment_outcome_pair: true
  anti_signals:
    - task: heterogeneous_treatment_effects

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [dowhy, statsmodels, sklearn, pandas]

# ——— DAG TEMPLATE ———
dag:
  - id: define_causal_question
    intent: >
      Identify roles and select confounder columns.
      You MUST use this exact Python code block:
      ```python
      import pandas as pd
      import numpy as np

      if 'df' not in globals():
          df = pd.read_csv('creditcard.csv')

      treatment_col = 'Amount'
      outcome_col = 'Class'

      temporal_id_patterns = {'time', 'date', 'timestamp', 'id', 'index'}
      confounder_cols = [
          c for c in df.columns
          if c not in (treatment_col, outcome_col)
          and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
          and not any(p in c.lower() for p in temporal_id_patterns)
      ]

      if len(confounder_cols) > 10:
          corrs = df[confounder_cols].corrwith(df[outcome_col]).abs()
          confounder_cols = corrs.nlargest(10).index.tolist()

      minority_class_frac = float(df[outcome_col].value_counts(normalize=True).min())

      print(f"Treatment: {treatment_col}")
      print(f"Outcome: {outcome_col}")
      print(f"Confounders: {confounder_cols}")
      print(f"Minority class fraction: {minority_class_frac}")
      ```
    worker_tier: T2
    produces: [treatment_col, outcome_col, confounder_cols, minority_class_frac]
    postconditions:
      - "isinstance(treatment_col, str)"
      - "isinstance(outcome_col, str)"
      - "len(confounder_cols) <= 10"

  - id: engineer_and_build_graph
    intent: >
      Engineer treatment and build graph.
      You MUST use this exact Python code block:
      ```python
      import pandas as pd
      import networkx as nx
      from dowhy import CausalModel

      if df[treatment_col].nunique() > 2:
          global_median = float(df[treatment_col].median())
          bin_col = f"high_{treatment_col}"
          df[bin_col] = (df[treatment_col] > global_median).astype(int)
          treatment_col = bin_col
          print(f"Binarized treatment: {treatment_col}")

      nodes = [treatment_col, outcome_col] + list(confounder_cols)
      node_idx = {n: i for i, n in enumerate(nodes)}
      node_str = "\n".join(f'  node [ id {i} label "{n}" ]' for i, n in enumerate(nodes))
      edges = (
          [(c, treatment_col) for c in confounder_cols]
          + [(c, outcome_col) for c in confounder_cols]
          + [(treatment_col, outcome_col)]
      )
      edge_str = "\n".join(
          f'  edge [ source {node_idx[s]} target {node_idx[t]} ]' for s, t in edges
      )
      gml_string = "\n".join(["graph [ directed 1", node_str, edge_str, "]"])

      causal_model = CausalModel(
          data=df,
          treatment=treatment_col,
          outcome=outcome_col,
          graph=gml_string,
      )
      ```
    depends_on: [define_causal_question]
    worker_tier: T2
    produces: [causal_model, treatment_col]
    postconditions:
      - "causal_model is not None"

  - id: identify_estimand
    intent: >
      Identify the estimand.
      You MUST use this exact Python code block:
      ```python
      identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
      print(identified_estimand)
      ```
    depends_on: [engineer_and_build_graph]
    worker_tier: T2
    produces: [identified_estimand]
    postconditions:
      - "identified_estimand is not None"

  - id: estimate_effect
    intent: >
      Call the native causal estimation node and unpack metrics.
      You MUST use this exact Python code block:
      ```python
      result = gads_causal_estimate_ate(df, treatment_col, outcome_col, confounder_cols)

      ate = float(result["ate"])
      placebo_new_effect = float(result["placebo_new_effect"])
      subset_new_effect = float(result["subset_new_effect"])
      causal_estimate = result["causal_estimate"]
      refutation_results = {
          "placebo_new_effect": placebo_new_effect,
          "subset_new_effect": subset_new_effect
      }

      print(f"ATE={ate:.6f}")
      print(f"placebo={placebo_new_effect:.6f}")
      print(f"subset={subset_new_effect:.6f}")

      gads_emit_insight("causal_effect", f"ATE={ate:.4f}, placebo={placebo_new_effect:.4f}, subset={subset_new_effect:.4f}")
      ```
    depends_on: [identify_estimand]
    worker_tier: T2
    produces: [causal_estimate, ate, placebo_new_effect, subset_new_effect, refutation_results]
    postconditions:
      - "causal_estimate is not None"
    required_metrics: [ate]

  - id: refute_estimate
    intent: >
      Read refutation metrics from the kernel.
      You MUST use this exact Python code block:
      ```python
      assert isinstance(placebo_new_effect, float), "placebo_new_effect missing"
      assert isinstance(subset_new_effect, float), "subset_new_effect missing"

      print(f"placebo_new_effect={placebo_new_effect:.6f}")
      print(f"subset_new_effect={subset_new_effect:.6f}")

      refutation_results = {
          "placebo_new_effect": placebo_new_effect,
          "subset_new_effect": subset_new_effect
      }

      gads_emit_insight("refutation", f"Placebo effect={placebo_new_effect:.4f} (should be ~0). Subset effect={subset_new_effect:.4f} (should be ~ATE).")
      ```
    depends_on: [estimate_effect]
    worker_tier: T2
    produces: [refutation_results, placebo_new_effect, subset_new_effect]
    postconditions:
      - "refutation_results is not None"
    required_metrics: [placebo_new_effect, subset_new_effect]

# ——— GLOBAL INVARIANTS ———
invariants:
  - "USE DOWHY LIBRARY: always use CausalModel.estimate_effect() — NEVER implement propensity scoring, DR/AIPW, or causal estimation manually from scratch. DoWhy handles all of this internally."
  - "LARGE DATASET PERFORMANCE: for datasets >20K rows, ALWAYS subsample to 20K before building CausalModel. Use df_sample = df.sample(20000, random_state=42). Build CausalModel(data=df_sample, ...). NEVER run PSM or DML on the full dataset — it will timeout."
  - "CONFOUNDERS: infer from schema as numeric columns that are not treatment, outcome, or temporal/ID. Never ask the user to list them."
  - "TREATMENT ENGINEERING: if treatment is continuous, always binarise at its median. Name the new column high_{original_name}."
  - "GML CONSTRUCTION: always build the GML string programmatically from the confounder list. Never hardcode node IDs. Use INTEGER id + STRING label: node [ id 0 label \"treatment\" ]. Edges use integer source/target indices. String ids fail with NetworkXError."
  - "NATIVE NODE: prefer gads_causal_estimate_ate(df, treatment_col, outcome_col, confounder_cols) — it handles all GML construction, subsampling, and refutation internally."
  - "ESTIMATOR SELECTION: check minority_class_frac. Use propensity_score_matching for rare outcomes (<5%), linear_regression otherwise. Never use propensity_score_weighting — it produces NaN on imbalanced data."
  - "CLASS IMBALANCE: never use imblearn/SMOTE (not installed). Use class_weight='balanced' in sklearn propensity models if needed."
  - "REFUTATION: always reuse the existing causal_model object — never rebuild it. Both refuters are mandatory. Store placebo_new_effect and subset_new_effect as plain Python floats."
  - "POSTCONDITION CONTRACTS: required_columns must contain only string column names from the actual dataset schema, never integer indices."
  - "Random state must be 42 everywhere."
  - "Store all metric values (ate, placebo_new_effect, subset_new_effect) as plain Python floats, not numpy scalars."
---

# Causal Effect Estimation with DoWhy (Observational Data)

## Rationale
This recipe implements the DoWhy four-step workflow: **Model → Identify → Estimate → Refute**. It encodes the methodology decisions that should not need to appear in a spec: confounder identification from the schema, treatment engineering from continuous variables, estimator selection based on outcome class balance, and programmatic GML construction.

## When to use
Use when the objective is to estimate *how much* a treatment causally affects an outcome in observational tabular data. The spec needs only to identify the treatment column, outcome column, and any columns that are purely temporal or identifiers (to exclude from confounders).

## Key Constraints
- Treatment and outcome must be identifiable from the schema and objective.
- Temporal/ID columns must be excluded from the confounder set — the recipe infers this automatically.
- Refutation is mandatory before reporting an ATE.
