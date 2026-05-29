---
id: causal_discovery.observational.constraint
version: 1.0.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [causal_inference]
  data_modality: [tabular]
  signals:
    - causal_structure_unknown: true
  anti_signals:
    - task: effect_estimation

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [causallearn, pandas, numpy, networkx, matplotlib]

# ——— DAG TEMPLATE ———
dag:
  - id: prepare_numeric_matrix
    intent: "Produce a clean numeric matrix from the dataset: drop non-numeric columns or encode categoricals, impute or drop missing values. Report the final variable list and matrix shape."
    worker_tier: T2
    produces: [data_matrix, var_names]
    postconditions:
      - "data_matrix is not None"
      - "data_matrix.shape[1] >= 2"

  - id: run_structure_learning
    intent: "Run the PC algorithm (causallearn.search.ConstraintBased.PC) with fisherz CI test and alpha=0.05. If latent confounders are suspected, use FCI instead. Store the resulting graph object in `discovered_graph`. Print the number of directed and undirected edges."
    depends_on: [prepare_numeric_matrix]
    worker_tier: T2
    produces: [discovered_graph]
    postconditions:
      - "discovered_graph is not None"

  - id: visualize_dag
    intent: "Render the discovered graph using networkx and matplotlib. Draw directed edges as arrows, undirected edges as plain lines, and bidirected edges (latent confounders) in red. Save as Figure 1."
    depends_on: [run_structure_learning]
    worker_tier: T2
    postconditions:
      - "discovered_graph is not None"

  - id: validate_assumptions
    intent: "Compare discovered edges against any domain priors provided in the objective. Flag edges that contradict domain knowledge. List all undirected or bidirected edges and explain their interpretation. Print a plain-text summary of the causal skeleton."
    depends_on: [visualize_dag]
    worker_tier: T2
    postconditions:
      - "discovered_graph is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "Always report the CI test type and alpha level used — results are sensitive to these choices."
  - "Never assert a definitive causal direction from an undirected edge in the output CPDAG."
  - "Bidirected edges in a PAG (from FCI) indicate a latent common cause, not direct causation."
  - "Do not use graphviz binary rendering — use networkx + matplotlib instead."
---

# Causal Discovery from Observational Data

## Rationale
When the causal structure among variables is unknown, **causal discovery algorithms** learn a DAG (or equivalence class) directly from data using conditional independence tests. This recipe uses the PC algorithm (constraint-based, sound and complete under faithfulness) via causal-learn. The result is a **CPDAG** (Completed Partially Directed Acyclic Graph) where directed edges represent identifiable causal directions and undirected edges represent Markov-equivalent alternatives.

## When to use
Use when the objective is to discover *which variables causally influence which others* without pre-specifying a graph. Examples: understanding disease pathways from biomarker data, finding drivers of a business KPI in high-dimensional feature sets, validating a hypothesised causal structure.

## Key Constraints
- Input must be numeric; encode categoricals before running.
- PC assumes no latent confounders (causal sufficiency); use FCI if this is doubtful.
- Results are Markov-equivalent — some edge directions may not be identifiable from data alone.
