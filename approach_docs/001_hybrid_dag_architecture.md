# Approach Doc: 001 - Hybrid DAG Architecture

## 1. Objective
Evolve GADS from a system that generates ephemeral Python code for every task into a high-fidelity orchestrator that mixes **LLM Reasoning** with **Audited Deterministic Primitives (Native Nodes)**.

## 2. The "Expert Chef" Analogy
- **LLM Nodes (The Chef)**: Decide *how* to season a steak based on the grain of the meat and the kitchen environment.
- **Deterministic Nodes (The Oven)**: A pre-calibrated tool that performs a specific, high-fidelity task (e.g., baking at 400°F) with 100% reliability.

## 3. Node Taxonomy
Every node in a GADS DAG must now be assigned one of three roles:

| Node Type | Description | Primary Engine | Example |
| :--- | :--- | :--- | :--- |
| **Reasoning** | Strategic decisions and synthesis. | Tier 1 LLM | "Which features are most predictive?" |
| **Stochastic** | One-off, custom data manipulation. | Tier 2/3 LLM + Code | "Parse this unique, non-standard timestamp." |
| **Deterministic**| Standardized, audited DS operations. | Python Code / API | `gads.native.split_data()` |

---

## 4. The Functional Registry (`gads.*`)
A centralized library of audited data science primitives stored in `src/gads/knowledge/native/`.

### Registry Schema (JSON)
Each native node must define:
- `id`: Unique identifier (e.g., `gads.preprocessing.impute`).
- `input_schema`: Pydantic/JSON Schema of required variables and types.
- `output_schema`: Declaration of variables produced.
- `error_taxonomy`: Mapping of Python exceptions to **Remediation Hints**.

---

## 5. The "Contract Gate" Protocol
To prevent "Schema Drift" (the silent killer of LLM-to-Deterministic handovers), we implement the following pipeline:

1. **Pre-Flight Check**: The Orchestrator introspects the Sandbox Blackboard (dtypes, columns) and compares them against the Native node's `input_schema`.
2. **The Gate**: If a mismatch is found, the DAG is interrupted and a **Repair Node (LLM)** is inserted to perform the necessary renames or casts.
3. **Execution**: The Native code is injected into the IPython kernel and executed.
4. **Fact Emission**: Upon success, the node emits a "Fact" to the Blackboard (e.g., `is_stratified: True`).

---

## 6. Structured Error Taxonomy
Native nodes will not return raw Python tracebacks to the LLM. Instead, they use a translation layer:

- **Raw Error**: `ValueError: Mean of empty slice`
- **GADS Translation**: `ERROR_IMPUTE_ALL_NULL: Column 'income' is 100% null. Remediate by dropping the column or using a constant fill.`

This ensures the LLM remains in the "Reasoning" loop rather than getting lost in low-level debugging.

---

## 7. Implementation Roadmap

### Phase 1: The Toolbelt (Core)
- Implement `src/gads/knowledge/native/preprocessing.py`.
- Extend `KnowledgeRegistry` to load native function manifests.
- Pre-load `gads.*` namespace into every Sandbox session.

### Phase 2: Contract Enforcement (Safety)
- Build the `SchemaValidator` tool.
- Integrate validation into `ExecutionManager.run_task`.

### Phase 3: Content-Addressable Caching (Efficiency)
- Implement hashing for Native node outputs.
- Enable skip-logic for retries based on `hash(func + input_data)`.

### Phase 4: Hybrid Planner (Orchestration)
- Update `DataSciencePlanner` to proactively select `node_type: NATIVE` when a matching intent is found in the SOP Wiki.
