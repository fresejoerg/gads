# GADS: Generative-augmented Data Science

## The GADS Multi-Agent System

GADS is a professional, multi-agent system designed to act as an end-to-end Data Science copilot. The system utilizes a **Planner-Executor-Blackboard** architecture to orchestrate specialized sub-agents for tasks ranging from exploratory data analysis (EDA) and visualization to complex NLP extraction and model training.

### 📖 User Guide
For best practices on how to write optimal instructions for GADS, please refer to the **[User Guide](USER_GUIDE.md)**.

### Core Philosophy: "Gemini First, Local Focused"
A foundational requirement of GADS is the ability to run on **local LLMs** and **Gemini models** (via LiteLLM) as the primary brains, with Claude models serving as an architectural safety net (fallback).

---

## Key Features

*   **Project Manager (The Planner)**: A high-reasoning agent that decomposes user objectives into a deterministic task list with unique "Figure N" assignments and strict postcondition contracts.
*   **Auditor (Plan Critique)**: A Senior Auditor agent that evaluates proposed plans *before* execution, rejecting "lazy" or incomplete sequences and providing feedback for automatic re-planning.
*   **Architect (The Router)**: Classifies user objectives into structured intents (e.g., `binary_classification.tabular`) and seamlessly consults the Knowledge Base to fetch matching expert SOPs.
*   **Predictive Runtime Oracle**: Proactively predicts the execution time of generated code using AST analysis and data dimensions. If a task is estimated to exceed safety thresholds (e.g., 5 minutes), it is gracefully bypassed to prevent system hangs.
*   **Reproducible Project Bundles**: Automatically generates a complete handover ZIP for bypassed tasks, containing exported data artifacts (Parquet/Pickle), an end-to-end training script, and environment requirements for offline local execution.
*   **Automated Model Escalation**: Built-in resilience that automatically shifts tasks to higher-tier models (e.g., Haiku -> Sonnet -> Opus) upon execution failure or contract violation.
*   **GADS Control Center (UI)**: A 3-panel **Streamlit** IDE featuring a unified, professional UI styling, persistent **Project Archive**, and real-time **Task Tracking**.
*   **Professional Reporting Engine**: Automatically assembles an integrated **Interactive HTML Dashboard** and a formal **Markdown Research Report**. Powered by a native **Jinja2 + Bootstrap 5** templating engine, it provides responsive layouts and uses an XSS-immune, single-JS Plotly JSON embedding strategy to keep file sizes incredibly small (< 3MB) even with complex interactive charts. It also supports embedding static base64 images and handover bundle links in a single narrative thread.
*   **Automatic Data Introspection**: Instantly extracts file schemas and data samples using fast, low-memory DuckDB queries the moment an external dataset is mounted.
*   **Stateful Python Sandbox**: A secure, Docker-isolated execution environment using **IPython Kernels** to maintain variable state across multiple agent turns.
    *   **ML Ready**: Pre-installed with `scikit-learn`, `pandas`, `plotly`, `kaleido`, and `joblib`.
    *   **Automatic Plot Capture**: Captures both interactive HTML and static base64 visualizations.

---

## Project Structure

- `src/gads/core/`: Foundation logic (LLM connectors, Blackboard state, Execution Hub, Database Models, Reporting Engine).
- `src/gads/agents/`: Agent definitions (Router, Planner, Worker implementations).
- `src/gads/knowledge/recipes/`: YAML-backed Markdown recipes for data science best practices.
- `src/gads/ui/`: Streamlit-based control center with real-time state polling.
- `src/gads/tools/`: Sandbox client and security validators.

---

## Getting Started

### 1. Prerequisites
*   [MyLocalStack](https://github.com/deepfrese/MyLocalStack) running (LiteLLM on 4000, Sandbox on 8000).
*   **uv** Python package manager installed.

### 2. Installation
```bash
git clone git@codeberg.org:deepfrese/GADS.git
cd GADS
uv sync
```

### 3. Configuration
Copy `.env.example` to `.env` and fill in your API keys.

#### Local Only Mode
To run GADS entirely on local models (e.g., via LM Studio), set the following in your `.env`:
```env
GADS_LOCAL_ONLY=true
```

### 4. Running the Stack

**Start the Backend:**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
uv run uvicorn gads.core.server:app --host 0.0.0.0 --port 8001
```

**Start the GADS Control Center (Streamlit):**
```bash
./scripts/run_streamlit.sh
```
The UI will be available at `http://localhost:8003`.
