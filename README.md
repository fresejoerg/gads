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
*   **Architect (The Router)**: Classifies user objectives into structured intents (e.g., `binary_classification.tabular`) to fetch matching expert SOPs.
*   **GADS Control Center (UI)**: A 3-panel **Streamlit** IDE featuring a strictly monochromatic high-contrast theme, persistent **Project Archive** with timestamps, and real-time **Task Tracking** via auto-refreshing fragments.
*   **Professional Reporting Engine**: Automatically assembles an integrated **Interactive HTML Dashboard** and a formal **Markdown Research Report** (anchored by Figure numbers) at the end of every analysis.
*   **Durable Project Memory**: All analytical results (Narratives, Takeaways) and Sandbox Variable States are persisted in a PostgreSQL database, ensuring projects are fully reloadable and stateful.
*   **Automated Maintenance**: Background tasks for periodic archive cleanup of old artifacts (>30 days) and orphaned task detection.
*   **Stateful Python Sandbox**: A secure, Docker-isolated execution environment using **IPython Kernels** to maintain variable state across multiple agent turns.
    *   **ML Ready**: Pre-installed with `scikit-learn`, `pandas`, `plotly`, `kaleido`, and `joblib`.
    *   **Automatic Plot Capture**: Captures **Plotly Express** and `matplotlib`/`seaborn` figures as interactive HTML or base64 artifacts.
    *   **Professional Defaults**: Pre-configured with the `plotly_white` template and professional color palettes.

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
