# GADS: Generative-augmented Data Science

## The GADS Multi-Agent System

GADS is a highly capable, multi-agent system designed to act as an end-to-end Data Science copilot. The system utilizes a **Planner-Executor-Blackboard** architecture to orchestrate specialized sub-agents for tasks ranging from exploratory data analysis (EDA) and visualization to complex NLP extraction and model training.

### Core Philosophy: "Gemini First, Local Focused"
A foundational requirement of GADS is the ability to run on **local LLMs** and **Gemini models** (via LiteLLM) as the primary brains, with Claude models serving as an architectural safety net (fallback).

---

## Key Features

*   **Project Manager (The Planner)**: A high-reasoning agent that decomposes user objectives into a deterministic task list with strict postcondition contracts.
*   **Architect (The Router)**: Classifies user objectives into structured intents (e.g., `binary_classification.tabular`) to fetch matching expert SOPs.
*   **Domain Expert SOPs (The Registry)**: Uses a YAML-backed library of Data Science best practices ("Recipes") to guide the Project Manager toward optimal workflows.
*   **Persistent Control Center (UI)**: Built with **Chainlit 2.11**, featuring a stable, real-time **Project State** side panel and an **Interactive File Explorer** for point-and-click artifact viewing.
*   **Authoritative Runtime Grounding**: Agents are forced to sync state directly from the sandbox kernel at the start of every request, preventing "semantic drift" or hallucinated data.
*   **Stateful Python Sandbox**: A secure, Docker-isolated execution environment using **IPython Kernels** to maintain variable state across multiple agent turns.
    *   **ML Ready**: Pre-installed with `scikit-learn`, `pandas`, and `joblib`.
    *   **Automatic Plot Capture**: Captures `matplotlib`/`seaborn` figures as base64 artifacts.
    *   **Self-Correction**: Automatic feedback loop that allows agents to fix their own code bugs via distilled tracebacks.
    *   **Resource Optimized**: Default support for up to 1000 processes/threads.

---

## Project Structure

- `src/gads/core/`: Foundation logic (LLM connectors, Blackboard state, Execution Hub, Model Registry, Knowledge Base).
- `src/gads/agents/`: Agent definitions (Router, Planner, Worker implementations).
- `src/gads/knowledge/recipes/`: YAML-backed Markdown recipes for data science best practices.
- `src/gads/ui/`: Chainlit-based control center with WebSocket event orchestration.
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
This will force all agents (Router, Planner, Workers) to use the `local_model` defined in your LiteLLM configuration.

### 4. Running the Stack

**Start the Backend:**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
uv run uvicorn gads.core.server:app --host 0.0.0.0 --port 8001
```

**Start the Control Center UI:**
```bash
uv run chainlit run src/gads/ui/app.py --port 8002
```
