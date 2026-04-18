# GADS: Generative-augmented Data Science

## The "Data Science Rockstar" Multi-Agent System

GADS is a repository for developing highly capable, multi-agent systems designed to act as an end-to-end Data Science copilot. The system utilizes a **Planner-Executor-Blackboard** architecture to orchestrate specialized sub-agents for tasks ranging from exploratory data analysis (EDA) and visualization to complex NLP extraction and model training.

### Core Philosophy: "Local First, Hybrid Ready"
A foundational requirement of GADS is the ability to run on **local LLMs** (via LM Studio, Ollama, etc.) while leveraging high-powered remote models (Claude 4.7, Gemini 3.1) for high-level planning and orchestration.

---

## Architecture

The system is built on a custom, lightweight framework using **LiteLLM**, **Pydantic**, and **Instructor**:

1.  **Project Manager (The Planner)**: A high-reasoning agent that decomposes user objectives into a deterministic Directed Acyclic Graph (DAG) of tasks.
2.  **Specialized Workers (The Sub-Agents)**:
    *   `NLPExtractor`: Precision entity and structured data extraction.
    *   `CodeGenerator`: Expert Python developer for data science tasks.
    *   `DataViz`: (In Progress) Specialized in generating visualization code.
3.  **The Blackboard**: A central Pydantic-based state repository where all agents store and retrieve artifacts.
4.  **Stateful Python Sandbox**: A secure, Docker-isolated execution environment using **IPython Kernels** to maintain variable state across multiple agent turns.
    *   **Automatic Plot Capture**: Captures `matplotlib`/`seaborn` figures as base64 artifacts.
    *   **Self-Correction**: Automatic feedback loop that allows agents to fix their own code bugs via distilled tracebacks.
    *   **Security**: AST-level validation to block dangerous imports and calls.

---

## Project Structure

- `src/gads/core/`: Foundation logic (LLM connectors, Blackboard state, Execution Manager).
- `src/gads/agents/`: Agent definitions (Base class, Planner).
- `src/gads/agents/workers/`: Specialized sub-agent implementations (`nlp.py`, `coder.py`).
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

### 3. Running the Demo
Execute the stateful sandbox demo to see a multi-turn data science workflow (Create -> Analyze -> Plot):
```bash
uv run main.py
```

---

## Development Roadmap

- [x] Foundation: custom multi-agent orchestration.
- [x] Planner-Worker-Blackboard architecture.
- [x] Reliable structured output for local models via Instructor.
- [x] **Stateful Python Sandbox with automatic plot capture.**
- [x] **Code-Execution-Feedback Loop for agent self-correction.**
- [ ] Implement `DataVizAgent` for automatic chart generation.
- [ ] Full automated Executor loop for complex multi-step DAGs.
