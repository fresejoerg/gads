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
    *   `DataViz`: (In Progress) Specialized in generating visualization code.
    *   `CodeRunner`: (In Progress) Safe, sandboxed Python execution for data processing.
3.  **The Blackboard**: A central Pydantic-based state repository where all agents store and retrieve artifacts, preventing context window bloat and ensuring data integrity.
4.  **Instructor Integration**: Uses the `instructor` library with `MD_JSON` mode to ensure local models strictly adhere to Pydantic schemas through automatic retries and validation.

---

## Project Structure

- `src/gads/core/`: Foundation logic (LLM connectors, Blackboard state).
- `src/gads/agents/`: Agent definitions (Base class, Planner).
- `src/gads/agents/workers/`: Specialized sub-agent implementations.
- `src/gads/tools/`: Functional tools and execution environments.

---

## Getting Started

### 1. Prerequisites
*   [MyLocalStack](https://github.com/deepfrese/MyLocalStack) running (LiteLLM proxy on port 4000).
*   **uv** Python package manager installed.

### 2. Installation
```bash
git clone git@codeberg.org:deepfrese/GADS.git
cd GADS
uv sync
```

### 3. Configuration
Create a `.env` file in the root directory (see `.env.example`):
```bash
LITELLM_BASE_URL=http://localhost:4000/v1
LITELLM_MASTER_KEY=sk-1234
ANTHROPIC_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

### 4. Running the Demo
Execute the main entry point to see the Planner and NLP Worker in action:
```bash
uv run main.py
```

---

## Development Roadmap

- [x] Foundation: custom multi-agent orchestration.
- [x] Planner-Worker-Blackboard architecture.
- [x] Reliable structured output for local models via Instructor.
- [ ] Implement `DataVizAgent` for automatic chart generation.
- [ ] Implement `Sandbox` for safe Python code execution.
- [ ] Full automated Executor loop for multi-step DAGs.
