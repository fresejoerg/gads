# GADS: Generative-augmented Data Science

## The "Data Science Rockstar" Multi-Agent System

GADS is a highly capable, multi-agent system designed to act as an end-to-end Data Science copilot. The system utilizes a **Planner-Executor-Blackboard** architecture to orchestrate specialized sub-agents for tasks ranging from exploratory data analysis (EDA) and visualization to complex NLP extraction and model training.

### Core Philosophy: "Local First, Hybrid Ready"
A foundational requirement of GADS is the ability to run on **local LLMs** (via LM Studio, Ollama, etc.) while leveraging high-powered remote models (Claude 4.7, Gemini 3.1) for high-level planning and orchestration.

---

## Key Features

*   **Project Manager (The Planner)**: A high-reasoning agent that decomposes user objectives into a deterministic task list with strict postcondition contracts.
*   **Persistent Control Center (UI)**: Built with **Chainlit 2.11**, featuring a stable, real-time **Project State** side panel that tracks workspace files and live sandbox memory.
*   **Authoritative Runtime Grounding**: Agents are forced to sync state directly from the sandbox kernel at the start of every request, preventing "semantic drift" or hallucinated data.
*   **Stateful Python Sandbox**: A secure, Docker-isolated execution environment using **IPython Kernels** to maintain variable state across multiple agent turns.
    *   **Automatic Plot Capture**: Captures `matplotlib`/`seaborn` figures as base64 artifacts.
    *   **Self-Correction**: Automatic feedback loop that allows agents to fix their own code bugs via distilled tracebacks.
    *   **Resource Optimized**: Default support for up to 1000 processes/threads to handle heavy data science loads.

---

## Project Structure

- `src/gads/core/`: Foundation logic (LLM connectors, Blackboard state, Execution Hub, Model Registry).
- `src/gads/agents/`: Agent definitions (Base class, Planner, Worker implementations).
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

### 3. Running the Control Center UI
```bash
uv run chainlit run src/gads/ui/app.py --port 8002
```

### 4. Running the Sandbox CLI
Execute the stateful sandbox demo to see a multi-turn data science workflow (Create -> Analyze -> Plot):
```bash
uv run main.py
```

---

## Development Roadmap

- [x] Foundation: custom multi-agent orchestration.
- [x] Planner-Worker-Blackboard architecture.
- [x] **Real-time Stateful UI with persistent sidebar.**
- [x] **Authoritative State Grounding from IPython Kernels.**
- [x] **Stateful Python Sandbox with automatic plot capture.**
- [x] **Code-Execution-Feedback Loop for agent self-correction.**
- [ ] Implement `DataVizAgent` for automatic chart generation.
- [ ] Full automated Executor loop for complex multi-step DAGs.
