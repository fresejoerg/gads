import httpx
import json
import sys

plan_text = """
# Plan: System Prompt Editor Integration

## Objective
To provide a dedicated interface within the GADS Control Center UI for viewing, editing, and persistently modifying the foundational system prompts that govern all worker and orchestrator agents (e.g., Planner, Router, Coder, Synthesizer).

## Background & Motivation
Currently, system prompts are hardcoded directly into individual Python files (e.g., `src/gads/agents/planner.py`, `src/gads/agents/workers/coder.py`). Modifying these prompts requires opening the source code and restarting the server. We will extract all system prompts into a centralized, dedicated configuration file (or directory of files). The UI will then be updated to act directly on these files, allowing data scientists and architects to rapidly tune agent behaviors and inject new global constraints dynamically.

## Proposed Solution: Centralized File-Based Prompts

### 1. Centralized Storage (`src/gads/prompts.yaml`)
- Create a new central configuration file, e.g., `src/gads/prompts.yaml` (or a `prompts/` directory with individual `.md`/`.txt` files per agent). A YAML file is generally preferred for mapping agent names to prompt strings.
- **Structure**:
  - `Planner`: "You are a Senior Data Science Planner..."
  - `Router`: "You are a Senior Data Science Architect..."
  - `Coder`: "You are an elite Data Science Coder..."
  - `Synthesizer`: "You are a Lead Data Scientist..."

### 2. Core Agent Refactoring (`src/gads/agents/base.py` & Subclasses)
- **Extraction**: Remove the hardcoded `*_SYSTEM_PROMPT` strings from all agent Python files.
- **Dynamic Loading**: Update the `BaseAgent` or individual agent classes to dynamically read their system prompt from the central `prompts.yaml` file during initialization or execution. This ensures the agent always uses the latest version saved to disk.

### 3. API Endpoints (`src/gads/core/server.py`)
- `GET /prompts`: Parses the `prompts.yaml` file and returns a JSON payload of all recognized agents and their current prompt strings.
- `POST /prompts/{agent_name}`: Receives an updated prompt string for a specific agent, modifies the `prompts.yaml` file on disk, and saves the changes.

### 4. UI Implementation (`src/gads/ui/streamlit_app.py`)
- Expand the **KNOWLEDGE BASE** panel (which currently holds "Recipes" and "Skills") to include a third radio toggle option: **"Prompts"**.
- When "Prompts" is selected:
  - Populate the selectbox with the recognized agent names parsed from the API.
  - Render a `st_ace` text editor displaying the active prompt for the selected agent.
  - Provide a **SAVE PROMPT** button (which calls the `POST` endpoint to overwrite the file on disk).

## Verification & Testing
1. **Extraction Verification**: Ensure the backend starts up successfully and agents load their prompts from the YAML file without `NameError` or missing variable issues.
2. **API Validation**: Hit the `GET /prompts` endpoint to ensure all agents and their strings are successfully loaded from disk.
3. **Persistence**: Modify the Coder's prompt via the UI, save it, and restart the backend. The UI should still show the edited value, and the `prompts.yaml` file on disk should reflect the change.
4. **Agent Adoption**: Launch a dummy workflow and inspect the LLM payload or logs to confirm the agent was injected with the newly customized system prompt read from the file.
"""

prompt = f"""
Please stress test and critique the following architectural plan for integrating a System Prompt Editor into a Python/FastAPI/Streamlit agentic framework (GADS). 

{plan_text}

Provide your critique focusing on:
1. Potential pitfalls with file-based prompt storage (e.g., concurrency, formatting/escaping).
2. UI/UX issues or edge cases.
3. Recommendations for making this bulletproof.
"""

try:
    resp = httpx.post(
        "http://localhost:4000/v1/chat/completions",
        headers={"Authorization": "Bearer sk-1234"},
        json={
            "model": "claude-opus-4.7",
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=120.0
    )
    print(resp.json()["choices"][0]["message"]["content"])
except Exception as e:
    print(f"Error querying LiteLLM proxy: {e}")
    sys.exit(1)
