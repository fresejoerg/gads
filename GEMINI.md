# GADS Project Instructions

## Launching the Stack

To launch the complete GADS stack, ensure **MyLocalStack** is running (ports 4000 and 8000), then execute the following in parallel as background processes:

### 1. GADS Backend (Port 8001)
The backend manages the multi-agent workflows and database state.
```bash
PYTHONPATH=src uv run uvicorn gads.core.server:app --host 0.0.0.0 --port 8001
```

### 2. GADS Control Center UI (Port 8003)
The Streamlit interface for project management.
```bash
./scripts/run_streamlit.sh
```

## Engineering Standards & Output Fidelity

### 1. Strict Requirement Enforcement (Critique)
The `CritiqueAgent` is configured with **STRICT** adherence rules. If a user objective contains a specific quantitative or structural request (e.g., "list the top 5", "show a table of X"), that request MUST be fulfilled by a visual artifact (Plotly chart or table). 
- A narrative description alone is a **FAILURE**.
- Missing artifacts for specific requests will trigger a **REJECTION** and a synthesis retry.

### 2. Tabular Visualization Standard
For any request involving specific records, "Top N" lists, or small dataframes, use the `tabular_visualization` skill:
- **Requirement**: Use `plotly.graph_objects.Table`.
- **Persistence**: Save as an interactive HTML artifact so it appears in the final dashboard.

## Verification & Health Checks
- **Backend Health**: `curl http://localhost:8001/health` -> `{"status": "ok"}`
- **UI Accessibility**: Verify port `8003` is listening.
- **Database**: The backend automatically initializes tables on startup via `init_db()`.
