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

## Verification & Health Checks
- **Backend Health**: `curl http://localhost:8001/health` -> `{"status": "ok"}`
- **UI Accessibility**: Verify port `8003` is listening.
- **Database**: The backend automatically initializes tables on startup via `init_db()`.
