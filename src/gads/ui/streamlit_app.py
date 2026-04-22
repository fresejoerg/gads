import streamlit as st
import httpx
import json
import asyncio
import websockets
import os
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional

# --- CONFIGURATION ---
BACKEND_URL = os.getenv("GADS_BACKEND_URL", "http://localhost:8001")
WS_URL = os.getenv("GADS_WS_URL", "ws://localhost:8001/ws")

st.set_page_config(
    page_title="GADS Control Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FOR STYLING (High Contrast / Professional) ---
st.markdown("""
<style>
    .stApp { background-color: #f1f5f9; }
    .task-card { 
        background-color: white; 
        padding: 20px; 
        border-radius: 4px; 
        border-left: 6px solid #1e293b;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    .status-label {
        font-family: monospace;
        font-size: 0.8rem;
        padding: 2px 6px;
        border-radius: 3px;
        text-transform: uppercase;
    }
    .status-pending { background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
    .status-running { background-color: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }
    .status-completed { background-color: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
    .status-failed { background-color: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
    
    .sidebar-project {
        padding: 10px;
        border-bottom: 1px solid #e2e8f0;
    }
    h1, h2, h3 { color: #0f172a !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "events" not in st.session_state:
    st.session_state.events = []
if "current_project_id" not in st.session_state:
    st.session_state.current_project_id = None
if "project_files" not in st.session_state:
    st.session_state.project_files = []
if "project_state" not in st.session_state:
    st.session_state.project_state = {}
if "tasks" not in st.session_state:
    st.session_state.tasks = {}
if "narrative" not in st.session_state:
    st.session_state.narrative = ""
if "takeaways" not in st.session_state:
    st.session_state.takeaways = []

# --- API HELPERS ---
async def fetch_projects():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BACKEND_URL}/projects")
            return resp.json()
    except Exception as e:
        st.error(f"Failed to fetch projects: {e}")
        return []

async def start_workflow(objective: str):
    try:
        async with httpx.AsyncClient() as client:
            payload = {"name": f"Project {datetime.now().strftime('%Y-%m-%d %H:%M')}", "objective": objective}
            resp = await client.post(f"{BACKEND_URL}/projects", json=payload)
            data = resp.json()
            st.session_state.current_project_id = data["project"]["id"]
            st.session_state.tasks = {}
            st.session_state.narrative = ""
            st.session_state.takeaways = []
            return data
    except Exception as e:
        st.error(f"Failed to start workflow: {e}")

# --- WEBSOCKET LISTENER ---
async def listen_to_ws():
    while True:
        try:
            last_seq = st.session_state.get("last_seq", 0)
            async with websockets.connect(f"{WS_URL}?last_seq={last_seq}") as ws:
                while True:
                    msg = await ws.recv()
                    event = json.loads(msg)
                    st.session_state.last_seq = event["seq"]
                    
                    # Update local state based on event
                    etype = event["type"]
                    payload = event["payload"]
                    
                    if etype == "TASK_CREATED":
                        st.session_state.tasks[payload["task_id"]] = {
                            "description": payload["description"],
                            "status": "pending",
                            "output": ""
                        }
                    elif etype == "TASK_STARTED":
                        if payload["task_id"] in st.session_state.tasks:
                            st.session_state.tasks[payload["task_id"]]["status"] = "running"
                    elif etype == "TASK_COMPLETED":
                        if payload["task_id"] in st.session_state.tasks:
                            st.session_state.tasks[payload["task_id"]]["status"] = "completed"
                            st.session_state.tasks[payload["task_id"]]["output"] = payload.get("result", {}).get("stdout", "")
                    elif etype == "TASK_FAILED":
                        if payload["task_id"] in st.session_state.tasks:
                            st.session_state.tasks[payload["task_id"]]["status"] = "failed"
                            st.session_state.tasks[payload["task_id"]]["output"] = payload.get("error", "")
                    elif etype == "STATE_UPDATED":
                        st.session_state.project_files = payload.get("files", [])
                        st.session_state.project_state = payload.get("state", {})
                    elif etype == "WORKFLOW_FINAL_RESULT":
                        st.session_state.narrative = payload["narrative"]
                        st.session_state.takeaways = payload["takeaways"]
                    
                    # Trigger a rerun to show new state
                    st.rerun()
    except Exception as e:
        # In a real app, you'd handle reconnection here
        pass

# --- UI COMPONENTS ---

def render_sidebar():
    st.sidebar.title("Project Archive")
    projects = asyncio.run(fetch_projects())
    for p in projects:
        with st.sidebar.expander(f"PROJECT: {p['name'][:30]}"):
            st.caption(f"ID: {p['id']}")
            st.markdown(f"[VIEW DASHBOARD]({BACKEND_URL}/projects/{p['id']}/files/final_dashboard.html)")
            st.markdown(f"[VIEW REPORT]({BACKEND_URL}/projects/{p['id']}/files/research_report.md)")

def render_main():
    st.title("GADS Control Center")
    st.markdown("Generative-augmented Data Science Orchestrator")
    
    objective = st.text_area("Research Objective", placeholder="Describe your data science goal...")
    
    if st.button("Launch Workflow", type="primary"):
        if objective:
            asyncio.run(start_workflow(objective))
            st.info("[SYSTEM] Workflow initiated.")
        else:
            st.warning("[SYSTEM] Objective required.")

    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Task Tracking")
        if not st.session_state.tasks:
            st.info("[SYSTEM] No active tasks.")
        for tid, tinfo in st.session_state.tasks.items():
            status = tinfo['status'].upper()
            st.markdown(f"""
            <div class="task-card">
                <span class="status-label status-{tinfo['status']}">{status}</span><br/>
                <div style="margin-top: 10px; color: #334155;">{tinfo['description']}</div>
            </div>
            """, unsafe_allow_html=True)
            if tinfo['output']:
                with st.expander("Logs"):
                    st.code(tinfo['output'])

    with col2:
        st.subheader("Synthesis")
        if st.session_state.narrative:
            st.markdown(st.session_state.narrative)
            if st.session_state.takeaways:
                st.markdown("#### Key Takeaways")
                for t in st.session_state.takeaways:
                    st.markdown(f"- {t}")
        else:
            st.info("[SYSTEM] Synthesis pending completion.")

        st.subheader("Ground Truth")
        tab1, tab2 = st.tabs(["Files", "Memory"])
        with tab1:
            if not st.session_state.project_files:
                st.write("No files in workspace.")
            for f in st.session_state.project_files:
                st.markdown(f"- `{f}` ([view]({BACKEND_URL}/projects/{st.session_state.current_project_id}/files/{f}))")
        with tab2:
            if not st.session_state.project_state:
                st.write("Sandbox memory is empty.")
            else:
                st.json(st.session_state.project_state)

# --- EXECUTION ---
render_sidebar()
render_main()

# Start WS listener in a way that Streamlit can handle
# Note: This POC uses a background task. For production, a more robust event loop would be needed.
if st.session_state.current_project_id and not st.session_state.get("ws_started"):
    st.session_state.ws_started = True
    asyncio.run(listen_to_ws())
