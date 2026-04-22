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
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FOR STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    .task-card { 
        background-color: white; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #3b82f6;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-pending { color: #64748b; font-weight: bold; }
    .status-running { color: #3b82f6; font-weight: bold; }
    .status-completed { color: #10b981; font-weight: bold; }
    .status-failed { color: #ef4444; font-weight: bold; }
    .artifact-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px;
        background: white;
        margin-top: 10px;
    }
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
    try:
        async with websockets.connect(f"{WS_URL}") as ws:
            while True:
                msg = await ws.recv()
                event = json.loads(msg)
                
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
    st.sidebar.title("📁 Project Archive")
    projects = asyncio.run(fetch_projects())
    for p in projects:
        with st.sidebar.expander(f"📝 {p['name'][:30]}..."):
            st.caption(f"ID: {p['id']}")
            st.markdown(f"[View Master Dashboard]({BACKEND_URL}/projects/{p['id']}/files/final_dashboard.html)")
            st.markdown(f"[View Research Report]({BACKEND_URL}/projects/{p['id']}/files/research_report.md)")

def render_main():
    st.title("🔬 GADS Control Center")
    st.markdown("### Agentic Data Science Orchestrator")
    
    objective = st.text_area("What is your research objective?", placeholder="e.g., Run a full EDA and predictive analysis on the Titanic dataset...")
    
    if st.button("🚀 Launch Expert Workflow", type="primary"):
        if objective:
            asyncio.run(start_workflow(objective))
            st.info("Workflow initiated. Monitoring events...")
        else:
            st.warning("Please enter an objective.")

    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("📋 Task Tracker")
        if not st.session_state.tasks:
            st.info("No active tasks.")
        for tid, tinfo in st.session_state.tasks.items():
            status_class = f"status-{tinfo['status']}"
            st.markdown(f"""
            <div class="task-card">
                <span class="{status_class}">{tinfo['status'].upper()}</span><br/>
                {tinfo['description']}
            </div>
            """, unsafe_allow_html=True)
            if tinfo['output']:
                with st.expander("View Logs"):
                    st.code(tinfo['output'])

    with col2:
        st.subheader("🧠 Research Synthesis")
        if st.session_state.narrative:
            st.markdown(st.session_state.narrative)
            if st.session_state.takeaways:
                st.markdown("#### Key Takeaways")
                for t in st.session_state.takeaways:
                    st.markdown(f"- {t}")
        else:
            st.info("Synthesis will appear here once the Lead Data Scientist completes the run.")

        st.subheader("📦 Ground Truth")
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
