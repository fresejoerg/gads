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
    /* Global Overrides */
    html, body, [class*="st-"] {
        font-size: 1.1rem;
        color: #0f172a;
    }
    
    .stApp { background-color: #ffffff; }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #ffffff;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Project Archive Cards */
    .stExpander {
        border: 1px solid #334155 !important;
        background-color: #1e293b !important;
        margin-bottom: 5px !important;
    }
    
    /* Task Tracking */
    .task-card { 
        background-color: #f8fafc; 
        padding: 24px; 
        border-radius: 2px; 
        border-left: 8px solid #0f172a;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .status-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        padding: 4px 8px;
        border-radius: 2px;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .status-pending { background-color: #e2e8f0; color: #0f172a; }
    .status-running { background-color: #2563eb; color: #ffffff; }
    .status-completed { background-color: #059669; color: #ffffff; }
    .status-failed { background-color: #dc2626; color: #ffffff; }
    
    h1, h2, h3 { color: #0f172a !important; font-weight: 800 !important; }
    
    /* Links */
    a { color: #2563eb !important; text-decoration: underline !important; font-weight: 600; }
    section[data-testid="stSidebar"] a { color: #38bdf8 !important; }
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
        except Exception:
            await asyncio.sleep(2)

# --- UI COMPONENTS ---

def render_sidebar():
    st.sidebar.title("Project Archive")
    st.sidebar.markdown("---")
    projects = asyncio.run(fetch_projects())
    for p in projects:
        # Parse timestamp for display
        dt = datetime.fromisoformat(p["created_at"].replace("Z", ""))
        ts_str = dt.strftime("%Y-%m-%d %H:%M")
        
        with st.sidebar.expander(f"{ts_str} | {p['name'][:20]}"):
            st.caption(f"PROJECT ID: {p['id']}")
            st.markdown(f"**[OPEN MASTER DASHBOARD]({BACKEND_URL}/projects/{p['id']}/files/final_dashboard.html)**")
            st.markdown(f"**[OPEN RESEARCH REPORT]({BACKEND_URL}/projects/{p['id']}/files/research_report.md)**")

def render_main():
    st.title("GADS Control Center")
    st.markdown("**Generative-augmented Data Science Orchestrator**")
    
    st.markdown("---")
    objective = st.text_area("**Research Objective**", placeholder="Describe your data science goal...")
    
    if st.button("Launch Workflow", type="primary", use_container_width=True):
        if objective:
            asyncio.run(start_workflow(objective))
            st.info("[SYSTEM] Workflow initiated.")
        else:
            st.warning("[SYSTEM] Objective required.")

    st.markdown("---")
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
                <div style="margin-top: 15px; color: #0f172a; font-weight: 500; font-size: 1.15rem;">{tinfo['description']}</div>
            </div>
            """, unsafe_allow_html=True)
            if tinfo['output']:
                with st.expander("View Execution Logs"):
                    st.code(tinfo['output'])

    with col2:
        st.subheader("Analysis Synthesis")
        if st.session_state.narrative:
            st.markdown(f"<div style='background-color: #f8fafc; padding: 25px; border: 1px solid #e2e8f0;'>{st.session_state.narrative}</div>", unsafe_allow_html=True)
            if st.session_state.takeaways:
                st.markdown("#### Key Takeaways")
                for t in st.session_state.takeaways:
                    st.markdown(f"- **{t}**")
        else:
            st.info("[SYSTEM] Synthesis pending completion of the workflow.")

        st.markdown("### Ground Truth")
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
