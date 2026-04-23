import streamlit as st
import httpx
import json
import asyncio
import websockets
import os
import base64
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

# --- CONFIGURATION ---
BACKEND_URL = os.getenv("GADS_BACKEND_URL", "http://localhost:8001")
WS_URL = os.getenv("GADS_WS_URL", "ws://localhost:8001/ws")
WORKSPACE_ROOT = "/home/joergf/projects/MyLocalStack/data/workspaces"

st.set_page_config(
    page_title="GADS Control Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- WEBSOCKET THREAD HANDLER ---
def run_ws_thread(loop, last_seq):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(listen_to_ws_background(last_seq))

async def listen_to_ws_background(start_seq):
    curr_seq = start_seq
    while True:
        try:
            async with websockets.connect(f"{WS_URL}?last_seq={curr_seq}") as ws:
                while True:
                    msg = await ws.recv()
                    event = json.loads(msg)
                    curr_seq = event["sequence"]
                    
                    # Update the global session state from background
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
                    
                    st.session_state.last_seq = curr_seq
                    # Signal a refresh
                    st.session_state.needs_rerun = True
        except Exception:
            await asyncio.sleep(2)

# --- CSS FOR STYLING (Strictly Monochromatic High Contrast) ---
st.markdown("""
<style>
    /* Global Overrides */
    html, body, [class*="st-"] {
        font-size: 1.1rem;
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    .stApp { background-color: #ffffff !important; }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #000000 !important;
    }
    
    /* FIX: Force '>>' collapse icon and all sidebar buttons to be black */
    [data-testid="stSidebarCollapse"] button,
    section[data-testid="stSidebar"] button {
        color: #000000 !important;
    }
    [data-testid="stSidebarCollapse"] svg,
    section[data-testid="stSidebar"] svg {
        fill: #000000 !important;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #000000 !important;
    }
    
    /* Project Archive Cards - High Contrast */
    .stExpander {
        border: 2px solid #000000 !important;
        background-color: #ffffff !important;
        margin-bottom: 10px !important;
    }
    
    /* Force text inside sidebar expanders to be black */
    section[data-testid="stSidebar"] .stExpander div[data-testid="stExpanderDetails"] p,
    section[data-testid="stSidebar"] .stExpander div[data-testid="stExpanderDetails"] span {
        color: #000000 !important;
    }
    
    /* Task Tracking */
    .task-card { 
        background-color: #ffffff; 
        padding: 16px; 
        border-radius: 0; 
        border: 2px solid #000000;
        margin-bottom: 12px;
    }
    
    .status-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        padding: 4px 8px;
        border-radius: 0;
        font-weight: 900;
        border: 2px solid #000000;
        display: inline-block;
        color: #000000 !important;
    }
    /* Monochromatic Status indicators */
    .status-pending { background-color: #ffffff; }
    .status-running { background-color: #ffffff; text-decoration: underline; }
    .status-completed { background-color: #ffffff; font-style: italic; }
    .status-failed { background-color: #ffffff; border-style: dashed; }
    
    h1, h2, h3 { color: #000000 !important; font-weight: 900 !important; }
    
    /* Links */
    a { color: #000000 !important; text-decoration: underline !important; font-weight: 700; }
    section[data-testid="stSidebar"] a { color: #000000 !important; }
    
    /* Input & Textarea Contrast Fix */
    .stTextArea textarea {
        color: #000000 !important;
        background-color: #ffffff !important;
        -webkit-text-fill-color: #000000 !important;
        caret-color: #000000 !important;
        border: 3px solid #000000 !important;
    }

    .stTextInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
        -webkit-text-fill-color: #000000 !important;
        caret-color: #000000 !important;
        border: 3px solid #000000 !important;
    }


    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border: 1px solid #000000 !important;
        padding: 8px 16px;
        background-color: #ffffff;
    }

    /* Force all text in the main area to be black */
    .main .stMarkdown p, .main .stMarkdown li {
        color: #000000 !important;
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
if "last_seq" not in st.session_state:
    st.session_state.last_seq = 0
if "needs_rerun" not in st.session_state:
    st.session_state.needs_rerun = False

# --- API HELPERS ---
def fetch_projects():
    try:
        with httpx.Client() as client:
            resp = client.get(f"{BACKEND_URL}/projects")
            return resp.json()
    except Exception as e:
        st.error(f"Failed to fetch projects: {e}")
        return []

def fetch_project_details(project_id: str):
    try:
        with httpx.Client() as client:
            resp = client.get(f"{BACKEND_URL}/projects/{project_id}")
            return resp.json()
    except Exception:
        return None

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

# --- UI COMPONENTS ---

@st.fragment(run_every=10)
def render_sidebar_content():
    st.title("Project Archive")
    st.markdown("---")
    projects = fetch_projects()
    if not projects:
        st.caption("No projects found.")
        return

    for p in projects:
        # Parse timestamp for display with fallback
        try:
            raw_ts = p.get("created_at")
            if raw_ts:
                dt = datetime.fromisoformat(raw_ts.replace("Z", ""))
                ts_str = dt.strftime("%Y-%m-%d %H:%M")
            else:
                ts_str = "N/A"
        except Exception:
            ts_str = "N/A"
        
        p_id = p['id']
        with st.expander(f"{ts_str} | {p.get('name', 'Unnamed')[:20]}"):
            st.caption(f"PROJECT ID: {p_id}")
            
            if st.button("SELECT PROJECT", key=f"sel_{p_id}", use_container_width=True):
                st.session_state.current_project_id = p_id
                st.session_state.tasks = {}
                st.session_state.narrative = ""
                st.session_state.takeaways = []
                st.rerun()

            if p.get("has_dashboard"):
                st.markdown(f"**[OPEN MASTER DASHBOARD]({BACKEND_URL}/projects/{p_id}/files/final_dashboard.html)**")
            else:
                st.caption("Dashboard: Not Available")
                
            if p.get("has_report"):
                st.markdown(f"**[OPEN RESEARCH REPORT]({BACKEND_URL}/projects/{p_id}/files/research_report.md)**")
            else:
                st.caption("Report: Not Available")

def fetch_current_tasks(project_id: str):
    """Fallback to fetch all tasks from DB if WS fails."""
    try:
        with httpx.Client() as client:
            # We'll need to add a backend endpoint for this or just rely on state
            pass
    except Exception:
        pass

def render_main():
    # 1. Header (Static)
    st.title("GADS Control Center")
    st.markdown("**Generative-augmented Data Science Orchestrator**")
    st.markdown("---")
    
    # 2. Fragment: Auto-refreshing Dashboard (Sidebar + Center + Right)
    @st.fragment(run_every=2)
    def render_dynamic_dashboard():
        # --- BACKGROUND DATA SYNC ---
        project_complete = False
        if st.session_state.current_project_id:
            details = fetch_project_details(st.session_state.current_project_id)
            if details:
                # 1. Sync tasks
                st.session_state.tasks = {t["id"]: t for t in details["tasks"]}
                
                # 2. Sync files and memory state
                st.session_state.project_state = details["project"].get("last_state_json", {})
                st.session_state.narrative = details["project"].get("narrative", "")
                st.session_state.takeaways = details["project"].get("takeaways", [])
                
                ws_path = f"{WORKSPACE_ROOT}/{st.session_state.current_project_id}"
                if os.path.exists(ws_path):
                    st.session_state.project_files = os.listdir(ws_path)

                # 3. Detect completion and sync narrative
                # Check ALL tasks for synthesis output
                for t in details["tasks"]:
                    res = t.get("result_json")
                    if res and isinstance(res, dict) and "narrative" in res:
                        st.session_state.narrative = res["narrative"]
                        st.session_state.takeaways = res.get("takeaways", [])
                
                # Terminal state check
                if details["tasks"] and all(t["status"] in ["completed", "failed"] for t in details["tasks"]):
                    # If we have no narrative yet, but tasks are done, it's a failure or pure technical run
                    project_complete = True
        
        # Split into Center and Right
        center_col, right_col = st.columns([1.5, 1])

        with center_col:
            st.subheader("Objective & Synthesis")
            
            if st.session_state.narrative:
                st.markdown(f"<div style='background-color: #ffffff; padding: 30px; border: 2px solid #000000;'>{st.session_state.narrative}</div>", unsafe_allow_html=True)
                if st.session_state.takeaways:
                    st.markdown("#### Key Takeaways")
                    for t in st.session_state.takeaways:
                        st.markdown(f"- **{t}**")
                
                if project_complete:
                    st.success("[SYSTEM] Project complete. Final reports generated and saved to workspace.")
            elif project_complete:
                st.warning("[SYSTEM] Workflow finished but no analytical synthesis was produced.")
            else:
                st.info("[SYSTEM] Analysis synthesis pending completion.")

        with right_col:
            st.subheader("System Status")
            track_tab, grounding_tab, action_tab = st.tabs(["Tasks", "Grounding", "Management"])
            
            with track_tab:
                if not st.session_state.tasks:
                    st.info("No active tasks.")
                
                # Sort tasks by created_at (now that we have the field)
                sorted_tasks = sorted(st.session_state.tasks.values(), key=lambda x: x.get("created_at", ""))
                
                for tinfo in sorted_tasks:
                    status = tinfo['status'].upper()
                    st.markdown(f"""
                    <div class="task-card">
                        <span class="status-label status-{tinfo['status']}">{status}</span><br/>
                        <div style="margin-top: 10px; color: #000000; font-weight: 500;">{tinfo['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Robust log extraction for ALL completed tasks
                    res_data = tinfo.get('result_json')
                    if res_data and isinstance(res_data, dict):
                        stdout = res_data.get('stdout', '')
                        if stdout:
                            with st.expander("View Execution Logs"):
                                st.code(stdout)

            with grounding_tab:
                st.markdown("#### Workspace Files")
                if not st.session_state.project_files:
                    st.write("*(Empty)*")
                for f in st.session_state.project_files:
                    st.markdown(f"- `{f}` ([view]({BACKEND_URL}/projects/{st.session_state.current_project_id}/files/{f}))")
                
                st.markdown("---")
                st.markdown("#### Sandbox Memory")
                if not st.session_state.project_state:
                    st.write("*(Empty)*")
                else:
                    st.json(st.session_state.project_state)

            with action_tab:
                # 1. External Path Registration (Local Picker)
                st.markdown("#### Register External Dataset")
                
                # Scan host datasets directory
                host_datasets_root = "/home/joergf/datasets"
                available_host_files = []
                if os.path.exists(host_datasets_root):
                    try:
                        available_host_files = sorted([f for f in os.listdir(host_datasets_root) if os.path.isfile(os.path.join(host_datasets_root, f))])
                    except Exception:
                        pass
                
                if available_host_files:
                    selected_file = st.selectbox("Pick Dataset from Host", options=["--- Select File ---"] + available_host_files)
                    if st.button("Mount Selected Dataset", use_container_width=True):
                        if selected_file != "--- Select File ---":
                            try:
                                project_id = st.session_state.current_project_id
                                # If no project active, create one on the fly
                                if not project_id:
                                    with httpx.Client(timeout=30.0) as client:
                                        payload = {
                                            "name": f"Analysis: {selected_file}",
                                            "objective": f"Explore and analyze the dataset: {selected_file}"
                                        }
                                        resp = client.post(f"{BACKEND_URL}/projects", json=payload)
                                        resp.raise_for_status()
                                        project_data = resp.json()
                                        project_id = project_data["project"]["id"]
                                        st.session_state.current_project_id = project_id
                                        st.info(f"[SYSTEM] Created new project context: {project_id}")

                                full_host_path = os.path.join(host_datasets_root, selected_file)
                                with httpx.Client(timeout=30.0) as client:
                                    resp = client.post(f"{BACKEND_URL}/projects/{project_id}/register-external", json={"path": full_host_path})
                                    resp.raise_for_status()
                                    st.success(f"[SYSTEM] Dataset '{selected_file}' mounted successfully.")
                                    st.session_state.needs_rerun = True
                            except Exception as e:
                                st.error(f"Mount failed: {e}")
                else:
                    st.info(f"No files found in {host_datasets_root}")
                    host_path = st.text_input("Manual Host Path", placeholder="/absolute/path/to/dataset.csv")
                    if st.button("Mount Manual Path", use_container_width=True):
                        # ... fallback logic
                        pass

                st.markdown("---")
                # 2. File uploader inside fragment
                uploaded_files = st.file_uploader("Upload local files", accept_multiple_files=True)
                if uploaded_files and st.button("Sync Uploads"):
                    payload = [{"name": f.name, "content_base64": base64.b64encode(f.getvalue()).decode("utf-8")} for f in uploaded_files]
                    try:
                        project_id = st.session_state.current_project_id
                        with httpx.Client(timeout=30.0) as client:
                            if project_id:
                                resp = client.post(f"{BACKEND_URL}/projects/{project_id}/files", json={"files": payload})
                            else:
                                resp = client.post(f"{BACKEND_URL}/projects", json={"name": "Upload Session", "objective": "", "files": payload})
                            resp.raise_for_status()
                            st.success(f"Uploaded {len(uploaded_files)} file(s).")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

                st.markdown("---")
                if st.button("Purge Old Projects (>30d)", use_container_width=True):
                    try:
                        with httpx.Client(timeout=60.0) as client:
                            resp = client.post(f"{BACKEND_URL}/maintenance/cleanup")
                            resp.raise_for_status()
                            data = resp.json()
                            st.success(f"[SYSTEM] Cleanup complete. Removed {data['removed_projects']} projects.")
                    except Exception as e:
                        st.error(f"Cleanup failed: {e}")

                if st.button("Reset Global Session", use_container_width=True):
                    st.session_state.current_project_id = None
                    st.session_state.tasks = {}
                    st.session_state.narrative = ""
                    st.session_state.takeaways = []
                    st.session_state.project_files = []
                    st.session_state.project_state = {}
                    st.rerun()

    # Place Objective input ABOVE the dynamic fragment so it doesn't reset every 2s
    objective = st.text_area("Research Objective", placeholder="Describe your data science goal...", height=100)
    if st.button("Launch Workflow", type="primary", use_container_width=True):
        if objective:
            asyncio.run(start_workflow(objective))
            st.info("[SYSTEM] Workflow initiated.")
        else:
            st.warning("[SYSTEM] Objective required.")
    
    st.markdown("---")
    # Render the dynamic panels
    render_dynamic_dashboard()

# --- EXECUTION ---
with st.sidebar:
    render_sidebar_content()
render_main()
