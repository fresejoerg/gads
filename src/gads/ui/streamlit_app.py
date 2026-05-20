import streamlit as st
from streamlit_ace import st_ace
import httpx
import json
import asyncio
import os
import base64
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

# --- CONFIGURATION ---
BACKEND_URL = os.getenv("GADS_BACKEND_URL", "http://127.0.0.1:8001")
WORKSPACE_ROOT = "/home/joergf/projects/MyLocalStack/data/workspaces"

st.set_page_config(
    page_title="GADS Workspace",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS FOR MODERN HIGH-LEGIBILITY WORKSPACE ---
st.markdown("""
<style>
    /* Global Styles */
    html, body {
        font-family: 'Inter', sans-serif;
    }

    /* Column Dividers */
    [data-testid="column"] {
        border-right: 1px solid #EDEDED;
        padding: 0 1.5rem !important;
    }
    [data-testid="column"]:last-child {
        border-right: none;
    }

    /* ------------------------------------------------------------- */
    /* UNIFIED PROFESSIONAL BUTTON STYLING                          */
    /* ------------------------------------------------------------- */

    .stButton > button, 
    button[kind="primary"], 
    button[kind="secondary"],
    div[class*="st-key-"] button {
        background-color: #262730 !important;
        color: #FFFFFF !important;
        border: 1px solid #464855 !important;
        box-shadow: none !important;
        text-transform: none !important;
        padding: 0px 1rem !important;
        min-height: 2.1rem !important;
        height: 2.1rem !important;
        font-size: 0.85rem !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 4px !important;
        transition: background-color 0.2s, border-color 0.2s !important;
    }

    .stButton > button:hover, 
    button[kind="primary"]:hover, 
    button[kind="secondary"]:hover,
    div[class*="st-key-"] button:hover {
        background-color: #31333F !important;
        border-color: #606477 !important;
        color: #FFFFFF !important;
    }

    /* Archive Styling */
    .archive-meta {
        font-size: 1.0rem;
        color: #666666;
        display: block;
        margin-top: 4px;
    }
    
    /* Active Project Indicator */
    .active-badge {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        background-color: #000000;
        color: #FFFFFF;
        padding: 4px 10px;
        border-radius: 2px;
        margin-bottom: 1rem;
        display: inline-block;
    }

    /* Log Containers */
    .stCodeBlock {
        border-radius: 4px !important;
        border: 1px solid #EDEDED !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "current_project_id" not in st.session_state:
    st.session_state.current_project_id = None
if "local_only_mode" not in st.session_state:
    st.session_state.local_only_mode = False

# --- API HELPERS ---
def api_get(path):
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{BACKEND_URL}{path}")
            return resp.json()
    except Exception as e:
        print(f"  [API] GET {path} failed: {e}")
        return None

def api_post(path, json_data=None):
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{BACKEND_URL}{path}", json=json_data)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"  [API] POST {path} failed: {e}")
        return None

def api_delete(path):
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(f"{BACKEND_URL}{path}")
            return resp.json()
    except Exception:
        return None

def render_banner():
    img_path = "/home/joergf/Generated Image May 05, 2026 - 10_53PM.png"
    if os.path.exists(img_path):
        try:
            with open(img_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-start; align-items: center; margin-top: -3rem; margin-bottom: 1rem;">
                        <img src="data:image/png;base64,{data}" style="height: 220px; filter: drop-shadow(0px 4px 8px rgba(0,0,0,0.1));">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Error loading banner: {e}")

# --- UI LOGIC ---

def render_knowledge_panel():
    st.markdown("### KNOWLEDGE")
    with st.expander("KNOWLEDGE BASE", expanded=False):
        k_type = st.radio("Type", ["Recipes", "Skills", "Prompts", "Tiers"], horizontal=True, label_visibility="collapsed")
        
        if k_type == "Tiers":
            # --- MODEL TIER DISPLAY (READ ONLY) ---
            hierarchy = api_get("/hierarchy") or {}
            for tier in ["T1", "T2", "T3", "T4"]:
                if tier in hierarchy:
                    data = hierarchy[tier]
                    st.markdown(f"**{tier} - {data['description']}**")
                    models_html = ""
                    for m in data["models"]:
                        models_html += f"<span style='background-color: #262730; color: #ffffff; padding: 2px 8px; border-radius: 4px; margin-right: 6px; font-family: monospace; font-size: 0.85rem; border: 1px solid #464855;'>{m}</span>"
                    st.markdown(models_html, unsafe_allow_html=True)
                    st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
            return

        if k_type == "Prompts":
            # --- SYSTEM PROMPT EDITOR ---
            prompts = api_get("/prompts") or []
            agent_options = [p["agent_name"] for p in prompts]
            
            sel_key = "prompts_sel"
            if sel_key not in st.session_state: st.session_state[sel_key] = agent_options[0] if agent_options else None
            
            selected_agent = st.selectbox("Select Agent", options=agent_options, index=agent_options.index(st.session_state[sel_key]) if st.session_state[sel_key] in agent_options else 0)
            
            if selected_agent != st.session_state[sel_key]:
                st.session_state[sel_key] = selected_agent
                if "prompts_buffer" in st.session_state: del st.session_state.prompts_buffer
                st.rerun()

            p_meta = next((p for p in prompts if p["agent_name"] == selected_agent), None)
            if p_meta:
                if "prompts_buffer" not in st.session_state:
                    st.session_state.prompts_buffer = p_meta["content"]
                
                status_color = "green" if not p_meta["is_override"] else "orange"
                status_label = "FACTORY DEFAULT" if not p_meta["is_override"] else "USER OVERRIDE"
                st.markdown(f"Status: :{status_color}[{status_label}]")
                
                if p_meta["required_vars"]:
                    st.caption(f"Required variables: {', '.join([f'{{{v}}}' for v in p_meta['required_vars']])}")

                new_content = st_ace(
                    value=st.session_state.prompts_buffer, 
                    language="markdown", 
                    theme="monokai", 
                    height=500, 
                    wrap=True, 
                    key=f"ace_prompt_{selected_agent}"
                )
                st.session_state.prompts_buffer = new_content
                
                col1, col2 = st.columns(2)
                if col1.button("SAVE OVERRIDE", use_container_width=True, type="primary"):
                    res = api_post(f"/prompts/{selected_agent}", {"content": new_content})
                    if res and res.get("status") == "ok":
                        st.toast("Override saved.")
                        st.rerun()
                    else:
                        st.error(res.get("detail", "Save failed."))
                
                if p_meta["is_override"]:
                    if col2.button("RESTORE DEFAULT", use_container_width=True):
                        api_delete(f"/prompts/{selected_agent}")
                        if "prompts_buffer" in st.session_state: del st.session_state.prompts_buffer
                        st.rerun()
            return

        base_path = "/recipes" if k_type == "Recipes" else "/skills"
        
        items = api_get(base_path) or []
        new_label = f"-- New {k_type[:-1]} --"
        options = [new_label] + items
        
        # Track selection in session state
        sel_key = f"{k_type.lower()}_sel"
        if sel_key not in st.session_state:
            st.session_state[sel_key] = options[0]
            
        selected = st.selectbox(f"Select {k_type[:-1]}", options=options, index=options.index(st.session_state[sel_key]) if st.session_state[sel_key] in options else 0)
        
        if selected != st.session_state[sel_key]:
            st.session_state[sel_key] = selected
            # Clear buffered content when switching
            for buf in ["yaml", "md", "skill"]:
                b_key = f"{k_type.lower()}_buffer_{buf}"
                if b_key in st.session_state: del st.session_state[b_key]
            st.rerun()

        filename = ""
        if k_type == "Recipes":
            # --- RECIPE EDITOR (SPLIT) ---
            if selected == new_label:
                filename = st.text_input("Filename", placeholder="e.g. classification.md", key="new_recipe_name")
                initial_yaml = "id: \nversion: 1.0.0\nauthor: \napplies_when:\n  task_type: []\n  data_modality: []\nrequires:\n  variables: []\n  capabilities: []\ndag: []\n"
                initial_md = "# Title\n\n## Rationale\n"
            else:
                filename = selected
                if "recipes_buffer_yaml" not in st.session_state or "recipes_buffer_md" not in st.session_state:
                    res = api_get(f"/recipes/{selected}")
                    content = res.get("content", "") if res else ""
                    match = re.search(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
                    if match:
                        st.session_state.recipes_buffer_yaml = match.group(1)
                        st.session_state.recipes_buffer_md = match.group(2)
                    else:
                        st.session_state.recipes_buffer_yaml = ""
                        st.session_state.recipes_buffer_md = content
                initial_yaml = st.session_state.recipes_buffer_yaml
                initial_md = st.session_state.recipes_buffer_md

            st.markdown("##### METADATA (YAML)")
            new_yaml = st_ace(value=initial_yaml, language="yaml", theme="monokai", height=200, key=f"ace_yaml_{selected}")
            st.session_state.recipes_buffer_yaml = new_yaml

            st.markdown("##### CONTENT (MARKDOWN)")
            new_md = st_ace(value=initial_md, language="markdown", theme="monokai", height=300, wrap=True, key=f"ace_md_{selected}")
            st.session_state.recipes_buffer_md = new_md
            full_content = f"---\n{new_yaml}\n---\n{new_md}"
        else:
            # --- SKILL EDITOR (UNIFIED) ---
            if selected == new_label:
                filename = st.text_input("Filename", placeholder="e.g. visualization.md", key="new_skill_name")
                initial_skill = "---\nid: \ntriggers: []\n---\n# Skill Description\n"
            else:
                filename = selected
                if "skills_buffer_skill" not in st.session_state:
                    res = api_get(f"/skills/{selected}")
                    st.session_state.skills_buffer_skill = res.get("content", "") if res else ""
                initial_skill = st.session_state.skills_buffer_skill

            full_content = st_ace(value=initial_skill, language="markdown", theme="monokai", height=500, wrap=True, key=f"ace_skill_{selected}")
            st.session_state.skills_buffer_skill = full_content

        if st.button(f"SAVE {k_type[:-1].upper()}", use_container_width=True, type="primary"):
            if not filename:
                st.error("Filename required.")
            else:
                res = api_post(f"{base_path}/{filename}", {"content": full_content})
                if res and res.get("status") == "success":
                    st.success(f"Saved {filename}")
                    st.session_state[sel_key] = filename
                    for buf in ["yaml", "md", "skill"]:
                        b_key = f"{k_type.lower()}_buffer_{buf}"
                        if b_key in st.session_state: del st.session_state[b_key]
                    st.rerun()
                else:
                    detail = res.get("detail", "Validation failed") if res else "Error"
                    st.error(detail)

def render_archive_panel():
    head_col, kill_col, refresh_col = st.columns([0.5, 0.25, 0.25])
    head_col.markdown("### ARCHIVE")
    
    if kill_col.button("KILL ALL", key="kill_all_btn", use_container_width=True, help="Forcefully terminate all active workflows system-wide."):
        res = api_post("/system/cancel-all")
        if res and res.get("status") == "ok":
            count = res.get("cancelled_count", 0)
            st.toast(f"Terminated {count} workflows.")
            st.rerun()
        else:
            st.error("Failed to kill workflows.")

    if refresh_col.button("REFRESH", key="refresh_archive_btn", use_container_width=True):
        st.rerun()
    
    st.text_input("Filter", placeholder="Search...", label_visibility="collapsed")
    
    projects = api_get("/projects")
    if not projects:
        st.caption("No projects found.")
        return

    # Scrollable Archive
    with st.container(height=800):
        for p in projects:
            p_id = p['id']
            created = p.get("created_at", "N/A")[5:16].replace("T", " ")
            
            status_indicators = []
            if p.get("has_failed_tasks"): status_indicators.append("❌")
            elif p.get("is_running"): status_indicators.append("🔵")
            elif p.get("has_report") and p.get("has_dashboard"): status_indicators.append("✅")
            else: status_indicators.append("⏳")

            cols = st.columns([0.1, 0.55, 0.35])
            cols[0].markdown(" ".join(status_indicators))
            
            # Metadata with project ID and first instruction
            meta_html = f"**{p.get('name', 'Project')}**<br><span style='font-size: 1.0rem; color: #888;'>ID: `{p_id}`</span><br><span class='archive-meta'>{created}</span>"
            if p.get('first_instruction'):
                meta_html += f"<br><span style='font-size: 1.05rem; font-style: italic; color: #bbb;'>{p['first_instruction']}</span>"
            
            cols[1].markdown(meta_html, unsafe_allow_html=True)

            # Action buttons (Stacked vertically)
            with cols[2]:
                if st.button("LOAD", key=f"load_{p_id}", help="Load this project into the orchestrator.", use_container_width=True):
                    st.session_state.current_project_id = p_id
                    st.rerun()
            
                if st.button("DELETE", key=f"del_{p_id}", use_container_width=True):
                    api_delete(f"/projects/{p_id}")
                    if st.session_state.current_project_id == p_id:
                        st.session_state.current_project_id = None
                    st.toast(f"Deleted {p['name']}")
                    st.rerun()

            st.markdown("---")

def render_orchestrator_panel():
    # Header with Toggles
    head_col1, head_col2, head_col3 = st.columns([2.0, 1.2, 1.0])
    head_col1.markdown("### ORCHESTRATOR")
    
    if "initial_config_synced" not in st.session_state:
        config = api_get("/config")
        if config:
            st.session_state.local_only_mode = config.get("local_only", False)
            st.session_state.random_routing_mode = config.get("random_routing", False)
            st.session_state.initial_config_synced = True
        else:
            st.session_state.local_only_mode = False
            st.session_state.random_routing_mode = False

    st.session_state.local_only_mode = head_col2.toggle(
        "LOCAL ONLY", 
        value=st.session_state.local_only_mode,
        help="Force the system to use local LLM models (e.g. Ollama) instead of cloud APIs."
    )
    st.session_state.random_routing_mode = head_col3.toggle(
        "RANDOM ROUTING", 
        value=st.session_state.random_routing_mode,
        help="Bypass the Planner and randomly route the objective to a worker agent (for testing)."
    )
    api_post("/config", {
        "local_only": st.session_state.local_only_mode,
        "random_routing": st.session_state.random_routing_mode
    })

    if st.session_state.current_project_id:
        st.markdown(f"<div class='active-badge'>ACTIVE PROJECT: {st.session_state.current_project_id}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='active-badge'>NO PROJECT LOADED</div>", unsafe_allow_html=True)

    # 1. Spec Launcher
    with st.expander("LAUNCH FROM SPEC", expanded=False):
        specs = api_get("/specs")
        if specs:
            spec_cols = st.columns([0.4, 0.2, 0.2, 0.2])
            selected_spec = spec_cols[0].selectbox("Select Spec Document", options=specs, label_visibility="collapsed")
            
            if spec_cols[1].button("VIEW SPEC", key="btn_view_spec", use_container_width=True):
                st.session_state.view_spec = selected_spec
                
            if spec_cols[2].button("LOAD SPEC", key="btn_load_spec", use_container_width=True):
                with st.spinner("Loading..."):
                    res = api_post("/projects/from-spec", {"filename": selected_spec, "launch_workflow": False})
                    if res and "project" in res:
                        st.toast("Spec loaded successfully!")
                        st.session_state.current_project_id = res["project"]["id"]
                        st.rerun()
                    else:
                        st.error(res.get("detail", "Failed to load spec.")) if res else st.error("Failed to connect to backend.")
                        
            if spec_cols[3].button("LAUNCH", key="btn_launch_spec", type="primary", use_container_width=True):
                with st.spinner("Launching..."):
                    res = api_post("/projects/from-spec", {"filename": selected_spec, "launch_workflow": True})
                    if res and "project" in res:
                        st.toast("Spec launched successfully!")
                        st.session_state.current_project_id = res["project"]["id"]
                        st.rerun()
                    else:
                        st.error(res.get("detail", "Failed to launch spec.")) if res else st.error("Failed to connect to backend.")
        else:
            st.info("No `.md` spec documents found in the `specs/` directory.")

    if "view_spec" in st.session_state and st.session_state.view_spec:
        @st.dialog(f"Edit Spec: {st.session_state.view_spec}", width="large")
        def show_spec():
            if "spec_buffer" not in st.session_state:
                spec_data = api_get(f"/specs/{st.session_state.view_spec}")
                st.session_state.spec_buffer = spec_data.get("content", "") if spec_data else ""
            
            new_content = st_ace(
                value=st.session_state.spec_buffer,
                language="markdown",
                theme="monokai",
                height=500,
                wrap=True,
                key="ace_spec_editor"
            )
            st.session_state.spec_buffer = new_content
            
            col1, col2 = st.columns(2)
            if col1.button("SAVE CHANGES", use_container_width=True, type="primary"):
                res = api_post(f"/specs/{st.session_state.view_spec}", {"content": new_content})
                if res and res.get("status") == "success":
                    st.toast("Spec saved successfully.")
                    del st.session_state.spec_buffer
                    st.session_state.view_spec = None
                    st.rerun()
                else:
                    st.error("Failed to save spec.")
            
            if col2.button("CLOSE", use_container_width=True):
                del st.session_state.spec_buffer
                st.session_state.view_spec = None
                st.rerun()
        show_spec()

    # 2. Objective Input
    objective = st.text_area("Research Objective", 
                            placeholder="Describe your data science goal...", 
                            height=100, 
                            label_visibility="collapsed")
    
    # 3. Control Bar
    ctrl_cols = st.columns(4)
    
    with ctrl_cols[0]:
        # --- PRE-FLIGHT CHECK LOGIC ---
        if st.button("LAUNCH", key="btn_launch", help="Start the multi-agent analysis workflow.", use_container_width=True):
            if not objective and not st.session_state.current_project_id:
                st.warning("Objective required.")
            else:
                # Check for files in current workspace (if project exists)
                has_files = False
                if st.session_state.current_project_id:
                    ws_path = f"{WORKSPACE_ROOT}/{st.session_state.current_project_id}"
                    if os.path.exists(ws_path) and len(os.listdir(ws_path)) > 0:
                        has_files = True
                
                def trigger_launch():
                    payload = {
                        "name": f"Project {datetime.now().strftime('%m-%d %H:%M')}", 
                        "objective": objective, 
                        "existing_project_id": st.session_state.current_project_id
                    }
                    res = api_post("/projects", payload)
                    if res:
                        st.session_state.current_project_id = res["project"]["id"]
                        st.rerun()

                if not has_files:
                    @st.dialog("Pre-Flight Check: Empty Workspace")
                    def confirm_empty_launch():
                        st.warning("⚠️ No datasets are mounted in this workspace.")
                        st.write("Most research objectives require a dataset. If you proceed, the agents will likely fail or halt.")
                        col1, col2 = st.columns(2)
                        if col1.button("MOUNT DATA FIRST", use_container_width=True):
                            st.rerun()
                        if col2.button("PROCEED ANYWAY", type="primary", use_container_width=True):
                            trigger_launch()
                    confirm_empty_launch()
                else:
                    trigger_launch()

    with ctrl_cols[1]:
        if st.button("CANCEL", key="btn_cancel", help="Interrupt and cancel the currently running workflow.", use_container_width=True):
            if st.session_state.current_project_id:
                api_post(f"/projects/{st.session_state.current_project_id}/cancel")
                st.toast("Cancellation requested.")

    with ctrl_cols[2]:
        if st.button("NEW", key="btn_new", help="Clear the workspace to start a new project from scratch.", use_container_width=True):
            st.session_state.current_project_id = None
            st.rerun()
        
    with ctrl_cols[3]:
        if st.button("REFRESH", key="btn_refresh", help="Manually refresh the UI state.", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # 3. Dynamic Workflow Polling
    @st.fragment(run_every=2)
    def render_active_workflow():
        if not st.session_state.current_project_id:
            st.info("Start a new project or select from the archive.")
            return

        details = api_get(f"/projects/{st.session_state.current_project_id}")
        if not details: return

        proj = details["project"]
        tasks = details["tasks"]
        
        if proj.get("narrative"):
            with st.expander("EXECUTIVE SUMMARY", expanded=True):
                st.markdown(proj["narrative"])
        
        st.markdown("#### LOGS")
        with st.container(height=500):
            if not tasks: st.caption("Preparing agent swarm...")
            
            # Map instructions for display
            instr_map = {i['id']: i for i in details.get("instructions", [])}
            last_instr_id = None

            for t in sorted(tasks, key=lambda x: x.get("created_at", "")):
                # Display instruction block when it changes
                curr_instr_id = t.get("instruction_id")
                if curr_instr_id and curr_instr_id != last_instr_id:
                    instr = instr_map.get(curr_instr_id)
                    if instr:
                        st.chat_message("user").markdown(f"**Instruction:** {instr['content']}")
                    last_instr_id = curr_instr_id

                st_icon = "⚪"
                if t['assigned_to'] == "Router": st_icon = "📐"
                elif t['assigned_to'] == "Planner": st_icon = "📋"
                elif t['status'] == "completed": st_icon = "🟢"
                elif t['status'] == "failed": st_icon = "🔴"
                elif t['status'] == "running": st_icon = "🔵"
                
                expanded = (t['status'] == "running") or (t['assigned_to'] in ["Router", "Planner"] and t['status'] == "completed")
                
                with st.expander(f"{st_icon} {t['description'][:85]}...", expanded=expanded):
                    res = t.get("result_json") or {}
                    model_name = res.get("model_used")
                    
                    if t['assigned_to'] == "Router":
                        st.caption(f"Architect: `{model_name or 'N/A'}`")
                    elif t['assigned_to'] == "Planner":
                        st.caption(f"Project Manager: `{model_name or 'N/A'}`")
                    elif t['assigned_to'] == "System":
                        st.caption(f"System Task")
                    else:
                        st.caption(f"Worker: `CodeGenerator`")
                        st.caption(f"Assigned Model: `{t['assigned_to']}`")
                        if model_name and model_name != t['assigned_to']:
                            st.caption(f"Executed Model: `{model_name}`")
                    
                    if t['status'] == "running":
                        stream_data = api_get(f"/tasks/{t['id']}/stream")
                        if stream_data:
                            reasoning = stream_data.get("reasoning", "")
                            stdout = stream_data.get("stdout", "")
                            if reasoning:
                                if len(reasoning) > 5000:
                                    reasoning = "... [truncated] ...\n" + reasoning[-5000:]
                                st.markdown("**🤔 Reasoning:**")
                                st.code(reasoning, language="markdown")
                            else:
                                st.caption("🤔 Agent is thinking...")

                            if stdout:
                                if len(stdout) > 5000:
                                    stdout = "... [truncated] ...\n" + stdout[-5000:]
                                st.markdown("**🖥️ Sandbox Output:**")
                                st.code(stdout, language="text")
                            else:
                                if t['assigned_to'] not in ["Router", "Planner"]:
                                    st.caption("🖥️ Sandbox is starting up...")
                        else:
                            st.caption("Initializing...")
                            
                    res = t.get("result_json")
                    if res and res.get("stdout"):
                        if t['assigned_to'] == "Router":
                            # Attempt to parse and format nicely
                            stdout = res["stdout"]
                            if "Decision Reasoning:" in stdout:
                                parts = stdout.split("--- KNOWLEDGE BASE ---")
                                st.markdown(parts[0])
                                if len(parts) > 1:
                                    with st.expander("KNOWLEDGE RETRIEVAL", expanded=True):
                                        st.markdown(parts[1])
                            else:
                                st.markdown(stdout)
                        else:
                            st.markdown(res["stdout"])
                    if t.get("error"):
                        st.error(t["error"])

    render_active_workflow()

def render_grounding_panel():
    st.markdown("### GROUNDING")
    
    st.markdown("#### DATASETS")
    host_datasets_root = "/home/joergf/datasets"
    if os.path.exists(host_datasets_root):
        files = sorted([f for f in os.listdir(host_datasets_root) if os.path.isfile(os.path.join(host_datasets_root, f))])
        selected = st.selectbox("Mount", options=["--- Select to Mount ---"] + files, label_visibility="collapsed")
        if st.button("MOUNT", key="btn_mount", help="Link the selected external dataset into the current project workspace.", use_container_width=True) and selected != "--- Select to Mount ---":
            # AUTO-CREATE PROJECT IF NONE LOADED
            if not st.session_state.current_project_id:
                payload = {
                    "name": f"Draft {datetime.now().strftime('%m-%d %H:%M')}", 
                    "objective": ""
                }
                res = api_post("/projects", payload)
                if res and "project" in res:
                    st.session_state.current_project_id = res["project"]["id"]
                else:
                    detail = res.get("detail", "Unknown error") if isinstance(res, dict) else "Connection failed"
                    st.error(f"Failed to create project session. Backend returned: {detail}")
            
            if st.session_state.current_project_id:
                api_post(f"/projects/{st.session_state.current_project_id}/register-external", {"path": os.path.join(host_datasets_root, selected)})
                st.toast(f"Mounted {selected}")
                st.rerun() # Refresh to show in workspace
    
    st.markdown("---")
    
    @st.fragment(run_every=3)
    def render_state_explorer():
        if not st.session_state.current_project_id: return

        details = api_get(f"/projects/{st.session_state.current_project_id}")
        if not details: return

        st.markdown("#### WORKSPACE")
        ws_path = f"{WORKSPACE_ROOT}/{st.session_state.current_project_id}"
        if os.path.exists(ws_path):
            files = sorted(os.listdir(ws_path))
            for f in files:
                url = f"{BACKEND_URL}/projects/{st.session_state.current_project_id}/files/{f}"
                st.markdown(f"- [`{f}`]({url})")
        else:
            st.caption("No files yet.")

        st.markdown("---")
        
        mem = details["project"].get("last_state_json", {}) or {}
        
        # Display Schemas if present
        schemas = mem.pop("__schemas__", None)
        if schemas:
            st.markdown("#### FILE SCHEMAS")
            for filename, schema_data in schemas.items():
                with st.expander(f"📊 {filename}", expanded=False):
                    if "schema" in schema_data and "sample" in schema_data:
                        st.markdown("**Schema:**")
                        st.json(schema_data["schema"])
                        st.markdown("**Sample Data (First 5 Rows):**")
                        st.dataframe(schema_data["sample"])
                    else:
                        st.json(schema_data)
            st.markdown("---")

        st.markdown("#### KERNEL STATE")
        if mem:
            st.json(mem)
        else:
            st.caption("Memory empty.")

    render_state_explorer()

# --- MAIN LAYOUT ---
render_banner()

col_archive, col_orch, col_ground = st.columns([1.6, 3.2, 1.2])

with col_archive:
    render_knowledge_panel()
    render_archive_panel()

with col_orch:
    render_orchestrator_panel()

with col_ground:
    render_grounding_panel()
