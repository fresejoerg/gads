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
BACKEND_URL = os.getenv("GADS_BACKEND_URL", "http://localhost:8001")
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
    /* BUTTON STYLING (Key-Targeted)                                 */
    /* ------------------------------------------------------------- */

    /* LAUNCH WORKFLOW - Green */
    .st-key-btn_launch button {
        background-color: #E8F5E9 !important;
        color: #2E7D32 !important;
        border: 1px solid #E8F5E9 !important;
    }
    .st-key-btn_launch button:hover, .st-key-btn_launch button:active, .st-key-btn_launch button:focus {
        background-color: #C8E6C9 !important;
        color: #1B5E20 !important;
        border-color: #C8E6C9 !important;
        box-shadow: none !important;
    }

    /* CANCEL - Red */
    .st-key-btn_cancel button {
        background-color: #FFEBEE !important;
        color: #C62828 !important;
        border: 1px solid #FFEBEE !important;
    }
    .st-key-btn_cancel button:hover, .st-key-btn_cancel button:active, .st-key-btn_cancel button:focus {
        background-color: #FFCDD2 !important;
        color: #B71C1C !important;
        border-color: #FFCDD2 !important;
        box-shadow: none !important;
    }

    /* NEW PROJECT - Yellow */
    .st-key-btn_new button {
        background-color: #FFFDE7 !important;
        color: #F57F17 !important;
        border: 1px solid #FFFDE7 !important;
    }
    .st-key-btn_new button:hover, .st-key-btn_new button:active, .st-key-btn_new button:focus {
        background-color: #FFF9C4 !important;
        color: #E65100 !important;
        border-color: #FFF9C4 !important;
        box-shadow: none !important;
    }

    /* REFRESH - Orange */
    .st-key-btn_refresh button {
        background-color: #FFF3E0 !important;
        color: #E65100 !important;
        border: 1px solid #FFF3E0 !important;
    }
    .st-key-btn_refresh button:hover, .st-key-btn_refresh button:active, .st-key-btn_refresh button:focus {
        background-color: #FFE0B2 !important;
        color: #BF360C !important;
        border-color: #FFE0B2 !important;
        box-shadow: none !important;
    }
    
    /* MOUNT - Blue */
    .st-key-btn_mount button {
        background-color: #E3F2FD !important;
        color: #1565C0 !important;
        border: 1px solid #E3F2FD !important;
    }
    .st-key-btn_mount button:hover, .st-key-btn_mount button:active, .st-key-btn_mount button:focus {
        background-color: #BBDEFB !important;
        color: #0D47A1 !important;
        border-color: #BBDEFB !important;
        box-shadow: none !important;
    }
    
    /* ARCHIVE LOAD - Blue */
    div[class*="st-key-load_"] button {
        background-color: #E3F2FD !important;
        color: #1565C0 !important;
        border: 1px solid #E3F2FD !important;
    }
    div[class*="st-key-load_"] button:hover, div[class*="st-key-load_"] button:active, div[class*="st-key-load_"] button:focus {
        background-color: #BBDEFB !important;
        color: #0D47A1 !important;
        border-color: #BBDEFB !important;
        box-shadow: none !important;
    }

    /* ARCHIVE DELETE - Red */
    div[class*="st-key-del_"] button {
        background-color: #FFEBEE !important;
        color: #C62828 !important;
        border: 1px solid #FFEBEE !important;
    }
    div[class*="st-key-del_"] button:hover, div[class*="st-key-del_"] button:active, div[class*="st-key-del_"] button:focus {
        background-color: #FFCDD2 !important;
        color: #B71C1C !important;
        border-color: #FFCDD2 !important;
        box-shadow: none !important;
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
    except Exception:
        return None

def api_post(path, json_data=None):
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{BACKEND_URL}{path}", json=json_data)
            return resp.json()
    except Exception:
        return None

def api_delete(path):
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(f"{BACKEND_URL}{path}")
            return resp.json()
    except Exception:
        return None

# --- UI LOGIC ---

def render_knowledge_panel():
    st.markdown("### KNOWLEDGE")
    with st.expander("KNOWLEDGE BASE", expanded=False):
        k_type = st.radio("Type", ["Recipes", "Skills"], horizontal=True, label_visibility="collapsed")
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
    st.markdown("### ARCHIVE")
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
            elif p.get("has_report") and p.get("has_dashboard"): status_indicators.append("✅")
            else: status_indicators.append("⏳")

            cols = st.columns([0.1, 0.55, 0.35])
            cols[0].markdown(" ".join(status_indicators))
            
            # Metadata with project ID and first instruction
            meta_html = f"**{p.get('name', 'Project')}**<br><span style='font-size: 1.0rem; color: #888;'>ID: `{p_id}`</span><br><span class='archive-meta'>{created}</span>"
            if p.get('first_instruction'):
                instr_preview = p['first_instruction'][:120] + ("..." if len(p['first_instruction']) > 120 else "")
                meta_html += f"<br><span style='font-size: 1.05rem; font-style: italic; color: #bbb;'>{instr_preview}</span>"
            
            cols[1].markdown(meta_html, unsafe_allow_html=True)

            btn_cols = cols[2].columns(2)
            # LOAD button (Primary)
            with btn_cols[0]:
                if st.button("LOAD", key=f"load_{p_id}", use_container_width=True, type="primary"):
                    st.session_state.current_project_id = p_id
                    st.rerun()
            
            # Delete button (Secondary)
            with btn_cols[1]:
                if st.button("🗑️", key=f"del_{p_id}", use_container_width=True):
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
        value=st.session_state.local_only_mode
    )
    st.session_state.random_routing_mode = head_col3.toggle(
        "RANDOM ROUTING", 
        value=st.session_state.random_routing_mode
    )
    api_post("/config", {
        "local_only": st.session_state.local_only_mode,
        "random_routing": st.session_state.random_routing_mode
    })

    if st.session_state.current_project_id:
        st.markdown(f"<div class='active-badge'>ACTIVE PROJECT: {st.session_state.current_project_id}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='active-badge'>NO PROJECT LOADED</div>", unsafe_allow_html=True)

    # 1. Objective Input
    objective = st.text_area("Research Objective", 
                            placeholder="Describe your data science goal...", 
                            height=100, 
                            label_visibility="collapsed")
    
    # 2. Control Bar
    ctrl_cols = st.columns(4)
    
    with ctrl_cols[0]:
        if st.button("LAUNCH WORKFLOW", key="btn_launch", use_container_width=True):
            if objective:
                payload = {
                    "name": f"Project {datetime.now().strftime('%m-%d %H:%M')}", 
                    "objective": objective, 
                    "existing_project_id": st.session_state.current_project_id
                }
                res = api_post("/projects", payload)
                if res:
                    st.session_state.current_project_id = res["project"]["id"]
                    st.rerun()
            else:
                st.warning("Objective required.")

    with ctrl_cols[1]:
        if st.button("CANCEL", key="btn_cancel", use_container_width=True):
            if st.session_state.current_project_id:
                api_post(f"/projects/{st.session_state.current_project_id}/cancel")
                st.toast("Cancellation requested.")

    with ctrl_cols[2]:
        if st.button("NEW PROJECT", key="btn_new", use_container_width=True):
            st.session_state.current_project_id = None
            st.rerun()
        
    with ctrl_cols[3]:
        if st.button("REFRESH", key="btn_refresh", use_container_width=True):
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
                if t['status'] == "completed": st_icon = "🟢"
                elif t['status'] == "failed": st_icon = "🔴"
                elif t['status'] == "running": st_icon = "🔵"
                
                with st.expander(f"{st_icon} {t['description'][:85]}...", expanded=(t['status'] == "running")):
                    st.caption(f"Worker: `{t['assigned_to']}`")
                    
                    if t['status'] == "running":
                        stream_data = api_get(f"/tasks/{t['id']}/stream")
                        if stream_data:
                            reasoning = stream_data.get("reasoning", "")
                            stdout = stream_data.get("stdout", "")
                            if reasoning:
                                # Truncate reasoning to avoid freezing the UI on massive outputs
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
                                st.caption("🖥️ Sandbox is starting up...")
                        else:
                            st.caption("Initializing...")
                            
                    res = t.get("result_json")
                    if res and res.get("stdout"):
                        st.code(res["stdout"], language="text")
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
        if st.button("MOUNT", key="btn_mount", use_container_width=True) and selected != "--- Select to Mount ---":
            # AUTO-CREATE PROJECT IF NONE LOADED
            if not st.session_state.current_project_id:
                payload = {
                    "name": f"Draft {datetime.now().strftime('%m-%d %H:%M')}", 
                    "objective": ""
                }
                res = api_post("/projects", payload)
                if res:
                    st.session_state.current_project_id = res["project"]["id"]
            
            if st.session_state.current_project_id:
                api_post(f"/projects/{st.session_state.current_project_id}/register-external", {"path": os.path.join(host_datasets_root, selected)})
                st.toast(f"Mounted {selected}")
                st.rerun() # Refresh to show in workspace
            else:
                st.error("Failed to create project session.")
    
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
        
        st.markdown("#### KERNEL STATE")
        mem = details["project"].get("last_state_json", {})
        if mem:
            st.json(mem)
        else:
            st.caption("Memory empty.")

    render_state_explorer()

# --- MAIN LAYOUT ---
col_archive, col_orch, col_ground = st.columns([2.0, 2.0, 1.0])

with col_archive:
    render_knowledge_panel()
    render_archive_panel()

with col_orch:
    render_orchestrator_panel()

with col_ground:
    render_grounding_panel()
