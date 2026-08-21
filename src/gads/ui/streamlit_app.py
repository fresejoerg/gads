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
if "routing_mode" not in st.session_state:
    st.session_state.routing_mode = "cloud"
if "run_mode" not in st.session_state:
    st.session_state.run_mode = "research"
if "pinned_model" not in st.session_state:
    st.session_state.pinned_model = None

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

def api_get_long(path, timeout=60.0):
    """GET for endpoints that run sandbox code (kernel probes) — the 10s default is too tight."""
    try:
        with httpx.Client(timeout=timeout) as client:
            return client.get(f"{BACKEND_URL}{path}").json()
    except Exception as e:
        print(f"  [API] GET(long) {path} failed: {e}")
        return None

def api_post_long(path, json_data=None, timeout=900.0):
    """POST for long operations — rehydration replays a run's code and can take minutes."""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{BACKEND_URL}{path}", json=json_data)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"  [API] POST(long) {path} failed: {e}")
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
            # --- RECIPE EDITOR (SPLIT: YAML + plain-text body with preview) ---
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
            recipe_preview = st.toggle("Preview", key=f"recipe_preview_{selected}", value=False)
            if recipe_preview:
                with st.container(border=True, height=300):
                    st.markdown(st.session_state.get("recipes_buffer_md") or initial_md)
                new_md = st.session_state.get("recipes_buffer_md") or initial_md
            else:
                new_md = st_ace(value=initial_md, language="plain_text", theme="monokai", height=300, wrap=True, key=f"ace_md_{selected}")
                st.session_state.recipes_buffer_md = new_md
            full_content = f"---\n{new_yaml}\n---\n{new_md}"
        else:
            # --- SKILL EDITOR (SPLIT: YAML metadata + plain-text body with preview) ---
            if selected == new_label:
                filename = st.text_input("Filename", placeholder="e.g. visualization.md", key="new_skill_name")
                initial_skill_yaml = "id: \ndescription: \"\"\ntriggers: []"
                initial_skill_md = "# Skill Description\n"
            else:
                filename = selected
                if "skills_buffer_yaml" not in st.session_state or "skills_buffer_md" not in st.session_state:
                    res = api_get(f"/skills/{selected}")
                    raw_skill = res.get("content", "") if res else ""
                    match = re.search(r"^---\s*\n(.*?)\n---\s*\n(.*)", raw_skill, re.DOTALL)
                    if match:
                        st.session_state.skills_buffer_yaml = match.group(1)
                        st.session_state.skills_buffer_md = match.group(2).strip()
                    else:
                        st.session_state.skills_buffer_yaml = ""
                        st.session_state.skills_buffer_md = raw_skill
                initial_skill_yaml = st.session_state.skills_buffer_yaml
                initial_skill_md = st.session_state.skills_buffer_md

            st.markdown("##### METADATA (YAML)")
            new_skill_yaml = st_ace(value=initial_skill_yaml, language="yaml", theme="monokai", height=120, key=f"ace_skill_yaml_{selected}")
            st.session_state.skills_buffer_yaml = new_skill_yaml

            st.markdown("##### GUIDANCE (MARKDOWN)")
            skill_preview = st.toggle("Preview", key=f"skill_preview_{selected}", value=False)
            if skill_preview:
                with st.container(border=True, height=360):
                    st.markdown(st.session_state.get("skills_buffer_md") or initial_skill_md)
                new_skill_md = st.session_state.get("skills_buffer_md") or initial_skill_md
            else:
                new_skill_md = st_ace(value=initial_skill_md, language="plain_text", theme="monokai", height=360, wrap=True, key=f"ace_skill_md_{selected}")
                st.session_state.skills_buffer_md = new_skill_md
            full_content = f"---\n{new_skill_yaml}\n---\n{new_skill_md}\n"

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
            if p.get("is_running"): status_indicators.append("🔵")
            elif p.get("has_report") and p.get("has_dashboard"): status_indicators.append("✅")
            elif p.get("has_failed_tasks"): status_indicators.append("❌")
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
    st.markdown("### ORCHESTRATOR")
    
    if "initial_config_synced" not in st.session_state:
        config = api_get("/config")
        if config:
            st.session_state.routing_mode = config.get("routing_mode", "cloud")
            st.session_state.run_mode = config.get("run_mode", "research")
            st.session_state.pinned_model = config.get("pinned_model")
            st.session_state.available_models = config.get("available_models", [])
            st.session_state.random_routing_mode = config.get("random_routing", False)
            st.session_state.initial_config_synced = True
        else:
            st.session_state.routing_mode = "cloud"
            st.session_state.run_mode = "research"
            st.session_state.pinned_model = None
            st.session_state.available_models = []
            st.session_state.random_routing_mode = False
        # Snapshot of the backend's current config: POST only when the user
        # actually changes something. (The previous unconditional POST on every
        # rerender silently clobbered config changes made outside this session,
        # e.g. reverting a routing-mode flip mid-run.)
        st.session_state.last_synced_config = (
            st.session_state.routing_mode,
            st.session_state.pinned_model,
            st.session_state.random_routing_mode,
        )

    proj = None
    last_state = {}
    proj_details = None
    if st.session_state.current_project_id:
        proj_details = api_get(f"/projects/{st.session_state.current_project_id}")
        if proj_details:
            proj = proj_details.get("project", {})
            last_state = proj.get("last_state_json") or {}
            
            # Sync session configuration with loaded project
            if st.session_state.get("last_synced_project_id") != st.session_state.current_project_id:
                st.session_state.fast_mode = last_state.get("fast_mode", False)
                st.session_state.disable_recipes = last_state.get("disable_recipes", False)
                st.session_state.last_synced_project_id = st.session_state.current_project_id

    # Initialize fast_mode and disable_recipes in session_state if not present
    if "fast_mode" not in st.session_state:
        st.session_state.fast_mode = False
    if "disable_recipes" not in st.session_state:
        st.session_state.disable_recipes = False

    # SYSTEM CONFIGURATION SECTION
    with st.container(border=True):
        st.markdown("**SYSTEM CONFIGURATION**")
        cfg_col1, cfg_col2, cfg_col3, cfg_col4, cfg_col5 = st.columns(5)

        _mode_options = ["cloud", "local", "hybrid", "cloud_pinned"]
        _mode_labels = {
            "cloud": "Cloud (tiered)",
            "local": "Local only",
            "hybrid": "Hybrid (cloud plan/report, local exec)",
            "cloud_pinned": "Cloud pinned (single model)",
        }
        _current_mode = st.session_state.get("routing_mode", "cloud")
        st.session_state.routing_mode = cfg_col1.selectbox(
            "Routing Mode",
            options=_mode_options,
            index=_mode_options.index(_current_mode) if _current_mode in _mode_options else 0,
            format_func=lambda m: _mode_labels.get(m, m),
            help=(
                "cloud: tiered hierarchy with the T3→T2→T1 escalation ladder. "
                "local: every stage on local_model. "
                "hybrid: plan construction + report writing on cloud, execution on local_model. "
                "cloud_pinned: one cloud model for all steps, no escalation ladder."
            )
        )
        if st.session_state.routing_mode == "cloud_pinned":
            _models = st.session_state.get("available_models") or []
            _pin = st.session_state.get("pinned_model")
            st.session_state.pinned_model = cfg_col2.selectbox(
                "Pinned Model",
                options=_models,
                index=_models.index(_pin) if _pin in _models else 0,
                help="Used for every workflow step. Failures retry on this model or fail — no escalation."
            ) if _models else cfg_col2.text_input("Pinned Model", value=_pin or "", help="LiteLLM model id")
        else:
            st.session_state.random_routing_mode = cfg_col2.checkbox(
                "Random Routing",
                value=st.session_state.random_routing_mode,
                help="Shuffle intra-tier model order (for testing)."
            )
        st.session_state.fast_mode = cfg_col3.checkbox(
            "Fast Mode",
            value=st.session_state.fast_mode,
            help="Subsample large datasets to 50K rows to prevent execution timeouts."
        )
        _run_mode_opts = ["research", "production"]
        _run_mode_labels = {
            "research": "Research (model-first)",
            "production": "Production (native-first)",
        }
        _cur_run_mode = st.session_state.get("run_mode", "research")
        st.session_state.run_mode = cfg_col5.selectbox(
            "Run Mode",
            options=_run_mode_opts,
            index=_run_mode_opts.index(_cur_run_mode) if _cur_run_mode in _run_mode_opts else 0,
            format_func=lambda m: _run_mode_labels.get(m, m),
            help=(
                "research: the model attempts every node itself; a node's native is only a "
                "post-exhaustion safety net. Keeps pass@model meaningful — that metric is "
                "undefined if the native runs first. Costs tokens and carries the "
                "failed-attempt risks.\n\n"
                "production: a node that declares a native uses it directly and the model is "
                "never asked — deterministic and cheaper on nodes with one defensible "
                "answer. Judgment nodes with no native stay model-generated either way."
            )
        )
        st.session_state.disable_recipes = cfg_col4.checkbox(
            "Disable Recipes",
            value=st.session_state.disable_recipes,
            help="Bypass Standard Operating Procedures (recipes) to run raw objective. Useful for side-by-side comparison."
        )

    # Sync backend settings ONLY when something actually changed in this session.
    _desired_config = (
        st.session_state.routing_mode,
        st.session_state.get("pinned_model"),
        st.session_state.random_routing_mode,
        st.session_state.run_mode,
    )
    if _desired_config != st.session_state.get("last_synced_config"):
        resp = api_post("/config", {
            "routing_mode": st.session_state.routing_mode,
            "pinned_model": st.session_state.get("pinned_model"),
            "random_routing": st.session_state.random_routing_mode,
            "run_mode": st.session_state.run_mode,
        })
        if resp:
            st.session_state.last_synced_config = _desired_config
            st.toast(f"Routing: {st.session_state.routing_mode} · "
                     f"Run mode: {st.session_state.run_mode}")
        else:
            st.warning("Failed to update backend config — check that the routing mode/pinned model is valid.")

    if st.session_state.current_project_id:
        badge_text = f"ACTIVE PROJECT: {st.session_state.current_project_id}"
        if proj:
            flags = []
            if last_state.get("fast_mode"):
                flags.append("FAST MODE")
            if last_state.get("disable_recipes"):
                flags.append("NO RECIPES")
            if flags:
                badge_text += f" ({' | '.join(flags)})"
        st.markdown(f"<div class='active-badge'>{badge_text}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='active-badge'>NO PROJECT LOADED</div>", unsafe_allow_html=True)

    # 1. Spec Launcher
    with st.expander("LAUNCH FROM SPEC", expanded=False):
        specs = api_get("/specs")
        if specs:
            def _fmt_ts(iso):
                if not iso:
                    return "never"
                try:
                    return datetime.fromisoformat(iso).strftime("%b %d %Y %H:%M")
                except Exception:
                    return iso

            import pandas as pd
            spec_df = pd.DataFrame([{
                "Spec": s["filename"],
                "Created": _fmt_ts(s.get("created_at")),
                "Last Used": _fmt_ts(s.get("last_used_at")),
            } for s in specs])

            sel = st.dataframe(
                spec_df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                height=min(38 + len(specs) * 35, 250),
            )

            selected_rows = sel.selection.rows if sel and sel.selection else []
            selected_spec = specs[selected_rows[0]]["filename"] if selected_rows else None

            if selected_spec:
                st.caption(f"Selected: **{selected_spec}**")
                btn_cols = st.columns(3)
                if btn_cols[0].button("VIEW SPEC", key="btn_view_spec", use_container_width=True):
                    st.session_state.view_spec = selected_spec

                if btn_cols[1].button("LOAD SPEC", key="btn_load_spec", use_container_width=True):
                    with st.spinner("Loading..."):
                        res = api_post("/projects/from-spec", {"filename": selected_spec, "launch_workflow": False, "fast_mode": st.session_state.fast_mode, "disable_recipes": st.session_state.disable_recipes})
                        if res and "project" in res:
                            st.toast("Spec loaded successfully!")
                            st.session_state.current_project_id = res["project"]["id"]
                            st.rerun()
                        else:
                            st.error(res.get("detail", "Failed to load spec.")) if res else st.error("Failed to connect to backend.")

                if btn_cols[2].button("LAUNCH", key="btn_launch_spec", type="primary", use_container_width=True):
                    with st.spinner("Launching..."):
                        res = api_post("/projects/from-spec", {"filename": selected_spec, "launch_workflow": True, "fast_mode": st.session_state.fast_mode, "disable_recipes": st.session_state.disable_recipes})
                        if res and "project" in res:
                            st.toast("Spec launched successfully!")
                            st.session_state.current_project_id = res["project"]["id"]
                            st.rerun()
                        else:
                            st.error(res.get("detail", "Failed to launch spec.")) if res else st.error("Failed to connect to backend.")
            else:
                st.caption("Click a row to select a spec.")
        else:
            st.info("No `.md` spec documents found in the `specs/` directory.")

    if "view_spec" in st.session_state and st.session_state.view_spec:
        @st.dialog(f"Edit Spec: {st.session_state.view_spec}", width="large")
        def show_spec():
            if "spec_buffer" not in st.session_state:
                spec_data = api_get(f"/specs/{st.session_state.view_spec}")
                raw = spec_data.get("content", "") if spec_data else ""
                # Split YAML frontmatter from markdown body
                parts = raw.split("---", 2)
                if len(parts) >= 3:
                    st.session_state.spec_yaml = parts[1].strip()
                    st.session_state.spec_md = parts[2].strip()
                else:
                    st.session_state.spec_yaml = ""
                    st.session_state.spec_md = raw.strip()
                st.session_state.spec_buffer = raw

            import hashlib
            spec_key = hashlib.md5(st.session_state.view_spec.encode()).hexdigest()[:8]
            st.markdown("##### FRONTMATTER (YAML)")
            new_yaml = st_ace(
                value=st.session_state.spec_yaml,
                language="yaml",
                theme="monokai",
                height=160,
                key=f"ace_spec_yaml_{spec_key}"
            )
            st.session_state.spec_yaml = new_yaml

            st.markdown("##### OBJECTIVE")
            preview_mode = st.toggle("Preview", key=f"spec_preview_{spec_key}", value=False)

            if preview_mode:
                with st.container(border=True, height=340):
                    st.markdown(st.session_state.spec_md)
                new_md = st.session_state.spec_md
            else:
                new_md = st_ace(
                    value=st.session_state.spec_md,
                    language="plain_text",
                    theme="monokai",
                    height=340,
                    wrap=True,
                    key=f"ace_spec_md_{spec_key}"
                )
                st.session_state.spec_md = new_md

            new_content = f"---\n{new_yaml}\n---\n{new_md}\n"
            st.session_state.spec_buffer = new_content
            
            col1, col2 = st.columns(2)
            if col1.button("SAVE CHANGES", use_container_width=True, type="primary"):
                res = api_post(f"/specs/{st.session_state.view_spec}", {"content": new_content})
                if res and res.get("status") == "success":
                    st.toast("Spec saved successfully.")
                    for _k in ("spec_buffer", "spec_yaml", "spec_md"):
                        st.session_state.pop(_k, None)
                    st.session_state.view_spec = None
                    st.rerun()
                else:
                    st.error("Failed to save spec.")
            
            if col2.button("CLOSE", use_container_width=True):
                for _k in ("spec_buffer", "spec_yaml", "spec_md"):
                    st.session_state.pop(_k, None)
                st.session_state.view_spec = None
                st.rerun()
        show_spec()

    # 2. Objective Input
    objective = st.text_area("Research Objective", 
                            placeholder="Describe your data science goal...", 
                            height=100, 
                            label_visibility="collapsed")
    
    # Check if active project has >50K rows scanned by DataAnalyzer
    show_advisory = False
    max_rows = 0
    if proj_details:
        schemas = last_state.get("__schemas__", {})
        for f_name, s_info in schemas.items():
            rc = s_info.get("row_count", 0)
            if rc > max_rows:
                max_rows = rc
        sample_rows_hint = last_state.get("sample_rows")
        if max_rows > 50000 and not sample_rows_hint:
            show_advisory = True

    if show_advisory and not st.session_state.fast_mode:
        st.warning(
            f"⚠️ **Advisory**: Dataset has {max_rows:,} rows. "
            "It is highly recommended to enable **Fast Mode** to speed up training and prevent sandbox timeouts."
        )
    
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
                        "existing_project_id": st.session_state.current_project_id,
                        "fast_mode": st.session_state.fast_mode,
                        "disable_recipes": st.session_state.disable_recipes
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
            st.session_state.fast_mode = False
            st.session_state.disable_recipes = False
            st.session_state.last_synced_project_id = None
            st.rerun()

    with ctrl_cols[3]:
        if st.button("REFRESH", key="btn_refresh", help="Manually refresh the UI state.", use_container_width=True):
            st.rerun()

    # --- FOLLOW-UP LANE (approach_docs/020) ---
    # Distinct from LAUNCH on purpose: LAUNCH re-runs the full autonomous pipeline, this runs
    # ONE user-directed instruction against the project's live kernel and appends its output
    # to the dashboard. Only offered once a project exists and has results to build on.
    if st.session_state.current_project_id and proj_details and proj_details["project"].get("narrative"):
        with st.expander("🔎 FOLLOW-UP — ask a question about these results", expanded=False):
            st.caption(
                "Runs a single instruction against this project's session state "
                "(rehydrated if needed). Output is appended to the dashboard as its own "
                "section; the original findings are not changed."
            )
            fu_text = st.text_area(
                "Follow-up instruction",
                placeholder="e.g. do error analysis on the false negatives; "
                            "plot recall against tumour grade",
                height=80, key="followup_text", label_visibility="collapsed",
            )
            fu_cols = st.columns([1, 1, 2])
            with fu_cols[0]:
                fu_rehydrate = st.checkbox("Rehydrate", value=True, key="fu_rehydrate",
                                           help="Restore prior kernel state (replays the run's "
                                                "code if the session went cold).")
            with fu_cols[1]:
                if st.button("RUN", key="btn_followup", use_container_width=True):
                    if not (fu_text or "").strip():
                        st.warning("Enter an instruction first.")
                    else:
                        res = api_post(
                            f"/projects/{st.session_state.current_project_id}/followup",
                            {"objective": fu_text.strip(), "rehydrate": fu_rehydrate},
                        )
                        if res and res.get("task_id"):
                            st.toast("Follow-up started.")
                            st.rerun()
                        else:
                            detail = res.get("detail") if isinstance(res, dict) else None
                            st.error(f"Follow-up failed to start: {detail or 'backend error'}")

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

            ORCHESTRATOR_ROLES = {"System", "DataAnalyzer", "Router", "Planner", "PlanCritique",
                                      "Synthesizer", "Critique", "CompletenessVerifier", "DataSampler"}
            coder_task_num = 0

            for t in sorted(tasks, key=lambda x: x.get("created_at", "")):
                # Display instruction block when it changes
                curr_instr_id = t.get("instruction_id")
                if curr_instr_id and curr_instr_id != last_instr_id:
                    instr = instr_map.get(curr_instr_id)
                    if instr:
                        st.chat_message("user").markdown(f"**Instruction:** {instr['content']}")
                    last_instr_id = curr_instr_id

                is_coder_task = t.get("assigned_to") not in ORCHESTRATOR_ROLES
                if is_coder_task:
                    coder_task_num += 1

                st_icon = "⚪"
                if t['assigned_to'] == "Router": st_icon = "📐"
                elif t['assigned_to'] == "Planner": st_icon = "📋"
                elif t['assigned_to'] == "DataSampler": st_icon = "✂️"
                elif t['status'] == "completed": st_icon = "🟢"
                elif t['status'] == "failed": st_icon = "🔴"
                elif t['status'] == "running": st_icon = "🔵"

                expanded = (t['status'] == "running") or (t['assigned_to'] in ["Router", "Planner"] and t['status'] == "completed")

                desc = t['description']
                header_text = desc[:80].rstrip()
                if len(desc) > 80:
                    header_text += "..."
                if is_coder_task:
                    header = f"{st_icon} #{coder_task_num} {header_text}"
                else:
                    header = f"{st_icon} {header_text}"

                with st.expander(header, expanded=expanded):
                    if is_coder_task:
                        st.caption(desc)
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

    # --- SESSION LIFECYCLE (#14) ---
    # Deliberately OUTSIDE the auto-refreshing fragment below: probing the live kernel runs
    # code in the sandbox, so it happens on demand, not every 3 seconds.
    if st.session_state.current_project_id:
        pid = st.session_state.current_project_id
        st.markdown("#### SESSION")

        if st.session_state.get("kernel_info_pid") != pid:
            st.session_state.kernel_info = None
            st.session_state.kernel_info_pid = pid

        k = st.session_state.get("kernel_info")
        if k:
            status = k.get("status", "unknown")
            badge = {"live": "🟢 live", "cold": "⚪ cold"}.get(status, f"• {status}")
            st.caption(f"Kernel: {badge} — {k.get('variable_count', 0)} variable(s)"
                       + (" · pinned" if k.get("pinned") else ""))
            if k.get("variables"):
                with st.expander("Live variables", expanded=False):
                    st.json(k["variables"])
            if status == "cold" and k.get("workspace_replay_available"):
                st.caption("Prior state can be replayed from this run's saved code.")
        else:
            st.caption("Kernel: not checked.")

        sess_cols = st.columns(3)
        with sess_cols[0]:
            if st.button("CHECK", key="btn_kernel_check", use_container_width=True,
                         help="Probe the sandbox session for live variables."):
                with st.spinner("Probing session..."):
                    st.session_state.kernel_info = api_get_long(f"/projects/{pid}/kernel")
                st.rerun()
        with sess_cols[1]:
            if st.button("REHYDRATE", key="btn_rehydrate", use_container_width=True,
                         help="Restore this project's kernel state by replaying its "
                              "completed code. Safe to repeat; may take a while."):
                with st.spinner("Replaying prior task code..."):
                    res = api_post_long(f"/projects/{pid}/rehydrate", {})
                if res:
                    st.session_state.kernel_info = {
                        "status": res.get("status"), "variables": res.get("variables", {}),
                        "variable_count": len(res.get("variables", {})),
                        "pinned": res.get("pinned"), "workspace_replay_available": True,
                    }
                    st.toast(f"Kernel {res.get('status')} "
                             f"({len(res.get('variables', {}))} vars, {res.get('seconds', 0)}s)")
                else:
                    st.error("Rehydration failed.")
                st.rerun()
        with sess_cols[2]:
            if st.button("FINISH", key="btn_finish", use_container_width=True,
                         help="End the interactive session and release the kernel. "
                              "Artifacts and the dashboard are kept."):
                res = api_post(f"/projects/{pid}/finish")
                if res and res.get("finished"):
                    st.session_state.kernel_info = None
                    st.toast("Project finished; kernel released.")
                    st.rerun()
                else:
                    detail = res.get("detail") if isinstance(res, dict) else None
                    st.error(f"Could not finish: {detail or 'backend error'}")

        # --- ANALYST NOTES ---
        if st.session_state.get("notes_pid") != pid:
            fetched = api_get(f"/projects/{pid}/notes") or {}
            st.session_state.notes_text = fetched.get("notes", "")
            st.session_state.notes_pid = pid
        notes_val = st.text_area(
            "Project notes", value=st.session_state.get("notes_text", ""), height=90,
            key="notes_area",
            help="Your own context for this project. Saved to user_notes.txt in the "
                 "workspace and given to the Synthesizer when the report is written.",
        )
        if st.button("SAVE NOTES", key="btn_save_notes", use_container_width=True):
            if api_post(f"/projects/{pid}/notes", {"notes": notes_val}) is not None:
                st.session_state.notes_text = notes_val
                st.toast("Notes saved.")
            else:
                st.error("Could not save notes.")

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
