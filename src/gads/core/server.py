import asyncio
import contextlib
import uuid
import traceback
import json
import textwrap
import base64
import os
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uuid
from gads.core.bus import bus, dispatcher_loop
from gads.core.execution_hub import watchdog_loop, ExecutionHub
from gads.core.database import init_db, engine
from gads.core.models import Project, Task, Artifact, Instruction
from gads.agents.planner import DataSciencePlanner, PlannerInput, PlannerTask, ReconciliationReport, FileMetadata
from gads.agents.router import DataScienceRouter, RouterInput
from gads.agents.plan_critique import PlanCritiqueAgent, PlanCritiqueInput
from gads.agents.spec_drafter import SpecDrafterAgent, SpecDraftInput
from gads.agents.workers.synthesizer import SynthesizerAgent, SynthesizerInput
from gads.agents.workers.critique import CritiqueAgent, CritiqueInput
from gads.agents.workers.completeness_verifier import CompletenessVerifierAgent, CompletenessVerifierInput
from gads.core.executor import ExecutionManager
from gads.tools.sandbox import SandboxClient
from gads.core.registry import get_model_hierarchy, get_local_only, set_local_only, get_random_routing, set_random_routing, get_next_model_dynamic
from gads.core.knowledge import KnowledgeRegistry
from gads.core.reporting import create_master_reports
from gads.core.notebook_exporter import export_python_script, export_notebook, copy_applied_recipe
from gads.core.introspection import summarize_artifact
from gads.core.distiller import distill_dashboard_to_markdown
from gads.core.history_renderer import HistoryRenderer
from gads.core.prompts import prompt_registry
from langfuse import Langfuse
from sqlmodel import select, Session
from dotenv import load_dotenv

load_dotenv()

# Initialize Observability
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

ACTIVE_WORKFLOWS: set[uuid.UUID] = set()

async def run_agent_workflow_wrapper(project_id: uuid.UUID, objective: str, instruction_id: Optional[uuid.UUID] = None):
    """Wrapper to manage the ACTIVE_WORKFLOWS lock."""
    try:
        await run_agent_workflow(project_id, objective, instruction_id)
    finally:
        if project_id in ACTIVE_WORKFLOWS:
            ACTIVE_WORKFLOWS.remove(project_id)

app = FastAPI(title="GADS Core API")

@app.middleware("http")
async def catch_exceptions_middleware(request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        import traceback
        with open("crash_debug.log", "a") as f:
            f.write(f"\n\n--- CRASH AT {datetime.now()} ---\n")
            f.write(f"URL: {request.url}\n")
            f.write(traceback.format_exc())
        raise e

registry = KnowledgeRegistry("src/gads/knowledge/recipes")
WORKSPACE_ROOT = "/home/joergf/projects/MyLocalStack/data/workspaces"

LIVE_STREAMS: Dict[str, Dict[str, str]] = {}

@app.get("/tasks/{task_id}/stream")
def get_task_stream(task_id: str):
    return LIVE_STREAMS.get(task_id, {"reasoning": "", "stdout": ""})

# --- RESPONSE MODELS ---
class ProjectRead(BaseModel):
    id: uuid.UUID
    name: str
    objective: str
    first_instruction: Optional[str] = None
    narrative: Optional[str] = None
    takeaways: Optional[List[str]] = None
    last_state_json: Optional[Dict[str, Any]] = None
    created_at: datetime
    has_dashboard: bool = False
    has_report: bool = False
    has_failed_tasks: bool = False
    is_running: bool = False

    class Config:
        from_attributes = True

class InstructionRead(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    content: str
    created_at: datetime

    class Config:
        from_attributes = True

class ProjectResponse(BaseModel):
    project: ProjectRead
    files: List[str] = []
    instructions: List[InstructionRead] = []

class ConfigUpdate(BaseModel):
    local_only: bool
    random_routing: bool

class FileUpload(BaseModel):
    name: str
    content_base64: str

class ProjectCreateRequest(BaseModel):
    name: str
    objective: str
    files: List[FileUpload] = []
    existing_project_id: Optional[str] = None
    fast_mode: bool = False

class FilesUploadRequest(BaseModel):
    files: List[FileUpload]

class ExternalPathRequest(BaseModel):
    path: str

class RecipeContent(BaseModel):
    content: str

class PromptUpdate(BaseModel):
    content: str

class ProjectSpecMetadata(BaseModel):
    name: Optional[str] = None
    datasets: List[str] = Field(default_factory=list)
    recipes: List[str] = Field(default_factory=list)
    target_column: Optional[str] = None
    feature_columns: List[str] = Field(default_factory=list)
    filters: Optional[str] = None
    domain: Optional[str] = None
    save_model: bool = False

@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(dispatcher_loop())
    asyncio.create_task(watchdog_loop())
    # # asyncio.create_task(archive_cleanup_loop())

@app.get("/recipes", response_model=List[str])
def list_recipes_files():
    return registry.list_recipe_files()

@app.get("/recipes/{filename}")
def get_recipe_raw(filename: str):
    try:
        return {"content": registry.get_raw_recipe(filename)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/recipes/{filename}")
def save_recipe_raw(filename: str, req: RecipeContent):
    try:
        registry.save_raw_recipe(filename, req.content)
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/prompts")
def list_prompts():
    return prompt_registry.list_prompts()

@app.get("/hierarchy")
async def get_hierarchy():
    return await get_model_hierarchy()

@app.post("/prompts/{agent_name}")
def update_prompt(agent_name: str, req: PromptUpdate):
    error = prompt_registry.save_override(agent_name, req.content)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return {"status": "ok"}

@app.delete("/prompts/{agent_name}")
def reset_prompt(agent_name: str):
    success = prompt_registry.delete_override(agent_name)
    if not success:
        raise HTTPException(status_code=404, detail="Override not found")
    return {"status": "ok"}

@app.get("/skills", response_model=List[str])
def list_skills_files():
    return registry.list_skill_files()

@app.get("/skills/{filename}")
def get_skill_raw(filename: str):
    try:
        return {"content": registry.get_raw_skill(filename)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/skills/{filename}")
def save_skill_raw(filename: str, req: RecipeContent):
    try:
        registry.save_raw_skill(filename, req.content)
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/config")
def get_config():
    return {
        "local_only": get_local_only(),
        "random_routing": get_random_routing()
    }

@app.post("/config")
def update_config(req: ConfigUpdate):
    set_local_only(req.local_only)
    set_random_routing(req.random_routing)
    return {
        "status": "success", 
        "local_only": get_local_only(),
        "random_routing": get_random_routing()
    }

async def archive_cleanup_loop():
    """Background task to delete project artifacts older than 30 days."""
    while True:
        try:
            print("  [Maintenance] Starting automated archive cleanup...", flush=True)
            count = await perform_cleanup()
            print(f"  [Maintenance] Cleanup complete. Removed {count} old projects.", flush=True)
        except Exception as e:
            print(f"  [Maintenance] Error during cleanup: {e}", flush=True)
        
        await asyncio.sleep(86400) # Run every 24 hours

async def perform_cleanup() -> int:
    """Logic to identify and delete projects older than 30 days."""
    import shutil
    from datetime import datetime, timedelta
    
    threshold = datetime.now() - timedelta(days=30)
    removed_count = 0
    
    with Session(engine) as session:
        # Find old projects
        old_projects = session.exec(select(Project).where(Project.created_at < threshold)).all()
        
        for p in old_projects:
            # 1. Delete workspace on disk
            workspace_dir = f"{WORKSPACE_ROOT}/{p.id}"
            if os.path.exists(workspace_dir):
                shutil.rmtree(workspace_dir)
            
            # 2. Delete project from DB (cascading will handle tasks/artifacts)
            session.delete(p)
            removed_count += 1
            
        session.commit()
    return removed_count

@app.post("/maintenance/cleanup")
async def manual_cleanup():
    """Manually trigger project archive cleanup."""
    count = await perform_cleanup()
    return {"status": "success", "removed_projects": count}

@app.post("/projects/{project_id}/cancel")
async def cancel_project(project_id: uuid.UUID):
    """Mark a project for cancellation."""
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project: raise HTTPException(status_code=404, detail="Project not found")

        # Mark narrative with cancelled status
        project.narrative = "[CANCELLED] User requested termination."
        session.add(project)

        # Mark all pending/running tasks as failed
        tasks = session.exec(select(Task).where(Task.project_id == project_id, Task.status.in_(["pending", "running"]))).all()
        for t in tasks:
            t.status = "failed"
            t.error = "Workflow terminated by user."
            session.add(t)

        session.commit()

        # IMMEDIATELY reset sandbox to stop any running local models
        await SandboxClient().reset_session(str(project_id))

        hub = ExecutionHub(session)
        hub.create_outbox_event("WORKFLOW_CANCELLED", {"project_id": str(project_id)})
        session.commit()

    return {"status": "cancelled"}

@app.delete("/projects/{project_id}")
async def delete_project(project_id: uuid.UUID):
    """Delete a project and all its artifacts."""
    import shutil
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project: raise HTTPException(status_code=404, detail="Project not found")
        
        # 1. Delete workspace on disk
        workspace_dir = f"{WORKSPACE_ROOT}/{project_id}"
        if os.path.exists(workspace_dir):
            shutil.rmtree(workspace_dir)
            
        # 2. Delete from DB (manually handle cascading to avoid FK constraints)
        tasks = session.exec(select(Task).where(Task.project_id == project_id)).all()
        for t in tasks: session.delete(t)
        
        artifacts = session.exec(select(Artifact).where(Artifact.project_id == project_id)).all()
        for a in artifacts: session.delete(a)
            
        instructions = session.exec(select(Instruction).where(Instruction.project_id == project_id)).all()
        for i in instructions: session.delete(i)
        
        session.delete(project)
        session.commit()
        
    return {"status": "success"}

async def _cleanup_stale_sessions(sandbox, current_project_id: uuid.UUID):
    """Reset sandbox sessions for all projects that have no active (pending/running) tasks.

    Session IDs are str(project_id), so we can enumerate them via the DB without
    needing a sandbox list-sessions endpoint. Called at the start of each execution
    phase so stale kernels from previous runs don't accumulate.
    """
    try:
        with Session(engine) as db:
            all_projects = db.exec(select(Project)).all()
        reset_count = 0
        for project in all_projects:
            if project.id == current_project_id:
                continue
            with Session(engine) as db:
                active = db.exec(
                    select(Task).where(
                        Task.project_id == project.id,
                        Task.status.in_(["pending", "running"])
                    )
                ).first()
            if active is None:
                try:
                    await sandbox.reset_session(str(project.id))
                    reset_count += 1
                except Exception:
                    pass
        if reset_count:
            print(f"  [SessionCleanup] Reset {reset_count} stale sandbox session(s).", flush=True)
    except Exception as e:
        print(f"  [SessionCleanup] Warning: cleanup failed: {e}", flush=True)

    # Always reset the current project's session for a clean kernel slate
    try:
        await sandbox.reset_session(str(current_project_id))
        print(f"  [SessionCleanup] Reset current session {current_project_id}.", flush=True)
    except Exception as e:
        print(f"  [SessionCleanup] Warning: could not reset current session: {e}", flush=True)


async def _probe_kernel_for_metrics(sandbox, project_id: uuid.UUID, session_id: str, required_metrics: List[str]) -> Dict[str, Any]:
    """Probe the IPython kernel for named scalar metric variables. Returns found {name: value} pairs."""
    probe_code = f"""
import json as _json_m
_metrics_found = {{}}
for _m_name in {json.dumps(required_metrics)}:
    _m_val = globals().get(_m_name)
    if _m_val is not None:
        try:
            _as_float = float(_m_val)
            _metrics_found[_m_name] = _as_float
        except (TypeError, ValueError):
            pass  # Not a numeric scalar — do not record; task will be retried
print("GADS_METRICS_JSON:" + _json_m.dumps(_metrics_found))
"""
    try:
        result = await sandbox.execute(probe_code, project_id=project_id, session_id=session_id)
        if "GADS_METRICS_JSON:" in result.stdout:
            raw = result.stdout.split("GADS_METRICS_JSON:")[1].strip().split("\n")[0]
            return json.loads(raw)
    except Exception as e:
        print(f"  [Metrics] Kernel probe failed: {e}")
    return {}

def _merge_metrics_json(workspace_dir: str, new_metrics: Dict[str, Any]):
    """Merge new scalar metrics into workspace metrics.json (creates if absent)."""
    metrics_path = os.path.join(workspace_dir, "metrics.json")
    existing: Dict[str, Any] = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(new_metrics)
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)

def _get_recursive_files(workspace_dir: str) -> List[Dict[str, Any]]:
    """Helper to list all files in workspace recursively with size metadata as JSON-serializable dicts."""
    all_files = []
    if not os.path.exists(workspace_dir):
        return []
    for root, _, files in os.walk(workspace_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, workspace_dir)
            try:
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
            except Exception:
                size_mb = 0.0
            all_files.append({"name": rel_path, "size_mb": size_mb})
    return sorted(all_files, key=lambda x: x["name"])

async def run_agent_workflow(project_id: uuid.UUID, objective: str, instruction_id: Optional[uuid.UUID] = None):
    from gads.core.llm import trace_context
    
    # 1. Create top-level Langfuse Trace
    trace = langfuse_client.trace(
        id=str(project_id),
        name="Project Workflow",
        user_id="default_user",
        metadata={"objective": objective},
        session_id=str(project_id)
    )

    ctx_token = trace_context.set({
        "project_id": str(project_id),
        "workflow_id": str(project_id),
        "user_id": "default_user",
        "langfuse_trace_id": trace.id
    })

    try:
        """Orchestrates the multi-agent workflow with Full Gemini Priority Strategy."""
        print(f"\n--- 🚀 Starting expert workflow for Project {project_id} ---", flush=True)

        async def is_cancelled():
            with Session(engine) as s:
                p = s.get(Project, project_id)
                return p and p.narrative and "[CANCELLED]" in p.narrative

        # 0. IMMEDIATELY CREATE AN ORCHESTRATOR TASK TO BLOCK DUPLICATES
        with Session(engine) as session:
            init_task = Task(
                project_id=project_id,
                instruction_id=instruction_id,
                description="System is grounding session state and re-syncing kernel...",
                assigned_to="System",
                status="running",
                heartbeat=datetime.now()
            )
            session.add(init_task)
            session.commit()
            init_task_id = init_task.id

        executor = ExecutionManager()
        with Session(engine) as session:
            p_obj = session.get(Project, project_id)
            if p_obj and p_obj.last_state_json and "__schemas__" in p_obj.last_state_json:
                executor.file_schemas = p_obj.last_state_json["__schemas__"]
                print(f"  [Workflow] Loaded {len(executor.file_schemas)} cached file schemas from Project state.")
        hierarchy = await get_model_hierarchy()


        # Check cancellation before start
        if await is_cancelled(): return

        # 0. SYNC GROUND TRUTH
        print(f"  [Workflow] Grounding session state...", flush=True)
        workspace_dir = f"{WORKSPACE_ROOT}/{project_id}"
        current_files_meta = _get_recursive_files(workspace_dir)

        # Check sandbox health instead of full execute to avoid heavy state introspection hangs
        try:
            await asyncio.wait_for(executor.sandbox.client.get(f"{executor.sandbox.base_url}/health"), timeout=5.0)
            print(f"  [Workflow] Sandbox is healthy. Proceeding.", flush=True)
        except Exception as e:
            print(f"  [Workflow] Grounding health check failed: {e}. Proceeding anyway.", flush=True)

        # Mark init task completed after health check (schema probe gets its own task below)
        with Session(engine) as session:
            itask = session.get(Task, init_task_id)
            if itask:
                itask.status = "completed"
                session.add(itask)
                session.commit()

        if await is_cancelled(): return

        # 0.5. DATA ANALYZER — rich schema + distribution profiling (runs once, outside retry loop)
        print(f"  [Workflow] DataAnalyzer: profiling dataset schemas and distributions...", flush=True)
        from gads.agents.planner import FileMetadata as FM
        with Session(engine) as session:
            analyzer_task = Task(
                project_id=project_id,
                instruction_id=instruction_id,
                description="DataAnalyzer is profiling dataset schemas, value distributions, and cardinality...",
                assigned_to="DataAnalyzer",
                status="running",
                heartbeat=datetime.now()
            )
            session.add(analyzer_task)
            session.commit()
            session.refresh(analyzer_task)
            analyzer_task_id = analyzer_task.id

        planner_files = []
        detected_schemas = {}
        PROFILABLE_EXTS = (".csv", ".parquet", ".xlsx", ".xls", ".json", ".txt", ".md", ".log")
        for f in current_files_meta:
            columns_dtypes = None
            if any(f["name"].endswith(ext) for ext in PROFILABLE_EXTS):
                columns_dtypes = await _probe_file_schema(executor, project_id, f["name"])
                if columns_dtypes and "error" not in columns_dtypes:
                    detected_schemas[f["name"]] = columns_dtypes
            planner_files.append(FM(name=f["name"], size_mb=f["size_mb"], columns_and_dtypes=columns_dtypes))

        if detected_schemas:
            executor.file_schemas = detected_schemas
            with Session(engine) as session:
                project = session.get(Project, project_id)
                if project:
                    state = project.last_state_json or {}
                    schemas = state.get("__schemas__", {})
                    schemas.update(detected_schemas)
                    state["__schemas__"] = schemas
                    project.last_state_json = state
                    session.add(project)
                    session.commit()

        profile_lines = []
        for fname, profile in detected_schemas.items():
            if "schema" in profile:
                ncols = len(profile["schema"])
                nrows = profile.get("row_count", "?")
                nrows_str = f"{nrows:,}" if isinstance(nrows, int) else str(nrows)
                profile_lines.append(f"**{fname}**: {ncols} columns, {nrows_str} rows")
            else:
                profile_lines.append(f"**{fname}**: profiled ({profile.get('type', 'file')})")

        with Session(engine) as session:
            at = session.get(Task, analyzer_task_id)
            if at:
                at.status = "completed"
                at.result_json = {
                    "stdout": (
                        "**Data Profiles Generated:**\n" +
                        "\n".join(f"- {s}" for s in profile_lines)
                        if profile_lines else "No profilable files found in workspace."
                    )
                }
                session.add(at)
                session.commit()

        if await is_cancelled(): return

        # 0.6. SPEC DRAFTING — generate or load structured workflow spec
        # Build files_with_schemas string for SpecDrafter
        files_schema_lines = []
        for f in planner_files:
            if f.columns_and_dtypes and f.columns_and_dtypes.get("schema"):
                cols = list(f.columns_and_dtypes["schema"].keys())
                files_schema_lines.append(f"'{f.name}' ({f.size_mb:.2f} MB) — columns: {cols}")
            else:
                files_schema_lines.append(f"'{f.name}' ({f.size_mb:.2f} MB)")
        files_schema_str = "\n".join(files_schema_lines) if files_schema_lines else "None"

        workflow_spec_path = Path(workspace_dir) / "workflow_spec.md"
        spec_hints: Dict[str, Any] = {}
        formalized_objective = objective  # default: use raw objective

        if workflow_spec_path.exists():
            # Re-use existing spec (from-spec path or prior run)
            print(f"  [SpecDrafter] Found existing workflow_spec.md — loading.", flush=True)
            try:
                spec_content = workflow_spec_path.read_text(encoding="utf-8")
                yaml_data = {}
                if spec_content.startswith("---"):
                    parts = spec_content.split("---", 2)
                    if len(parts) >= 3:
                        yaml_data = yaml.safe_load(parts[1]) or {}
                        fo = spec_content.split("---", 2)[2].strip()
                        if fo:
                            formalized_objective = fo
                # Pull hints from YAML frontmatter
                for key in ("target_column", "feature_columns", "filters", "domain", "recipe_id", "save_model", "sample_rows"):
                    if key in yaml_data:
                        spec_hints[key] = yaml_data[key]
            except Exception as e:
                print(f"  [SpecDrafter] Failed to parse workflow_spec.md: {e}. Using raw objective.", flush=True)
        else:
            # Run SpecDrafter to generate the spec
            spec_model_fallback = ["local_model"] if get_local_only() else ["claude-haiku-4.5"]
            spec_model = hierarchy.get("T3", {}).get("models", spec_model_fallback)[0]
            available_recipe_ids = [r["id"] for r in registry.get_recipes_summary() if "id" in r]

            with Session(engine) as session:
                spec_task = Task(
                    project_id=project_id,
                    instruction_id=instruction_id,
                    description=f"SpecDrafter ({spec_model}) is formalizing the project specification...",
                    assigned_to="SpecDrafter",
                    status="running",
                    heartbeat=datetime.now()
                )
                session.add(spec_task)
                session.commit()
                session.refresh(spec_task)
                spec_task_id = spec_task.id

            spec_draft = None
            try:
                spec_span = trace.span(name="Spec Drafting", metadata={"task_id": str(spec_task_id)})
                trace_context.get().update({
                    "agent_name": "SpecDrafter",
                    "task_id": str(spec_task_id),
                    "parent_observation_id": spec_span.id
                })

                spec_agent = SpecDrafterAgent(model=spec_model)
                spec_res = await asyncio.wait_for(
                    spec_agent.run(SpecDraftInput(
                        objective=objective,
                        files_with_schemas=files_schema_str,
                        available_recipe_ids=available_recipe_ids
                    )),
                    timeout=120.0
                )
                spec_draft = spec_res.content
                spec_span.end(output=spec_draft.model_dump())

                formalized_objective = spec_draft.formalized_objective
                for key in ("target_column", "feature_columns", "filters", "domain", "recipe_id"):
                    val = getattr(spec_draft, key, None)
                    if val:
                        spec_hints[key] = val

                # Check if fast_mode is enabled for the project
                fast_mode = False
                with Session(engine) as session:
                    proj = session.get(Project, project_id)
                    if proj and proj.last_state_json:
                        fast_mode = proj.last_state_json.get("fast_mode", False)

                # Write workflow_spec.md to workspace
                frontmatter: Dict[str, Any] = {"name": spec_draft.name, "datasets": spec_draft.datasets}
                if fast_mode:
                    frontmatter["sample_rows"] = 50000
                    spec_hints["sample_rows"] = 50000
                if spec_draft.recipe_id:
                    frontmatter["recipe_id"] = spec_draft.recipe_id
                if spec_draft.target_column:
                    frontmatter["target_column"] = spec_draft.target_column
                if spec_draft.feature_columns:
                    frontmatter["feature_columns"] = spec_draft.feature_columns
                if spec_draft.filters:
                    frontmatter["filters"] = spec_draft.filters
                if spec_draft.domain:
                    frontmatter["domain"] = spec_draft.domain
                spec_md = f"---\n{yaml.dump(frontmatter, default_flow_style=False).strip()}\n---\n\n{formalized_objective}\n"
                workflow_spec_path.write_text(spec_md, encoding="utf-8")
                print(f"  [SpecDrafter] Wrote workflow_spec.md to workspace.", flush=True)

                with Session(engine) as session:
                    st = session.get(Task, spec_task_id)
                    if st:
                        st.status = "completed"
                        st.result_json = {
                            "stdout": f"**Formalized Objective:**\n{formalized_objective}\n\n**Hints:**\n{json.dumps(spec_hints, indent=2)}",
                            "model_used": spec_model
                        }
                        session.add(st)
                        session.commit()

            except Exception as e:
                print(f"  [SpecDrafter] Failed: {e}. Writing minimal spec and continuing.", flush=True)
                # Fallback: write minimal spec so workflow_spec.md always exists
                minimal_spec = f"---\nname: \"{objective[:60]}\"\ndatasets: {json.dumps([f['name'] for f in current_files_meta])}\n---\n\n{objective}\n"
                try:
                    workflow_spec_path.write_text(minimal_spec, encoding="utf-8")
                except Exception:
                    pass
                with Session(engine) as session:
                    st = session.get(Task, spec_task_id)
                    if st:
                        st.status = "failed"
                        st.error = str(e)
                        session.add(st)
                        session.commit()

        print(f"  [SpecDrafter] Formalized objective: {formalized_objective[:80]}...", flush=True)

        # 1. ROUTING (Resilient & Resourced)
        router_fallback = ["local_model"] if get_local_only() else ["gemini-3.1-flash-lite-preview"]
        router_model = hierarchy.get("T3", {}).get("models", router_fallback)[0]

        intent = None
        knowledge_report = None

        while True:
            with Session(engine) as session:
                route_task = Task(
                    project_id=project_id,
                    instruction_id=instruction_id,
                    description=f"Architect ({router_model}) is classifying intent and consulting knowledge base...",
                    assigned_to="Router",
                    status="running",
                    heartbeat=datetime.now()
                )
                session.add(route_task)
                session.commit()
                session.refresh(route_task)

                # Create a Span for the Architect
                span = trace.span(name="Architect Routing", metadata={"task_id": str(route_task.id)})
                trace_context.get().update({
                    "agent_name": "Router",
                    "task_id": str(route_task.id),
                    "parent_observation_id": span.id
                })

                rid_str = str(route_task.id)
                LIVE_STREAMS[rid_str] = {"reasoning": "", "stdout": ""}
                async def stream_router_callback(token: str):
                    LIVE_STREAMS[rid_str]["reasoning"] += token

                try:
                    router = DataScienceRouter(model=router_model)
                    router_res = await router.run(RouterInput(
                        objective=formalized_objective,
                        available_recipes=registry.get_recipes_summary()
                    ), stream_callback=stream_router_callback)
                    intent = router_res.content
                    span.end(output=intent.model_dump())

                    # Inline Knowledge Retrieval — prefer Router match, fall back to SpecDrafter hint
                    recipe_id = intent.matched_recipe_id
                    recipe = registry.get_recipe(recipe_id) if recipe_id else None
                    if not recipe and spec_hints.get("recipe_id"):
                        recipe_id = spec_hints.get("recipe_id")
                        recipe = registry.get_recipe(recipe_id) if recipe_id else None

                    
                    recipe_info = "No specific recipe found. Proceeding with general data science reasoning."
                    if recipe:
                        knowledge_report = ReconciliationReport(
                            recipe_id=recipe.id,
                            rationale=recipe.rationale,
                            recommended_dag_nodes=[node.dict() for node in recipe.dag],
                            skippable_nodes=[],
                            schema_warnings=[]
                        )
                        recipe_info = f"Applied SOP: {recipe.id}\nRationale: {recipe.rationale}"

                    route_task.status = "completed"
                    route_task.result_json = {
                        "stdout": f"**Decision Reasoning:**\n{intent.reasoning}\n\n**Intent Classification:**\n- Task Type: `{intent.task_type}`\n- Modality: `{intent.data_modality}`\n- Confidence: `{intent.confidence}`\n- Matched Recipe: `{intent.matched_recipe_id or 'None'}`\n\n--- KNOWLEDGE BASE ---\n{recipe_info}",
                        "model_used": router_model
                    }
                    session.add(route_task)
                    session.commit()
                    if rid_str in LIVE_STREAMS: del LIVE_STREAMS[rid_str]
                    break # Success
                except Exception as e:
                    print(f"  [Router] Call failed: {e}. Attempting escalation...", flush=True)
                    span.end(output={"error": str(e)})
                    if rid_str in LIVE_STREAMS: del LIVE_STREAMS[rid_str]
                    
                    next_model = get_next_model_dynamic(router_model, hierarchy)
                    if next_model and next_model != router_model:
                        route_task.status = "failed"
                        route_task.error = f"Service unavailable, retrying with {next_model}: {str(e)}"
                        session.add(route_task)
                        session.commit()
                        router_model = next_model
                        continue
                    else:
                        raise e

        print(f"  [Router] Intent: {intent.task_type} (Recipe: {intent.matched_recipe_id})", flush=True)

        if await is_cancelled(): return

        # --- SAMPLE BUDGET ADVISOR (deterministic, no LLM) ---
        # Checks dataset row count against task type and injects a sampling hint into
        # spec_hints so the Planner includes subsampling in its first task description.
        # Threshold: 50K rows for ML training recipes; 200K for lighter analysis.
        # Override: spec frontmatter `sample_rows: N` always takes precedence.
        _TRAINING_TASK_TYPES = {
            "binary_classification", "multiclass_classification", "regression",
            "classification", "automl", "tabular_modeling", "time_series_forecasting",
            "forecasting", "time_series", "causal_inference",
        }
        _SAMPLE_THRESHOLD_TRAINING = 50_000
        _SAMPLE_THRESHOLD_ANALYSIS = 200_000

        if "sample_rows" not in spec_hints:
            max_rows = 0
            for f in planner_files:
                rc = (f.columns_and_dtypes or {}).get("row_count", 0) or 0
                max_rows = max(max_rows, rc)

            if max_rows > 0:
                is_training = intent.task_type in _TRAINING_TASK_TYPES
                threshold = _SAMPLE_THRESHOLD_TRAINING if is_training else _SAMPLE_THRESHOLD_ANALYSIS
                if max_rows > threshold:
                    spec_hints["sample_rows"] = threshold
                    print(
                        f"  [SampleBudget] Dataset has {max_rows:,} rows "
                        f"(task_type={intent.task_type}, threshold={threshold:,}). "
                        f"Injecting sample_rows={threshold} hint for Planner.",
                        flush=True
                    )
                    
                    # Create DataSampler task in DB
                    with Session(engine) as session:
                        sampler_task = Task(
                            project_id=project_id,
                            instruction_id=instruction_id,
                            description=f"DataSampler: Auto-sampling large dataset to prevent timeout ({threshold:,} rows cap)",
                            assigned_to="DataSampler",
                            status="completed",
                            result_json={
                                "stdout": f"[SampleBudget] Dataset has {max_rows:,} rows, exceeding the threshold of {threshold:,} rows for {intent.task_type}.\nAuto-injected sample_rows={threshold} constraint to prevent sandbox timeouts."
                            },
                            heartbeat=datetime.now()
                        )
                        session.add(sampler_task)
                        session.commit()
        else:
            # sample_rows is already present (e.g. set via fast_mode or explicitly in spec)
            with Session(engine) as session:
                existing_sampler = session.exec(select(Task).where(Task.project_id == project_id, Task.assigned_to == "DataSampler")).first()
                if not existing_sampler:
                    val = spec_hints["sample_rows"]
                    sampler_task = Task(
                        project_id=project_id,
                        instruction_id=instruction_id,
                        description=f"DataSampler: Applying row limit ({val:,} rows cap)",
                        assigned_to="DataSampler",
                        status="completed",
                        result_json={
                            "stdout": f"[SampleBudget] Applying sample_rows={val} constraint from spec frontmatter / Fast Mode."
                        },
                        heartbeat=datetime.now()
                    )
                    session.add(sampler_task)
                    session.commit()

        # --- MAIN WORKFLOW LOOP (Planning -> Execution -> Synthesis -> Critique) ---
        MAX_WORKFLOW_ATTEMPTS = 3
        workflow_attempt = 0
        critique_feedback = None
        final_synth = None
        redundant_plots = []
        task_ids = []
        workflow_succeeded = False

        while workflow_attempt < MAX_WORKFLOW_ATTEMPTS:
            workflow_attempt += 1
            print(f"\n  [Workflow] Starting attempt {workflow_attempt}/{MAX_WORKFLOW_ATTEMPTS}...", flush=True)
            
            if await is_cancelled(): return

            # 3. PLANNING (Resilient Decomposition)
            planner_fallback = ["local_model"] if get_local_only() else ["gemini-3-flash-preview"]
            planner_model = hierarchy.get("T2", {}).get("models", planner_fallback)[0]
            # planner_files and spec_hints are pre-computed above (outside this retry loop)

            planner_res = None
            while True:
                # Create a Task for the Planner to show in the UI
                with Session(engine) as session:
                    plan_task = Task(
                        project_id=project_id,
                        instruction_id=instruction_id,
                        description=f"Planner ({planner_model}) is decomposing objective into discrete tasks (Attempt {workflow_attempt})...",
                        assigned_to="Planner",
                        status="running",
                        heartbeat=datetime.now()
                    )
                    session.add(plan_task)
                    session.commit()
                    session.refresh(plan_task)

                    pid_str = str(plan_task.id)
                    LIVE_STREAMS[pid_str] = {"reasoning": "", "stdout": ""}
                    async def stream_planner_callback(token: str):
                        LIVE_STREAMS[pid_str]["reasoning"] += token

                    span = trace.span(name=f"Project Planning (Attempt {workflow_attempt})", metadata={"task_id": pid_str})
                    trace_context.get().update({
                        "agent_name": "Planner",
                        "task_id": pid_str,
                        "parent_observation_id": span.id
                    })

                    try:
                        planner = DataSciencePlanner(model=planner_model)
                        planner_res = await planner.run(PlannerInput(
                            objective=formalized_objective,
                            available_models_hierarchy=hierarchy,
                            available_files=planner_files,
                            knowledge_report=knowledge_report,
                            available_skills=registry.get_skills_summary(),
                            critique_feedback=critique_feedback,
                            user_hints={k: v for k, v in spec_hints.items() if k != "save_model"} or None
                        ), stream_callback=stream_planner_callback)
                        span.end(output=planner_res.content.model_dump())

                        # --- RECIPE ENFORCER (deterministic post-Planner override) ---
                        # The local model reliably ignores the MANDATORY recipe instruction
                        # in the prompt. When a recipe is matched, replace the LLM's tasks
                        # verbatim with tasks constructed from the recipe DAG nodes.
                        # The node `intent` field is already a full Coder-ready description.
                        if knowledge_report and knowledge_report.recommended_dag_nodes:
                            enforced_steps = []
                            for idx, node in enumerate(knowledge_report.recommended_dag_nodes):
                                tier = node.get("worker_tier", "T2")
                                model_id = (
                                    hierarchy.get(tier, hierarchy.get("T2", {}))
                                    .get("models", ["local_model"])[0]
                                )

                                postcondition: Dict[str, Any] = {"output_type": "value"}
                                if node.get("required_metrics"):
                                    postcondition["required_metrics"] = node["required_metrics"]
                                if node.get("produces"):
                                    postcondition["required_variables"] = node["produces"]

                                description = node.get("intent", node.get("id", "")).strip()
                                if idx == 0:
                                    hint_lines = []
                                    if spec_hints.get("target_column"):
                                        hint_lines.append(f"Target column: `{spec_hints['target_column']}`.")
                                    if spec_hints.get("sample_rows"):
                                        n = spec_hints["sample_rows"]
                                        hint_lines.append(
                                            f"SAMPLING CONSTRAINT: immediately after loading, apply "
                                            f"`df = df.sample({n}, random_state=42).reset_index(drop=True)` "
                                            f"to cap the dataset at {n:,} rows."
                                        )
                                    if hint_lines:
                                        description += "\n\n[SPEC HINTS: " + " ".join(hint_lines) + "]"

                                skill_matches = registry.find_skills_scored(description)
                                attached = [s.id for s, _ in skill_matches]
                                if "sandbox_environment" not in attached:
                                    attached.append("sandbox_environment")

                                enforced_steps.append(PlannerTask(
                                    description=description,
                                    assigned_to=model_id,
                                    postcondition=postcondition,
                                    attached_skills=attached
                                ))

                            original_count = len(planner_res.content.steps)
                            planner_res.content.steps = enforced_steps
                            print(
                                f"  [RecipeEnforcer] Replaced {original_count} Planner tasks with "
                                f"{len(enforced_steps)} recipe DAG nodes "
                                f"(recipe: {knowledge_report.recipe_id})",
                                flush=True
                            )

                        plan_task.status = "completed"
                        plan_task.result_json = {
                            "stdout": f"Successfully decomposed objective into {len(planner_res.content.steps)} discrete tasks.",
                            "model_used": planner_model
                        }
                        session.add(plan_task)
                        session.commit()
                        if pid_str in LIVE_STREAMS: del LIVE_STREAMS[pid_str]
                        break # Success
                    except Exception as e:
                        print(f"  [Planner] Call failed: {e}. Attempting escalation...", flush=True)
                        span.end(output={"error": str(e)})
                        if pid_str in LIVE_STREAMS: del LIVE_STREAMS[pid_str]

                        next_model = get_next_model_dynamic(planner_model, hierarchy)
                        if next_model and next_model != planner_model:
                            plan_task.status = "failed"
                            plan_task.error = f"Service unavailable, retrying with {next_model}: {str(e)}"
                            session.add(plan_task)
                            session.commit()
                            planner_model = next_model
                            continue
                        else:
                            raise e

            
            # 3.1 PLAN CRITIQUE
            plan_critique_fallback = ["local_model"] if get_local_only() else ["gemini-3-flash-preview"]
            plan_critique_model = hierarchy.get("T2", {}).get("models", plan_critique_fallback)[0]
            
            plan_critique = None
            while True:
                with Session(engine) as session:
                    pc_task = Task(
                        project_id=project_id,
                        instruction_id=instruction_id,
                        description=f"Auditor ({plan_critique_model}) is evaluating the proposed plan (Attempt {workflow_attempt})...",
                        assigned_to="PlanCritique",
                        status="running",
                        heartbeat=datetime.now()
                    )
                    session.add(pc_task)
                    session.commit()
                    session.refresh(pc_task)

                    span = trace.span(name=f"Plan Critique (Attempt {workflow_attempt})", metadata={"task_id": str(pc_task.id)})
                    trace_context.get().update({
                        "agent_name": "PlanCritique", 
                        "task_id": str(pc_task.id),
                        "parent_observation_id": span.id
                    })

                    try:
                        pc_agent = PlanCritiqueAgent(model=plan_critique_model)
                        pc_res = await pc_agent.run(PlanCritiqueInput(
                            objective=objective,
                            proposed_steps=planner_res.content.steps,
                            knowledge_report=knowledge_report,
                            available_files=[f["name"] for f in current_files_meta]
                        ))
                        plan_critique = pc_res.content
                        span.end(output=plan_critique.model_dump())

                        pc_task.status = "completed"
                        pc_task.result_json = {
                            "stdout": f"**Plan Evaluation:**\n- Approved: `{plan_critique.is_approved}`\n- Terminal Failure: `{plan_critique.is_terminal_failure}`\n- Missing: `{', '.join(plan_critique.missing_requirements) if plan_critique.missing_requirements else 'None'}`\n\n**Feedback:**\n{plan_critique.feedback}",
                            "model_used": plan_critique_model
                        }
                        session.add(pc_task)
                        session.commit()
                        break # Successfully audited
                    except Exception as e:
                        print(f"  [PlanCritique] Call failed: {e}. Attempting escalation...", flush=True)
                        span.end(output={"error": str(e)})
                        
                        next_model = get_next_model_dynamic(plan_critique_model, hierarchy)
                        if next_model and next_model != plan_critique_model:
                            pc_task.status = "failed"
                            pc_task.error = f"Service unavailable, retrying with {next_model}: {str(e)}"
                            session.add(pc_task)
                            session.commit()
                            plan_critique_model = next_model
                            continue
                        else:
                            raise e

            if not plan_critique.is_approved:
                print(f"  [Workflow] ❌ Plan attempt {workflow_attempt} rejected by Auditor. Feedback: {plan_critique.feedback[:100]}...", flush=True)
                
                if plan_critique.is_terminal_failure:
                    print(f"  [Workflow] 🛑 TERMINAL FAILURE detected by Auditor. Halting.", flush=True)
                    with Session(engine) as session:
                        proj = session.get(Project, project_id)
                        if proj:
                            proj.narrative = f"[HALTED] {plan_critique.feedback}"
                            session.add(proj)
                            
                        # Also mark all current workflow attempts as failed to clear the UI
                        statement = select(Task).where(Task.project_id == project_id, Task.status == "running")
                        running_tasks = session.exec(statement).all()
                        for rt in running_tasks:
                            rt.status = "failed"
                            rt.error = "Workflow halted due to terminal environmental issue."
                            session.add(rt)
                        session.commit()
                    return # Exit the workflow entirely

                critique_feedback = plan_critique.feedback
                continue # Back to Planning


            tasks_to_run = []
            with Session(engine) as session:
                hub = ExecutionHub(session)
                
                # Flatten hierarchy to get valid models
                valid_models = []
                for tier_data in hierarchy.values():
                    valid_models.extend(tier_data.get("models", []))
                fallback_model = hierarchy.get("T3", {}).get("models", ["local_model"])[0]

                for step in planner_res.content.steps:
                    # Sanitize assigned_to to prevent hallucinations from causing un-routable tasks
                    assigned_model = step.assigned_to
                    if assigned_model not in valid_models:
                        print(f"  [Workflow] Warning: Planner hallucinated model '{assigned_model}'. Falling back to '{fallback_model}'.", flush=True)
                        assigned_model = fallback_model

                    new_task = Task(
                        project_id=project_id,
                        instruction_id=instruction_id,
                        description=step.description,
                        assigned_to=assigned_model,
                        postcondition_json=step.postcondition,
                        attached_skills=step.attached_skills,
                        status="pending"
                    )
                    session.add(new_task)
                    tasks_to_run.append(new_task)

                # POSTCONDITION SANITIZER (runs before commit so changes persist atomically)
                # Strip required_columns entries that cannot belong to this task:
                # any column that isn't in any known file schema AND isn't in any prior
                # task's declared outputs AND isn't mentioned in this task's description.
                # This prevents the local_model from assigning downstream-derived target
                # columns (e.g. 'winner') to upstream merge tasks that never create them.
                _pc_known: set = set()
                for _schema in detected_schemas.values():
                    _pc_known.update(_schema.keys())
                for _t in tasks_to_run:
                    _contract = _t.postcondition_json
                    if _contract and _contract.get("output_type") == "dataframe":
                        _required = _contract.get("required_columns", [])
                        _desc_lower = _t.description.lower()
                        _stripped, _kept = [], []
                        for _col in _required:
                            if _col in _pc_known or _col.lower() in _desc_lower:
                                _kept.append(_col)
                            else:
                                _stripped.append(_col)
                        if _stripped:
                            print(f"  [PlanSanitizer] '{_t.description[:60]}': stripped downstream columns {_stripped}", flush=True)
                            _t.postcondition_json = {**_contract, "required_columns": _kept}
                    _pc_known.update((_t.postcondition_json or {}).get("required_columns", []))

                session.commit()

                for t in tasks_to_run:
                    session.refresh(t)
                    hub.create_outbox_event("TASK_CREATED", {"task_id": str(t.id), "description": t.description})
                session.commit()
                task_ids = [t.id for t in tasks_to_run]

            # 4. EXECUTION
            await _cleanup_stale_sessions(executor.sandbox, project_id)
            for task_id in task_ids:
                if await is_cancelled(): return

                # Create a Span for this specific worker task
                task_span = trace.span(name="Task Execution", metadata={"task_id": str(task_id)})

                while True:
                    desc, assigned_to = "", ""
                    with Session(engine) as session:
                        hub = ExecutionHub(session)
                        task_obj = session.get(Task, task_id)
                        if not task_obj: break
                        if hub.claim_task(task_id):
                            executor.coder.model = task_obj.assigned_to
                            executor.coder.model_str = task_obj.assigned_to
                            desc = task_obj.description
                        else: break 

                    # Lazy two-tier skill loading (index always; full body only when needed)
                    # Planner-attached skills always get full body (curated, intentional).
                    # Keyword-matched skills need >=2 trigger hits to load full body (avoids weak-match bloat).
                    KEYWORD_HIT_THRESHOLD = 2
                    assigned_ids = set(task_obj.attached_skills or [])
                    scored_matches = registry.find_skills_scored(desc)

                    full_body_ids = set(assigned_ids)
                    for skill, hits in scored_matches:
                        if skill.id not in full_body_ids and hits >= KEYWORD_HIT_THRESHOLD:
                            full_body_ids.add(skill.id)

                    loaded_skills = [registry.skills[sid] for sid in full_body_ids if sid in registry.skills]

                    ctx_parts = []
                    index_lines = [f"- {s.id}: {s.description}" for s in registry.skills.values()]
                    if index_lines:
                        ctx_parts.append("### SKILLS INDEX (all available)\n" + "\n".join(index_lines))
                    if loaded_skills:
                        bodies = "\n\n".join(f"#### {s.id}\n{s.content}" for s in loaded_skills)
                        ctx_parts.append(f"### LOADED SKILL GUIDANCE\n{bodies}")
                        print(f"    [Workflow] Applied skills: {[s.id for s in loaded_skills]}", flush=True)
                    skills_ctx = "\n\n".join(ctx_parts) if ctx_parts else None

                    # 4.1 CONTEXT HARDENING: SLIDING WINDOW & INTROSPECTION
                    all_tasks = session.exec(select(Task).where(Task.project_id == project_id).order_by(Task.created_at.asc())).all()
                    current_task_idx = next((idx for idx, t in enumerate(all_tasks) if t.id == task_id), -1)
                    context = HistoryRenderer.build_coder_context(all_tasks, current_task_idx)

                    # 4.2 KERNEL NAMESPACE SNAPSHOT (Preventing Amnesia)
                    # Before each task, we probe the kernel for live variables and dtypes
                    snapshot_code = """
import json
import pandas as pd
import numpy as np
_vars = dir()
_summary = {}
for _v in _vars:
    if _v.startswith('_'): continue
    try:
        _obj = globals()[_v]
        if isinstance(_obj, pd.DataFrame):
            _summary[_v] = f"DataFrame ({_obj.shape[0]}x{_obj.shape[1]}) - Columns: {list(_obj.columns)}"
        elif isinstance(_obj, (list, dict, np.ndarray)):
            _summary[_v] = f"{type(_obj).__name__} (len: {len(_obj)})"
        elif isinstance(_obj, (str, int, float, bool)):
            _summary[_v] = _obj
        elif hasattr(_obj, 'predict') and hasattr(_obj, 'fit'):
            _summary[_v] = f"Model ({type(_obj).__name__})"
        else:
            _summary[_v] = f"{type(_obj).__module__}.{type(_obj).__name__}"
    except: pass
print("GADS_STATE_SNAPSHOT:" + json.dumps(_summary))
"""
                    namespace_summary = "No live variables detected."
                    try:
                        snap_res = await asyncio.wait_for(executor.sandbox.execute(snapshot_code, project_id=project_id, session_id=str(project_id)), timeout=5.0)
                        if "GADS_STATE_SNAPSHOT:" in snap_res.stdout:
                            raw_snap = snap_res.stdout.split("GADS_STATE_SNAPSHOT:")[1].strip().split("\n")[0]
                            namespace_summary = json.loads(raw_snap)
                    except Exception as e:
                        print(f"    [Workflow] Warning: Namespace snapshot failed: {e}")

                    state_summary_str = json.dumps(namespace_summary, indent=2)

                    tid_str = str(task_id)
                    LIVE_STREAMS[tid_str] = {"reasoning": "", "stdout": ""}

                    async def stream_reasoning_callback(token: str):
                        LIVE_STREAMS[tid_str]["reasoning"] += token

                    async def stream_stdout_callback(text: str):
                        LIVE_STREAMS[tid_str]["stdout"] = text

                    trace_context.get().update({
                        "agent_name": "CodeGenerator",
                        "task_id": tid_str,
                        "parent_observation_id": task_span.id
                    })

                    # Keepalive: local-model LLM + execution can exceed the 5-min watchdog window.
                    # Refresh the heartbeat every 60s while run_task is awaited.
                    async def _heartbeat_loop(tid: uuid.UUID):
                        while True:
                            await asyncio.sleep(60)
                            try:
                                with Session(engine) as _s:
                                    ExecutionHub(_s).heartbeat(tid)
                            except Exception:
                                pass

                    _hb_task = asyncio.create_task(_heartbeat_loop(task_id))
                    try:
                        res, model_used = await executor.run_task(
                            desc,
                            project_id=project_id,
                            session_id=str(project_id),
                            skills_context=skills_ctx,
                            task_id=task_id,
                            stdout_callback=stream_stdout_callback,
                            stream_callback=stream_reasoning_callback,
                            cancel_check=is_cancelled,
                            state_summary=state_summary_str
                        )
                    finally:
                        _hb_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await _hb_task

                    # Update task span with result details
                    task_span.update(output={"model": model_used, "code_len": len(res.code) if res.code else 0})

                    with Session(engine) as session:
                        hub = ExecutionHub(session)
                        task_obj = session.get(Task, task_id)
                        
                        # Handle Bypassed (Handover)
                        if res.result and res.result.startswith("HANDOVER_BUNDLE:"):
                            bundle_file = res.result.split(":")[1]
                            hub.bypass_task(task_id, {"stdout": res.stdout, "bundle_file": bundle_file, "model_used": model_used})
                            
                            art = Artifact(
                                project_id=project_id, 
                                type="handover_bundle", 
                                description=f"Reproducible Project Bundle", 
                                content_json={"filename": bundle_file}, 
                                agent_id="CodeGenerator"
                            )
                            session.add(art)
                            session.commit()
                            hub.create_outbox_event("ARTIFACT_CREATED", {"type": "handover_bundle", "description": art.description, "content_json": art.content_json})
                            break 

                        # 2. Handle Errors & Contract Violations
                        if res.error:
                            error_msg = res.error.get("evalue", "Unknown error")
                        else:
                            error_msg = await hub.validate_contract(
                                task_obj, res.stdout, executor.authoritative_state,
                                semantic_insights=res.semantic_insights,
                                validation_model=task_obj.assigned_to,
                            )

                        # --- HALLUCINATION GUARD ---
                        hallucination_tokens = [
                            "no files provided", "no data available", "simulating data", 
                            "mock data", "dummy data", "environment is empty",
                            "no file available to process"
                        ]
                        if not error_msg and any(token in res.stdout.lower() for token in hallucination_tokens):
                            error_msg = "Task Failed: Agent detected missing environment files and attempted to simulate/skip instead of erroring."

                        if error_msg:
                            if hub.escalate_task(task_id, error_msg, hierarchy):
                                session.commit()
                                continue 
                            else:
                                hub.fail_task(task_id, error_msg, result={"stdout": res.stdout, "stderr": res.stderr, "code": res.code})
                                task_span.end(output={"error": error_msg})
                                session.commit()
                                break
                        else:
                            # 3. Handle Success
                            # ORCHESTRATOR-SIDE INTROSPECTION (No Hallucinations)
                            new_files_after = _get_recursive_files(workspace_dir)
                            new_names = set([f["name"] for f in new_files_after]) - set([f["name"] for f in current_files_meta])

                            artifact_summaries = []
                            for nf in new_names:
                                fpath = os.path.join(workspace_dir, nf)
                                if os.path.exists(fpath):
                                    artifact_summaries.append(summarize_artifact(fpath))

                            orchestrator_summary = "; ".join(artifact_summaries) if artifact_summaries else "Task completed successfully with no new files."

                            # --- METRICS GUARANTEE ---
                            required_metrics = (task_obj.postcondition_json or {}).get("required_metrics", [])
                            if required_metrics:
                                found_metrics = await _probe_kernel_for_metrics(
                                    executor.sandbox, project_id, str(project_id), required_metrics
                                )
                                missing = [m for m in required_metrics if m not in found_metrics]

                                if found_metrics:
                                    _merge_metrics_json(workspace_dir, found_metrics)
                                    metric_strs = ", ".join(
                                        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in found_metrics.items()
                                    )
                                    orchestrator_summary += f" | Metrics captured: {metric_strs}"
                                    print(f"  [Metrics] Captured: {metric_strs}", flush=True)

                                if missing:
                                    missing_str = ", ".join(missing)
                                    # Try escalation first (only on first attempt)
                                    if task_obj.escalation_count == 0 and hub.escalate_task(task_id, f"Required metrics not found in kernel: {missing_str}", hierarchy):
                                        print(f"  [Metrics] Escalating task — missing: {missing_str}", flush=True)
                                        session.commit()
                                        continue
                                    # Can't escalate — fail the task so execution feedback triggers a replan.
                                    # A silent warning here would let a task with no actual output pass as success.
                                    error_msg = f"Required metrics not produced: {missing_str}. Code must bind these as top-level scalar variables."
                                    print(f"  [Metrics] ❌ Failing task — metrics not produced: {missing_str}", flush=True)
                                    hub.fail_task(task_id, error_msg, result={"stdout": res.stdout, "code": res.code, "orchestrator_summary": orchestrator_summary})
                                    task_span.end(output={"error": error_msg})
                                    session.commit()
                                    break

                            hub.complete_task(task_id, {
                                "stdout": res.stdout,
                                "model_used": model_used,
                                "code": res.code,
                                "orchestrator_summary": orchestrator_summary # Persist ground truth
                            })
                            files_after = new_files_after

                            project = session.get(Project, project_id)
                            if project:
                                project.last_state_json = executor.authoritative_state
                                session.add(project)
                                session.commit()

                            new_files_names = set([f["name"] for f in files_after]) - set([f["name"] for f in current_files_meta])
                            
                            has_explicit_plots = any(nf.lower().endswith((".json", ".html")) and not nf.endswith(".meta.json") and nf != "final_dashboard.html" for nf in new_files_names)

                            if not has_explicit_plots:
                                for i, plot_b64 in enumerate(res.plots):
                                    art = Artifact(project_id=project_id, type="plot", description=f"In-memory plot {i+1}", content_json={"image_base64": plot_b64}, agent_id="CodeGenerator")
                                    session.add(art)
                                    session.commit()
                                    hub.create_outbox_event("ARTIFACT_CREATED", {"type": "plot", "description": art.description, "content_json": art.content_json})

                            for nf in new_files_names:
                                if nf == "final_dashboard.html": continue 
                                full_path = os.path.join(workspace_dir, nf)
                                if nf.lower().endswith(".png"):
                                    try:
                                        with open(full_path, "rb") as img_file:
                                            img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                                        art = Artifact(project_id=project_id, type="plot", description=f"Workspace artifact: {nf}", content_json={"image_base64": img_b64}, agent_id="CodeGenerator")
                                        session.add(art)
                                        session.commit()
                                        hub.create_outbox_event("ARTIFACT_CREATED", {"type": "plot", "description": art.description, "content_json": art.content_json})
                                    except Exception: pass
                                elif nf.lower().endswith(".json") and not nf.endswith(".meta.json"):
                                    try:
                                        # Deterministic Server-Side Hardening (Purge Binary Data)
                                        from gads.core.introspection import harden_json_artifact
                                        harden_json_artifact(full_path)
                                        
                                        art = Artifact(project_id=project_id, type="json_plot", description=f"Interactive: {nf}", content_json={"filename": nf}, agent_id="CodeGenerator")
                                        session.add(art)
                                        session.commit()
                                        hub.create_outbox_event("ARTIFACT_CREATED", {"type": "json_plot", "description": art.description, "content_json": art.content_json, "project_id": str(project_id)})
                                    except Exception: pass

                            current_files_meta = files_after
                            task_span.end(output={"stdout": res.stdout, "model_used": model_used, "code": res.code})
                            session.commit()
                            break

                # HARD ABORT MECHANISM (Per Task)
                with Session(engine) as session:
                    task_obj = session.get(Task, task_id)
                    if task_obj and task_obj.status == "failed":
                        print(f"    [Workflow] 🛑 ABORT ATTEMPT: Task failed.", flush=True)
                        break 

            # Collect execution failures to feed back into the next replan attempt
            with Session(engine) as session:
                current_attempt_tasks = session.exec(
                    select(Task).where(Task.id.in_(task_ids))
                ).all()
                failed_tasks = [t for t in current_attempt_tasks if t.status == "failed"]
                pending_tasks = [t for t in current_attempt_tasks if t.status == "pending"]

                if failed_tasks or pending_tasks:
                    lines = []
                    for t in failed_tasks:
                        err = (t.error or "Unknown error")[:200]
                        lines.append(f"- FAILED: '{t.description[:80]}' → {err}")
                    for t in pending_tasks:
                        lines.append(f"- SKIPPED (never ran due to prior failure): '{t.description[:80]}'")

                    execution_feedback = (
                        f"EXECUTION FAILURES FROM ATTEMPT {workflow_attempt}:\n" +
                        "\n".join(lines) + "\n\n"
                        "CRITICAL: Revise your plan to address these failures. "
                        "Ensure each task's postcondition only requires columns that task's own code explicitly produces. "
                        "Do NOT add a column to a task's postcondition if that column is built by a later task."
                    )
                    critique_feedback = execution_feedback + ("\n\n" + critique_feedback if critique_feedback else "")

            # 4b. DETERMINISTIC MODEL SAVE (if spec requested it)
            if spec_hints.get("save_model") and not failed_tasks and not pending_tasks:
                print(f"  [Workflow] save_model=true — persisting model binary to model.joblib", flush=True)
                _save_code = textwrap.dedent("""
                    import joblib as _jl, os as _os
                    _fitted = {k: v for k, v in globals().items()
                               if not k.startswith('_')
                               and hasattr(v, 'fit') and hasattr(v, 'predict')
                               and hasattr(v, 'classes_')}
                    if not _fitted:
                        raise RuntimeError("save_model: no fitted classifier found in kernel namespace")
                    _mname = sorted(_fitted.keys())[0]
                    _jl.dump(_fitted[_mname], 'model.joblib')
                    print(f"Saved model '{_mname}' → model.joblib ({_os.path.getsize('model.joblib')} bytes)")
                """).strip()
                try:
                    _save_res = await asyncio.wait_for(
                        executor.sandbox.execute(_save_code, project_id=project_id, session_id=str(project_id)),
                        timeout=30.0
                    )
                    if _save_res.error:
                        print(f"  [Workflow] ⚠️  Model save failed: {_save_res.error}", flush=True)
                    else:
                        print(f"  [Workflow] ✅ Model saved: {_save_res.stdout.strip()}", flush=True)
                except Exception as _save_exc:
                    print(f"  [Workflow] ⚠️  Model save exception: {_save_exc}", flush=True)

            # 4c. SEMANTIC COMPLETENESS VERIFIER
            # Catches two gaps PlanCritique structurally cannot:
            # (1) required_insights is a soft-fail — tasks complete without emitting interpretive work
            # (2) bypassed/handover tasks never executed but don't fail the workflow
            # Only fires when execution succeeded and a replan is still possible.
            if not (failed_tasks or pending_tasks) and workflow_attempt < MAX_WORKFLOW_ATTEMPTS:
                cv_model_fallback = ["local_model"] if get_local_only() else ["gemini-3-flash-preview"]
                cv_model = hierarchy.get("T2", {}).get("models", cv_model_fallback)[0]

                with Session(engine) as session:
                    cv_task = Task(
                        project_id=project_id,
                        instruction_id=instruction_id,
                        description=f"Completeness Auditor ({cv_model}) is verifying analytical coverage...",
                        assigned_to="CompletenessVerifier",
                        status="running",
                        heartbeat=datetime.now()
                    )
                    session.add(cv_task)
                    session.commit()
                    session.refresh(cv_task)
                    cv_task_id = cv_task.id

                    exec_tasks = session.exec(select(Task).where(Task.id.in_(task_ids))).all()
                    completed_summaries = [
                        f"[{t.description}]: "
                        f"{(t.result_json or {}).get('orchestrator_summary', 'completed')}"
                        for t in exec_tasks if t.status == "completed"
                    ]

                # Load metrics.json — gives the verifier ground-truth scalar evidence
                metrics_data = None
                metrics_path = os.path.join(workspace_dir, "metrics.json")
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path) as mf:
                            metrics_data = json.load(mf)
                    except Exception:
                        pass

                cv_span = trace.span(
                    name=f"Completeness Verification (Attempt {workflow_attempt})",
                    metadata={"task_id": str(cv_task_id)}
                )
                trace_context.get().update({
                    "agent_name": "CompletenessVerifier",
                    "task_id": str(cv_task_id),
                    "parent_observation_id": cv_span.id
                })

                try:
                    cv_agent = CompletenessVerifierAgent(model=cv_model)
                    cv_res = await asyncio.wait_for(
                        cv_agent.run(CompletenessVerifierInput(
                            objective=formalized_objective,  # matches what was planned against
                            completed_task_summaries=completed_summaries,
                            produced_artifact_names=[f["name"] for f in current_files_meta],
                            metrics_json=metrics_data
                        )),
                        timeout=120.0
                    )
                    cv_out = cv_res.content
                    cv_span.end(output=cv_out.model_dump())

                    with Session(engine) as session:
                        ct = session.get(Task, cv_task_id)
                        if ct:
                            ct.status = "completed"
                            ct.result_json = {
                                "stdout": (
                                    f"**Completeness: {'✅ Complete' if cv_out.is_complete else '❌ Gaps Found'}**\n"
                                    f"Verdict: {cv_out.verdict}"
                                    + (
                                        "\n\n**Missing analyses:**\n" +
                                        "\n".join(f"- {m}" for m in cv_out.missing_analyses)
                                        if cv_out.missing_analyses else ""
                                    )
                                ),
                                "model_used": cv_model
                            }
                            session.add(ct)
                            session.commit()

                    # Guard: only replan if there are concrete named gaps (not just is_complete=False)
                    if not cv_out.is_complete and cv_out.missing_analyses:
                        missing_str = "; ".join(cv_out.missing_analyses)
                        critique_feedback = (
                            f"COMPLETENESS GAPS (attempt {workflow_attempt}):\n{missing_str}\n\n"
                            f"Verifier verdict: {cv_out.verdict}\n\n"
                            "Revise the plan to include ALL missing analyses listed above."
                            + (f"\n\n{critique_feedback}" if critique_feedback else "")
                        )
                        print(
                            f"  [CompletenessVerifier] ❌ Gaps — triggering replan: {missing_str[:100]}",
                            flush=True
                        )
                        continue  # → back to Planning with enriched critique_feedback
                    else:
                        print(f"  [CompletenessVerifier] ✅ Execution is analytically complete.", flush=True)

                except Exception as cv_exc:
                    # Fail-open: verifier failure must never block the workflow
                    print(
                        f"  [CompletenessVerifier] ⚠️ Failed ({cv_exc}). Proceeding to synthesis.",
                        flush=True
                    )
                    cv_span.end(output={"error": str(cv_exc)})
                    with Session(engine) as session:
                        ct = session.get(Task, cv_task_id)
                        if ct:
                            ct.status = "failed"
                            ct.error = str(cv_exc)
                            session.add(ct)
                            session.commit()
                    # Fail-open: fall through to synthesis

            # 4d. FINAL ARTIFACT HARDENING PASS
            # Re-harden all JSON files in the workspace. A task can overwrite a previously
            # hardened file (e.g. task 2 re-saves rating_distribution_chart.json), which
            # re-introduces bdata encoding. This pass runs once per attempt before synthesis.
            for _jf in _get_recursive_files(workspace_dir):
                if _jf["name"].endswith(".json") and not _jf["name"].endswith(".meta.json"):
                    try:
                        harden_json_artifact(os.path.join(workspace_dir, _jf["name"]))
                    except Exception:
                        pass

            # 5. SYNTHESIS & CRITIQUE LOOP
            synthesizer_fallback = ["local_model"] if get_local_only() else ["gemini-3-flash-preview"]
            synthesizer_model = hierarchy.get("T2", {}).get("models", synthesizer_fallback)[0]
            
            critique_fallback = ["local_model"] if get_local_only() else ["gemini-3-flash-preview"]
            critique_model = hierarchy.get("T2", {}).get("models", hierarchy.get("T3", {}).get("models", critique_fallback))[0]

            while True:
                with Session(engine) as session:
                    synth_task = Task(
                        project_id=project_id, 
                        instruction_id=instruction_id,
                        description=f"Lead Data Scientist ({synthesizer_model}) is synthesizing results (Attempt {workflow_attempt})...", 
                        assigned_to="Synthesizer", 
                        status="running",
                        heartbeat=datetime.now()
                    )
                    session.add(synth_task)
                    session.commit()
                    session.refresh(synth_task)

                    span = trace.span(name=f"Synthesis Attempt {workflow_attempt}")
                    trace_context.get().update({"agent_name": "Synthesizer", "task_id": str(synth_task.id), "parent_observation_id": span.id})

                    all_tasks = session.exec(select(Task).where(Task.project_id == project_id)).all()
                    task_log_parts = []
                    for t in all_tasks:
                         status_str = t.status.upper()
                         summary = (t.result_json or {}).get("orchestrator_summary", "No summary available.")
                         task_log_parts.append(f"- Task: {t.description}\n  Status: {status_str}\n  Result: {summary}")
                    
                    artifacts = session.exec(select(Artifact).where(Artifact.project_id == project_id)).all()
                    artifact_evidence = []
                    for a in artifacts:
                        fname = a.content_json.get("filename")
                        fpath = os.path.join(workspace_dir, fname) if fname else None
                        # Use the actual summarized truth from the file
                        summary = summarize_artifact(fpath) if fpath else f"Artifact: {a.description}"
                        artifact_evidence.append(f"### ARTIFACT: {a.description}\nSource: {fname or 'In-Memory'}\nContent Summary: {summary}")
                    
                    context = "### TASK EXECUTION LOG\n" + "\n\n".join(task_log_parts) + \
                              "\n\n### GENERATED ARTIFACTS (THE BLACKBOARD)\n" + \
                              ( "\n\n".join(artifact_evidence) if artifact_evidence else "NO ARTIFACTS GENERATED.")

                    try:
                        synthesizer = SynthesizerAgent(model=synthesizer_model)
                        synth_res = await synthesizer.run(SynthesizerInput(
                            objective=objective, 
                            context_artifacts=context,
                            existing_narrative=None,
                            existing_takeaways=None
                        ))
                        final_synth = synth_res.content
                        span.end(output=final_synth.model_dump())

                        synth_task.status = "completed"
                        synth_task.result_json = {
                            "stdout": "Draft generated.", 
                            "narrative": final_synth.narrative, 
                            "takeaways": final_synth.key_takeaways,
                            "artifact_insights": [ins.dict() for ins in final_synth.artifact_insights],
                            "model_used": synthesizer_model
                        }
                        session.add(synth_task)
                        session.commit()
                        break 
                    except Exception as e:
                        next_model = get_next_model_dynamic(synthesizer_model, hierarchy)
                        if next_model:
                            synth_task.status = "failed"
                            session.add(synth_task)
                            session.commit()
                            synthesizer_model = next_model
                            continue
                        else: raise e

            # Generate distilled markdown preview for Critique
            with Session(engine) as session:
                artifacts = session.exec(select(Artifact).where(Artifact.project_id == project_id)).all()
                distiller_cards = []
                for a in artifacts:
                    fname = a.content_json.get("filename")
                    fpath = os.path.join(workspace_dir, fname) if fname else None
                    distiller_cards.append({
                        "description": a.description,
                        "type": a.type,
                        "caption": "Artifact preview.",
                        "metadata_summary": summarize_artifact(fpath) if fpath else ""
                    })

                dashboard_md = distill_dashboard_to_markdown(
                    narrative=final_synth.narrative,
                    takeaways=final_synth.key_takeaways,
                    cards=distiller_cards
                )

            # 6. CRITIQUE
            print(f"  [Workflow] Quality Assurance Critique (Attempt {workflow_attempt})...", flush=True)
            while True:
                with Session(engine) as session:
                    critique_task = Task(
                        project_id=project_id,
                        instruction_id=instruction_id,
                        description=f"QA Specialist ({critique_model}) is evaluating synthesis quality (Attempt {workflow_attempt})...",
                        assigned_to="Critique",
                        status="running",
                        heartbeat=datetime.now()
                    )
                    session.add(critique_task)
                    session.commit()
                    session.refresh(critique_task)

                    span = trace.span(name=f"Critique Attempt {workflow_attempt}")
                    trace_context.get().update({"agent_name": "Critique", "task_id": str(critique_task.id), "parent_observation_id": span.id})

                    try:
                        critique_agent = CritiqueAgent(model=critique_model) 
                        critique_res = await critique_agent.run(CritiqueInput(
                            objective=objective,
                            context_artifacts=context,
                            synthesis_narrative=final_synth.narrative,
                            synthesis_takeaways=final_synth.key_takeaways,
                            dashboard_html=dashboard_md # Now passing distilled Markdown!
                        ))
                        critique = critique_res.content
                        span.end(output=critique.model_dump())

                        critique_task.status = "completed"
                        critique_task.result_json = {"stdout": f"Approved: {critique.is_approved}", "model_used": critique_model}
                        session.add(critique_task)
                        session.commit()
                        break 
                    except Exception as e:
                        next_model = get_next_model_dynamic(critique_model, hierarchy)
                        if next_model:
                            critique_task.status = "failed"
                            session.add(critique_task)
                            session.commit()
                            critique_model = next_model
                            continue
                        else: raise e

            if critique.is_approved:
                redundant_plots = critique.redundant_artifacts
                workflow_succeeded = True
                break
            else:
                critique_feedback = critique.critique_feedback

        # Clean up pending tasks left over when MAX_WORKFLOW_ATTEMPTS is exhausted.
        # Without this, those tasks stay "pending" forever and block monitoring loops.
        if not workflow_succeeded and task_ids:
            with Session(engine) as session:
                leftover = session.exec(
                    select(Task).where(Task.id.in_(task_ids), Task.status == "pending")
                ).all()
                for _t in leftover:
                    _t.status = "failed"
                    _t.error = "Workflow exhausted max planning attempts; task never ran."
                    session.add(_t)
                if leftover:
                    print(f"  [Workflow] Marked {len(leftover)} permanently-pending task(s) as failed after max attempts.", flush=True)
                session.commit()

        # 7. FINAL REPORTING
        with Session(engine) as session:
            reporting_task = Task(
                project_id=project_id, instruction_id=instruction_id,
                description="Publishing final dashboard and research reports...",
                assigned_to="System", status="running", heartbeat=datetime.now()
            )
            session.add(reporting_task)
            session.commit()

            artifacts = session.exec(select(Artifact).where(Artifact.project_id == project_id)).all()
            filtered_artifacts = [a for a in artifacts if not any(r.lower() in a.description.lower() for r in (redundant_plots or []))]

            create_master_reports(
                project_id=project_id, workspace_dir=workspace_dir,
                narrative=final_synth.narrative if final_synth else "Workflow halted.",
                takeaways=final_synth.key_takeaways if final_synth else ["No takeaways."],
                artifacts=filtered_artifacts,
                artifact_insights=final_synth.artifact_insights if final_synth else []
            )

            # Export code bundle: .py script and .ipynb notebook
            try:
                py_path = export_python_script(project_id, workspace_dir)
                nb_path = export_notebook(project_id, workspace_dir)
                print(f"  [Reporting] Exported code bundle: {os.path.basename(py_path)}, {os.path.basename(nb_path)}", flush=True)
            except Exception as e:
                print(f"  [Reporting] Warning: Code bundle export failed: {e}", flush=True)

            # Copy applied recipe into workspace for reproducibility
            try:
                applied_recipe_id = (knowledge_report.recipe_id if knowledge_report else None) or spec_hints.get("recipe_id")
                if applied_recipe_id:
                    recipe_src = registry.get_recipe_filepath(applied_recipe_id)
                    dest = copy_applied_recipe(recipe_src, workspace_dir)
                    if dest:
                        print(f"  [Reporting] Copied applied recipe: {os.path.basename(dest)}", flush=True)
            except Exception as e:
                print(f"  [Reporting] Warning: Recipe copy failed: {e}", flush=True)

            proj = session.get(Project, project_id)
            if proj:
                proj.narrative = final_synth.narrative if final_synth else "Workflow halted."
                proj.takeaways = final_synth.key_takeaways if final_synth else ["No takeaways."]
            
            rtask = session.get(Task, reporting_task.id)
            if rtask: rtask.status = "completed"
            session.add(proj)
            session.commit()
            hub = ExecutionHub(session)
            hub.create_outbox_event("WORKFLOW_FINAL_RESULT", {"project_id": str(project_id)})
            session.commit()

    except Exception as e:
        print(f"❌ FATAL ERROR: {traceback.format_exc()}")
        with Session(engine) as session:
            _mark_workflow_failed(session, project_id, f"Fatal system error: {str(e)}")
    finally:
        from gads.core.llm import trace_context
        trace_context.reset(ctx_token)
        langfuse_client.flush()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, last_seq: int = 0):
    await bus.connect(websocket)
    if last_seq > 0: await bus.replay_events(websocket, last_seq)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: bus.disconnect(websocket)

@app.get("/specs")
def list_specs():
    specs_dir = Path("specs").resolve()
    if not specs_dir.exists():
        return []
    files = [f.name for f in specs_dir.iterdir() if f.is_file() and f.suffix == ".md"]
    return sorted(files)

@app.get("/specs/{filename}")
def get_spec_content(filename: str):
    specs_dir = Path("specs").resolve()
    target = (specs_dir / filename).resolve()
    if not target.is_relative_to(specs_dir) or target.suffix != ".md":
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Spec file not found")
    try:
        return {"content": target.read_text(encoding='utf-8')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read spec: {e}")

class SpecContent(BaseModel):
    content: str

@app.post("/specs/{filename}")
def save_spec_content(filename: str, req: SpecContent):
    specs_dir = Path("specs").resolve()
    target = (specs_dir / filename).resolve()
    if not target.is_relative_to(specs_dir) or target.suffix != ".md":
        raise HTTPException(status_code=400, detail="Invalid path")
    try:
        target.write_text(req.content, encoding='utf-8')
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save spec: {e}")

class SpecLaunchRequest(BaseModel):
    filename: str
    launch_workflow: bool = True
    fast_mode: bool = False

@app.post("/projects/from-spec", response_model=ProjectResponse)
async def launch_from_spec(req: SpecLaunchRequest, background_tasks: BackgroundTasks):
    specs_dir = Path("specs").resolve()
    target = (specs_dir / req.filename).resolve()
    if not target.is_relative_to(specs_dir) or target.suffix != ".md":
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Spec file not found")
    
    datasets_root = Path(os.getenv("GADS_DATASETS_ROOT", "/home/joergf/datasets")).resolve()

    try:
        content = target.read_text(encoding='utf-8')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read spec: {e}")

    # Parse YAML frontmatter
    yaml_data = {}
    objective = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                yaml_data = yaml.safe_load(parts[1]) or {}
                objective = parts[2].strip()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid YAML frontmatter: {e}")

    try:
        meta = ProjectSpecMetadata(**yaml_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Schema validation failed: {e}")

    # Pre-flight validation
    # 1. Datasets
    for ds in meta.datasets:
        ds_path = (datasets_root / ds).resolve()
        if not ds_path.is_relative_to(datasets_root) or not ds_path.exists():
            raise HTTPException(status_code=400, detail=f"Dataset not found or invalid: {ds}")
            
    # 2. Recipes
    for recipe in meta.recipes:
        try:
            registry.get_raw_recipe(recipe)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Recipe not found: {recipe}")

    # Transactional Execution
    with Session(engine) as session:
        project_name = meta.name or f"Project {datetime.now().strftime('%m-%d %H:%M')} (from spec)"
        project = Project(name=project_name, objective=objective, last_state_json={"fast_mode": req.fast_mode})
        session.add(project)
        session.flush() # Get ID without fully committing yet
        
        instruction_content = objective
        if meta.recipes:
            instruction_content += f"\\n\\n--- PREFERRED RECIPES ---\\nPlease prioritize the following recipes for this objective: {', '.join(meta.recipes)}"
            
        instruction = Instruction(project_id=project.id, content=instruction_content)
        session.add(instruction)
        
        workspace_dir = f"{WORKSPACE_ROOT}/{project.id}"
        
        try:
            os.makedirs(workspace_dir, exist_ok=True)
            for ds in meta.datasets:
                ds_path = (datasets_root / ds).resolve()
                _mount_external_dataset(workspace_dir, str(ds_path))
            
            # If fast_mode is enabled, inject sample_rows: 50000 into frontmatter
            spec_content_to_write = content
            if req.fast_mode:
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        try:
                            frontmatter_data = yaml.safe_load(parts[1]) or {}
                            frontmatter_data["sample_rows"] = 50000
                            new_yaml_str = yaml.dump(frontmatter_data, default_flow_style=False).strip()
                            spec_content_to_write = f"---\n{new_yaml_str}\n---\n{parts[2]}"
                        except Exception as e:
                            print(f"Error updating frontmatter for fast_mode: {e}")
                else:
                    spec_content_to_write = f"---\nsample_rows: 50000\n---\n{content}"

            # Write spec to workspace so the workflow's SpecDrafter stage can skip drafting
            spec_dest = Path(workspace_dir) / "workflow_spec.md"
            spec_dest.write_text(spec_content_to_write, encoding="utf-8")
        except Exception as e:
            # Rollback DB and filesystem
            session.rollback()
            if os.path.exists(workspace_dir):
                import shutil
                shutil.rmtree(workspace_dir)
            raise HTTPException(status_code=500, detail=f"Failed to setup workspace: {e}")

        session.commit()
        session.refresh(project)
        
        if req.launch_workflow:
            if project.id in ACTIVE_WORKFLOWS: raise HTTPException(status_code=400, detail="Workflow already in progress")
            ACTIVE_WORKFLOWS.add(project.id)
            background_tasks.add_task(run_agent_workflow_wrapper, project.id, objective, instruction.id)
        
        current_files = _get_recursive_files(workspace_dir)
        instructions = session.exec(select(Instruction).where(Instruction.project_id == project.id).order_by(Instruction.created_at.asc())).all()
        
        return ProjectResponse(
            project=ProjectRead.from_orm(project), 
            files=[f["name"] for f in current_files],
            instructions=[InstructionRead.from_orm(i) for i in instructions]
        )


@app.get("/projects/{project_id}", response_model=Dict[str, Any])
async def get_project_details(project_id: uuid.UUID):
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project: raise HTTPException(status_code=404, detail="Project not found")
        tasks = session.exec(select(Task).where(Task.project_id == project_id)).all()
        artifacts = session.exec(select(Artifact).where(Artifact.project_id == project_id)).all()
        instructions = session.exec(select(Instruction).where(Instruction.project_id == project_id).order_by(Instruction.created_at.asc())).all()

        p_data = ProjectRead.from_orm(project)
        if instructions:
            p_data.first_instruction = instructions[0].content
        
        # Check if running
        threshold = datetime.now() - timedelta(minutes=2)
        running_tasks = [t for t in tasks if t.status == "running" and t.heartbeat and t.heartbeat > threshold]
        p_data.is_running = len(running_tasks) > 0
        p_data.has_dashboard = os.path.exists(f"{WORKSPACE_ROOT}/{project_id}/final_dashboard.html")
        p_data.has_report = os.path.exists(f"{WORKSPACE_ROOT}/{project_id}/research_report.md")

        return {
            "project": p_data,
            "tasks": tasks, 
            "artifacts": artifacts,
            "instructions": instructions
        }

@app.get("/projects", response_model=List[ProjectRead])
async def list_projects():
    with Session(engine) as session:
        projects = session.exec(select(Project).order_by(Project.created_at.desc())).all()
        results = []
        for p in projects:
            p_data = ProjectRead.from_orm(p)
            p_data.has_dashboard = os.path.exists(f"{WORKSPACE_ROOT}/{p.id}/final_dashboard.html")
            p_data.has_report = os.path.exists(f"{WORKSPACE_ROOT}/{p.id}/research_report.md")

            # Fetch first instruction
            first_instr = session.exec(select(Instruction).where(Instruction.project_id == p.id).order_by(Instruction.created_at.asc())).first()
            if first_instr:
                p_data.first_instruction = first_instr.content

            # Check for failed tasks
            failed_tasks = session.exec(select(Task).where(Task.project_id == p.id, Task.status == "failed")).all()
            p_data.has_failed_tasks = len(failed_tasks) > 0

            # Check if running
            threshold = datetime.now() - timedelta(minutes=2)
            active_running = session.exec(select(Task).where(
                Task.project_id == p.id, 
                Task.status == "running", 
                Task.heartbeat > threshold
            )).all()
            p_data.is_running = len(active_running) > 0

            results.append(p_data)
        return results
@app.post("/projects", response_model=ProjectResponse)
async def create_project(req: ProjectCreateRequest, background_tasks: BackgroundTasks):
    with Session(engine) as session:
        if req.existing_project_id:
            project = session.get(Project, uuid.UUID(req.existing_project_id))
            if not project: raise HTTPException(status_code=404, detail="Project not found")

            # Reset cancelled status if resuming
            if project.narrative and "[CANCELLED]" in project.narrative:
                project.narrative = None
            
            # Save fast_mode state
            state = project.last_state_json or {}
            state["fast_mode"] = req.fast_mode
            project.last_state_json = state
            
            # Update objective if provided (for shell projects)
            if req.objective:
                project.objective = req.objective
            session.add(project)
            session.commit()
            session.refresh(project)
        else:
            project = Project(name=req.name, objective=req.objective, last_state_json={"fast_mode": req.fast_mode})
            session.add(project)
            session.commit()
            session.refresh(project)

        # 0. Duplicate Prevention
        # Check if there are already active tasks for this project
        active_tasks = session.exec(select(Task).where(Task.project_id == project.id, Task.status.in_(["pending", "running"]))).all()
        if active_tasks:
            print(f"    [Workflow] Duplicate thread blocked for project {project.id} (found {len(active_tasks)} active tasks).")
            # Return current state without spawning a new background task
            current_files = _get_recursive_files(f"{WORKSPACE_ROOT}/{project.id}")
            instructions = session.exec(select(Instruction).where(Instruction.project_id == project.id).order_by(Instruction.created_at.asc())).all()
            return ProjectResponse(
                project=ProjectRead.from_orm(project), 
                files=[f["name"] for f in current_files],
                instructions=[InstructionRead.from_orm(i) for i in instructions]
            )

        # Create Instruction record
        instr_id = None
        if req.objective.strip():
            instruction = Instruction(project_id=project.id, content=req.objective)
            session.add(instruction)
            session.commit()
            session.refresh(instruction)
            instr_id = instruction.id

        workspace_dir = f"{WORKSPACE_ROOT}/{project.id}"
        os.makedirs(workspace_dir, exist_ok=True)
        current_files = _get_recursive_files(workspace_dir)

        if req.objective.strip():
            ACTIVE_WORKFLOWS.add(project.id)
            background_tasks.add_task(run_agent_workflow_wrapper, project.id, req.objective, instr_id)

        # Fetch instructions to return
        instructions = session.exec(select(Instruction).where(Instruction.project_id == project.id).order_by(Instruction.created_at.asc())).all()

        return ProjectResponse(
            project=ProjectRead.from_orm(project), 
            files=[f["name"] for f in current_files],
            instructions=[InstructionRead.from_orm(i) for i in instructions]
        )
def _mount_external_dataset(workspace_dir: str, host_path: str):
    """Copies an external dataset into the workspace.

    Previously used symlinks, but symlinks allow task code to overwrite the
    source file (writes go through the link to the original). Copying protects
    the source dataset from accidental overwrites.
    """
    import shutil
    if not os.path.lexists(host_path):
        raise FileNotFoundError(f"Path not found: {host_path}")

    os.makedirs(workspace_dir, exist_ok=True)
    filename = os.path.basename(host_path)
    target_path = os.path.join(workspace_dir, filename)

    try:
        if os.path.lexists(target_path):
            if os.path.islink(target_path): os.unlink(target_path)
            else: os.remove(target_path)

        shutil.copy2(host_path, target_path)
    except Exception as e:
        raise Exception(f"Failed to copy dataset: {e}")

async def _probe_file_schema(executor: ExecutionManager, project_id: uuid.UUID, filename: str) -> Optional[Dict[str, Any]]:
    """Rich, time-capped schema + distribution profiling for CSV, Parquet, and text files."""
    print(f"    [DataAnalyzer] Profiling {filename}...", flush=True)

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext in ("csv", "parquet"):
        code = f"""
import pandas as pd, numpy as np, json
try:
    if "{filename}".endswith(".csv"):
        df = pd.read_csv("{filename}", nrows=5000)
    else:
        import pyarrow.parquet as pq
        df = pq.read_table("{filename}").to_pandas().head(5000)

    schema = {{str(k): str(v) for k, v in df.dtypes.items()}}
    row_count = len(df)

    null_rates = {{k: round(float(v), 4)
                   for k, v in df.isnull().mean().items() if v > 0}}

    cardinality = {{}}
    imbalance_stats = {{}}
    for col in df.columns:
        n_uniq = df[col].nunique()
        if n_uniq <= 20 or (row_count > 0 and n_uniq / row_count < 0.005):
            vc = df[col].value_counts(normalize=True).head(8)
            cardinality[col] = {{str(k): round(float(v), 4) for k, v in vc.items()}}
            
            vc_counts = df[col].value_counts()
            if len(vc_counts) >= 2:
                min_count = float(vc_counts.min())
                max_count = float(vc_counts.max())
                ratio = round(max_count / min_count, 4) if min_count > 0 else 1.0
                minority_frac = round(min_count / max(row_count, 1), 4)
                imbalance_stats[col] = {{
                    "imbalance_ratio": ratio,
                    "minority_class_frac": minority_frac
                }}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_stats = {{}}
    for col in numeric_cols[:8]:
        s = df[col].dropna()
        if len(s):
            numeric_stats[col] = {{
                "min": float(s.min()), "max": float(s.max()),
                "mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4)
            }}

    def _ser(v):
        if isinstance(v, float) and v != v: return None
        if hasattr(v, "item"): return v.item()
        return v if isinstance(v, (str, int, float, bool, type(None))) else str(v)

    sample = [{{k: _ser(row[k]) for k in row}} for row in df.head(3).to_dict(orient="records")]

    print("PROFILE_JSON:" + json.dumps({{
        "schema": schema, "row_count": row_count,
        "null_rates": null_rates, "cardinality": cardinality,
        "imbalance_stats": imbalance_stats,
        "numeric_stats": numeric_stats, "sample": sample
    }}))
except Exception as e:
    print("PROFILE_JSON:" + json.dumps({{"error": str(e)}}))
""".strip()

    elif ext in ("xlsx", "xls"):
        code = f"""
import pandas as pd, json
try:
    xl = pd.ExcelFile("{filename}")
    sheets = xl.sheet_names
    profiles = {{}}
    for sheet in sheets[:3]:
        df = xl.parse(sheet, nrows=1000)
        profiles[sheet] = {{
            "schema": {{str(k): str(v) for k, v in df.dtypes.items()}},
            "rows_sampled": len(df),
            "sample": df.head(3).fillna("").to_dict(orient="records")
        }}
    print("PROFILE_JSON:" + json.dumps({{"type": "excel", "excel_sheets": sheets, "profiles": profiles}}))
except Exception as e:
    print("PROFILE_JSON:" + json.dumps({{"error": str(e)}}))
""".strip()

    elif ext == "json":
        code = f"""
import json, pandas as pd
try:
    with open("{filename}") as f:
        data = json.load(f)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        df = pd.DataFrame(data[:1000])
        schema = {{str(k): str(v) for k, v in df.dtypes.items()}}
        result = {{"type": "records", "schema": schema, "row_count": len(data), "sample": data[:3]}}
    elif isinstance(data, dict):
        result = {{"type": "object", "top_level_keys": list(data.keys())[:20],
                   "sample": {{k: str(v)[:100] for k, v in list(data.items())[:5]}}}}
    else:
        result = {{"type": "array", "length": len(data)}}
    print("PROFILE_JSON:" + json.dumps(result))
except Exception as e:
    print("PROFILE_JSON:" + json.dumps({{"error": str(e)}}))
""".strip()

    elif ext in ("txt", "md", "log"):
        code = f"""
import json
try:
    with open("{filename}", encoding="utf-8", errors="replace") as f:
        content = f.read(10000)
    result = {{"type": "text", "chars": len(content),
               "words": len(content.split()), "lines": content.count("\\n") + 1,
               "preview": content[:400]}}
    print("PROFILE_JSON:" + json.dumps(result))
except Exception as e:
    print("PROFILE_JSON:" + json.dumps({{"error": str(e)}}))
""".strip()

    else:
        return None

    try:
        res = await asyncio.wait_for(
            executor.sandbox.execute(
                code, project_id=project_id,
                session_id=f"probe_{project_id}",
                workspace_id=str(project_id)
            ),
            timeout=30.0
        )
        for line in reversed(res.stdout.strip().split("\n")):
            if line.startswith("PROFILE_JSON:"):
                data = json.loads(line[len("PROFILE_JSON:"):])
                if "error" not in data:
                    print(f"    [DataAnalyzer] Profiled {filename}", flush=True)
                    return data
                print(f"    [DataAnalyzer] Profile error for {filename}: {data['error']}", flush=True)
    except Exception as e:
        print(f"    [DataAnalyzer] Probe failed for {filename}: {e}", flush=True)
    return None

async def _background_probe_and_update(project_id: uuid.UUID, filename: str):
    """Background worker to probe schema and update project state."""
    executor = ExecutionManager()
    schema = await _probe_file_schema(executor, project_id, filename)
    if schema:
        with Session(engine) as session:
            project = session.get(Project, project_id)
            if project:
                state = project.last_state_json or {}
                schemas = state.get("__schemas__", {})
                schemas[filename] = schema
                state["__schemas__"] = schemas
                project.last_state_json = state
                session.add(project)
                
                # Update UI via outbox
                workspace_dir = f"{WORKSPACE_ROOT}/{project_id}"
                current_files = _get_recursive_files(workspace_dir)
                hub = ExecutionHub(session)
                hub.create_outbox_event("STATE_UPDATED", {
                    "files": current_files, 
                    "state": state
                })
                session.commit()
                print(f"    [Introspection] Persisted schema for {filename} to project state.", flush=True)

@app.post("/projects/{project_id}/register-external", response_model=ProjectResponse)
async def register_external_file(project_id: uuid.UUID, req: ExternalPathRequest, background_tasks: BackgroundTasks):
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project: raise HTTPException(status_code=404, detail="Project not found")
        
        workspace_dir = f"{WORKSPACE_ROOT}/{project_id}"
        try:
            _mount_external_dataset(workspace_dir, req.path)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Path not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        filename = os.path.basename(req.path)
        if filename.endswith(".csv") or filename.endswith(".parquet"):
            background_tasks.add_task(_background_probe_and_update, project_id, filename)

        current_files = _get_recursive_files(workspace_dir)
        hub = ExecutionHub(session)
        hub.create_outbox_event("STATE_UPDATED", {"files": current_files, "state": project.last_state_json or {}})
        session.commit()
        return ProjectResponse(project=ProjectRead.from_orm(project), files=[f["name"] for f in current_files])

@app.get("/projects/{project_id}/files/{file_path:path}")
async def download_file(project_id: uuid.UUID, file_path: str, download: bool = False):
    workspace_dir = f"{WORKSPACE_ROOT}/{project_id}"
    full_path = os.path.join(workspace_dir, file_path)
    if not os.path.abspath(full_path).startswith(os.path.abspath(workspace_dir)): raise HTTPException(status_code=403, detail="Denied")
    if not os.path.exists(full_path): raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(full_path, filename=os.path.basename(full_path)) if download else FileResponse(full_path)

def _mark_workflow_failed(session: Session, project_id: uuid.UUID, reason: str):
    """Idempotently halts a workflow and cancels all pending tasks."""
    project = session.get(Project, project_id)
    if project:
        project.narrative = f"[HALTED] {reason}"
        session.add(project)
    
    # Mark all pending/running tasks as failed
    tasks = session.exec(select(Task).where(Task.project_id == project_id, Task.status.in_(["pending", "running"]))).all()
    for t in tasks:
        t.status = "failed"
        t.error = reason
        session.add(t)
    
    session.commit()
    hub = ExecutionHub(session)
    hub.create_outbox_event("WORKFLOW_CANCELLED", {"project_id": str(project_id), "reason": reason})
    session.commit()

@app.get("/health")
def health(): return {"status": "ok"}
