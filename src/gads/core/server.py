import asyncio
import contextlib
import uuid
import traceback
import json
import textwrap
import base64
import os
import re
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uuid
from gads.core.bus import bus, dispatcher_loop
from gads.core.execution_hub import watchdog_loop, ExecutionHub
from gads.core.database import init_db, engine
from gads.core.models import Project, Task, Artifact, Instruction
from gads.agents.planner import DataSciencePlanner, PlannerInput, PlannerOutput, PlannerTask, ReconciliationReport, FileMetadata
from gads.agents.base import AgentResponse
from gads.agents.router import DataScienceRouter, RouterInput, RouterOutput
from gads.agents.plan_critique import PlanCritiqueAgent, PlanCritiqueInput, PlanCritiqueOutput
from gads.agents.spec_drafter import SpecDrafterAgent, SpecDraftInput
from gads.agents.workers.synthesizer import SynthesizerAgent, SynthesizerInput
from gads.agents.workers.critique import CritiqueAgent, CritiqueInput
from gads.agents.workers.completeness_verifier import CompletenessVerifierAgent, CompletenessVerifierInput
from gads.core.executor import ExecutionManager
from gads.tools.sandbox import SandboxClient
from gads.core.registry import (
    get_model_hierarchy, get_local_only, set_local_only, get_random_routing,
    set_random_routing, get_next_model_dynamic, get_routing_mode, set_routing_mode,
    get_pinned_model, resolve_stage_model, get_available_models, VALID_ROUTING_MODES,
    get_local_fallback, set_local_fallback, VALID_LOCAL_FALLBACKS
)
from gads.core.knowledge import KnowledgeRegistry
from gads.core.dial import compiled_plan_dial, drafted_plan_dial, append_ledger
from gads.core.reporting import create_master_reports
from gads.core.notebook_exporter import export_python_script, export_notebook, copy_applied_recipe
from gads.core.introspection import summarize_artifact, looks_like_plotly_figure
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

# Projects whose sandbox session is pinned for interactive follow-up work: their kernel holds
# rehydrated state and must survive another project's stale-session sweep (approach_docs/020).
# In-process only, like ACTIVE_WORKFLOWS — a backend restart clears it (and the kernels too).
PINNED_SESSIONS: set[uuid.UUID] = set()

async def run_agent_workflow_wrapper(project_id: uuid.UUID, objective: str, instruction_id: Optional[uuid.UUID] = None):
    """Wrapper to manage the ACTIVE_WORKFLOWS lock."""
    try:
        await run_agent_workflow(project_id, objective, instruction_id)
    finally:
        if project_id in ACTIVE_WORKFLOWS:
            ACTIVE_WORKFLOWS.remove(project_id)

app = FastAPI(title="GADS Core API")

# The Knowledge Studio SPA is served from its own origin (Vite dev server on :5173,
# or a static bundle). Allow local origins so it can call the Knowledge API directly.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# Source-dataset root, bind mounted READ-ONLY into the sandbox at the identical path.
# Datasets under this root are symlinked into workspaces rather than copied (see
# _mount_external_dataset); the read-only mount is what makes that safe.
DATASETS_ROOT = os.getenv("GADS_DATASETS_ROOT", "/home/joergf/datasets")

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
    # Legacy toggle (old clients): true ≡ mode "local", false ≡ mode "cloud".
    # Ignored when routing_mode is provided.
    local_only: Optional[bool] = None
    random_routing: bool = False
    # New-style routing: cloud | local | hybrid | cloud_pinned
    routing_mode: Optional[str] = None
    pinned_model: Optional[str] = None
    # Local retry-exhaustion fallback: none | native | cloud | native_then_cloud
    local_fallback: Optional[str] = None

class NotesRequest(BaseModel):
    notes: str = ""


class FollowUpRequest(BaseModel):
    objective: str                       # the user's instruction, used verbatim as the task
    rehydrate: bool = True               # restore prior kernel state before running
    # "auto"/"none" = no recipe guidance; "recipe_id#node_id" injects ONE node's intent.
    use_recipe: str = "auto"


class RehydrateRequest(BaseModel):
    # Only replay if these variables are missing; None = replay when the session is empty.
    required_vars: Optional[List[str]] = None
    force: bool = False          # replay even if the session already looks live
    timeout: float = 900.0       # replay re-runs the original computation; allow for fits


class FileUpload(BaseModel):
    name: str
    content_base64: str

class ProjectCreateRequest(BaseModel):
    name: str
    objective: str
    files: List[FileUpload] = []
    existing_project_id: Optional[str] = None
    fast_mode: bool = False
    disable_recipes: bool = False

class FilesUploadRequest(BaseModel):
    files: List[FileUpload]

class ExternalPathRequest(BaseModel):
    path: str

class RecipeContent(BaseModel):
    content: str

class PromptUpdate(BaseModel):
    content: str

class KnowledgeValidateRequest(BaseModel):
    type: str          # recipe | skill | native
    content: str
    filename: Optional[str] = None

class ProjectSpecMetadata(BaseModel):
    name: Optional[str] = None
    datasets: List[str] = Field(default_factory=list)
    recipes: List[str] = Field(default_factory=list)
    target_column: Optional[str] = None
    feature_columns: List[str] = Field(default_factory=list)
    filters: Optional[str] = None
    domain: Optional[str] = None
    save_model: bool = False
    # Reuse a finished project's outputs: its artifacts are linked into `upstream/`
    # (approach_docs/021 §6). The value is that project's UUID.
    artifacts_from: Optional[str] = None
    # Force the drafted-plan lane (no recipe pin, no Router match) — used by
    # delegation-dial D0/D1 specs so the rung doesn't depend on a launch flag.
    disable_recipes: bool = False

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

@app.post("/knowledge/validate")
def knowledge_validate(req: KnowledgeValidateRequest):
    """Dry-run deep validation of a knowledge item (recipe|skill|native) without
    writing it — powers inline errors/warnings in the Knowledge Studio
    (approach_docs/017 §3). Never mutates state."""
    from gads.core.knowledge_validation import validate
    result = validate(req.type, req.content, registry=registry, filename=req.filename)
    return {"valid": not result["errors"], **result}

@app.get("/knowledge/items")
def knowledge_items(type: Optional[str] = None):
    """Unified library listing (recipes + skills + native) with metadata + provenance
    (shipped | overlay | overridden) — one call for the studio library view."""
    return registry.list_items(type)

@app.post("/knowledge/{item_type}/{filename}/reset")
def knowledge_reset(item_type: str, filename: str):
    """Revert an overridden item to its shipped version by deleting the overlay copy."""
    if item_type not in ("recipes", "skills", "native"):
        raise HTTPException(status_code=400, detail="item_type must be recipes|skills|native")
    try:
        registry.reset_to_shipped(item_type, filename)
        return {"status": "reset", "provenance": "shipped"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/native", response_model=List[str])
def list_native_files():
    return registry.list_native_files()

@app.get("/native/{filename}")
def get_native_raw(filename: str):
    try:
        return {"content": registry.get_raw_native(filename)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/native/{filename}")
def save_native_raw(filename: str, req: RecipeContent):
    """Deep-validate (AST + hazard scan) and save a native module to the overlay. The
    executor does not yet load overlay native modules — write-only for now (017 §4)."""
    try:
        registry.save_raw_native(filename, req.content)
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Knowledge Studio insight endpoints (read-only; approach_docs/017 §2) ------ #
@app.get("/knowledge/graph")
def knowledge_graph():
    """recipe→skill (attached) + recipe→native (mechanized) dependency graph."""
    from gads.core import knowledge_insights as ki
    return ki.build_graph(registry)

@app.get("/knowledge/coverage")
def knowledge_coverage():
    """task_type × dial-rung matrix over the recipe library, plus orphan analysis."""
    from gads.core import knowledge_insights as ki
    return ki.build_coverage(registry)

@app.get("/knowledge/{item_type}/{item_id}/history")
def knowledge_history(item_type: str, item_id: str, limit: int = 50):
    """Git commit history for a shipped item's file (overlay is git-ignored)."""
    from gads.core import knowledge_insights as ki
    if registry.norm_type(item_type) is None:
        raise HTTPException(status_code=400, detail="item_type must be recipe(s)|skill(s)|native")
    try:
        return ki.item_history(registry, item_type, item_id, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/knowledge/{item_type}/{item_id}/diff")
def knowledge_diff(item_type: str, item_id: str,
                   from_ref: Optional[str] = None, to_ref: Optional[str] = None):
    """Diff a shipped item across commits (from_ref/to_ref) or vs working tree."""
    from gads.core import knowledge_insights as ki
    if registry.norm_type(item_type) is None:
        raise HTTPException(status_code=400, detail="item_type must be recipe(s)|skill(s)|native")
    try:
        return ki.item_diff(registry, item_type, item_id, from_ref=from_ref, to_ref=to_ref)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/knowledge/{item_type}/{item_id}/impact")
def knowledge_impact(item_type: str, item_id: str):
    """What references this id — specs that pin a recipe, recipes that attach a skill
    or mechanize a native module (rename/deprecate guard)."""
    from gads.core import knowledge_insights as ki
    if registry.norm_type(item_type) is None:
        raise HTTPException(status_code=400, detail="item_type must be recipe(s)|skill(s)|native")
    return ki.item_impact(registry, item_type, item_id)

@app.get("/knowledge/{item_type}/{item_id}/evidence")
def knowledge_evidence(item_type: str, item_id: str):
    """Per-engine benchmark pass rate from the dial ledger (recipes only)."""
    from gads.core import knowledge_insights as ki
    if registry.norm_type(item_type) != "recipes":
        raise HTTPException(status_code=400, detail="evidence is available for recipes only")
    return ki.item_evidence(registry, item_id)

@app.get("/knowledge/{item_type}/{item_id}")
def knowledge_item_detail(item_type: str, item_id: str):
    """Raw content + parsed frontmatter + provenance for one item (studio read view)."""
    if registry.norm_type(item_type) is None:
        raise HTTPException(status_code=400, detail="item_type must be recipe(s)|skill(s)|native")
    try:
        return registry.get_item_detail(item_type, item_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# --- Project taxonomy (approach_docs/018) -------------------------------------- #
@app.get("/taxonomy")
def taxonomy_vocab():
    """The controlled vocabularies (facets, schema, crosswalk) — Studio tag pickers."""
    from gads.core import taxonomy as tx
    return tx.load_vocab()

@app.post("/taxonomy/validate")
def taxonomy_validate(block: Dict[str, Any] = Body(...)):
    """Validate a spec's taxonomy block against the vocabulary. Never mutates state."""
    from gads.core import taxonomy as tx
    result = tx.validate_tags(block)
    return {"valid": not result["errors"], **result}

@app.get("/taxonomy/coverage")
def taxonomy_coverage():
    """Intent × task-family matrix over the tagged spec library, plus per-axis
    distributions and populated cells — the coverage map (018 §6)."""
    from gads.core import taxonomy as tx
    return tx.coverage()

@app.get("/taxonomy/specs")
def taxonomy_specs():
    """Per-spec taxonomy tags (untagged specs included so gaps are visible)."""
    from gads.core import taxonomy as tx
    return tx.spec_index()

@app.get("/taxonomy/recipes")
def taxonomy_recipes():
    """Recipe library projected onto the taxonomy — intent × task-family matrix +
    per-axis distributions, derived from each recipe's applies_when via the crosswalk.
    `unmapped` flags recipes whose declared task_type the crosswalk doesn't cover."""
    from gads.core import taxonomy as tx
    return tx.recipe_coverage(registry)

@app.get("/taxonomy/runs")
def taxonomy_runs(limit: int = 100):
    """Per-run taxonomy classification (every launched project that has been routed),
    newest first — the classification for any run, ad-hoc included (018)."""
    with Session(engine) as session:
        projects = session.exec(select(Project).order_by(Project.created_at.desc()).limit(limit)).all()
        out = []
        for p in projects:
            tax = (p.last_state_json or {}).get("taxonomy")
            if not tax:
                continue
            out.append({
                "project_id": str(p.id),
                "name": p.name,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "source": (p.last_state_json or {}).get("taxonomy_source"),
                "taxonomy": tax,
            })
        return out

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

@app.get("/skills/match")
def match_skills(query: str):
    """Debug: show how a task description would rank against the skill library
    with both matchers (keyword triggers + semantic embeddings)."""
    return {
        "query": query[:300],
        "matches": [
            {"skill": s.id, "source": src, "score": round(score, 4)}
            for s, src, score in registry.find_skills_combined(query)
        ],
    }

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

def _config_payload() -> Dict[str, Any]:
    return {
        "local_only": get_local_only(),
        "random_routing": get_random_routing(),
        "routing_mode": get_routing_mode(),
        "pinned_model": get_pinned_model(),
        "valid_routing_modes": list(VALID_ROUTING_MODES),
        "local_fallback": get_local_fallback(),
        "valid_local_fallbacks": list(VALID_LOCAL_FALLBACKS),
    }

@app.get("/config")
async def get_config():
    payload = _config_payload()
    # Best-effort model list for the UI's pinned-model picker.
    try:
        payload["available_models"] = [m for m in await get_available_models() if m != "local_model"]
    except Exception:
        payload["available_models"] = []
    return payload

@app.post("/config")
def update_config(req: ConfigUpdate):
    try:
        if req.routing_mode is not None:
            set_routing_mode(req.routing_mode, pinned_model=req.pinned_model)
        elif req.local_only is not None:
            set_local_only(req.local_only)
        if req.local_fallback is not None:
            set_local_fallback(req.local_fallback)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    set_random_routing(req.random_routing)
    return {"status": "success", **_config_payload()}

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

@app.get("/projects/{project_id}/kernel")
async def get_project_kernel(project_id: uuid.UUID):
    """Live namespace of a project's sandbox session (no side effects).

    `status` is 'live' when variables are present, 'cold' when the session is empty or
    unreachable — i.e. whether follow-up work can build on prior state as-is.
    """
    workspace_dir = f"{WORKSPACE_ROOT}/{project_id}"
    with Session(engine) as session:
        known = session.get(Project, project_id) is not None
    if not known and not os.path.isdir(workspace_dir):
        raise HTTPException(status_code=404, detail="Project not found")
    from gads.core.kernel_state import (snapshot_kernel, replayable_tasks,
                                        replay_code_from_workspace, meaningful_variables)
    variables = meaningful_variables(await snapshot_kernel(SandboxClient(), project_id))
    n_db = len(replayable_tasks(project_id))
    return {
        "project_id": str(project_id),
        "status": "live" if variables else "cold",
        "variables": variables or {},
        "variable_count": len(variables or {}),
        "replayable_tasks": n_db,
        # Replay can also be recovered from the workspace export when DB rows are gone.
        "workspace_replay_available": bool(replay_code_from_workspace(workspace_dir)) if not n_db else True,
        "in_database": known,
        "pinned": project_id in PINNED_SESSIONS,
    }


@app.post("/projects/{project_id}/rehydrate")
async def rehydrate_project(project_id: uuid.UUID, req: Optional[RehydrateRequest] = None):
    """Restore a completed project's kernel state so follow-up work can build on it.

    Idempotent: if the session already holds the state this is a no-op probe. Otherwise the
    project's COMPLETED task code is replayed into the session. Fail-soft — a failed replay
    returns status 'cold' with the error rather than raising, so callers can proceed against
    workspace files instead.

    Note the returned state is *replayed*, not restored: re-running the original code can
    rebuild subtly different objects where a step was not fully deterministic.
    """
    req = req or RehydrateRequest()
    workspace_dir = f"{WORKSPACE_ROOT}/{project_id}"
    with Session(engine) as session:
        known = session.get(Project, project_id) is not None
    # A workspace with no DB row is still rehydratable (archive recovery), so accept either.
    if not known and not os.path.isdir(workspace_dir):
        raise HTTPException(status_code=404, detail="Project not found")
    if project_id in ACTIVE_WORKFLOWS:
        raise HTTPException(status_code=409,
                            detail="A workflow is currently running for this project; "
                                   "rehydration would race its kernel.")
    from gads.core.kernel_state import ensure_kernel_state
    # Pin BEFORE replaying so a concurrent run's sweep cannot wipe the work in progress.
    PINNED_SESSIONS.add(project_id)
    result = await ensure_kernel_state(
        SandboxClient(), project_id,
        required_vars=req.required_vars, force=req.force, timeout=req.timeout,
        workspace_dir=workspace_dir,
    )
    if result["status"] in ("cold", "empty") and not result.get("variables"):
        PINNED_SESSIONS.discard(project_id)   # nothing worth protecting
    result["pinned"] = project_id in PINNED_SESSIONS
    return result


@app.post("/projects/{project_id}/followup")
async def create_followup(project_id: uuid.UUID, req: FollowUpRequest,
                          background_tasks: BackgroundTasks):
    """Run ONE user-directed instruction against a completed project's live kernel.

    The short lane (approach_docs/020): instruction → one task → rehydrate → Coder → execute
    → artifacts. Skips SpecDrafter/Router/Planner/PlanCritique/CompletenessVerifier/Critique —
    the user is the planner and the critic. Use `POST /projects` (with `existing_project_id`)
    when a full autonomous re-plan is actually wanted.

    Returns immediately with the task id; stream progress from `GET /tasks/{id}/stream`.
    """
    objective = (req.objective or "").strip()
    if not objective:
        raise HTTPException(status_code=400, detail="objective is required")

    workspace_dir = f"{WORKSPACE_ROOT}/{project_id}"
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project and not os.path.isdir(workspace_dir):
            raise HTTPException(status_code=404, detail="Project not found")
        if project_id in ACTIVE_WORKFLOWS:
            raise HTTPException(status_code=409,
                                detail="This project already has work in flight.")
        busy = session.exec(select(Task).where(Task.project_id == project_id,
                                               Task.status.in_(["pending", "running"]))).first()
        if busy:
            raise HTTPException(status_code=409,
                                detail="This project has a running task; wait for it to finish.")

        # A workspace recovered from disk (no DB row) still needs a Project to hang the
        # instruction/task off — recreate a minimal shell rather than refusing the request.
        if not project:
            project = Project(id=project_id, name=f"Recovered {str(project_id)[:8]}",
                              objective="(recovered from workspace)")
            session.add(project)
            session.commit()

        instruction = Instruction(project_id=project_id, content=objective)
        session.add(instruction)
        session.commit()
        session.refresh(instruction)

        model = resolve_stage_model("Coder", "local_model")
        task = Task(project_id=project_id, instruction_id=instruction.id,
                    description=objective, assigned_to=model,
                    status="pending", heartbeat=datetime.now())
        session.add(task)
        session.commit()
        session.refresh(task)
        instruction_id, task_id = instruction.id, task.id

    ACTIVE_WORKFLOWS.add(project_id)
    from gads.core.followup import run_followup_wrapper
    background_tasks.add_task(run_followup_wrapper, project_id, instruction_id, task_id,
                              objective, req.rehydrate, req.use_recipe)
    return {"project_id": str(project_id), "instruction_id": str(instruction_id),
            "task_id": str(task_id), "status": "started",
            "stream": f"/tasks/{task_id}/stream"}


NOTES_FILENAME = "user_notes.txt"


@app.get("/projects/{project_id}/notes")
async def get_project_notes(project_id: uuid.UUID):
    """Analyst notes for a project (free text, stored in the workspace)."""
    path = os.path.join(f"{WORKSPACE_ROOT}/{project_id}", NOTES_FILENAME)
    notes = ""
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                notes = f.read()
        except Exception:
            pass
    return {"project_id": str(project_id), "notes": notes}


@app.post("/projects/{project_id}/notes")
async def save_project_notes(project_id: uuid.UUID, req: NotesRequest):
    """Persist analyst notes to the workspace.

    Kept as a workspace file rather than a DB column: it survives the archive (and the DB
    losses that motivated approach_docs/020), travels with the exported bundle, and is
    surfaced to the Synthesizer so the analyst's own context reaches the report.
    """
    workspace_dir = f"{WORKSPACE_ROOT}/{project_id}"
    os.makedirs(workspace_dir, exist_ok=True)
    path = os.path.join(workspace_dir, NOTES_FILENAME)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(req.notes or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save notes: {e}")
    return {"project_id": str(project_id), "saved": True, "chars": len(req.notes or "")}


def _read_project_notes(workspace_dir: str) -> str:
    path = os.path.join(workspace_dir, NOTES_FILENAME)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


@app.post("/projects/{project_id}/finish")
async def finish_project(project_id: uuid.UUID):
    """End a project's interactive session and release its kernel.

    A project stays *active* (kernel pinned, state warm, follow-ups cheap) until the analyst
    explicitly finishes it — the user-driven lifecycle from #14. Finishing unpins the session
    so normal cleanup can reclaim it and resets the kernel to free sandbox memory. The
    workspace, artifacts, dashboard and DB records are untouched; the project can still be
    reopened and rehydrated (#18).
    """
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project and not os.path.isdir(f"{WORKSPACE_ROOT}/{project_id}"):
            raise HTTPException(status_code=404, detail="Project not found")
        if project_id in ACTIVE_WORKFLOWS:
            raise HTTPException(status_code=409,
                                detail="Work is still in flight; cancel or wait before finishing.")
        if project:
            state = dict(project.last_state_json or {})
            state["finished_at"] = datetime.now().isoformat(timespec="seconds")
            project.last_state_json = state
            session.add(project)
            session.commit()

    PINNED_SESSIONS.discard(project_id)
    released = False
    try:
        await SandboxClient().reset_session(str(project_id))
        released = True
    except Exception as e:
        print(f"  [Finish] Warning: could not reset session: {e}", flush=True)
    print(f"  [Finish] Project {project_id} finished; kernel released={released}.", flush=True)
    return {"project_id": str(project_id), "finished": True, "kernel_released": released}


@app.post("/projects/{project_id}/unpin")
async def unpin_project_session(project_id: uuid.UUID):
    """Release a pinned session so normal stale-session cleanup can reclaim its kernel."""
    was = project_id in PINNED_SESSIONS
    PINNED_SESSIONS.discard(project_id)
    return {"project_id": str(project_id), "was_pinned": was, "pinned": False}


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

async def _cleanup_stale_sessions(sandbox, current_project_id: uuid.UUID, reset_current: bool = True):
    """Reset sandbox sessions for all projects that have no active (pending/running) tasks.

    Session IDs are str(project_id), so we can enumerate them via the DB without
    needing a sandbox list-sessions endpoint. Called at the start of each execution
    phase so stale kernels from previous runs don't accumulate.

    `reset_current=False` PRESERVES this project's kernel — used on a replan of a
    deterministic recipe plan so completed upstream nodes' state survives and the
    resume-from-failed-node path can skip re-running them (see the execution loop).
    """
    try:
        with Session(engine) as db:
            all_projects = db.exec(select(Project)).all()
        reset_count = 0
        for project in all_projects:
            if project.id == current_project_id:
                continue
            # Kernel-state pin: a project with a live/rehydrated session for interactive
            # follow-up work must not be wiped by another project's run (approach_docs/020).
            if project.id in PINNED_SESSIONS:
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

    # Reset the current project's session for a clean kernel slate — unless the caller
    # asked to preserve it (resume-from-failed-node on a deterministic replan).
    if reset_current:
        try:
            await sandbox.reset_session(str(current_project_id))
            print(f"  [SessionCleanup] Reset current session {current_project_id}.", flush=True)
        except Exception as e:
            print(f"  [SessionCleanup] Warning: could not reset current session: {e}", flush=True)
    else:
        print(f"  [SessionCleanup] Preserving current session {current_project_id} (resume mode).", flush=True)


def _resolve_cloud_fallback_model(hierarchy: Dict[str, Any]) -> Optional[str]:
    """Pick a cloud model for the local retry-exhaustion cloud fallback (approach_docs/019).

    Prefers the cheapest tier first (T3 → T2 → T1) and never returns local_model. Used only
    when the operator opted into cloud fallback and the local model has exhausted its retries.
    """
    for tier in ("T3", "T2", "T1"):
        for m in ((hierarchy.get(tier, {}) or {}).get("models", []) or []):
            if m and m != "local_model":
                return m
    return None


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

def _compute_prompt_version() -> str:
    """SHA-256 over effective prompts + recipe/skill files.

    Labels every trace/generation with the prompt regime that produced it, so
    traces collected under superseded prompts can be excluded from training and
    eval pools (telemetry plan 010, Phase 1c).
    """
    import hashlib
    from gads.core.prompts import FACTORY_DEFAULTS
    h = hashlib.sha256()
    for agent_name in sorted(FACTORY_DEFAULTS.keys()):
        h.update(agent_name.encode())
        h.update(prompt_registry.get_prompt(agent_name).encode())
    knowledge_dir = Path(__file__).resolve().parent.parent / "knowledge"
    for sub in ("recipes", "skills"):
        d = knowledge_dir / sub
        if d.is_dir():
            for f in sorted(d.glob("*.md")):
                h.update(f.name.encode())
                h.update(f.read_bytes())
    return h.hexdigest()[:12]

def _finalize_workflow_trace(trace, project_id: uuid.UUID, prompt_version: str, recipe_id: Optional[str] = None):
    """Write outcome labels at the single workflow exit point (telemetry plan 010, Phase 1d).

    Runs in the workflow's `finally` block so success, failure (_mark_workflow_failed)
    and cancellation all get labeled through one code path.
    """
    with Session(engine) as s:
        proj = s.get(Project, project_id)
        tasks = s.exec(select(Task).where(Task.project_id == project_id)).all()

    outcome = "completed"
    if proj and proj.narrative:
        if "[CANCELLED]" in proj.narrative:
            outcome = "cancelled"
        elif proj.narrative.startswith("[HALTED]"):
            outcome = "failed"
    status_counts: Dict[str, int] = {}
    for t in tasks:
        status_counts[t.status] = status_counts.get(t.status, 0) + 1
    if outcome == "completed" and not status_counts.get("completed"):
        outcome = "failed"

    tags = [f"outcome:{outcome}", f"local_only:{get_local_only()}", f"routing_mode:{get_routing_mode()}"]
    if recipe_id:
        tags.append(f"recipe:{recipe_id}")
    trace.update(
        tags=tags,
        metadata={
            "outcome": outcome,
            "prompt_version": prompt_version,
            "recipe_id": recipe_id,
            "task_status_counts": status_counts,
            "task_models": sorted({t.assigned_to for t in tasks if t.assigned_to}),
            "total_escalations": sum(t.escalation_count or 0 for t in tasks),
        },
    )
    print(f"  [Telemetry] Trace labeled outcome:{outcome} ({len(tasks)} tasks)", flush=True)

async def run_agent_workflow(project_id: uuid.UUID, objective: str, instruction_id: Optional[uuid.UUID] = None):
    from gads.core.llm import trace_context

    prompt_version = _compute_prompt_version()

    # 1. Create top-level Langfuse Trace
    trace = langfuse_client.trace(
        id=str(project_id),
        name="Project Workflow",
        user_id="default_user",
        metadata={"objective": objective, "prompt_version": prompt_version},
        session_id=str(project_id)
    )

    ctx_token = trace_context.set({
        "project_id": str(project_id),
        "workflow_id": str(project_id),
        "user_id": "default_user",
        "langfuse_trace_id": trace.id,
        "prompt_version": prompt_version
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
        # A recipe_id read from workflow_spec.md is an explicit, launch-validated pin
        # and must override the Router's LLM match. A SpecDrafter-guessed recipe_id
        # (populated into spec_hints below) stays a soft fallback only.
        spec_pinned_recipe: Optional[str] = None

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
                for key in ("target_column", "feature_columns", "filters", "domain", "recipe_id", "save_model", "sample_rows", "taxonomy"):
                    if key in yaml_data:
                        spec_hints[key] = yaml_data[key]
                spec_pinned_recipe = spec_hints.get("recipe_id")
            except Exception as e:
                print(f"  [SpecDrafter] Failed to parse workflow_spec.md: {e}. Using raw objective.", flush=True)
        else:
            # Run SpecDrafter to generate the spec
            spec_model_fallback = ["local_model"] if get_local_only() else ["claude-haiku-4.5"]
            spec_model = resolve_stage_model("SpecDrafter", hierarchy.get("T3", {}).get("models", spec_model_fallback)[0])
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
                    "parent_observation_id": spec_span.id,
                    "stage": "Spec Drafting",
                    "attempt": None,
                    "escalation_count": None
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

        # Get disable_recipes from project state
        disable_recipes = False
        with Session(engine) as session:
            p_state = session.get(Project, project_id)
            if p_state and p_state.last_state_json:
                disable_recipes = p_state.last_state_json.get("disable_recipes", False)

        # 1. ROUTING (Resilient & Resourced)
        router_fallback = ["local_model"] if get_local_only() else ["gemini-3.5-flash-lite"]
        router_model = resolve_stage_model("Router", hierarchy.get("T3", {}).get("models", router_fallback)[0])

        intent = None
        knowledge_report = None
        intent = None

        # DETERMINISTIC ROUTING: when the spec pins a valid recipe, the Router LLM
        # call is redundant — the pin overrides its match anyway (see the HARD PIN
        # block below). On local models that call is also the workflow's most
        # fragile stage: unbounded `reasoning` rambling hits max_tokens, the JSON
        # never closes, and with no T4 escalation target the whole run halts.
        # Derive the intent from the recipe's own routing metadata instead.
        if not disable_recipes and spec_pinned_recipe:
            pinned = registry.get_recipe(spec_pinned_recipe)
            if pinned:
                def _first(v: Any, default: str) -> str:
                    if isinstance(v, list):
                        return str(v[0]) if v else default
                    return str(v) if v else default
                aw = pinned.applies_when or {}
                intent = RouterOutput(
                    task_type=_first(aw.get("task_type"), "unknown"),
                    data_modality=_first(aw.get("data_modality"), "tabular"),
                    matched_recipe_id=pinned.id,
                    confidence=1.0,
                    reasoning=f"Deterministic route: spec pins recipe '{pinned.id}'. Router LLM call skipped."
                )
                knowledge_report = ReconciliationReport(
                    recipe_id=pinned.id,
                    rationale=pinned.rationale,
                    recommended_dag_nodes=[node.dict() for node in pinned.dag],
                    invariants=pinned.invariants,
                    skippable_nodes=[],
                    schema_warnings=[]
                )
                trace_context.get().update({"recipe_id": pinned.id})
                print(f"  [Router] Spec pins recipe '{pinned.id}' — deterministic route, LLM call skipped.", flush=True)
                with Session(engine) as session:
                    route_task = Task(
                        project_id=project_id,
                        instruction_id=instruction_id,
                        description="Architect (deterministic) routed via spec-pinned recipe — no LLM call needed.",
                        assigned_to="Router",
                        status="completed",
                        heartbeat=datetime.now(),
                        result_json={
                            "stdout": (
                                f"**Deterministic Routing (spec pin):**\n"
                                f"- Task Type: `{intent.task_type}`\n- Modality: `{intent.data_modality}`\n"
                                f"- Recipe: `{pinned.id}`\n\n--- KNOWLEDGE BASE ---\n"
                                f"Applied SOP: {pinned.id}\nRationale: {pinned.rationale}"
                            ),
                            "model_used": "none (deterministic)"
                        }
                    )
                    session.add(route_task)
                    session.commit()

        while intent is None:
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
                    "parent_observation_id": span.id,
                    "stage": "Architect Routing",
                    "attempt": None,
                    "escalation_count": None
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
                    if disable_recipes:
                        recipe_id = None
                        recipe = None
                        if intent:
                            intent.matched_recipe_id = None
                    else:
                        recipe_id = intent.matched_recipe_id
                        recipe = registry.get_recipe(recipe_id) if recipe_id else None
                        # HARD PIN: a recipe declared in the spec file wins over the
                        # Router's LLM classification. Without this, a Router mismatch
                        # is fatal — the Recipe Enforcer replaces the Planner's tasks
                        # with the wrong recipe's DAG on every replan attempt, so
                        # PlanCritique rejects identically until the workflow halts.
                        if spec_pinned_recipe:
                            pinned = registry.get_recipe(spec_pinned_recipe)
                            if pinned and pinned.id != recipe_id:
                                print(
                                    f"  [Router] Spec pins recipe '{pinned.id}' — overriding "
                                    f"Router match '{recipe_id or 'None'}'.",
                                    flush=True
                                )
                                recipe_id = pinned.id
                                recipe = pinned
                                intent.matched_recipe_id = pinned.id
                        if not recipe and spec_hints.get("recipe_id"):
                            recipe_id = spec_hints.get("recipe_id")
                            recipe = registry.get_recipe(recipe_id) if recipe_id else None

                    recipe_info = "No specific recipe found. Proceeding with general data science reasoning."
                    if disable_recipes:
                        recipe_info = "Recipe search disabled by user configuration. Proceeding with general data science reasoning."

                    if recipe:
                        knowledge_report = ReconciliationReport(
                            recipe_id=recipe.id,
                            rationale=recipe.rationale,
                            recommended_dag_nodes=[node.dict() for node in recipe.dag],
                            invariants=recipe.invariants,
                            skippable_nodes=[],
                            schema_warnings=[]
                        )
                        recipe_info = f"Applied SOP: {recipe.id}\nRationale: {recipe.rationale}"
                        trace_context.get().update({"recipe_id": recipe.id})

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

        # --- RUN TAXONOMY (approach_docs/018) — classify EVERY run, ad-hoc included ---
        # Spec-launched runs carry a `taxonomy:` block; ad-hoc runs are classified
        # deterministically from the Router's task_type/data_modality + domain hint.
        # Persisted on the project so any run has a viewable classification.
        try:
            from gads.core import taxonomy as _tx
            _existing = spec_hints.get("taxonomy")
            run_tax = _tx.derive_run_taxonomy(
                existing=_existing if isinstance(_existing, dict) else None,
                task_type=intent.task_type,
                data_modality=intent.data_modality,
                domain_text=spec_hints.get("domain"),
            )
            with Session(engine) as session:
                proj = session.get(Project, project_id)
                if proj:
                    st = dict(proj.last_state_json or {})
                    st["taxonomy"] = run_tax["taxonomy"]
                    st["taxonomy_source"] = run_tax["source"]
                    proj.last_state_json = st
                    session.add(proj)
                    session.commit()
            # Tag the generated spec file too, so the ad-hoc spec is itself classified.
            try:
                if workflow_spec_path.exists():
                    _txt = workflow_spec_path.read_text(encoding="utf-8")
                    _new = _tx.inject_into_frontmatter(_txt, run_tax["taxonomy"])
                    if _new != _txt:
                        workflow_spec_path.write_text(_new, encoding="utf-8")
            except Exception as _e:
                print(f"  [taxonomy] could not tag workflow_spec.md: {_e}", flush=True)
            _t = run_tax["taxonomy"]
            print(f"  [taxonomy] run classified: {_t.get('intent')} / {_t.get('task')} / "
                  f"{_t.get('domain')} (source={run_tax['source']})", flush=True)
            for _w in run_tax["warnings"]:
                print(f"  [taxonomy] warning: {_w}", flush=True)
        except Exception as _e:
            print(f"  [taxonomy] run classification skipped: {_e}", flush=True)

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
        dial_info = None  # delegation-dial rung info (approach_docs/013), set per plan

        while workflow_attempt < MAX_WORKFLOW_ATTEMPTS:
            workflow_attempt += 1
            print(f"\n  [Workflow] Starting attempt {workflow_attempt}/{MAX_WORKFLOW_ATTEMPTS}...", flush=True)
            
            if await is_cancelled(): return

            # 3. PLANNING (Resilient Decomposition)
            planner_fallback = ["local_model"] if get_local_only() else ["gemini-3.6-flash"]
            planner_model = resolve_stage_model("Planner", hierarchy.get("T2", {}).get("models", planner_fallback)[0])
            # planner_files and spec_hints are pre-computed above (outside this retry loop)

            planner_res = None
            plan_is_deterministic = False

            # --- RECIPE PLAN COMPILER (deterministic, no LLM) ---
            # When a recipe is matched, the plan IS the recipe DAG: the Recipe
            # Enforcer used to discard the Planner's LLM output and rebuild it
            # from the recipe anyway. On local models that LLM call is worse
            # than wasted — the model transcribes the recipe intents until it
            # hits max_tokens and the truncated JSON halts the workflow.
            # Compile the plan directly from the DAG nodes instead, and skip
            # PlanCritique below (auditing a deterministic plan can only reject
            # into a replan that recompiles the identical plan).
            if knowledge_report and knowledge_report.recommended_dag_nodes:
                enforced_steps = []
                # Recipe invariants are global failure-mode guardrails; append a
                # compact block to every task so each Coder call carries them.
                invariants_block = ""
                if knowledge_report.invariants:
                    inv_lines = "\n".join(f"- {inv}" for inv in knowledge_report.invariants)
                    invariants_block = (
                        "\n\n[RECIPE INVARIANTS — these rules are MANDATORY for every step:\n"
                        + inv_lines + "\n]"
                    )
                for idx, node in enumerate(knowledge_report.recommended_dag_nodes):
                    tier = node.get("worker_tier", "T2")
                    model_id = resolve_stage_model("Coder", (
                        hierarchy.get(tier, hierarchy.get("T2", {}))
                        .get("models", ["local_model"])[0]
                    ))

                    postcondition: Dict[str, Any] = {"output_type": "value"}
                    if node.get("required_metrics"):
                        postcondition["required_metrics"] = node["required_metrics"]
                    if node.get("produces"):
                        postcondition["required_variables"] = node["produces"]
                    # Native safety net for the local retry-exhaustion fallback (approach_docs/019).
                    if node.get("fallback_native"):
                        postcondition["fallback_native"] = node["fallback_native"]
                    if node.get("fallback_call"):
                        postcondition["fallback_call"] = node["fallback_call"]

                    description = node.get("intent", node.get("id", "")).strip()
                    if idx == 0:
                        hint_lines = []
                        # Recipe intents routinely reference "the objective" (e.g. the causal
                        # recipe derives treatment/outcome from it), but the Coder only ever
                        # sees the task description — so the objective must ride along here.
                        if formalized_objective:
                            hint_lines.append(f"OBJECTIVE: {formalized_objective[:600]}")
                        if spec_hints.get("target_column"):
                            hint_lines.append(f"Target column: `{spec_hints['target_column']}`.")
                        if spec_hints.get("sample_rows"):
                            n = spec_hints["sample_rows"]
                            # Collaborative filtering must NOT be randomly row-capped: an
                            # interaction log is long-tailed, so a random subset shares almost
                            # no users/items and the k-core filter downstream collapses the
                            # matrix (see #22). Express the same budget as dense-core sampling.
                            _dag_text = " ".join(
                                str(x.get("intent", "")) for x in
                                (knowledge_report.recommended_dag_nodes or [])
                            )
                            _is_cf = ("gads_build_interaction_matrix" in _dag_text
                                      or "recommendation" in str(knowledge_report.recipe_id or "")
                                      or "collaborative" in str(knowledge_report.recipe_id or ""))
                            if _is_cf:
                                hint_lines.append(
                                    f"SAMPLING CONSTRAINT: cap the data at {n:,} interactions by "
                                    f"passing `max_rows={n}` to gads_build_interaction_matrix "
                                    f"(dense-core sampling). Do NOT call `df.sample(...)` — "
                                    f"randomly subsetting an interaction log destroys the "
                                    f"co-occurrence structure and collapses the matrix."
                                )
                            else:
                                hint_lines.append(
                                    f"SAMPLING CONSTRAINT: immediately after loading, apply "
                                    f"`df = df.sample({n}, random_state=42).reset_index(drop=True)` "
                                    f"to cap the dataset at {n:,} rows."
                                )
                        if hint_lines:
                            description += "\n\n[SPEC HINTS: " + " ".join(hint_lines) + "]"

                    if invariants_block:
                        description += invariants_block

                    # Prefer skills the recipe node explicitly declares; only fall back
                    # to discovery (keyword + semantic) when the node names none, so
                    # curated recipes keep byte-stable prompts.
                    #
                    # An explicitly EMPTY list means "no curated skill" and suppresses
                    # discovery — distinct from the key being absent. Discovery cannot
                    # tell which library a node is meant to use (it matches on the task
                    # description, which is library-agnostic), so on a node pinned to a
                    # specific API it injects whatever is semantically nearest: the
                    # statsmodels/sklearn causal recipes were both handed the DoWhy and
                    # EconML skills, contradicting their own intent. It also silently
                    # lifts a D3 node's effective rung, since D3 is defined as
                    # "no curated skill" (approach_docs/014).
                    declared = node.get("attached_skills")
                    node_skills = [s for s in (declared or []) if s in registry.skills]
                    if node_skills:
                        attached = list(node_skills)
                    elif isinstance(declared, list) and not declared:
                        attached = []
                    else:
                        matches = registry.find_skills_combined(description)
                        attached = [s.id for s, _src, _score in matches]
                        if matches:
                            print(
                                "  [RecipeCompiler] discovered skills for "
                                f"'{node.get('id')}': "
                                + ", ".join(f"{s.id}({src}:{score:.2f})" for s, src, score in matches),
                                flush=True
                            )
                    # sandbox_environment is force-loaded at the executor.

                    enforced_steps.append(PlannerTask(
                        description=description,
                        assigned_to=model_id,
                        postcondition=postcondition,
                        attached_skills=attached
                    ))

                planner_res = AgentResponse(
                    content=PlannerOutput(steps=enforced_steps),
                    model_used="none (deterministic)"
                )
                plan_is_deterministic = True
                print(
                    f"  [RecipeCompiler] Compiled {len(enforced_steps)} tasks directly from "
                    f"recipe DAG '{knowledge_report.recipe_id}' — Planner LLM call skipped.",
                    flush=True
                )
                with Session(engine) as session:
                    plan_task = Task(
                        project_id=project_id,
                        instruction_id=instruction_id,
                        description=f"Planner (deterministic): plan compiled from recipe '{knowledge_report.recipe_id}' (Attempt {workflow_attempt}).",
                        assigned_to="Planner",
                        status="completed",
                        heartbeat=datetime.now(),
                        result_json={
                            "stdout": f"Compiled {len(enforced_steps)} tasks from recipe DAG `{knowledge_report.recipe_id}` — no LLM call needed.",
                            "model_used": "none (deterministic)"
                        }
                    )
                    session.add(plan_task)
                    session.commit()

            while planner_res is None:
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
                        "parent_observation_id": span.id,
                        "stage": "Project Planning",
                        "attempt": workflow_attempt,
                        "escalation_count": None
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
                        # NOTE: recipe-matched plans never reach this LLM path — they
                        # are compiled deterministically above (RECIPE PLAN COMPILER),
                        # which replaced the old post-Planner Recipe Enforcer.

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

            
            # 3.1 PLAN CRITIQUE (skipped for deterministic recipe plans: an LLM
            # audit of a compiled plan can only reject into a replan that
            # recompiles the identical plan, or halt a user-pinned recipe — and
            # on local models the audit call itself is a fatal-failure risk)
            plan_critique = None
            if plan_is_deterministic:
                plan_critique = PlanCritiqueOutput(
                    is_approved=True,
                    feedback=f"Plan compiled deterministically from recipe '{knowledge_report.recipe_id}' — audit skipped."
                )
                print(f"  [PlanCritique] Skipped — plan is deterministic from recipe DAG.", flush=True)
                with Session(engine) as session:
                    pc_task = Task(
                        project_id=project_id,
                        instruction_id=instruction_id,
                        description=f"Auditor skipped: plan compiled deterministically from recipe (Attempt {workflow_attempt}).",
                        assigned_to="PlanCritique",
                        status="completed",
                        heartbeat=datetime.now(),
                        result_json={"stdout": plan_critique.feedback, "model_used": "none (deterministic)"}
                    )
                    session.add(pc_task)
                    session.commit()

            plan_critique_fallback = ["local_model"] if get_local_only() else ["gemini-3.6-flash"]
            plan_critique_model = resolve_stage_model("PlanCritique", hierarchy.get("T2", {}).get("models", plan_critique_fallback)[0])

            while plan_critique is None:
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
                        "parent_observation_id": span.id,
                        "stage": "Plan Critique",
                        "attempt": workflow_attempt,
                        "escalation_count": None
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
                    # Routing-mode override AFTER hallucination sanitization: hybrid
                    # sends all execution tasks to local_model (not in the cloud
                    # hierarchy, so it must be applied here, not via valid_models);
                    # cloud_pinned forces the pinned model.
                    assigned_model = resolve_stage_model("Coder", assigned_model)

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

            # DIAL RUNG (approach_docs/013): place this run on the delegation ladder
            # from the same inputs the plan was built from, and persist it on the
            # project so it is inspectable while the run is live.
            if knowledge_report and knowledge_report.recommended_dag_nodes:
                _selection = "pinned" if spec_hints.get("recipe_id") == knowledge_report.recipe_id else "routed"
                dial_info = compiled_plan_dial(knowledge_report.recommended_dag_nodes, _selection)
                dial_info["recipe_id"] = knowledge_report.recipe_id
            else:
                dial_info = drafted_plan_dial(spec_hints)
            print(
                f"  [Dial] Project rung: {dial_info['rung']} "
                f"(selection={dial_info['selection']}, tasks={dial_info['task_rungs'] or 'drafted'})",
                flush=True
            )
            with Session(engine) as session:
                proj = session.get(Project, project_id)
                if proj:
                    _state = dict(proj.last_state_json or {})
                    _state["dial"] = dial_info
                    proj.last_state_json = _state
                    session.add(proj)
                    session.commit()

            # 4. EXECUTION
            # Preserve the kernel across replans of a deterministic recipe plan so the
            # resume-from-failed-node path can skip already-completed upstream nodes.
            # Attempt 1 always starts clean; drafted (non-deterministic) plans always
            # reset (their descriptions vary run-to-run, so nothing is safely resumable).
            _reset_kernel = (workflow_attempt == 1) or (not plan_is_deterministic)
            await _cleanup_stale_sessions(executor.sandbox, project_id, reset_current=_reset_kernel)
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
                    # Three-state, matching RecipeTask.attached_skills: None = nobody
                    # curated this task (discover freely); [] = declared skill-free, which
                    # must suppress BOTH matchers — a node pinned to a specific library
                    # gets actively wrong guidance from matchers that only see the
                    # library-agnostic description (approach_docs/014); non-empty = curated.
                    # Gated on plan_is_deterministic: only a RECIPE can declare a node
                    # skill-free. `PlannerTask.attached_skills` defaults to [], so a
                    # drafted-lane task the Planner left blank is indistinguishable from
                    # a deliberate empty declaration — and that is exactly where the
                    # discovery safety net matters most (weak model, no curation).
                    curated = task_obj.attached_skills
                    declared_skill_free = (
                        plan_is_deterministic and isinstance(curated, list) and not curated
                    )
                    assigned_ids = set(curated or [])

                    full_body_ids = set(assigned_ids)
                    if not declared_skill_free:
                        for skill, hits in registry.find_skills_scored(desc):
                            if skill.id not in full_body_ids and hits >= KEYWORD_HIT_THRESHOLD:
                                full_body_ids.add(skill.id)

                        # Semantic discovery — only for uncurated tasks. Tasks with
                        # recipe/Planner-attached skills keep byte-stable prompts (the
                        # frozen benchmarks depend on that); tasks nobody curated get the
                        # embedding matcher as a safety net beyond keyword triggers.
                        if not assigned_ids:
                            sem_matches = registry.find_skills_semantic(desc)
                            for skill, score in sem_matches:
                                if skill.id not in full_body_ids:
                                    full_body_ids.add(skill.id)
                                    print(f"    [SkillSemantics] attached '{skill.id}' (cos={score:.2f})", flush=True)
                    else:
                        print("    [Workflow] Node declared skill-free — discovery suppressed.", flush=True)

                    # The sandbox core constraints are mandatory for every Coder call.
                    # (The Coder is single-shot — it cannot request skills, so no index
                    # of unloaded skills is included; only full bodies are useful to it.)
                    full_body_ids.add("sandbox_environment")
                    loaded_skills = [registry.skills[sid] for sid in full_body_ids if sid in registry.skills]

                    skills_ctx = None
                    if loaded_skills:
                        bodies = "\n\n".join(f"#### {s.id}\n{s.content}" for s in loaded_skills)
                        skills_ctx = f"### LOADED SKILL GUIDANCE\n{bodies}"
                        print(f"    [Workflow] Applied skills: {[s.id for s in loaded_skills]}", flush=True)

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

                    # RESUME-FROM-FAILED-NODE: on a replan of a deterministic recipe plan
                    # the kernel is preserved (see _reset_kernel above), so a node that
                    # already completed in a PRIOR attempt — with its declared output
                    # variables still live in the kernel — need not run again. Skip it and
                    # reuse the prior result. This turns a single failed step from a full
                    # upstream-DAG re-run into a re-run of just the failed node + downstream.
                    # Conservative: only skips when the outputs are VERIFIABLY present.
                    if plan_is_deterministic and workflow_attempt > 1 and isinstance(namespace_summary, dict):
                        with Session(engine) as _rs:
                            _cur = _rs.get(Task, task_id)
                            _req_vars = (_cur.postcondition_json or {}).get("required_variables", []) if _cur else []
                            _prior = _rs.exec(
                                select(Task).where(
                                    Task.project_id == project_id,
                                    Task.description == desc,
                                    Task.status == "completed",
                                    Task.id != task_id,
                                ).order_by(Task.created_at.desc())
                            ).first()
                            if _prior is not None and _req_vars and all(v in namespace_summary for v in _req_vars):
                                hub = ExecutionHub(_rs)
                                _cur.status = "completed"
                                _cur.heartbeat = datetime.now()
                                _cur.result_json = {**(_prior.result_json or {}), "resumed_from_prior_attempt": True}
                                _rs.add(_cur)
                                _rs.commit()
                                hub.create_outbox_event("TASK_COMPLETED", {"task_id": str(task_id), "description": desc})
                                print(f"  [Resume] ✓ Skipping '{desc[:60]}' — completed in a prior "
                                      f"attempt; outputs {_req_vars} live in kernel.", flush=True)
                                task_span.end(output={"resumed": True})
                                break  # task satisfied without re-execution; on to the next node

                    tid_str = str(task_id)
                    LIVE_STREAMS[tid_str] = {"reasoning": "", "stdout": ""}

                    async def stream_reasoning_callback(token: str):
                        LIVE_STREAMS[tid_str]["reasoning"] += token

                    async def stream_stdout_callback(text: str):
                        LIVE_STREAMS[tid_str]["stdout"] = text

                    trace_context.get().update({
                        "agent_name": "CodeGenerator",
                        "task_id": tid_str,
                        "parent_observation_id": task_span.id,
                        "stage": "Task Execution",
                        "attempt": None,  # set per coder attempt by the executor retry loop
                        "escalation_count": None
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

                    # Local retry-exhaustion fallback (approach_docs/019): if enabled and this
                    # node declares a native safety net, pass it so the executor can invoke it
                    # deterministically after the model exhausts its retries.
                    _fb_mode = get_local_fallback()
                    _fb_native = _fb_call = None
                    if _fb_mode != "none":
                        with Session(engine) as _fbs:
                            _fbt = _fbs.get(Task, task_id)
                            _pc = (_fbt.postcondition_json or {}) if _fbt else {}
                        _fb_native = _pc.get("fallback_native")
                        _fb_call = _pc.get("fallback_call")

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
                            state_summary=state_summary_str,
                            recipe_id=(knowledge_report.recipe_id if knowledge_report else None),
                            fallback_native=_fb_native,
                            fallback_call=_fb_call,
                            fallback_mode=_fb_mode,
                        )
                    finally:
                        _hb_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await _hb_task

                    # CLOUD FALLBACK (opt-in, post-exhaustion): the local model exhausted its
                    # retries and any native fallback did not satisfy the node. Escalate this
                    # ONE task to a cloud model for a SINGLE attempt (max_attempts=1) — a
                    # deliberate, gated exception to the "local never escalates" mandate,
                    # reached only after the local model provably could not do it. See
                    # approach_docs/019.
                    if (res.error and _fb_mode in ("cloud", "native_then_cloud")
                            and not str(model_used).startswith("native_fallback")
                            and get_local_only()):
                        try:
                            _cloud_hier = await get_model_hierarchy(force_cloud=True)
                        except Exception as _che:
                            _cloud_hier = {}
                            print(f"  [Workflow] ⚠ Cloud fallback: could not fetch cloud hierarchy: {_che}", flush=True)
                        _cloud_model = _resolve_cloud_fallback_model(_cloud_hier)
                        if _cloud_model:
                            print(f"  [Workflow] ☁ Cloud fallback: local exhausted — one attempt "
                                  f"on '{_cloud_model}'.", flush=True)
                            _prev_model, _prev_str = executor.coder.model, executor.coder.model_str
                            executor.coder.model = _cloud_model
                            executor.coder.model_str = _cloud_model
                            _hb2 = asyncio.create_task(_heartbeat_loop(task_id))
                            try:
                                res2, model_used2 = await executor.run_task(
                                    desc,
                                    project_id=project_id,
                                    session_id=str(project_id),
                                    skills_context=skills_ctx,
                                    task_id=task_id,
                                    stdout_callback=stream_stdout_callback,
                                    stream_callback=stream_reasoning_callback,
                                    cancel_check=is_cancelled,
                                    state_summary=state_summary_str,
                                    recipe_id=(knowledge_report.recipe_id if knowledge_report else None),
                                    fallback_mode="none",
                                    max_attempts=1,
                                )
                                if res2.error is None:
                                    res, model_used = res2, f"cloud_fallback:{model_used2}"
                                    print(f"  [Workflow] ✅ Cloud fallback succeeded on '{model_used2}'.", flush=True)
                                else:
                                    print(f"  [Workflow] ⚠ Cloud fallback also failed: "
                                          f"{res2.error.get('evalue', '')[:120]}", flush=True)
                            finally:
                                _hb2.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await _hb2
                                executor.coder.model, executor.coder.model_str = _prev_model, _prev_str

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

                            # Fallback observability: when a node completed via a fallback
                            # (not the assigned model), emit a distinct event so the UI can
                            # surface it and pass@model reporting can count it.
                            if str(model_used).startswith(("native_fallback:", "cloud_fallback:")):
                                _fb_kind, _, _fb_model = str(model_used).partition(":")
                                hub.create_outbox_event("TASK_FALLBACK", {
                                    "task_id": str(task_id), "description": desc[:120],
                                    "fallback_kind": _fb_kind, "model": _fb_model,
                                })
                                print(f"  [Workflow] 📎 Fallback recorded: {_fb_kind} via "
                                      f"'{_fb_model}' for '{desc[:50]}'", flush=True)
                            files_after = new_files_after

                            project = session.get(Project, project_id)
                            if project:
                                # Kernel snapshots share the JSON column with project
                                # metadata (spec_filename/fast_mode/dial). Full
                                # replacement erased those keys on the first completed
                                # task — the retroactive spec_filename matching in
                                # list_projects exists because of exactly this. Merge:
                                # snapshot wins on kernel keys, metadata is preserved.
                                _meta_keys = ("fast_mode", "disable_recipes", "spec_filename", "dial")
                                _prev = project.last_state_json or {}
                                _merged = dict(executor.authoritative_state or {})
                                for _k in _meta_keys:
                                    if _k in _prev and _k not in _merged:
                                        _merged[_k] = _prev[_k]
                                project.last_state_json = _merged
                                session.add(project)
                                session.commit()

                            new_files_names = set([f["name"] for f in files_after]) - set([f["name"] for f in current_files_meta])
                            
                            has_explicit_plots = any(
                                (nf.lower().endswith(".html") and nf != "final_dashboard.html")
                                or (nf.lower().endswith(".json") and not nf.endswith(".meta.json")
                                    and looks_like_plotly_figure(os.path.join(workspace_dir, nf)))
                                for nf in new_files_names)

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
                                elif (nf.lower().endswith(".json") and not nf.endswith(".meta.json")
                                      and looks_like_plotly_figure(full_path)):
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

            # A recipe-structural execution failure warrants a fresh planning attempt,
            # steered by the execution_feedback assembled above. Without this the failure
            # falls straight through to synthesis and only replans if the synthesis critique
            # happens to reject — so the resume-from-failed-node path was unreachable in its
            # own target scenario. For a deterministic recipe plan the kernel is preserved
            # across the replan (see _reset_kernel), so resume skips the already-completed
            # upstream nodes on the next attempt instead of re-running the whole DAG; only the
            # failed node + its downstream re-run. Falls through to synthesis (report what we
            # have) once attempts are exhausted.
            if (failed_tasks or pending_tasks) and workflow_attempt < MAX_WORKFLOW_ATTEMPTS:
                print(f"  [Workflow] ↻ Replanning after execution failure "
                      f"(attempt {workflow_attempt}/{MAX_WORKFLOW_ATTEMPTS}).", flush=True)
                continue

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
                cv_model_fallback = ["local_model"] if get_local_only() else ["gemini-3.6-flash"]
                cv_model = resolve_stage_model("CompletenessVerifier", hierarchy.get("T2", {}).get("models", cv_model_fallback)[0])

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
                    "parent_observation_id": cv_span.id,
                    "stage": "Completeness Verification",
                    "attempt": workflow_attempt,
                    "escalation_count": None
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

                    # Materially-complete guard: if the recipe declared required_metrics
                    # and metrics.json contains them ALL, the quantitative objective is
                    # demonstrably met. Do NOT let a local verifier's soft "missing_analyses"
                    # discard a complete run and gamble on a replan — an over-eager gap on
                    # gemma-4-12b threw away a successful attempt and the replan regressed
                    # into buggy code. Hard metric evidence outranks a fuzzy interpretive gap.
                    required_metrics: set = set()
                    if knowledge_report is not None:
                        for _n in (getattr(knowledge_report, "recommended_dag_nodes", None) or []):
                            _rm = _n.get("required_metrics") if isinstance(_n, dict) else getattr(_n, "required_metrics", None)
                            if _rm:
                                required_metrics.update(_rm)
                    metrics_satisfied = (
                        bool(required_metrics)
                        and isinstance(metrics_data, dict)
                        and required_metrics.issubset(set(metrics_data.keys()))
                    )

                    # Guard: only replan if there are concrete named gaps AND the run's
                    # required metrics are not already all present.
                    if not cv_out.is_complete and cv_out.missing_analyses and not metrics_satisfied:
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
                    elif metrics_satisfied and not cv_out.is_complete:
                        print(
                            f"  [CompletenessVerifier] Soft gaps flagged but all required metrics "
                            f"{sorted(required_metrics)} are present — accepting the run, skipping replan.",
                            flush=True
                        )
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
                if (_jf["name"].endswith(".json") and not _jf["name"].endswith(".meta.json")
                        and looks_like_plotly_figure(os.path.join(workspace_dir, _jf["name"]))):
                    try:
                        harden_json_artifact(os.path.join(workspace_dir, _jf["name"]))
                    except Exception:
                        pass

            # 5. SYNTHESIS & CRITIQUE LOOP
            synthesizer_fallback = ["local_model"] if get_local_only() else ["gemini-3.6-flash"]
            synthesizer_model = resolve_stage_model("Synthesizer", hierarchy.get("T2", {}).get("models", synthesizer_fallback)[0])

            critique_fallback = ["local_model"] if get_local_only() else ["gemini-3.6-flash"]
            critique_model = resolve_stage_model("Critique", hierarchy.get("T2", {}).get("models", hierarchy.get("T3", {}).get("models", critique_fallback))[0])

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
                    trace_context.get().update({"agent_name": "Synthesizer", "task_id": str(synth_task.id), "parent_observation_id": span.id, "stage": "Synthesis", "attempt": workflow_attempt, "escalation_count": None})

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
                        # Analyst notes (user_notes.txt) are the human's own context for this
                        # project — domain caveats, what they actually care about. Appended to
                        # the artifact context rather than added as a prompt placeholder, so
                        # no factory-default/format() coupling is introduced.
                        _notes = _read_project_notes(workspace_dir)
                        _synth_context = context
                        if _notes:
                            _synth_context += (
                                "\n\n### ANALYST NOTES (written by the user for this project)\n"
                                f"{_notes[:4000]}\n"
                                "Treat these as context and priorities from the analyst; "
                                "reflect them in the narrative where they are relevant."
                            )
                            print(f"  [Synthesis] Including {len(_notes)} chars of analyst notes.", flush=True)

                        synth_res = await synthesizer.run(SynthesizerInput(
                            objective=objective,
                            context_artifacts=_synth_context,
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
                    trace_context.get().update({"agent_name": "Critique", "task_id": str(critique_task.id), "parent_observation_id": span.id, "stage": "Critique", "attempt": workflow_attempt, "escalation_count": None})

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

            # DIAL LEDGER (approach_docs/013): one record per completed run — the
            # accumulating evidence for the rung × engine pass/fail grid. "pass"
            # requires an approved synthesis AND zero failed tasks (a run can produce
            # a dashboard while an execution task failed, e.g. 835cf36c). Fail-open
            # ADVISORY agents are excluded: the CompletenessVerifier is designed so
            # its own crash never affects the workflow, so its failure is harness
            # noise, not a failure of the delegated work (first hit: run 7d07a611).
            _advisory_agents = ("CompletenessVerifier",)
            failed_count = len([
                t for t in session.exec(
                    select(Task).where(Task.project_id == project_id, Task.status == "failed")
                ).all()
                if t.assigned_to not in _advisory_agents
            ])
            # pass@model vs pass@model+fallback (approach_docs/019): how much of the run the
            # assigned model did itself vs. what a fallback (native/cloud) had to rescue.
            # Keep the two SEPARATE — a fallback-assisted pass must never read as a model pass,
            # or the delegation-dial measurement is contaminated. Deduped per recipe node
            # (last completed instance wins across attempts).
            _by_node: Dict[str, str] = {}
            for t in session.exec(
                select(Task).where(Task.project_id == project_id, Task.status == "completed")
                .order_by(Task.created_at)
            ).all():
                _rj = t.result_json or {}
                _mu = str(_rj.get("model_used", ""))
                if _rj.get("code") or _mu.startswith(("native_fallback:", "cloud_fallback:")):
                    _by_node[t.description] = _mu
            _n_exec = len(_by_node)
            _n_native_fb = sum(1 for m in _by_node.values() if m.startswith("native_fallback:"))
            _n_cloud_fb = sum(1 for m in _by_node.values() if m.startswith("cloud_fallback:"))
            _n_fb = _n_native_fb + _n_cloud_fb
            _n_model = _n_exec - _n_fb
            _pass_at_model = round(_n_model / _n_exec, 3) if _n_exec else None
            if _n_exec:
                print(f"  [Dial] pass@model={_n_model}/{_n_exec} ({_pass_at_model}) | "
                      f"fallback-assisted: {_n_fb} ({_n_native_fb} native, {_n_cloud_fb} cloud)",
                      flush=True)

            append_ledger({
                "project_id": str(project_id),
                "spec": ((proj.last_state_json or {}).get("spec_filename") if proj else None),
                "recipe_id": (dial_info or {}).get("recipe_id"),
                "rung": (dial_info or {}).get("rung"),
                "task_rungs": (dial_info or {}).get("task_rungs"),
                "selection": (dial_info or {}).get("selection"),
                "routing_mode": get_routing_mode(),
                "pinned_model": get_pinned_model(),
                "outcome": "pass" if (workflow_succeeded and failed_count == 0) else "fail",
                "workflow_succeeded": workflow_succeeded,
                "failed_tasks": failed_count,
                "workflow_attempts": workflow_attempt,
                # pass@model reporting — model-only vs fallback-assisted (never collapsed).
                "exec_nodes": _n_exec,
                "model_pass": _n_model,
                "fallback_pass": _n_fb,
                "native_fallback": _n_native_fb,
                "cloud_fallback": _n_cloud_fb,
                "pass_at_model": _pass_at_model,
            })

    except Exception as e:
        print(f"❌ FATAL ERROR: {traceback.format_exc()}")
        with Session(engine) as session:
            _mark_workflow_failed(session, project_id, f"Fatal system error: {str(e)}")
    finally:
        from gads.core.llm import trace_context
        try:
            _ctx = trace_context.get() or {}
            _finalize_workflow_trace(trace, project_id, prompt_version, recipe_id=_ctx.get("recipe_id"))
        except Exception as label_exc:
            print(f"  [Telemetry] Failed to label trace outcome: {label_exc}", flush=True)
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
    files = sorted([f for f in specs_dir.iterdir() if f.is_file() and f.suffix == ".md"], key=lambda f: f.name)

    # Read spec contents once for retroactive matching
    spec_contents: Dict[str, str] = {}
    for f in files:
        try:
            spec_contents[f.name] = f.read_text(encoding="utf-8")
        except Exception:
            pass

    def _strip_injected(content: str) -> str:
        """Strip lines injected by fast_mode (sample_rows) for content comparison."""
        return re.sub(r"sample_rows:\s*\d+\n?", "", content)

    # Build last_used_at map from DB: max project.created_at per spec_filename.
    # Also retroactively tag old projects that predate spec_filename tracking by
    # matching their workspace workflow_spec.md content against known spec files.
    last_used_map: Dict[str, str] = {}
    with Session(engine) as session:
        projects = session.exec(select(Project)).all()
        needs_flush = False
        for p in projects:
            state = p.last_state_json or {}
            fname = state.get("spec_filename")

            # Retroactive match for projects missing spec_filename
            if not fname:
                ws_spec = Path(f"{WORKSPACE_ROOT}/{p.id}/workflow_spec.md")
                if ws_spec.exists():
                    try:
                        ws_content = _strip_injected(ws_spec.read_text(encoding="utf-8"))
                        for candidate, raw in spec_contents.items():
                            if ws_content == _strip_injected(raw):
                                fname = candidate
                                state["spec_filename"] = fname
                                p.last_state_json = state
                                session.add(p)
                                needs_flush = True
                                break
                    except Exception:
                        pass

            if fname:
                ts = p.created_at.isoformat()
                if fname not in last_used_map or ts > last_used_map[fname]:
                    last_used_map[fname] = ts

        if needs_flush:
            session.commit()

    # Sort: most recently used first, then by created_at desc for unused
    def _sort_key(f):
        used = last_used_map.get(f.name)
        return (0 if used else 1, used or "", -f.stat().st_mtime)

    files = sorted(files, key=_sort_key)

    result = []
    for f in files:
        stat = f.stat()
        result.append({
            "filename": f.name,
            "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "last_used_at": last_used_map.get(f.name),
        })
    return result

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
    disable_recipes: bool = False

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

    # Taxonomy tags (018) — fail-soft: log problems, don't block the launch.
    if isinstance(yaml_data.get("taxonomy"), dict):
        try:
            from gads.core import taxonomy as tx
            tax_result = tx.validate_tags(yaml_data["taxonomy"])
            for msg in tax_result["errors"]:
                print(f"[taxonomy] spec '{req.filename}': ERROR {msg}")
            for msg in tax_result["warnings"]:
                print(f"[taxonomy] spec '{req.filename}': warning {msg}")
        except Exception as e:
            print(f"[taxonomy] validation skipped for '{req.filename}': {e}")

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

    # 3. Upstream project (artifacts_from). Validated BEFORE the workspace is created so a
    # bad reference fails the launch cleanly rather than mid-transaction.
    upstream_id: Optional[uuid.UUID] = None
    if meta.artifacts_from:
        try:
            upstream_id = uuid.UUID(str(meta.artifacts_from))
        except ValueError:
            raise HTTPException(status_code=400,
                                detail=f"artifacts_from is not a UUID: {meta.artifacts_from}")
        with Session(engine) as _s:
            _up = _s.get(Project, upstream_id)
            if not _up:
                raise HTTPException(status_code=400,
                                    detail=f"artifacts_from project not found: {upstream_id}")
            # Require it to have produced something. A run still in flight would be linked
            # mid-write, and its manifest may not exist yet.
            _running = _s.exec(select(Task).where(Task.project_id == upstream_id,
                                                  Task.status.in_(["pending", "running"]))).first()
            if _running:
                raise HTTPException(
                    status_code=409,
                    detail=f"artifacts_from project {upstream_id} still has work in flight; "
                           "wait for it to finish.")
        if not os.path.isdir(f"{WORKSPACE_ROOT}/{upstream_id}"):
            raise HTTPException(status_code=400,
                                detail=f"artifacts_from workspace missing on disk: {upstream_id}")

    # Transactional Execution
    with Session(engine) as session:
        project_name = meta.name or f"Project {datetime.now().strftime('%m-%d %H:%M')} (from spec)"
        project = Project(name=project_name, objective=objective, last_state_json={"fast_mode": req.fast_mode, "disable_recipes": meta.disable_recipes or req.disable_recipes, "spec_filename": req.filename})
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

            # Upstream artifacts land in `upstream/`, after the datasets so a name clash
            # cannot overwrite one of them (they are in different directories anyway).
            if upstream_id is not None:
                _mount_upstream_artifacts(workspace_dir, upstream_id)
            
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
            "instructions": instructions,
            "taxonomy": (project.last_state_json or {}).get("taxonomy"),
            "taxonomy_source": (project.last_state_json or {}).get("taxonomy_source"),
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
            state["disable_recipes"] = req.disable_recipes
            project.last_state_json = state
            
            # Update objective if provided (for shell projects)
            if req.objective:
                project.objective = req.objective
            session.add(project)
            session.commit()
            session.refresh(project)
        else:
            project = Project(name=req.name, objective=req.objective, last_state_json={"fast_mode": req.fast_mode, "disable_recipes": req.disable_recipes})
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
    """Make an external dataset available in a project workspace.

    Datasets under `GADS_DATASETS_ROOT` are **symlinked**, not copied. That root is bind
    mounted into the sandbox READ-ONLY at the identical path, so the link resolves on both
    host and container while task code physically cannot write through it. This is what
    makes the link safe: the 2026-05 incident (generated code overwrote an original Kaggle
    train.csv) happened because the mount was writable, and copying was the workaround —
    at the cost of ~17GB of duplicated datasets across workspaces (one 144MB file had 89
    copies). The read-only mount removes the hazard without the duplication.

    Anything outside that root is still **copied**: an arbitrary host path is not mounted
    into the sandbox, so a symlink to it would dangle inside the container.
    """
    import shutil
    if not os.path.lexists(host_path):
        raise FileNotFoundError(f"Path not found: {host_path}")

    os.makedirs(workspace_dir, exist_ok=True)
    filename = os.path.basename(host_path)
    target_path = os.path.join(workspace_dir, filename)

    if os.path.lexists(target_path):
        if os.path.islink(target_path) or os.path.isfile(target_path):
            os.unlink(target_path) if os.path.islink(target_path) else os.remove(target_path)

    src = os.path.realpath(host_path)
    root = os.path.realpath(DATASETS_ROOT)
    inside_mounted_root = os.path.commonpath([src, root]) == root if os.path.isdir(root) else False

    if inside_mounted_root:
        try:
            os.symlink(src, target_path)
            print(f"  [Dataset] Linked '{filename}' (read-only source, no copy)", flush=True)
            return
        except Exception as e:
            print(f"  [Dataset] Symlink failed ({e}); falling back to copy.", flush=True)

    try:
        shutil.copy2(host_path, target_path)
        print(f"  [Dataset] Copied '{filename}' (outside the read-only datasets root)", flush=True)
    except Exception as e:
        raise Exception(f"Failed to copy dataset: {e}")

#

# Noise from the upstream run that a downstream task never needs: the rendered report,
# the replay notebook/script, and the spec/recipe copies. Data and manifests are what
# make a run reusable.
_UPSTREAM_SKIP = ("final_dashboard.html", "workflow_execution.ipynb",
                  "workflow_execution.py", "workflow_spec.md")


def _mount_upstream_artifacts(workspace_dir: str, upstream_id: uuid.UUID) -> List[str]:
    """Expose a finished project's artifacts inside `<workspace>/upstream/`.

    Linked with RELATIVE symlinks (`../../<upstream_id>/<file>`), which is what makes this
    work at all: the sandbox bind-mounts the whole workspaces root, but at a DIFFERENT path
    inside the container (`/app/workspaces`) than on the host. An absolute host symlink
    would therefore dangle in the sandbox — only the relative form resolves identically in
    both. (Datasets can use absolute links because that root is mounted at the same path;
    workspaces cannot.) Verified in-container before this was built.

    Mounted under `upstream/` rather than the workspace root so an upstream file can never
    shadow the downstream spec's own `datasets:` entry — an EDA workspace still contains the
    source CSV it profiled, so a flat mount would collide by construction.

    Read-only by convention, not by permission: these are symlinks into a live workspace, so
    downstream code MUST NOT write through them. The transform recipe writes its outputs to
    the downstream workspace root.
    """
    src_dir = f"{WORKSPACE_ROOT}/{upstream_id}"
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Upstream workspace not found: {upstream_id}")

    dest_dir = os.path.join(workspace_dir, "upstream")
    os.makedirs(dest_dir, exist_ok=True)

    linked: List[str] = []
    for name in sorted(os.listdir(src_dir)):
        if name in _UPSTREAM_SKIP or name.startswith("."):
            continue
        if not os.path.isfile(os.path.join(src_dir, name)):
            continue
        target = os.path.join(dest_dir, name)
        if os.path.lexists(target):
            os.unlink(target)
        os.symlink(os.path.join("..", "..", str(upstream_id), name), target)
        linked.append(name)

    print(f"  [Upstream] Linked {len(linked)} artifact(s) from {upstream_id} "
          f"into upstream/ ({', '.join(linked[:6])}{'…' if len(linked) > 6 else ''})",
          flush=True)
    return linked


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
