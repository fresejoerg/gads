import asyncio
import uuid
import traceback
import json
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
from gads.agents.planner import DataSciencePlanner, PlannerInput, ReconciliationReport, FileMetadata
from gads.agents.router import DataScienceRouter, RouterInput
from gads.agents.plan_critique import PlanCritiqueAgent, PlanCritiqueInput
from gads.agents.workers.synthesizer import SynthesizerAgent, SynthesizerInput
from gads.agents.workers.critique import CritiqueAgent, CritiqueInput
from gads.core.executor import ExecutionManager
from gads.core.registry import get_model_hierarchy, get_local_only, set_local_only, get_random_routing, set_random_routing, get_next_model_dynamic
from gads.core.knowledge import KnowledgeRegistry
from gads.core.reporting import create_master_reports
from gads.core.introspection import summarize_artifact
from gads.core.distiller import distill_dashboard_to_markdown
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
        executor = ExecutionManager()
        await executor.sandbox.reset_session(str(project_id))

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
        hierarchy = await get_model_hierarchy()

        # Check cancellation before start
        if await is_cancelled(): return

        # 0. SYNC GROUND TRUTH
        print(f"  [Workflow] Grounding session state...", flush=True)
        workspace_dir = f"{WORKSPACE_ROOT}/{project_id}"
        current_files_meta = _get_recursive_files(workspace_dir)

        # Dummy execute to trigger state introspection
        try:
            res = await asyncio.wait_for(executor.sandbox.execute("pass", project_id=project_id, session_id=str(project_id)), timeout=10.0)
            if res.kernel_state:
                executor.authoritative_state.update(res.kernel_state)
                print(f"  [Workflow] Synchronized {len(executor.authoritative_state)} variables from kernel.", flush=True)
                with Session(engine) as session:
                    project = session.get(Project, project_id)
                    if project:
                        project.last_state_json = executor.authoritative_state
                        session.add(project)
                        session.commit()
        except Exception as e:
            print(f"  [Workflow] Grounding failed or timed out: {e}. Proceeding with empty state.", flush=True)

        # Mark init task as completed
        with Session(engine) as session:
            itask = session.get(Task, init_task_id)
            if itask:
                itask.status = "completed"
                session.add(itask)
                session.commit()

                if await is_cancelled(): return

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
                        objective=objective,
                        available_recipes=registry.get_recipes_summary()
                    ), stream_callback=stream_router_callback)
                    intent = router_res.content
                    span.end(output=intent.model_dump())

                    # Inline Knowledge Retrieval
                    recipe_id = intent.matched_recipe_id
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

        # --- MAIN WORKFLOW LOOP (Planning -> Execution -> Synthesis -> Critique) ---
        MAX_WORKFLOW_ATTEMPTS = 3
        workflow_attempt = 0
        critique_feedback = None
        final_synth = None
        redundant_plots = []
        
        while workflow_attempt < MAX_WORKFLOW_ATTEMPTS:
            workflow_attempt += 1
            print(f"\n  [Workflow] Starting attempt {workflow_attempt}/{MAX_WORKFLOW_ATTEMPTS}...", flush=True)
            
            if await is_cancelled(): return

            # 3. PLANNING (Resilient Decomposition)
            planner_fallback = ["local_model"] if get_local_only() else ["gemini-3-flash-preview"]
            planner_model = hierarchy.get("T2", {}).get("models", planner_fallback)[0]
            
            from gads.agents.planner import FileMetadata as FM
            planner_files = []
            detected_schemas = {}
            for f in current_files_meta:
                columns_dtypes = None
                if f["name"].endswith(".csv") or f["name"].endswith(".parquet"):
                    columns_dtypes = await _probe_file_schema(executor, project_id, f["name"])
                    if columns_dtypes:
                        detected_schemas[f["name"]] = columns_dtypes
                planner_files.append(FM(name=f["name"], size_mb=f["size_mb"], columns_and_dtypes=columns_dtypes))

            if detected_schemas:
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
                            objective=objective,
                            available_models_hierarchy=hierarchy,
                            available_files=planner_files,
                            knowledge_report=knowledge_report,
                            available_skills=registry.get_skills_summary(),
                            critique_feedback=critique_feedback
                        ), stream_callback=stream_planner_callback)
                        span.end(output=planner_res.content.model_dump())

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
                for step in planner_res.content.steps:
                    new_task = Task(
                        project_id=project_id,
                        instruction_id=instruction_id,
                        description=step.description,
                        assigned_to=step.assigned_to,
                        postcondition_json=step.postcondition,
                        attached_skills=step.attached_skills,
                        status="pending"
                    )
                    session.add(new_task)
                    tasks_to_run.append(new_task)
                session.commit()

                for t in tasks_to_run:
                    session.refresh(t)
                    hub.create_outbox_event("TASK_CREATED", {"task_id": str(t.id), "description": t.description})
                session.commit()
                task_ids = [t.id for t in tasks_to_run]

            # 4. EXECUTION
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
                            desc = task_obj.description
                        else: break 

                    # Dynamic Skill Discovery (Planner-led + Keyword Fallback)
                    assigned_skills = task_obj.attached_skills or []
                    matched_skills = registry.find_skills(desc)

                    # Deduplicate and Fetch Full Content
                    all_skill_ids = list(set(assigned_skills + [s.id for s in matched_skills]))
                    skills_to_inject = []
                    for sid in all_skill_ids:
                        s_obj = registry.skills.get(sid)
                        if s_obj: skills_to_inject.append(s_obj)

                    skills_ctx = None
                    if skills_to_inject:
                        skills_ctx = "\n\n".join([f"### Skill: {s.id}\n{s.content}" for s in skills_to_inject])
                        print(f"    [Workflow] Applied skills: {[s.id for s in skills_to_inject]}", flush=True)

                    # 4.1 CONTEXT HARDENING: SLIDING WINDOW & INTROSPECTION
                    all_tasks = session.exec(select(Task).where(Task.project_id == project_id).order_by(Task.created_at.asc())).all()
                    
                    # Sliding Window (2+1 Model: First Task + Last 2 Tasks)
                    # All others are reduced to their metadata summary
                    hardened_context_parts = []
                    for i, t in enumerate(all_tasks):
                        # Determine if this task should be in full detail
                        is_first = (i == 0)
                        is_recent = (i >= len(all_tasks) - 2)
                        
                        stdout_str = (t.result_json or {}).get("stdout", "")
                        code_str = (t.result_json or {}).get("code", "")
                        
                        if is_first or is_recent:
                            # Full detail
                            task_ctx = f"### TASK: {t.description}\nStatus: {t.status}\nCode Used:\n```python\n{code_str}\n```\nStdout Output:\n{stdout_str[:2000]}"
                        else:
                            # Distilled metadata only
                            summary = (t.result_json or {}).get("orchestrator_summary", f"Completed: {t.description}")
                            task_ctx = f"### TASK (HISTORICAL): {t.description}\nStatus: {t.status}\nSummary: {summary}"
                        
                        hardened_context_parts.append(task_ctx)

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
        elif hasattr(_obj, 'predict') and hasattr(_obj, 'fit'):
            _summary[_v] = f"Model ({type(_obj).__name__})"
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

                    # Authoritative prompt building
                    context = "\n\n---\n\n".join(hardened_context_parts)
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

                    res, model_used = await executor.run_task(
                        desc, 
                        project_id=project_id, 
                        session_id=str(project_id), 
                        skills_context=skills_ctx, 
                        task_id=task_id,
                        stdout_callback=stream_stdout_callback,
                        stream_callback=stream_reasoning_callback,
                        cancel_check=is_cancelled,
                        state_summary=state_summary_str # Pass the ground truth
                    )

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
                        error_msg = res.error.get("evalue", "Unknown error") if res.error else hub.validate_contract(task_obj, res.stdout, executor.authoritative_state)

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
                    context_parts = []
                    for t in all_tasks:
                         summary = (t.result_json or {}).get("orchestrator_summary", f"Completed: {t.description}")
                         context_parts.append(f"Task: {t.description}\nSummary: {summary}\nOutput: {(t.result_json or {}).get('stdout','')[:2000]}")
                    
                    artifacts = session.exec(select(Artifact).where(Artifact.project_id == project_id)).all()
                    artifact_summaries = []
                    for a in artifacts:
                        fname = a.content_json.get("filename")
                        fpath = os.path.join(workspace_dir, fname) if fname else None
                        summary = summarize_artifact(fpath) if fpath else f"Artifact: {a.description}"
                        artifact_summaries.append(summary)
                    
                    context = "\n\n---\n\n".join(context_parts) + "\n\n--- ARTIFACT SUMMARIES ---\n" + "\n".join(artifact_summaries)

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
                break
            else:
                critique_feedback = critique.critique_feedback

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
        project = Project(name=project_name, objective=objective)
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
            # Update objective if provided (for shell projects)
            if req.objective:
                project.objective = req.objective
                session.add(project)
                session.commit()
                session.refresh(project)
        else:
            project = Project(name=req.name, objective=req.objective)
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
    """Securely symlinks an external dataset into the workspace."""
    if not os.path.lexists(host_path):
        raise FileNotFoundError(f"Path not found: {host_path}")
        
    os.makedirs(workspace_dir, exist_ok=True)
    filename = os.path.basename(host_path)
    target_path = os.path.join(workspace_dir, filename)
    
    try:
        if os.path.lexists(target_path):
            if os.path.islink(target_path): os.unlink(target_path)
            else: os.remove(target_path)
        
        os.symlink(host_path, target_path)
    except Exception as e:
        raise Exception(f"Failed to create symlink: {e}")

async def _probe_file_schema(executor: ExecutionManager, project_id: uuid.UUID, filename: str) -> Optional[Dict[str, Any]]:
    """Safe, time-capped schema and dtype extraction using DuckDB."""
    print(f"    [Introspection] Probing schema for {filename}...", flush=True)
    
    # Strictly capped DuckDB query for safe type inference
    code = f"""
import duckdb
import json
try:
    # Use read_csv_auto with limited sample for speed and safety
    if "{filename}".endswith(".csv"):
        res = duckdb.query("DESCRIBE SELECT * FROM read_csv_auto('{filename}', sample_size=1024)").to_df()
        sample_df = duckdb.query("SELECT * FROM read_csv_auto('{filename}', sample_size=1024) LIMIT 5").to_df()
    elif "{filename}".endswith(".parquet"):
        res = duckdb.query("DESCRIBE SELECT * FROM '{filename}'").to_df()
        sample_df = duckdb.query("SELECT * FROM '{filename}' LIMIT 5").to_df()
    else:
        print(json.dumps({{"error": "Unsupported format"}}))
        exit()
    
    # Map DuckDB types to simple string descriptions
    schema = dict(zip(res['column_name'], res['column_type']))
    
    # Convert sample to json-friendly records
    sample_json_str = sample_df.to_json(orient="records", date_format="iso")
    sample_records = json.loads(sample_json_str)

    print(json.dumps({{"schema": schema, "sample": sample_records}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
    try:
        # 30s timeout for schema probe, using a dedicated session to avoid interfering with project kernel
        res = await asyncio.wait_for(
            executor.sandbox.execute(code, project_id=project_id, session_id=f"probe_{project_id}"),
            timeout=30.0
        )
        if res.stdout:
            # Parse the last line as JSON to handle any pre-output
            lines = res.stdout.strip().split('\\n')
            data = None
            for line in reversed(lines):
                try:
                    data = json.loads(line)
                    break
                except Exception: continue
                
            if data and "error" not in data:
                print(f"    [Introspection] Detected schema for {filename}", flush=True)
                return data
    except Exception as e:
        print(f"    [Introspection] Probe failed for {filename}: {e}", flush=True)
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
