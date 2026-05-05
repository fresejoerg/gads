import asyncio
import uuid
import traceback
import json
import base64
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
from gads.core.bus import bus, dispatcher_loop
from gads.core.execution_hub import watchdog_loop, ExecutionHub
from gads.core.database import init_db, engine
from gads.core.models import Project, Task, Artifact, Instruction
from gads.agents.planner import DataSciencePlanner, PlannerInput, ReconciliationReport, FileMetadata
from gads.agents.router import DataScienceRouter, RouterInput
from gads.agents.workers.synthesizer import SynthesizerAgent, SynthesizerInput
from gads.agents.workers.critique import CritiqueAgent, CritiqueInput
from gads.core.executor import ExecutionManager
from gads.core.registry import get_model_hierarchy, get_local_only, set_local_only, get_random_routing, set_random_routing
from gads.core.knowledge import KnowledgeRegistry
from gads.core.reporting import create_master_reports
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

app = FastAPI(title="GADS Core API")
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

@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(dispatcher_loop())
    asyncio.create_task(watchdog_loop())
    asyncio.create_task(archive_cleanup_loop())

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

@app.get("/skills"
, response_model=List[str])
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

        executor = ExecutionManager()
        hierarchy = await get_model_hierarchy()

        # Check cancellation before start
        if await is_cancelled(): return

        # 0. SYNC GROUND TRUTH
        print(f"  [Workflow] Grounding session state...", flush=True)
        workspace_dir = f"{WORKSPACE_ROOT}/{project_id}"
        current_files_meta = _get_recursive_files(workspace_dir)

        # Dummy execute to trigger state introspection
        res = await executor.sandbox.execute("pass", project_id=project_id, session_id=str(project_id))
        if res.kernel_state:
            executor.authoritative_state.update(res.kernel_state)
            print(f"  [Workflow] Synchronized {len(executor.authoritative_state)} variables from kernel.", flush=True)
            with Session(engine) as session:
                project = session.get(Project, project_id)
                if project:
                    project.last_state_json = executor.authoritative_state
                    session.add(project)
                    session.commit()

        if await is_cancelled(): return

        # 1. ROUTING (Full Gemini Priority)
        router_fallback = ["local_model"] if get_local_only() else ["gemini-3.1-flash-lite-preview"]
        router_model = hierarchy.get("T3", {}).get("models", router_fallback)[0]
        
        # Select Critique model (Prefer T2 for quality, fallback to T3)
        critique_fallback = ["local_model"] if get_local_only() else ["gemini-3-flash-preview"]
        critique_model = hierarchy.get("T2", {}).get("models", hierarchy.get("T3", {}).get("models", critique_fallback))[0]

        router = DataScienceRouter(model=router_model)

        intent = None
        with Session(engine) as session:
            route_task = Task(
                project_id=project_id,
                instruction_id=instruction_id,
                description=f"Architect ({router_model}) is classifying intent and data modality...",
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

            router_res = await router.run(RouterInput(
                objective=objective,
                available_recipes=registry.get_recipes_summary()
            ))
            intent = router_res.content
            span.end(output=intent.model_dump())

            route_task.status = "completed"
            route_task.result_json = {
                "stdout": f"Task Type: {intent.task_type}\nModality: {intent.data_modality}\nMatched Recipe: {intent.matched_recipe_id}\nConfidence: {intent.confidence}",
                "model_used": router_model
            }
            session.add(route_task)
            session.commit()

        print(f"  [Router] Intent: {intent.task_type} (Recipe: {intent.matched_recipe_id})", flush=True)

        if await is_cancelled(): return

        # 2. KNOWLEDGE RETRIEVAL
        knowledge_report = None
        with Session(engine) as session:
            search_task = Task(
                project_id=project_id,
                instruction_id=instruction_id,
                description="Consulting Best Practices Wiki for matching Data Science recipes...",
                assigned_to="KnowledgeRegistry",
                status="running",
                heartbeat=datetime.now()
            )
            session.add(search_task)
            session.commit()
            session.refresh(search_task)

            # Logic shift: The Router (Architect) now explicitly picks the recipe ID
            recipe_id = intent.matched_recipe_id
            recipe = registry.get_recipe(recipe_id) if recipe_id else None

            if recipe:
                knowledge_report = ReconciliationReport(
                    recipe_id=recipe.id,
                    rationale=recipe.rationale,
                    recommended_dag_nodes=[node.dict() for node in recipe.dag],
                    skippable_nodes=[],
                    schema_warnings=[]
                )
                search_task.status = "completed"
                search_task.result_json = {"stdout": f"Applied SOP: {recipe.id}\nRationale: {recipe.rationale}"}
            else:
                search_task.status = "completed"
                search_task.result_json = {"stdout": "No specific recipe found. Proceeding with general data science reasoning."}
            session.add(search_task)
            session.commit()

        if await is_cancelled(): return

        # 3. PLANNING
        planner_fallback = ["local_model"] if get_local_only() else ["gemini-3.1-pro-preview"]
        planner_model = hierarchy.get("T1", {}).get("models", planner_fallback)[0]
        planner = DataSciencePlanner(model=planner_model)

        from gads.agents.planner import FileMetadata as FM
        planner_files = []
        for f in current_files_meta:
            columns_dtypes = None
            if f["name"].endswith(".csv") or f["name"].endswith(".parquet"):
                columns_dtypes = await _probe_file_schema(executor, project_id, f["name"])
            planner_files.append(FM(name=f["name"], size_mb=f["size_mb"], columns_and_dtypes=columns_dtypes))

        planner_res = None
        span = trace.span(name="Project Planning")
        trace_context.get().update({
            "agent_name": "Planner", 
            "task_id": "planning-phase", 
            "parent_observation_id": span.id
        })

        planner_res = await planner.run(PlannerInput(
            objective=objective,
            available_models_hierarchy=hierarchy,
            available_files=planner_files,
            knowledge_report=knowledge_report,
            available_skills=registry.get_skills_summary()
        ))
        span.end(output=planner_res.content.model_dump())

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

                tid_str = str(task_id)
                LIVE_STREAMS[tid_str] = {"reasoning": "", "stdout": ""}

                async def stream_reasoning_callback(token: str):
                    LIVE_STREAMS[tid_str]["reasoning"] += token

                async def stream_stdout_callback(text: str):
                    print(f"    [Workflow] Updating LIVE_STREAMS for {tid_str} ({len(text)} chars)", flush=True)
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
                    cancel_check=is_cancelled
                )

                # Update task span with result details
                task_span.update(output={"model": model_used, "code_len": len(res.code) if res.code else 0})

                with Session(engine) as session:
                    hub = ExecutionHub(session)
                    task_obj = session.get(Task, task_id)
                    error_msg = res.error.get("evalue", "Unknown error") if res.error else hub.validate_contract(task_obj, res.stdout, executor.authoritative_state)

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
                        hub.complete_task(task_id, {"stdout": res.stdout, "model_used": model_used, "code": res.code})
                        files_after = _get_recursive_files(workspace_dir)
                        hub.create_outbox_event("STATE_UPDATED", {"files": files_after, "state": executor.authoritative_state})

                        project = session.get(Project, project_id)
                        if project:
                            project.last_state_json = executor.authoritative_state
                            session.add(project)
                            session.commit()

                        for i, plot_b64 in enumerate(res.plots):
                            art = Artifact(project_id=project_id, type="plot", description=f"In-memory plot {i+1}", content_json={"image_base64": plot_b64}, agent_id="CodeGenerator")
                            session.add(art)
                            session.commit()
                            hub.create_outbox_event("ARTIFACT_CREATED", {"type": "plot", "description": art.description, "content_json": art.content_json})

                        new_files_names = set([f["name"] for f in files_after]) - set([f["name"] for f in current_files_meta])
                        for nf in new_files_names:
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
                            elif nf.lower().endswith(".html"):
                                try:
                                    art = Artifact(project_id=project_id, type="interactive_plot", description=f"Interactive: {nf}", content_json={"filename": nf}, agent_id="CodeGenerator")
                                    session.add(art)
                                    session.commit()
                                    hub.create_outbox_event("ARTIFACT_CREATED", {"type": "interactive_plot", "description": art.description, "content_json": art.content_json, "project_id": str(project_id)})
                                except Exception: pass

                        current_files_meta = files_after
                        task_span.end(output={"stdout": res.stdout, "model_used": model_used, "code": res.code})
                        session.commit()
                        break

            # HARD ABORT MECHANISM
            with Session(engine) as session:
                task_obj = session.get(Task, task_id)
                if task_obj and task_obj.status == "failed":
                    reason = f"Terminal failure in prerequisite task (ID: {task_id}). Halting workflow to prevent cascade failures."
                    print(f"    [Workflow] 🛑 HARD ABORT: {reason}", flush=True)
                    _mark_workflow_failed(session, project_id, reason)
                    return 

        if await is_cancelled(): return

        # 5. SYNTHESIS & CRITIQUE LOOP
        MAX_ATTEMPTS = 2
        attempt = 0
        final_synth = None
        redundant_plots = []

        while attempt < MAX_ATTEMPTS:
            attempt += 1
            print(f"  [Workflow] Synthesis attempt {attempt}/{MAX_ATTEMPTS}...", flush=True)

            with Session(engine) as session:
                synth_task = Task(
                    project_id=project_id, 
                    instruction_id=instruction_id,
                    description=f"Lead Data Scientist is synthesizing results (Attempt {attempt})...", 
                    assigned_to="Synthesizer", 
                    status="running",
                    heartbeat=datetime.now()
                )
                session.add(synth_task)
                session.commit()
                session.refresh(synth_task)

                span = trace.span(name=f"Synthesis Attempt {attempt}")
                trace_context.get().update({
                    "agent_name": "Synthesizer", 
                    "task_id": str(synth_task.id),
                    "parent_observation_id": span.id
                })

                all_tasks = session.exec(select(Task).where(Task.project_id == project_id)).all()
                context_parts = [f"Task: {t.description}\nStatus: {t.status}\nOutput: {(t.result_json or {}).get('stdout','')[:4000]}" for t in all_tasks]
                
                # Fetch actual artifacts to include in context
                artifacts = session.exec(select(Artifact).where(Artifact.project_id == project_id)).all()
                artifact_descriptions = [f"Artifact: {a.description} (Type: {a.type})" for a in artifacts]
                
                if artifact_descriptions:
                    context_parts.append("--- GENERATED ARTIFACTS ---\n" + "\n".join(artifact_descriptions))

                context = "\n\n---\n\n".join(context_parts)

                proj = session.get(Project, project_id)
                existing_narrative = proj.narrative if proj else None
                existing_takeaways = proj.takeaways if proj else None

                synthesizer = SynthesizerAgent(model=planner_model)
                synth_res = await synthesizer.run(SynthesizerInput(
                    objective=objective, 
                    context_artifacts=context,
                    existing_narrative=existing_narrative,
                    existing_takeaways=existing_takeaways
                ))
                final_synth = synth_res.content
                span.end(output=final_synth.model_dump())

                stask = session.get(Task, synth_task.id)
                if stask:
                    stask.status = "completed"
                    stask.result_json = {
                        "stdout": "Draft generated.", 
                        "narrative": final_synth.narrative, 
                        "takeaways": final_synth.key_takeaways,
                        "model_used": planner_model # Synthesizer currently uses planner_model
                    }
                session.commit()

            # Generate draft dashboard for Critique
            with Session(engine) as session:
                artifacts = session.exec(select(Artifact).where(Artifact.project_id == project_id)).all()
                draft_html = create_master_reports(
                    project_id=project_id, 
                    workspace_dir=workspace_dir, 
                    narrative=final_synth.narrative, 
                    takeaways=final_synth.key_takeaways, 
                    artifacts=artifacts,
                    artifact_insights=final_synth.artifact_insights
                )
            
            # Sanitize HTML for Critique context (strip scripts/data blobs)
            import re
            sanitized_html = re.sub(r'<script.*?>.*?</script>', '<script>/* JS OMITTED FOR BREVITY */</script>', draft_html, flags=re.DOTALL)
            if len(sanitized_html) > 100000:
                sanitized_html = sanitized_html[:50000] + "\n... [TRUNCATED] ...\n" + sanitized_html[-50000:]

            # 6. CRITIQUE
            if await is_cancelled(): return
            print(f"  [Workflow] Quality Assurance Critique (Attempt {attempt})...", flush=True)
            with Session(engine) as session:
                critique_task = Task(
                    project_id=project_id,
                    instruction_id=instruction_id,
                    description=f"QA Specialist is evaluating synthesis quality (Attempt {attempt})...",
                    assigned_to="Critique",
                    status="running",
                    heartbeat=datetime.now()
                )
                session.add(critique_task)
                session.commit()
                session.refresh(critique_task)

                span = trace.span(name=f"Critique Attempt {attempt}")
                trace_context.get().update({
                    "agent_name": "Critique",
                    "task_id": str(critique_task.id),
                    "parent_observation_id": span.id
                })

                critique_agent = CritiqueAgent(model=critique_model) # Use specialized model for critique
                critique_res = await critique_agent.run(CritiqueInput(
                    objective=objective,
                    context_artifacts=context,
                    synthesis_narrative=final_synth.narrative,
                    synthesis_takeaways=final_synth.key_takeaways,
                    dashboard_html=sanitized_html
                ))
                critique = critique_res.content
                span.end(output=critique.model_dump())

                ctask = session.get(Task, critique_task.id)
                if ctask:
                    ctask.status = "completed"
                    ctask.result_json = {
                        "stdout": f"Approved: {critique.is_approved}\nFeedback: {critique.critique_feedback}",
                        "model_used": critique_model
                    }
                session.commit()

                if critique.is_approved or attempt >= MAX_ATTEMPTS:
                    redundant_plots = critique.redundant_artifacts
                    break
                else:
                    # Inject feedback for next attempt
                    feedback_msg = f"\n\n--- CRITIQUE FEEDBACK (REJECTED) ---\n{critique.critique_feedback}\n\nPlease revise the narrative and takeaways to address these points."
                    with Session(engine) as session:
                        proj = session.get(Project, project_id)
                        if proj:
                            proj.narrative = (proj.narrative or "") + feedback_msg
                            session.add(proj)
                            session.commit()

        # 7. FINAL REPORTING & PRUNING
        with Session(engine) as session:
            reporting_task = Task(
                project_id=project_id,
                instruction_id=instruction_id,
                description="Publishing final dashboard and research reports...",
                assigned_to="System",
                status="running",
                heartbeat=datetime.now()
            )
            session.add(reporting_task)
            session.commit()
            session.refresh(reporting_task)
            hub = ExecutionHub(session)
            hub.create_outbox_event("TASK_CREATED", {"task_id": str(reporting_task.id), "description": reporting_task.description})
            session.commit()

            artifacts = session.exec(select(Artifact).where(Artifact.project_id == project_id)).all()
            
            # Pruning logic
            filtered_artifacts = []
            for a in artifacts:
                is_redundant = False
                # Check description and filename
                desc = a.description.lower()
                fname = a.content_json.get("filename", "").lower()
                for r in (redundant_plots or []):
                    r_low = r.lower()
                    if r_low in desc or (fname and r_low in fname):
                        is_redundant = True
                        break
                if not is_redundant:
                    filtered_artifacts.append(a)
                else:
                    print(f"  [Workflow] Pruning redundant artifact: {a.description}", flush=True)

            create_master_reports(
                project_id=project_id, 
                workspace_dir=workspace_dir, 
                narrative=final_synth.narrative, 
                takeaways=final_synth.key_takeaways, 
                artifacts=filtered_artifacts,
                artifact_insights=final_synth.artifact_insights
            )

            proj = session.get(Project, project_id)
            if proj:
                proj.narrative = final_synth.narrative
                proj.takeaways = final_synth.key_takeaways
            
            # Mark reporting task complete
            rtask = session.get(Task, reporting_task.id)
            if rtask:
                rtask.status = "completed"
                rtask.result_json = {"stdout": "Dashboard published successfully (QA warnings may apply)." if not critique.is_approved else "Dashboard published successfully."}
            session.commit()

            hub = ExecutionHub(session)
            hub.create_outbox_event("WORKFLOW_FINAL_RESULT", {"narrative": final_synth.narrative, "takeaways": final_synth.key_takeaways})
            hub.create_outbox_event("STATE_UPDATED", {"files": _get_recursive_files(workspace_dir), "state": executor.authoritative_state})
            hub.create_outbox_event("STEP_COMPLETED", {"message": "Project complete."})
            session.commit()

    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        with Session(engine) as session:
            _mark_workflow_failed(session, project_id, f"Fatal system error: {str(e)}")
    finally:
        from gads.core.llm import trace_context
        trace_context.reset(ctx_token)
        # Final flush to ensure everything is sent to Langfuse
        langfuse_client.flush()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, last_seq: int = 0):
    await bus.connect(websocket)
    if last_seq > 0: await bus.replay_events(websocket, last_seq)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: bus.disconnect(websocket)

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
            background_tasks.add_task(run_agent_workflow, project.id, req.objective, instr_id)

        # Fetch instructions to return
        instructions = session.exec(select(Instruction).where(Instruction.project_id == project.id).order_by(Instruction.created_at.asc())).all()

        return ProjectResponse(
            project=ProjectRead.from_orm(project), 
            files=[f["name"] for f in current_files],
            instructions=[InstructionRead.from_orm(i) for i in instructions]
        )
@app.post("/projects/{project_id}/register-external", response_model=ProjectResponse)
async def register_external_file(project_id: uuid.UUID, req: ExternalPathRequest):
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project: raise HTTPException(status_code=404, detail="Project not found")
        host_path = req.path
        if not os.path.lexists(host_path): raise HTTPException(status_code=400, detail="Path not found")
        
        workspace_dir = f"{WORKSPACE_ROOT}/{project_id}"
        os.makedirs(workspace_dir, exist_ok=True)
        filename = os.path.basename(host_path)
        target_path = os.path.join(workspace_dir, filename)
        
        try:
            if os.path.lexists(target_path):
                if os.path.islink(target_path): os.unlink(target_path)
                else: os.remove(target_path)
            
            os.symlink(host_path, target_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create symlink: {e}")
        
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

async def _probe_file_schema(executor: ExecutionManager, project_id: uuid.UUID, filename: str) -> Optional[Dict[str, str]]:
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
    elif "{filename}".endswith(".parquet"):
        res = duckdb.query("DESCRIBE SELECT * FROM '{filename}'").to_df()
    else:
        print(json.dumps({{"error": "Unsupported format"}}))
        exit()
    
    # Map DuckDB types to simple string descriptions
    schema = dict(zip(res['column_name'], res['column_type']))
    print(json.dumps(schema))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
    try:
        # 30s timeout for schema probe
        res = await asyncio.wait_for(
            executor.sandbox.execute(code, project_id=project_id, session_id=str(project_id)),
            timeout=30.0
        )
        if res.stdout:
            data = json.loads(res.stdout.strip().split('\n')[-1])
            if "error" not in data:
                print(f"    [Introspection] Detected schema: {data}", flush=True)
                return data
    except Exception as e:
        print(f"    [Introspection] Probe failed for {filename}: {e}", flush=True)
    return None

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
