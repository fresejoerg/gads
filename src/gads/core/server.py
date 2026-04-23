import asyncio
import uuid
import traceback
import json
import base64
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
from gads.core.bus import bus, dispatcher_loop
from gads.core.execution_hub import watchdog_loop, ExecutionHub
from gads.core.database import init_db, engine
from gads.core.models import Project, Task, Artifact
from gads.agents.planner import DataSciencePlanner, PlannerInput, ReconciliationReport
from gads.agents.router import DataScienceRouter, RouterInput
from gads.agents.workers.synthesizer import SynthesizerAgent, SynthesizerInput
from gads.core.executor import ExecutionManager
from gads.core.registry import get_model_hierarchy, GADS_LOCAL_ONLY
from gads.core.knowledge import KnowledgeRegistry
from gads.core.reporting import create_master_reports
from sqlmodel import select, Session
import base64
import os
import asyncio
import traceback

app = FastAPI(title="GADS Core API")
registry = KnowledgeRegistry("src/gads/knowledge/recipes")
WORKSPACE_ROOT = "/home/joergf/projects/MyLocalStack/data/workspaces"

# --- RESPONSE MODELS ---
class ProjectRead(BaseModel):
    id: uuid.UUID
    name: str
    objective: str
    narrative: Optional[str] = None
    takeaways: Optional[List[str]] = None
    last_state_json: Optional[Dict[str, Any]] = None
    created_at: datetime
    has_dashboard: bool = False
    has_report: bool = False

    class Config:
        from_attributes = True

class ProjectResponse(BaseModel):
    project: ProjectRead
    files: List[str] = []

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

@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(dispatcher_loop())
    asyncio.create_task(watchdog_loop())
    asyncio.create_task(archive_cleanup_loop())

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

def _get_recursive_files(workspace_dir: str) -> List[str]:
    """Helper to list all files in workspace recursively, relative to root."""
    all_files = []
    for root, _, files in os.walk(workspace_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, workspace_dir)
            all_files.append(rel_path)
    return sorted(all_files)

async def run_agent_workflow(project_id: uuid.UUID, objective: str):
    """Orchestrates the multi-agent workflow with Full Gemini Priority Strategy."""
    print(f"\n--- 🚀 Starting expert workflow for Project {project_id} ---", flush=True)
    
    try:
        executor = ExecutionManager()
        hierarchy = await get_model_hierarchy()
        
        # 0. SYNC GROUND TRUTH
        print(f"  [Workflow] Grounding session state...", flush=True)
        workspace_dir = f"{WORKSPACE_ROOT}/{project_id}"
        current_files = _get_recursive_files(workspace_dir)
        
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

        # 1. ROUTING (Full Gemini Priority)
        # Update Routing fallback to Gemini 3.1 Flash Lite
        router_fallback = ["local_model"] if GADS_LOCAL_ONLY else ["gemini-3.1-flash-lite-preview"]
        router_model = hierarchy.get("T3", {}).get("models", router_fallback)[0]
        router = DataScienceRouter(model=router_model)
        
        with Session(engine) as session:
            # Create Routing Task
            route_task = Task(
                project_id=project_id,
                description=f"Architect ({router_model}) is classifying intent and data modality...",
                assigned_to="Router",
                status="running"
            )
            session.add(route_task)
            session.commit()
            session.refresh(route_task)
            
            router_res = await router.run(RouterInput(objective=objective))
            intent = router_res.content
            
            route_task.status = "completed"
            route_task.result_json = {"stdout": f"Task Type: {intent.task_type}\nModality: {intent.data_modality}\nConfidence: {intent.confidence}"}
            session.add(route_task)
            session.commit()
            
        print(f"  [Router] Intent: {intent.task_type} on {intent.data_modality} (Conf: {intent.confidence})", flush=True)

        # 2. KNOWLEDGE RETRIEVAL
        knowledge_report = None
        
        with Session(engine) as session:
            # Create a task for Knowledge Retrieval visibility
            search_task = Task(
                project_id=project_id,
                description="Consulting Best Practices Wiki for matching Data Science recipes...",
                assigned_to="KnowledgeRegistry",
                status="running"
            )
            session.add(search_task)
            session.commit()
            session.refresh(search_task)
            
            matches = registry.find_matches({"task_type": intent.task_type, "data_modality": intent.data_modality})
            
            if matches and intent.confidence > 0.7:
                recipe = matches[0]
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

        # 3. PLANNING (Full Gemini Priority)
        # Update Planning fallback to Gemini 3.1 Pro
        planner_fallback = ["local_model"] if GADS_LOCAL_ONLY else ["gemini-3.1-pro-preview"]
        planner_model = hierarchy.get("T1", {}).get("models", planner_fallback)[0]
        planner = DataSciencePlanner(model=planner_model)
        
        with Session(engine) as session:
            hub = ExecutionHub(session)
            hub.create_outbox_event("STEP_STARTED", {"message": f"Project Manager ({planner_model}) is generating DAG..."})
            session.commit()
        
        planner_res = await planner.run(PlannerInput(
            objective=objective,
            available_models_hierarchy=hierarchy,
            available_files=current_files,
            knowledge_report=knowledge_report
        ))
        
        tasks_to_run = []
        with Session(engine) as session:
            hub = ExecutionHub(session)
            for step in planner_res.content.steps:
                new_task = Task(
                    project_id=project_id,
                    description=step.description,
                    assigned_to=step.assigned_to,
                    postcondition_json=step.postcondition,
                    status="pending"
                )
                session.add(new_task)
                tasks_to_run.append(new_task)
            session.commit()
            
            for t in tasks_to_run:
                session.refresh(t)
                hub.create_outbox_event("TASK_CREATED", {
                    "task_id": str(t.id),
                    "description": t.description
                })
            session.commit()
            task_ids = [t.id for t in tasks_to_run]

        # 4. EXECUTION
        for task_id in task_ids:
            while True:
                desc, assigned_to = "", ""
                with Session(engine) as session:
                    hub = ExecutionHub(session)
                    task_obj = session.get(Task, task_id)
                    if not task_obj: break
                    
                    if hub.claim_task(task_id):
                        executor.coder.model = task_obj.assigned_to
                        desc = task_obj.description
                    else:
                        break 
                
                res, model_used = await executor.run_task(
                    desc, project_id=project_id, session_id=str(project_id)
                )
                
                with Session(engine) as session:
                    hub = ExecutionHub(session)
                    task_obj = session.get(Task, task_id)
                    error_msg = None
                    if res.error:
                        error_msg = res.error.get("evalue", "Unknown error")
                    else:
                        error_msg = hub.validate_contract(task_obj, res.stdout)

                    if error_msg:
                        if hub.escalate_task(task_id, error_msg, hierarchy):
                            session.commit()
                            continue 
                        else:
                            hub.fail_task(task_id, error_msg)
                            session.commit()
                            break
                    else:
                        hub.complete_task(task_id, {"stdout": res.stdout, "model_used": model_used})
                        files_after = _get_recursive_files(workspace_dir)
                        hub.create_outbox_event("STATE_UPDATED", {
                            "files": files_after,
                            "state": executor.authoritative_state
                        })
                        
                        # Sync state to DB Project object for durable memory
                        with Session(engine) as session:
                            project = session.get(Project, project_id)
                            if project:
                                project.last_state_json = executor.authoritative_state
                                session.add(project)
                                session.commit()
                        
                        # 1. Capture in-memory plots (Matplotlib/Plotly display_data)
                        for i, plot_b64 in enumerate(res.plots):
                            art = Artifact(
                                project_id=project_id,
                                type="plot",
                                description=f"In-memory visualization {i+1} for: {task_obj.description[:50]}",
                                content_json={"image_base64": plot_b64},
                                agent_id="CodeGenerator"
                            )
                            session.add(art)
                            session.commit()
                            session.refresh(art)
                            hub.create_outbox_event("ARTIFACT_CREATED", {
                                "type": "plot",
                                "description": art.description,
                                "content_json": art.content_json
                            })

                        # 2. Capture newly created disk artifacts (Plotly write_image / write_html)
                        new_files = set(files_after) - set(current_files)
                        for nf in new_files:
                            full_path = os.path.join(workspace_dir, nf)
                            
                            # PNG: Convert to base64 for inline display
                            if nf.lower().endswith(".png"):
                                try:
                                    with open(full_path, "rb") as img_file:
                                        img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                                    
                                    art = Artifact(
                                        project_id=project_id,
                                        type="plot",
                                        description=f"Workspace artifact: {nf}",
                                        content_json={"image_base64": img_b64},
                                        agent_id="CodeGenerator"
                                    )
                                    session.add(art)
                                    session.commit()
                                    session.refresh(art)
                                    hub.create_outbox_event("ARTIFACT_CREATED", {
                                        "type": "plot",
                                        "description": art.description,
                                        "content_json": art.content_json
                                    })
                                except Exception as e:
                                    print(f"  [Workflow] Warning: Failed to capture PNG artifact {nf}: {e}")
                            
                            # HTML: Record as interactive plot
                            elif nf.lower().endswith(".html"):
                                try:
                                    art = Artifact(
                                        project_id=project_id,
                                        type="interactive_plot",
                                        description=f"Interactive visualization: {nf}",
                                        content_json={"filename": nf},
                                        agent_id="CodeGenerator"
                                    )
                                    session.add(art)
                                    session.commit()
                                    session.refresh(art)
                                    hub.create_outbox_event("ARTIFACT_CREATED", {
                                        "type": "interactive_plot",
                                        "description": art.description,
                                        "content_json": art.content_json,
                                        "project_id": str(project_id) # Needed for the link
                                    })
                                except Exception as e:
                                    print(f"  [Workflow] Warning: Failed to capture HTML artifact {nf}: {e}")

                        # Update ground truth for next task
                        current_files = files_after
                        session.commit()
                        break

        # 5. SYNTHESIS (Full Gemini Priority)
        with Session(engine) as session:
            # Create Synthesis Task
            synth_task = Task(
                project_id=project_id,
                description="Lead Data Scientist is synthesizing final results into a research report...",
                assigned_to="Synthesizer",
                status="running"
            )
            session.add(synth_task)
            session.commit()
            session.refresh(synth_task)

            artifacts = session.exec(select(Artifact).where(Artifact.project_id == project_id)).all()
            all_tasks = session.exec(select(Task).where(Task.project_id == project_id)).all()

            context_parts: List[str] = []
            for t in all_tasks:
                header = f"Task: {t.description}\nStatus: {t.status}"
                if t.status == "completed":
                    stdout = (t.result_json or {}).get("stdout", "").strip()
                    snippet = stdout if len(stdout) < 4000 else stdout[:4000] + "\n...[truncated]"
                    context_parts.append(f"{header}\nOutput:\n{snippet or '(empty stdout)'}")
                elif t.status == "failed":
                    err = (t.error or "unknown error").strip()
                    context_parts.append(f"{header}\nError:\n{err[:2000]}")
                else:
                    context_parts.append(header)
            for a in artifacts:
                filename = a.content_json.get("filename", "N/A")
                context_parts.append(f"Artifact {a.id} ({a.type}): {a.description} (File: {filename})")

            context = "\n\n---\n\n".join(context_parts) if context_parts else "(no outputs)"

        # Use same model as planner for final synthesis (which now defaults to Gemini)
        synthesizer = SynthesizerAgent(model=planner_model)
        synth_res = await synthesizer.run(SynthesizerInput(objective=objective, context_artifacts=context))
        
        # --- NEW: SYSTEM-LED MASTER REPORTING ---
        print(f"  [Workflow] Assembling final research report and dashboard...", flush=True)
        create_master_reports(
            project_id=project_id,
            workspace_dir=workspace_dir,
            narrative=synth_res.content.narrative,
            takeaways=synth_res.content.key_takeaways,
            artifacts=artifacts
        )

        with Session(engine) as session:
            # Update Synthesis Task status
            synth_task_db = session.get(Task, synth_task.id)
            if synth_task_db:
                synth_task_db.status = "completed"
                synth_task_db.result_json = {"stdout": "Narrative generated and master documents assembled.", "narrative": synth_res.content.narrative, "takeaways": synth_res.content.key_takeaways}
                session.add(synth_task_db)

            project = session.get(Project, project_id)
            if project:
                project.narrative = synth_res.content.narrative
                project.takeaways = synth_res.content.key_takeaways
                session.add(project)
            
            session.commit()

            hub = ExecutionHub(session)
            hub.create_outbox_event("WORKFLOW_FINAL_RESULT", {
                "narrative": synth_res.content.narrative,
                "takeaways": synth_res.content.key_takeaways
            })
            
            # Emit an extra update so the UI sees the new master files immediately
            current_files = _get_recursive_files(workspace_dir)
            hub.create_outbox_event("STATE_UPDATED", {"files": current_files, "state": executor.authoritative_state})
            
            hub.create_outbox_event("STEP_COMPLETED", {"message": "Project complete. Reports generated."})
            session.commit()
            
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        traceback.print_exc()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, last_seq: int = 0):
    await bus.connect(websocket)
    if last_seq > 0:
        await bus.replay_events(websocket, last_seq)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        bus.disconnect(websocket)

@app.get("/projects/{project_id}", response_model=Dict[str, Any])
async def get_project_details(project_id: uuid.UUID):
    """Returns full project details including tasks and artifacts."""
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        tasks = session.exec(select(Task).where(Task.project_id == project_id)).all()
        artifacts = session.exec(select(Artifact).where(Artifact.project_id == project_id)).all()
        
        return {
            "project": ProjectRead.from_orm(project),
            "tasks": tasks,
            "artifacts": artifacts
        }

@app.get("/projects", response_model=List[ProjectRead])
async def list_projects():
    """Lists all past and current projects with artifact flags."""
    with Session(engine) as session:
        projects = session.exec(select(Project).order_by(Project.created_at.desc())).all()
        results = []
        for p in projects:
            p_data = ProjectRead.from_orm(p)
            workspace_dir = f"{WORKSPACE_ROOT}/{p.id}"
            p_data.has_dashboard = os.path.exists(f"{workspace_dir}/final_dashboard.html")
            p_data.has_report = os.path.exists(f"{workspace_dir}/research_report.md")
            results.append(p_data)
        return results

@app.post("/projects", response_model=ProjectResponse)
async def create_project(req: ProjectCreateRequest, background_tasks: BackgroundTasks):
    with Session(engine) as session:
        hub = ExecutionHub(session)
        if req.existing_project_id:
            project = session.get(Project, uuid.UUID(req.existing_project_id))
            if not project:
                raise HTTPException(status_code=404, detail="Session project not found")
        else:
            project = Project(name=req.name, objective=req.objective)
            session.add(project)
            session.commit()
            session.refresh(project)
        
        workspace_dir = f"{WORKSPACE_ROOT}/{project.id}"
        os.makedirs(workspace_dir, exist_ok=True)
        if req.files:
            for f in req.files:
                safe_name = os.path.basename(f.name)
                file_path = os.path.join(workspace_dir, safe_name)
                with open(file_path, "wb") as out:
                    out.write(base64.b64decode(f.content_base64))
        
        current_files = _get_recursive_files(workspace_dir)
        hub.create_outbox_event("STATE_UPDATED", {"files": current_files, "state": {}})
        session.commit()
        
        if req.objective.strip():
            background_tasks.add_task(run_agent_workflow, project.id, req.objective)
            
        return ProjectResponse(project=project, files=current_files)

@app.post("/projects/{project_id}/files", response_model=ProjectResponse)
async def upload_files_to_project(project_id: uuid.UUID, req: FilesUploadRequest):
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        workspace_dir = f"{WORKSPACE_ROOT}/{project.id}"
        os.makedirs(workspace_dir, exist_ok=True)
        for f in req.files:
            safe_name = os.path.basename(f.name)
            file_path = os.path.join(workspace_dir, safe_name)
            with open(file_path, "wb") as out:
                out.write(base64.b64decode(f.content_base64))
        
        current_files = _get_recursive_files(workspace_dir)
        hub = ExecutionHub(session)
        hub.create_outbox_event("STATE_UPDATED", {"files": current_files, "state": {}})
        session.commit()
        return ProjectResponse(project=project, files=current_files)

@app.post("/projects/{project_id}/register-external", response_model=ProjectResponse)
async def register_external_file(project_id: uuid.UUID, req: ExternalPathRequest):
    """Registers an external host file by creating a symlink in the workspace."""
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        host_path = req.path
        if not os.path.lexists(host_path):
            raise HTTPException(status_code=400, detail=f"Host path does not exist: {host_path}")
            
        workspace_dir = f"{WORKSPACE_ROOT}/{project_id}"
        os.makedirs(workspace_dir, exist_ok=True)
        
        filename = os.path.basename(host_path)
        target_path = os.path.join(workspace_dir, filename)
        
        try:
            # Remove existing file/symlink if it exists
            if os.path.lexists(target_path):
                if os.path.islink(target_path):
                    os.unlink(target_path)
                else:
                    os.remove(target_path)
                    
            os.symlink(host_path, target_path)
            print(f"  [Server] Registered external file: {host_path} -> {target_path}", flush=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create symlink: {e}")
            
        current_files = _get_recursive_files(workspace_dir)
        hub = ExecutionHub(session)
        hub.create_outbox_event("STATE_UPDATED", {"files": current_files, "state": project.last_state_json or {}})
        session.commit()
        return ProjectResponse(project=ProjectRead.from_orm(project), files=current_files)

@app.get("/projects/{project_id}/files/{file_path:path}")
async def download_file(project_id: uuid.UUID, file_path: str, download: bool = False):
    """Serves a file from the workspace for viewing or download."""
    workspace_dir = f"{WORKSPACE_ROOT}/{project_id}"
    full_path = os.path.join(workspace_dir, file_path)
    
    # Security: Ensure the file is inside the workspace
    if not os.path.abspath(full_path).startswith(os.path.abspath(workspace_dir)):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    if download:
        return FileResponse(full_path, filename=os.path.basename(full_path))
    return FileResponse(full_path)

@app.get("/health")
def health():
    return {"status": "ok"}
