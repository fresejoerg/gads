import asyncio
import uuid
import traceback
import json
import base64
import os
from typing import List, Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel
from gads.core.bus import bus, dispatcher_loop
from gads.core.execution_hub import watchdog_loop, ExecutionHub
from gads.core.database import init_db, engine
from gads.core.models import Project, Task, Artifact
from gads.agents.planner import DataSciencePlanner, PlannerInput
from gads.agents.workers.synthesizer import SynthesizerAgent, SynthesizerInput
from gads.core.executor import ExecutionManager
from gads.core.registry import get_model_hierarchy
from sqlmodel import select, Session

app = FastAPI(title="GADS Core API")

class FileUpload(BaseModel):
    name: str
    content_base64: str

class ProjectCreateRequest(BaseModel):
    name: str
    objective: str
    files: List[FileUpload] = []

@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(dispatcher_loop())
    asyncio.create_task(watchdog_loop())

async def run_agent_workflow(project_id: uuid.UUID, objective: str):
    """Orchestrates the multi-agent workflow with Dynamic Routing & Escalation."""
    print(f"\n--- 🚀 Starting Workflow for Project {project_id} ---", flush=True)
    
    try:
        executor = ExecutionManager()
        hierarchy = await get_model_hierarchy()
        
        # 1. Planning
        t1_models = hierarchy.get("T1", {}).get("models", ["claude-opus-4.7"])
        planner_model = t1_models[0]
        planner = DataSciencePlanner(model=planner_model)
        
        print(f"  [Planner] Analyzing objective with {planner_model}...", flush=True)
        with Session(engine) as session:
            hub = ExecutionHub(session)
            hub.create_outbox_event("STEP_STARTED", {"message": f"Project Manager ({planner_model}) is planning the objective..."})
            session.commit()
        
        planner_res = await planner.run(PlannerInput(
            objective=objective,
            available_models_hierarchy=hierarchy
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

        print(f"  [Workflow] Saved {len(task_ids)} tasks.", flush=True)

        # 2. Sequential Execution
        for task_id in task_ids:
            while True:
                desc, assigned_to = "", ""
                with Session(engine) as session:
                    hub = ExecutionHub(session)
                    task_obj = session.get(Task, task_id)
                    if not task_obj: break
                    
                    print(f"  [Executor] Attempting claim for task: {task_id} ({task_obj.assigned_to})")
                    if hub.claim_task(task_id):
                        executor.coder.model = task_obj.assigned_to
                        desc = task_obj.description
                        assigned_to = task_obj.assigned_to
                    else:
                        break # Already claimed or completed
                
                # Run task OUTSIDE session
                res, model_used = await executor.run_task(
                    desc, project_id=project_id, session_id=str(project_id)
                )
                
                # Update status
                with Session(engine) as session:
                    hub = ExecutionHub(session)
                    task_obj = session.get(Task, task_id)
                    
                    error_msg = None
                    if res.error:
                        error_msg = res.error.get("evalue", "Unknown error")
                    else:
                        error_msg = hub.validate_contract(task_obj, res.stdout)

                    if error_msg:
                        print(f"  [Executor] ❌ Failure: {error_msg}")
                        if hub.escalate_task(task_id, error_msg, hierarchy):
                            session.commit()
                            continue 
                        else:
                            hub.fail_task(task_id, error_msg)
                            session.commit()
                            break
                    else:
                        print(f"  [Executor] ✅ Success.")
                        hub.complete_task(task_id, {"stdout": res.stdout, "model_used": model_used})
                        
                        # EMIT LIVE STATE UPDATE
                        files = executor.sandbox.list_workspace_files(project_id)
                        hub.create_outbox_event("STATE_UPDATED", {
                            "files": files,
                            "state": executor.authoritative_state
                        })
                        
                        if res.plots:
                            art = Artifact(
                                project_id=project_id,
                                type="plot",
                                description=f"Visualization for task: {task_obj.description[:50]}",
                                content_json={"image_base64": res.plots[0]},
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
                        session.commit()
                        break

        # 3. Final Synthesis
        print(f"  [Synthesizer] Storytelling...")
        with Session(engine) as session:
            hub = ExecutionHub(session)
            hub.create_outbox_event("STEP_STARTED", {"message": "Lead Data Scientist is synthesizing the results..."})
            session.commit()

            statement = select(Artifact).where(Artifact.project_id == project_id)
            artifacts = session.exec(statement).all()
            context = "\n".join([f"Artifact {a.id} ({a.type}): {a.description}" for a in artifacts])

        synthesizer = SynthesizerAgent(model=planner_model)
        synth_res = await synthesizer.run(SynthesizerInput(
            objective=objective,
            context_artifacts=context
        ))
        
        with Session(engine) as session:
            hub = ExecutionHub(session)
            hub.create_outbox_event("WORKFLOW_FINAL_RESULT", {
                "narrative": synth_res.content.narrative,
                "takeaways": synth_res.content.key_takeaways
            })
            hub.create_outbox_event("STEP_COMPLETED", {"message": "Project workflow complete."})
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

@app.post("/projects", response_model=Project)
async def create_project(req: ProjectCreateRequest, background_tasks: BackgroundTasks):
    with Session(engine) as session:
        project = Project(name=req.name, objective=req.objective)
        session.add(project)
        session.commit()
        session.refresh(project)
        
        # Save files to workspace directory
        if req.files:
            # Note: we use {project.id}_{project.id} to match Sandbox session format
            workspace_dir = f"/home/jfrese/projects/MyLocalStack/data/workspaces/{project.id}_{project.id}"
            os.makedirs(workspace_dir, exist_ok=True)
            for f in req.files:
                file_path = os.path.join(workspace_dir, f.name)
                print(f"  [Server] Saving uploaded file: {file_path}", flush=True)
                with open(file_path, "wb") as out:
                    out.write(base64.b64decode(f.content_base64))
        
        background_tasks.add_task(run_agent_workflow, project.id, req.objective)
        return project

@app.get("/health")
def health():
    return {"status": "ok"}
