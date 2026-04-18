import asyncio
import uuid
import traceback
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from gads.core.bus import bus, dispatcher_loop
from gads.core.execution_hub import watchdog_loop, ExecutionHub
from gads.core.database import init_db, engine
from gads.core.models import Project, Task, Artifact
from gads.agents.planner import DataSciencePlanner, PlannerInput
from gads.core.executor import ExecutionManager
from sqlmodel import select, Session

app = FastAPI(title="GADS Core API")

@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(dispatcher_loop())
    asyncio.create_task(watchdog_loop())

async def run_agent_workflow(project_id: uuid.UUID, objective: str):
    """Orchestrates the multi-agent workflow for a project."""
    print(f"--- Starting Workflow for Project {project_id} ---")
    
    try:
        with Session(engine) as session:
            hub = ExecutionHub(session)
            executor = ExecutionManager()
            planner = DataSciencePlanner()
            
            # 1. Planning
            print(f"  [Workflow] Planning for objective: {objective}")
            hub.create_outbox_event("STEP_STARTED", {"message": "Project Manager is planning the objective..."})
            session.commit()
            
            planner_res = await planner.run(PlannerInput(objective=objective))
            
            # Filter and save tasks
            valid_tasks = []
            for step in planner_res.content.steps:
                # Ensure the task is a real data science task and not conversational filler
                if any(kw in step.description.lower() for kw in ["create", "calculate", "plot", "extract", "analyze", "run", "load"]):
                    new_task = Task(
                        project_id=project_id,
                        description=step.description,
                        assigned_to=step.assigned_to,
                        status="pending"
                    )
                    session.add(new_task)
                    valid_tasks.append(new_task)
            
            session.commit()
            
            for t in valid_tasks:
                session.refresh(t)
                hub.create_outbox_event("TASK_CREATED", {
                    "task_id": str(t.id),
                    "description": t.description
                })
            session.commit()

            # 2. Sequential Execution
            for task in valid_tasks:
                print(f"  [Workflow] Claiming task: {task.id}")
                if hub.claim_task(task.id):
                    res = await executor.run_task(
                        task.description, 
                        project_id=project_id, 
                        session=session, 
                        session_id=str(project_id)
                    )
                    
                    if res.error:
                        print(f"  [Workflow] Task {task.id} failed.")
                        hub.fail_task(task.id, res.error.get("evalue", "Unknown error"))
                    else:
                        print(f"  [Workflow] Task {task.id} completed.")
                        hub.complete_task(task.id, {"stdout": res.stdout})
                        
                        if res.plots:
                            print(f"  [Workflow] Found plot artifact.")
                            art = Artifact(
                                project_id=project_id,
                                type="plot",
                                description=f"Visualization for task: {task.description}",
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

            hub.create_outbox_event("STEP_COMPLETED", {"message": "Project workflow complete."})
            session.commit()
            
    except Exception as e:
        print(f"ERROR in run_agent_workflow: {e}")
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
async def create_project(name: str, objective: str, background_tasks: BackgroundTasks):
    with Session(engine) as session:
        project = Project(name=name, objective=objective)
        session.add(project)
        session.commit()
        session.refresh(project)
        background_tasks.add_task(run_agent_workflow, project.id, objective)
        return project

@app.get("/health")
def health():
    return {"status": "ok"}
