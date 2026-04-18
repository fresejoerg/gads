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
from gads.agents.workers.synthesizer import SynthesizerAgent, SynthesizerInput
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
    print(f"\n--- 🚀 Starting Workflow for Project {project_id} ---")
    
    try:
        with Session(engine) as session:
            hub = ExecutionHub(session)
            executor = ExecutionManager()
            planner = DataSciencePlanner()
            synthesizer = SynthesizerAgent()
            
            # 1. Planning
            print(f"  [Planner] Analyzing objective: {objective}")
            hub.create_outbox_event("STEP_STARTED", {"message": "Project Manager is planning the objective..."})
            session.commit()
            
            planner_res = await planner.run(PlannerInput(objective=objective))
            
            # Save all tasks
            valid_tasks = []
            for step in planner_res.content.steps:
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
            context_summary = []
            for task in valid_tasks:
                print(f"  [Executor] Claiming task: {task.id}")
                if hub.claim_task(task.id):
                    res, model_used = await executor.run_task(
                        task.description, 
                        project_id=project_id, 
                        session=session, 
                        session_id=str(project_id)
                    )
                    
                    if res.error:
                        hub.fail_task(task.id, res.error.get("evalue", "Unknown error"))
                        context_summary.append(f"Task '{task.description}' failed with error: {res.error.get('evalue')}")
                    else:
                        hub.complete_task(task.id, {"stdout": res.stdout, "model_used": model_used})
                        context_summary.append(f"Task '{task.description}' completed. Output: {res.stdout[:200]}...")
                        
                        if res.plots:
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

            # 3. Final Synthesis
            print(f"  [Synthesizer] Generating final story...")
            hub.create_outbox_event("STEP_STARTED", {"message": "Lead Data Scientist is synthesizing the results..."})
            session.commit()

            synth_res = await synthesizer.run(SynthesizerInput(
                objective=objective,
                context_artifacts="\n".join(context_summary)
            ))
            
            hub.create_outbox_event("WORKFLOW_FINAL_RESULT", {
                "narrative": synth_res.content.narrative,
                "takeaways": synth_res.content.key_takeaways
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
