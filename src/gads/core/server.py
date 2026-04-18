import asyncio
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from gads.core.bus import bus, dispatcher_loop
from gads.core.execution_hub import watchdog_loop
from gads.core.database import init_db
from gads.core.models import Project
from sqlmodel import select, Session
from gads.core.database import engine

app = FastAPI(title="GADS Core API")

@app.on_event("startup")
async def startup_event():
    # Ensure tables exist
    init_db()
    # Start background loops
    asyncio.create_task(dispatcher_loop())
    asyncio.create_task(watchdog_loop())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, last_seq: int = 0):
    await bus.connect(websocket)
    # Replay missed events if client provides last_seq
    if last_seq > 0:
        await bus.replay_events(websocket, last_seq)
        
    try:
        while True:
            # Just keep connection open, we don't expect messages from client yet
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        bus.disconnect(websocket)

@app.get("/projects", response_model=List[Project])
async def list_projects():
    with Session(engine) as session:
        statement = select(Project)
        return session.exec(statement).all()

@app.post("/projects", response_model=Project)
async def create_project(name: str, objective: str):
    with Session(engine) as session:
        project = Project(name=name, objective=objective)
        session.add(project)
        session.commit()
        session.refresh(project)
        return project

@app.get("/health")
def health():
    return {"status": "ok"}
