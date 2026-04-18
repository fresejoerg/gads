import asyncio
import json
from typing import List, Dict, Any
from fastapi import WebSocket
from sqlmodel import select, Session
from gads.core.database import engine
from gads.core.models import OutboxEvent

class EventBus:
    """Manages WebSocket connections and event dispatching."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Send a message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Connection might be dead
                pass

    async def replay_events(self, websocket: WebSocket, since_sequence: int):
        """Replay missed events to a client."""
        with Session(engine) as session:
            statement = select(OutboxEvent).where(OutboxEvent.sequence > since_sequence).order_by(OutboxEvent.sequence.asc())
            missed_events = session.exec(statement).all()
            
            for event in missed_events:
                await websocket.send_json({
                    "seq": event.sequence,
                    "type": event.type,
                    "payload": event.payload_json
                })

bus = EventBus()

async def dispatcher_loop():
    """Background task to poll the Outbox and dispatch events via the Bus."""
    while True:
        with Session(engine) as session:
            # Find unprocessed events
            statement = select(OutboxEvent).where(OutboxEvent.processed == False).order_by(OutboxEvent.sequence.asc())
            new_events = session.exec(statement).all()
            
            for event in new_events:
                print(f"Dispatcher: Sending event {event.sequence} ({event.type})")
                await bus.broadcast({
                    "seq": event.sequence,
                    "type": event.type,
                    "payload": event.payload_json
                })
                event.processed = True
                session.add(event)
            
            session.commit()
            
        await asyncio.sleep(0.5)  # Poll every 500ms
