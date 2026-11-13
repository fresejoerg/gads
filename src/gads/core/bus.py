import asyncio
import json
from typing import List, Dict, Any, Optional
from fastapi import WebSocket
from sqlmodel import select, Session
from gads.core.database import engine
from gads.core.models import OutboxEvent

class EventBus:
    """Manages WebSocket connections and event dispatching.

    Each connection carries its caller (core/tenancy.Caller). Broadcast AND replay deliver an
    event to a non-admin only if it belongs to one of their projects (tenancy.event_visible):
    the feed used to replay everything, leaking other users' task names and summaries.
    """

    def __init__(self):
        self.active_connections: Dict[WebSocket, Any] = {}

    async def connect(self, websocket: WebSocket, caller: Any = None):
        await websocket.accept()
        self.active_connections[websocket] = caller

    def disconnect(self, websocket: WebSocket):
        self.active_connections.pop(websocket, None)

    @staticmethod
    def _visible(caller: Any, payload: Dict[str, Any], session: Session) -> bool:
        if caller is None:  # connected without identity: deliver nothing
            return False
        from gads.core.tenancy import event_visible
        return event_visible(caller, payload, session)

    async def broadcast(self, message: dict, session: Optional[Session] = None):
        """Send a message to every connected client allowed to see it."""
        own = session or Session(engine)
        try:
            for connection, caller in list(self.active_connections.items()):
                if not self._visible(caller, message.get("payload") or {}, own):
                    continue
                try:
                    await connection.send_json(message)
                except Exception:
                    # Connection might be dead
                    pass
        finally:
            if session is None:
                own.close()

    async def replay_events(self, websocket: WebSocket, since_sequence: int):
        """Replay missed events to a client (only those it may see)."""
        caller = self.active_connections.get(websocket)
        with Session(engine) as session:
            statement = select(OutboxEvent).where(OutboxEvent.sequence > since_sequence).order_by(OutboxEvent.sequence.asc())
            missed_events = session.exec(statement).all()
            
            for event in missed_events:
                if not self._visible(caller, event.payload_json or {}, session):
                    continue
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
                }, session=session)
                event.processed = True
                session.add(event)
            
            session.commit()
            
        await asyncio.sleep(0.5)  # Poll every 500ms
