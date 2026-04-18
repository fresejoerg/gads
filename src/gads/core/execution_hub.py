import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
from sqlmodel import select, Session
from gads.core.database import engine
from gads.core.models import Task, OutboxEvent
import uuid

class ExecutionHub:
    """Manages task lifecycle, heartbeats, and durable execution."""
    
    def __init__(self, session: Session):
        self.session = session

    def create_outbox_event(self, event_type: str, payload: dict):
        """Creates a transactional outbox event."""
        # Get the current highest sequence number
        statement = select(OutboxEvent).order_by(OutboxEvent.sequence.desc()).limit(1)
        last_event = self.session.exec(statement).first()
        next_seq = (last_event.sequence + 1) if last_event else 1
        
        event = OutboxEvent(
            sequence=next_seq,
            type=event_type,
            payload_json=payload
        )
        self.session.add(event)

    def claim_task(self, task_id: uuid.UUID) -> bool:
        """Atomically claim a task for execution."""
        task = self.session.get(Task, task_id)
        if task and task.status == "pending":
            task.status = "running"
            task.heartbeat = datetime.now()
            self.session.add(task)
            
            # Record in outbox
            self.create_outbox_event("TASK_STARTED", {"task_id": str(task_id)})
            
            self.session.commit()
            return True
        return False

    def heartbeat(self, task_id: uuid.UUID):
        """Update the heartbeat for a running task."""
        task = self.session.get(Task, task_id)
        if task and task.status == "running":
            task.heartbeat = datetime.now()
            self.session.add(task)
            self.session.commit()

    def complete_task(self, task_id: uuid.UUID, result: dict):
        """Mark a task as completed."""
        task = self.session.get(Task, task_id)
        if task:
            task.status = "completed"
            task.result_json = result
            self.session.add(task)
            
            # Record in outbox
            self.create_outbox_event("TASK_COMPLETED", {"task_id": str(task_id), "result": result})
            
            self.session.commit()

    def fail_task(self, task_id: uuid.UUID, error: str):
        """Mark a task as failed."""
        task = self.session.get(Task, task_id)
        if task:
            task.status = "failed"
            task.error = error
            self.session.add(task)
            
            # Record in outbox
            self.create_outbox_event("TASK_FAILED", {"task_id": str(task_id), "error": error})
            
            self.session.commit()

async def watchdog_loop():
    """Background task to recover orphaned (crashed) tasks."""
    HEARTBEAT_TIMEOUT = timedelta(minutes=5)
    
    while True:
        with Session(engine) as session:
            # Find tasks that are 'running' but have timed out heartbeats
            timeout_threshold = datetime.now() - HEARTBEAT_TIMEOUT
            statement = select(Task).where(Task.status == "running").where(Task.heartbeat < timeout_threshold)
            orphaned_tasks = session.exec(statement).all()
            
            for task in orphaned_tasks:
                print(f"Watchdog: Recovering orphaned task {task.id}")
                task.status = "pending"  # Reset to pending for retry
                task.error = "Orphaned (heartbeat timeout)"
                session.add(task)
            
            session.commit()
        
        await asyncio.sleep(60)  # Run every minute
