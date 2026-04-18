import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlmodel import select, Session
from gads.core.database import engine
from gads.core.models import Task, OutboxEvent
import uuid

# Escalation ladder: defines the next model to try upon failure
ESCALATION_LADDER = {
    "local_model": "claude-haiku-4.5",
    "claude-haiku-4.5": "claude-sonnet-4.6",
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-preview",
    "claude-sonnet-4.6": "claude-opus-4.7",
    "gemini-3.1-flash-preview": "gemini-3.1-pro-preview"
}

class ExecutionHub:
    """Manages task lifecycle, heartbeats, and durable execution."""
    
    def __init__(self, session: Session):
        self.session = session

    def create_outbox_event(self, event_type: str, payload: dict):
        """Creates a transactional outbox event."""
        statement = select(OutboxEvent).order_by(OutboxEvent.sequence.desc()).limit(1)
        last_event = self.session.exec(statement).first()
        next_seq = (last_event.sequence + 1) if last_event else 1
        
        event = OutboxEvent(
            sequence=next_seq,
            type=event_type,
            payload_json=payload
        )
        self.session.add(event)

    def validate_contract(self, task: Task, stdout: str) -> Optional[str]:
        """
        Validates the sandbox output against the task's postcondition contract.
        Returns an error message if invalid, else None.
        """
        if not task.postcondition_json:
            return None
            
        contract = task.postcondition_json
        ctype = contract.get("output_type")
        
        try:
            if ctype == "dataframe":
                # Very basic check: look for columns in stdout
                cols = contract.get("required_columns", [])
                for col in cols:
                    if col not in stdout:
                        return f"Contract Violation: Column '{col}' not found in output."
                
                # Check row count (heuristic)
                min_rows = contract.get("min_rows", 0)
                if min_rows > 0:
                    lines = [l for l in stdout.split('\n') if len(l.strip()) > 0]
                    if len(lines) < min_rows:
                        return f"Contract Violation: Output seems too short. Expected ~{min_rows} rows."
            
            elif ctype == "list":
                min_items = contract.get("min_items", 1)
                # Heuristic: look for list-like indicators
                if '[' not in stdout and '-' not in stdout:
                    return f"Contract Violation: Output does not appear to be a list."
                    
        except Exception as e:
            return f"Validation Error: {str(e)}"
            
        return None

    def claim_task(self, task_id: uuid.UUID) -> bool:
        """Atomically claim a task for execution."""
        task = self.session.get(Task, task_id)
        if task and task.status == "pending":
            task.status = "running"
            task.heartbeat = datetime.now()
            self.session.add(task)
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

    def escalate_task(self, task_id: uuid.UUID, error: str) -> bool:
        """
        Upgrades a task to a more powerful model and re-queues it.
        Returns True if escalated, False if limit reached.
        """
        task = self.session.get(Task, task_id)
        if not task:
            return False
            
        next_model = ESCALATION_LADDER.get(task.assigned_to)
        
        if next_model and task.escalation_count < 2:
            print(f"  [Escalation] Upgrading Task {task_id} from {task.assigned_to} to {next_model}")
            
            # Prepare Failure Packet Context
            failure_context = (
                f"\n\n--- PREVIOUS ATTEMPT FAILED ---\n"
                f"Model: {task.assigned_to}\n"
                f"Error: {error}\n"
                f"Note: DO NOT repeat the mistake above. Ground your solution in the objective below.\n"
            )
            
            task.description += failure_context
            task.assigned_to = next_model
            task.escalation_count += 1
            task.status = "pending" # Re-queue
            task.error = f"Escalated: {error}"
            
            self.session.add(task)
            self.create_outbox_event("ESCALATION_STARTED", {
                "task_id": str(task_id),
                "next_model": next_model,
                "error": error
            })
            self.session.commit()
            return True
            
        return False

    def complete_task(self, task_id: uuid.UUID, result: dict):
        """Mark a task as completed."""
        task = self.session.get(Task, task_id)
        if task:
            task.status = "completed"
            task.result_json = result
            self.session.add(task)
            self.create_outbox_event("TASK_COMPLETED", {"task_id": str(task_id), "result": result})
            self.session.commit()

    def fail_task(self, task_id: uuid.UUID, error: str):
        """Mark a task as failed."""
        task = self.session.get(Task, task_id)
        if task:
            task.status = "failed"
            task.error = error
            self.session.add(task)
            self.create_outbox_event("TASK_FAILED", {"task_id": str(task_id), "error": error})
            self.session.commit()

async def watchdog_loop():
    """Background task to recover orphaned (crashed) tasks."""
    HEARTBEAT_TIMEOUT = timedelta(minutes=5)
    while True:
        with Session(engine) as session:
            timeout_threshold = datetime.now() - HEARTBEAT_TIMEOUT
            statement = select(Task).where(Task.status == "running").where(Task.heartbeat < timeout_threshold)
            orphaned_tasks = session.exec(statement).all()
            for task in orphaned_tasks:
                print(f"Watchdog: Recovering orphaned task {task.id}")
                task.status = "pending"
                task.error = "Orphaned (heartbeat timeout)"
                session.add(task)
            session.commit()
        await asyncio.sleep(60)
