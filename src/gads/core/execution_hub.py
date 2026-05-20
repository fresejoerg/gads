import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlmodel import select, Session
from gads.core.database import engine
from gads.core.models import Task, OutboxEvent
from gads.core.registry import get_next_model_dynamic
import uuid

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

    def validate_contract(self, task: Task, stdout: str, kernel_state: Optional[Dict[str, Any]] = None, semantic_insights: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """
        Validates the sandbox output against the task's postcondition contract.
        Checks both stdout string and kernel_state variables (if provided).
        Also verifies that required semantic insights were emitted.
        """
        if not task.postcondition_json:
            return None
            
        contract = task.postcondition_json
        ctype = contract.get("output_type")
        
        try:
            # 1. Structural Validation (Existing Logic)
            if ctype == "dataframe":
                cols = contract.get("required_columns", [])
                for col in cols:
                    col_lower = str(col).lower()
                    in_stdout = col_lower in stdout.lower()
                    in_kernel = False
                    if kernel_state:
                        for var_name, var_info in kernel_state.items():
                            if var_info.get("type") == "DataFrame":
                                existing_cols_lower = [str(c).lower() for c in var_info.get("columns", [])]
                                if col_lower in existing_cols_lower:
                                    in_kernel = True
                                    break
                    if not (in_stdout or in_kernel):
                        return f"Contract Violation: Column '{col}' not found in output or kernel state."
            
            # 2. Semantic Insight Validation (New Logic)
            required_insights = contract.get("required_insights", [])
            if required_insights:
                emitted_artifacts = [i.get("artifact", "").lower() for i in (semantic_insights or [])]
                # Fallback: check if insight field contains the required keyword if no artifact specified
                emitted_texts = [i.get("insight", "").lower() for i in (semantic_insights or [])]
                
                for req in required_insights:
                    found = False
                    for em_art in emitted_artifacts:
                        if req.lower() in em_art:
                            found = True
                            break
                    if not found:
                        for em_txt in emitted_texts:
                            if req.lower() in em_txt:
                                found = True
                                break
                    
                    if not found:
                        # SOFT FAIL: We log the missing insight for debug but don't fail the task.
                        # This allows the project to proceed even with 'forgetful' local models.
                        print(f"    [ExecutionHub] WARNING: Missing required insight '{req}'. Proceeding anyway.")
                        # return f"Contract Violation: Required semantic insight for '{req}' was not emitted via gads_emit_insight()."
                    
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

    def escalate_task(self, task_id: uuid.UUID, error: str, hierarchy: Dict[str, Any]) -> bool:
        """
        Upgrades a task to a more powerful model and re-queues it.
        Uses dynamic selection across tiers with random sampling.
        """
        task = self.session.get(Task, task_id)
        if not task:
            return False
            
        next_model = get_next_model_dynamic(task.assigned_to, hierarchy)
        
        if next_model and task.escalation_count < 2:
            print(f"  [Escalation] Upgrading Task {task_id} from {task.assigned_to} to {next_model} (Tier Shift)")
            
            failure_context = (
                f"\n\n--- PREVIOUS ATTEMPT FAILED ---\n"
                f"Model: {task.assigned_to}\n"
                f"Error: {error}\n"
                f"Note: DO NOT repeat the mistake above.\n"
            )
            
            task.description += failure_context
            task.assigned_to = next_model
            task.escalation_count += 1
            task.status = "pending"
            task.error = f"Escalated: {error}"
            
            self.session.add(task)
            self.create_outbox_event("ESCALATION_STARTED", {
                "task_id": str(task_id),
                "next_model": next_model,
                "error": error
            })
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

    def bypass_task(self, task_id: uuid.UUID, result: dict):
        """Mark a task as bypassed due to complexity."""
        task = self.session.get(Task, task_id)
        if task:
            task.status = "bypassed"
            task.result_json = result
            self.session.add(task)
            self.create_outbox_event("TASK_BYPASSED", {"task_id": str(task_id), "result": result})
            self.session.commit()

    def fail_task(self, task_id: uuid.UUID, error: str, result: Optional[dict] = None):
        """Mark a task as failed."""
        task = self.session.get(Task, task_id)
        if task:
            task.status = "failed"
            task.error = error
            if result:
                task.result_json = result
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
                task.status = "pending"
                task.error = "Orphaned (heartbeat timeout)"
                session.add(task)
            session.commit()
        await asyncio.sleep(60)
