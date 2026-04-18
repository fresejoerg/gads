import asyncio
from typing import Optional, List, Tuple
from gads.agents.workers.coder import CodeGeneratorAgent, CoderInput
from gads.tools.sandbox import SandboxClient, ExecutionResult
from gads.core.models import Artifact
from sqlmodel import Session
import uuid

class ExecutionManager:
    """Manages the Code-Execution-Feedback loop and persists results to DB."""

    def __init__(self, sandbox_url: str = "http://localhost:8000"):
        self.coder = CodeGeneratorAgent()
        self.sandbox = SandboxClient(base_url=sandbox_url)

    async def run_task(
        self, 
        task_description: str, 
        project_id: uuid.UUID, 
        session: Session, 
        session_id: str = "default"
    ) -> Tuple[ExecutionResult, str]:
        """
        Runs the full loop: Generate -> Validate -> Execute -> (Retry if fail).
        Persists artifacts directly to the database session.
        Returns (ExecutionResult, model_used).
        """
        max_retries = 2
        retry_count = 0
        error_feedback = None
        previous_code = ""

        while retry_count <= max_retries:
            print(f"  [Coder] Task: {task_description} (Attempt {retry_count + 1})")
            
            # 1. Generate Code
            coder_input = CoderInput(
                task_description=task_description,
                previous_code=previous_code,
                error_feedback=error_feedback
            )
            coder_res = await self.coder.run(coder_input)
            
            # 2. Execute Code
            exec_result = await self.sandbox.execute(coder_res.code, project_id=project_id, session_id=session_id)

            # 3. Handle Result
            if exec_result.error is None:
                # Add artifact to DB
                artifact = Artifact(
                    project_id=project_id,
                    type="code_execution",
                    description=f"Execution result for: {task_description}",
                    content_json={"code": coder_res.code, "stdout": exec_result.stdout},
                    agent_id="CodeGenerator"
                )
                session.add(artifact)
                return exec_result, coder_res.model_used
            else:
                # 4. Feedback Loop
                ename = exec_result.error.get("ename", "Error")
                evalue = exec_result.error.get("evalue", "Unknown error")
                print(f"  [Failure] {ename}: {evalue}")
                
                error_feedback = f"{ename}: {evalue}"
                previous_code = coder_res.code
                retry_count += 1

        return exec_result, self.coder.model
