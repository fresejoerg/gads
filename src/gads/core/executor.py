import asyncio
import uuid
import time
from typing import Optional, List, Tuple
from gads.agents.workers.coder import CodeGeneratorAgent, CoderInput
from gads.tools.sandbox import SandboxClient, ExecutionResult
from gads.core.models import Artifact
from sqlmodel import Session

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
        Runs the full loop with strict timeouts and detailed logging.
        """
        max_retries = 2
        retry_count = 0
        error_feedback = None
        previous_code = ""

        while retry_count <= max_retries:
            print(f"    [Executor] --- Step {retry_count + 1} for: {task_description[:30]}... ---")
            
            try:
                # 0. List available files
                available_files = self.sandbox.list_workspace_files(project_id)
                print(f"    [Executor] Available files: {available_files}")

                # 1. Generate Code (with timeout)
                print(f"    [Executor] Calling CodeGenerator ({self.coder.model})...")
                start_time = time.time()
                
                # Add a 60-second timeout to the LLM call to prevent infinite hangs
                coder_res = await asyncio.wait_for(
                    self.coder.run(CoderInput(
                        task_description=task_description,
                        available_files=available_files,
                        previous_code=previous_code,
                        error_feedback=error_feedback
                    )),
                    timeout=60.0
                )
                
                # NOTE: coder_res is an AgentResponse object. We need to access .content.code
                current_code = coder_res.content.code
                
                print(f"    [Executor] Code generated in {time.time() - start_time:.2f}s")
                
                # 2. Execute Code
                print(f"    [Executor] Sending code to Sandbox (session: {session_id})...")
                exec_result = await self.sandbox.execute(current_code, project_id=project_id, session_id=session_id)

                # 3. Handle Result
                if exec_result.error is None:
                    print(f"    [Executor] ✅ Execution successful.")
                    artifact = Artifact(
                        project_id=project_id,
                        type="code_execution",
                        description=f"Execution result for: {task_description}",
                        content_json={"code": current_code, "stdout": exec_result.stdout},
                        agent_id="CodeGenerator"
                    )
                    session.add(artifact)
                    return exec_result, coder_res.model_used
                else:
                    ename = exec_result.error.get("ename", "Error")
                    evalue = exec_result.error.get("evalue", "Unknown error")
                    print(f"    [Executor] ❌ Execution failed: {ename}")
                    
                    error_feedback = f"{ename}: {evalue}"
                    previous_code = current_code
                    retry_count += 1

            except asyncio.TimeoutError:
                print(f"    [Executor] ❌ Code generation timed out after 60s")
                return ExecutionResult(
                    stdout="", stderr="", 
                    error={"ename": "TimeoutError", "evalue": "LLM generation timed out"},
                    execution_time_ms=60000
                ), self.coder.model
            except Exception as e:
                print(f"    [Executor] ❌ Unexpected error in run_task: {e}")
                return ExecutionResult(
                    stdout="", stderr="", 
                    error={"ename": "RuntimeError", "evalue": str(e)},
                    execution_time_ms=0
                ), self.coder.model

        return exec_result, self.coder.model
