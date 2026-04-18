import asyncio
import uuid
import time
from typing import Optional, List, Tuple, Dict, Any
from gads.agents.workers.coder import CodeGeneratorAgent, CoderInput
from gads.tools.sandbox import SandboxClient, ExecutionResult
from gads.core.models import Artifact
from sqlmodel import Session

class ExecutionManager:
    """Manages the Code-Execution-Feedback loop and persists results to DB."""

    def __init__(self, sandbox_url: str = "http://localhost:8000"):
        self.coder = CodeGeneratorAgent()
        self.sandbox = SandboxClient(base_url=sandbox_url)
        self.authoritative_state: Dict[str, Any] = {} # Persistent memory for the project run

    async def run_task(
        self, 
        task_description: str, 
        project_id: uuid.UUID, 
        session: Session, 
        session_id: str = "default"
    ) -> Tuple[ExecutionResult, str]:
        """
        Runs the full loop with State Introspection to prevent hallucinations.
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
                
                # 1. Generate Code (with state injection)
                print(f"    [Executor] Calling CodeGenerator with {len(self.authoritative_state)} variables in memory...")
                coder_res = await asyncio.wait_for(
                    self.coder.run(CoderInput(
                        task_description=task_description,
                        available_files=available_files,
                        authoritative_state=self.authoritative_state,
                        previous_code=previous_code,
                        error_feedback=error_feedback
                    )),
                    timeout=90.0 # High reasoning models can take time
                )
                
                current_code = coder_res.content.code
                
                # 2. Execute Code
                print(f"    [Executor] Sending code to Sandbox...")
                exec_result = await self.sandbox.execute(current_code, project_id=project_id, session_id=session_id)

                # 3. Handle Result
                if exec_result.error is None:
                    print(f"    [Executor] ✅ Execution successful.")
                    
                    # UPDATE AUTHORITATIVE STATE
                    # We merge the new kernel state into our memory
                    if exec_result.kernel_state:
                        self.authoritative_state.update(exec_result.kernel_state)
                        print(f"    [Executor] Memory updated. Total variables: {len(self.authoritative_state)}")

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
                print(f"    [Executor] ❌ Code generation timed out")
                return ExecutionResult(
                    stdout="", stderr="", 
                    error={"ename": "TimeoutError", "evalue": "LLM generation timed out"},
                    execution_time_ms=60000,
                    kernel_state={}
                ), self.coder.model
            except Exception as e:
                print(f"    [Executor] ❌ Unexpected error in run_task: {e}")
                import traceback
                traceback.print_exc()
                return ExecutionResult(
                    stdout="", stderr="", 
                    error={"ename": "RuntimeError", "evalue": str(e)},
                    execution_time_ms=0,
                    kernel_state={}
                ), self.coder.model

        return exec_result, self.coder.model
