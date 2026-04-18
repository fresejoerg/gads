import asyncio
from typing import Optional, List
from gads.agents.workers.coder import CodeGeneratorAgent, CoderInput
from gads.tools.sandbox import SandboxClient, ExecutionResult
from gads.core.state import Blackboard, Artifact

class ExecutionManager:
    """Manages the Code-Execution-Feedback loop."""

    def __init__(self, sandbox_url: str = "http://localhost:8000"):
        self.coder = CodeGeneratorAgent()
        self.sandbox = SandboxClient(base_url=sandbox_url)

    async def run_task(self, task_description: str, blackboard: Blackboard, session_id: str = "default") -> ExecutionResult:
        """
        Runs the full loop: Generate -> Validate -> Execute -> (Retry if fail).
        """
        max_retries = 2
        retry_count = 0
        error_feedback = None
        previous_code = ""

        while retry_count <= max_retries:
            # 1. Generate Code
            print(f"  [Coder] Attempt {retry_count + 1}...")
            coder_input = CoderInput(
                task_description=task_description,
                previous_code=previous_code,
                error_feedback=error_feedback
            )
            coder_res = await self.coder.run(coder_input)
            
            # 2. Execute Code (SandboxClient handles AST validation internally)
            print(f"  [Sandbox] Executing...")
            exec_result = await self.sandbox.execute(coder_res.code, session_id=session_id)

            # 3. Handle Result
            if exec_result.error is None:
                print("  [Success] Code ran perfectly.")
                # Add artifacts to blackboard
                artifact = Artifact(
                    id=f"code_{session_id}_{retry_count}",
                    type="code_execution",
                    description=f"Result of: {task_description}",
                    content={"code": coder_res.code, "stdout": exec_result.stdout},
                    agent_id="CodeGenerator"
                )
                blackboard.add_artifact(artifact)
                return exec_result
            else:
                # 4. Feedback Loop
                ename = exec_result.error.get("ename", "Error")
                evalue = exec_result.error.get("evalue", "Unknown error")
                print(f"  [Failure] {ename}: {evalue}")
                
                error_feedback = f"{ename}: {evalue}"
                previous_code = coder_res.code
                retry_count += 1

        return exec_result
