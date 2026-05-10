import asyncio
import uuid
import time
import contextlib
import re
from typing import Optional, List, Tuple, Dict, Any
from gads.agents.workers.coder import CodeGeneratorAgent, CoderInput
from gads.tools.sandbox import SandboxClient, ExecutionResult
from gads.core.models import Artifact, Task
from gads.core.database import engine
from gads.core.bus import bus
from gads.core.runtime_oracle import RuntimeOracle
from gads.core.handover import HandoverManager
from sqlmodel import Session

class ExecutionManager:
    """Manages the Code-Execution-Feedback loop."""

    def __init__(self, sandbox_url: str = "http://localhost:8000"):
        self.coder = CodeGeneratorAgent()
        self.sandbox = SandboxClient(base_url=sandbox_url)
        self.handover = HandoverManager(sandbox_url=sandbox_url)
        self.authoritative_state: Dict[str, Any] = {} # Persistent memory for the project run

    async def run_task(
        self, 
        task_description: str, 
        project_id: uuid.UUID, 
        session_id: str = "default",
        skills_context: Optional[str] = None,
        task_id: Optional[uuid.UUID] = None,
        stdout_callback = None,
        stream_callback = None,
        cancel_check = None,
        state_summary: Optional[str] = None
    ) -> Tuple[ExecutionResult, str]:
        """
        Runs the full loop with State Introspection. 
        Does NOT handle DB persistence (caller must do that).
        """
        max_retries = 2
        retry_count = 0
        error_feedback = None
        previous_code = ""

        # Buffer for reasoning debouncing
        reasoning_buffer = []
        last_emit = 0.0

        async def debounced_stream_callback(token: str):
            nonlocal last_emit
            if not stream_callback: return
            reasoning_buffer.append(token)
            now = time.time()
            if now - last_emit > 0.15:
                delta = "".join(reasoning_buffer)
                reasoning_buffer.clear()
                last_emit = now
                await stream_callback(delta)

        async def poll_logs_loop():
            if not stdout_callback: return
            offset_out = 0
            offset_err = 0
            accumulated_out = ""
            accumulated_err = ""
            print(f"    [Executor] Starting log poller for session {session_id}", flush=True)
            while True:
                await asyncio.sleep(1.0)
                logs = await self.sandbox.poll_logs(session_id, offset_out, offset_err)
                if logs:
                    new_out = logs.get("stdout", "")
                    new_err = logs.get("stderr", "")
                    offset_out = logs.get("offset_out", offset_out)
                    offset_err = logs.get("offset_err", offset_err)
                    
                    if new_out or new_err:
                        print(f"    [Executor] Polled {len(new_out)} chars of new stdout from {session_id}", flush=True)
                        accumulated_out += new_out
                        accumulated_err += new_err
                        combined = accumulated_out + "\n" + accumulated_err
                        
                        # Collapse carriage returns for tqdm
                        lines = combined.split('\n')
                        cleaned_lines = []
                        for line in lines:
                            if '\r' in line:
                                line = line.split('\r')[-1]
                            cleaned_lines.append(line)
                        cleaned_text = '\n'.join(cleaned_lines)
                        
                        await stdout_callback(cleaned_text)

        while retry_count <= max_retries:
            print(f"    [Executor] --- Step {retry_count + 1} for: {task_description[:30]}... ---", flush=True)
            
            try:
                # 0. Early Exit if Cancelled
                if cancel_check and await cancel_check():
                    print(f"    [Executor] 🛑 Aborting task due to user cancellation.", flush=True)
                    return ExecutionResult(
                        stdout="", stderr="Workflow cancelled by user.", 
                        error={"ename": "Cancelled", "evalue": "User requested abort"},
                        execution_time_ms=0,
                        kernel_state={}
                    ), self.coder.model

                available_files = self.sandbox.list_workspace_files(project_id)
                
                print(f"    [Executor] Calling {self.coder.model} with {len(self.authoritative_state)} variables...", flush=True)
                
                # Fetch postcondition contract for the worker
                contract = None
                with Session(engine) as session:
                    from gads.core.models import Task as DBTask
                    t_obj = session.get(DBTask, task_id)
                    if t_obj: contract = t_obj.postcondition_json

                coder_res = await asyncio.wait_for(
                    self.coder.run(CoderInput(
                        task_description=task_description,
                        available_files=available_files,
                        authoritative_state=self.authoritative_state,
                        previous_code=previous_code,
                        error_feedback=error_feedback,
                        skills_context=skills_context,
                        task_id=str(task_id) if task_id else None,
                        postcondition_contract=contract,
                        state_summary=state_summary
                    ), stream_callback=debounced_stream_callback),
                    timeout=300.0
                )
                
                # Flush remaining reasoning
                if stream_callback and reasoning_buffer:
                    delta = "".join(reasoning_buffer)
                    reasoning_buffer.clear()
                    await stream_callback(delta)
                
                current_code = coder_res.content.code
                
                # --- PREDICTIVE RUNTIME ORACLE ---
                # 1. Gather Data Dimensions
                n_rows, m_cols = 0, 0
                for var_info in self.authoritative_state.values():
                    if var_info.get("type") == "DataFrame":
                        shape = var_info.get("shape", [0, 0])
                        n_rows = max(n_rows, shape[0])
                        m_cols = max(m_cols, shape[1])
                
                # 2. Estimate
                est_seconds = RuntimeOracle.estimate_runtime(current_code, n_rows, m_cols)
                print(f"    [Oracle] Estimated Runtime: {est_seconds:.1f}s (N={n_rows}, M={m_cols})", flush=True)

                if task_id:
                    with Session(engine) as session:
                        from gads.core.models import Task as DBTask
                        t_obj = session.get(DBTask, task_id)
                        if t_obj: 
                            t_obj.estimated_runtime_s = est_seconds
                            session.add(t_obj)
                            session.commit()

                # 3. Decision Branch (300s limit)
                if est_seconds > 280.0: # 20s buffer
                    print(f"    [Executor] ⚠️ TASK BYPASSED: Likely to exceed safety limit. Generating handover bundle...", flush=True)
                    bundle_file = await self.handover.create_bundle(project_id, current_code, est_seconds)
                    
                    return ExecutionResult(
                        stdout=f"BYPASSED: Task estimated to take {est_seconds/60:.1f} minutes. Handover bundle created: {bundle_file}",
                        stderr="",
                        result=f"HANDOVER_BUNDLE:{bundle_file}",
                        execution_time_ms=0,
                        kernel_state={}
                    ), coder_res.model_used

                # 0.5. Check Cancellation before Sandbox execution
                if cancel_check and await cancel_check():
                    print(f"    [Executor] 🛑 Aborting execution due to user cancellation.", flush=True)
                    return ExecutionResult(
                        stdout="", stderr="Workflow cancelled by user.", 
                        error={"ename": "Cancelled", "evalue": "User requested abort"},
                        execution_time_ms=0,
                        kernel_state={},
                        code=current_code
                    ), self.coder.model

                print(f"    [Executor] Executing code in sandbox...", flush=True)
                
                poller = asyncio.create_task(poll_logs_loop())
                try:
                    exec_result = await self.sandbox.execute(current_code, project_id=project_id, session_id=session_id)
                    exec_result.code = current_code # Attach code for persistence
                    
                    # LOG ACTUAL RUNTIME FOR LEARNING
                    # Scan for first estimator to log
                    estimators = RuntimeOracle.analyze_code(current_code)
                    if estimators:
                        RuntimeOracle.log_execution(
                            estimators[0].name, n_rows, m_cols, 
                            estimators[0].params, exec_result.execution_time_ms / 1000.0
                        )
                finally:
                    poller.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await poller
                        
                    # Final flush
                    if task_id:
                        # Send a final update to ensure UI sees the end state before COMPLETE event
                        pass

                if exec_result.error is None:
                    print(f"    [Executor] ✅ Execution successful.", flush=True)
                    if exec_result.kernel_state:
                        self.authoritative_state.update(exec_result.kernel_state)
                        print(f"    [Executor] Memory updated. Total variables: {len(self.authoritative_state)}", flush=True)
                    return exec_result, coder_res.model_used
                else:
                    ename = exec_result.error.get("ename", "Error")
                    evalue = exec_result.error.get("evalue", "Unknown error")
                    print(f"    [Executor] ❌ Failure: {ename} - {evalue}", flush=True)
                    
                    error_feedback = f"{ename}: {evalue}"
                    previous_code = current_code
                    retry_count += 1

            except asyncio.TimeoutError:
                print(f"    [Executor] ❌ Timeout", flush=True)
                return ExecutionResult(
                    stdout="", stderr="", 
                    error={"ename": "TimeoutError", "evalue": "LLM generation timed out"},
                    execution_time_ms=180000,
                    kernel_state={}
                ), self.coder.model
            except Exception as e:
                print(f"    [Executor] ❌ Unexpected error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return ExecutionResult(
                    stdout="", stderr="", 
                    error={"ename": "RuntimeError", "evalue": str(e)},
                    execution_time_ms=0,
                    kernel_state={}
                ), self.coder.model

        return exec_result, self.coder.model
