import ast
import httpx
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import uuid

class ExecutionResult(BaseModel):
    stdout: str
    stderr: str
    code: Optional[str] = None
    result: Optional[str] = None
    plots: List[str] = []
    displays: List[Dict[str, Any]] = []
    error: Optional[Dict[str, Any]] = None
    execution_time_ms: int
    kernel_state: Dict[str, Any] = {} # New field for state introspection

class CodeValidator:
    """Proactively scans code for security and syntax issues."""
    
    # Standard DS libraries are allowed. Block system-level access.
    BLACKLISTED_MODULES = {"subprocess", "shutil", "socket", "requests", "urllib", "pickle", "marshal", "shelve"}

    @staticmethod
    def validate(code: str) -> Optional[str]:
        """
        Returns an error message if code is invalid or dangerous, else None.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax Error: {e.msg} at line {e.lineno}"

        for node in ast.walk(tree):
            # 1. Security: Block dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in CodeValidator.BLACKLISTED_MODULES:
                        return f"Security Error: Import of '{alias.name}' is not allowed."
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in CodeValidator.BLACKLISTED_MODULES:
                    return f"Security Error: From-Import of '{node.module}' is not allowed."
            
            # 2. Security: Block dangerous calls (Allow 'open' for DS work)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in {"eval", "exec", "compile", "getattr", "setattr"}:
                        return f"Security Error: Use of '{node.func.id}' is not allowed."
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"system", "popen"}:
                        return f"Security Error: Call to dangerous method '{node.func.attr}' is not allowed."

        return None

class SandboxClient:
    """Client for interacting with the MyLocalStack Sandbox with Project Isolation."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0) # Extended timeout for massive datasets

    def list_workspace_files(self, project_id: uuid.UUID) -> List[str]:
        """Lists files available in the project's host workspace directory."""
        host_path = f"/home/joergf/projects/MyLocalStack/data/workspaces/{project_id}"
        if not os.path.exists(host_path):
            return []
        return os.listdir(host_path)

    async def poll_logs(self, session_id: str, offset_out: int = 0, offset_err: int = 0) -> Optional[Dict[str, Any]]:
        """Polls the sandbox for incremental stdout and stderr."""
        try:
            response = await self.client.get(
                f"{self.base_url}/sessions/{session_id}/logs",
                params={"offset_out": offset_out, "offset_err": offset_err},
                timeout=5.0
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    async def reset_session(self, session_id: str):
        """Forcefully kills the kernel and resets the session in the sandbox container."""
        try:
            await self.client.post(f"{self.base_url}/sessions/{session_id}/reset")
        except Exception:
            pass

    async def execute(self, code: str, project_id: uuid.UUID, session_id: str = "default") -> ExecutionResult:
        """Validates and executes code in the sandbox, isolated by project."""
        
        # 1. Local AST Validation
        validation_error = CodeValidator.validate(code)
        if validation_error:
            return ExecutionResult(
                stdout="",
                stderr="",
                error={"ename": "ValidationError", "evalue": validation_error, "traceback": []},
                execution_time_ms=0,
                kernel_state={}
            )

        # 2. Remote Execution (Session ID is just the project ID for simple isolation)
        try:
            response = await self.client.post(
                f"{self.base_url}/execute",
                json={"code": code, "session_id": str(project_id)}
            )
            response.raise_for_status()
            return ExecutionResult(**response.json())
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                error={"ename": "ConnectionError", "evalue": f"Failed to reach sandbox: {e}", "traceback": []},
                execution_time_ms=0,
                kernel_state={}
            )

    async def close(self):
        await self.client.aclose()
