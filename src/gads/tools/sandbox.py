import ast
import httpx
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import uuid

class ExecutionResult(BaseModel):
    stdout: str
    stderr: str
    result: Optional[str] = None
    plots: List[str] = []
    displays: List[Dict[str, Any]] = []
    error: Optional[Dict[str, Any]] = None
    execution_time_ms: int

class CodeValidator:
    """Proactively scans code for security and syntax issues."""
    
    # Blacklisted modules that are dangerous even in a sandbox
    BLACKLISTED_MODULES = {"os", "subprocess", "shutil", "socket", "requests", "urllib", "pickle", "marshal", "shelve"}

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
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in CodeValidator.BLACKLISTED_MODULES:
                        return f"Security Error: Import of '{alias.name}' is not allowed."
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in CodeValidator.BLACKLISTED_MODULES:
                    return f"Security Error: From-Import of '{node.module}' is not allowed."
            
            # Check calls (e.g., eval, exec)
            if isinstance(node, ast.Call):
                # Handle direct calls like eval()
                if isinstance(node.func, ast.Name):
                    if node.func.id in {"eval", "exec", "open", "compile", "getattr", "setattr"}:
                        return f"Security Error: Use of '{node.func.id}' is not allowed."
                # Handle attribute calls like pandas.read_pickle()
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"read_pickle", "to_pickle", "system", "popen"}:
                        return f"Security Error: Call to dangerous method '{node.func.attr}' is not allowed."

        return None

class SandboxClient:
    """Client for interacting with the MyLocalStack Sandbox with Project Isolation."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=70.0)

    async def execute(self, code: str, project_id: uuid.UUID, session_id: str = "default") -> ExecutionResult:
        """Validates and executes code in the sandbox, isolated by project."""
        
        # 1. Local AST Validation
        validation_error = CodeValidator.validate(code)
        if validation_error:
            return ExecutionResult(
                stdout="",
                stderr="",
                error={"ename": "ValidationError", "evalue": validation_error, "traceback": []},
                execution_time_ms=0
            )

        # 2. Remote Execution
        # Note: we use (project_id + session_id) to create a unique kernel namespace
        internal_session_id = f"{project_id}_{session_id}"
        
        try:
            response = await self.client.post(
                f"{self.base_url}/execute",
                json={"code": code, "session_id": internal_session_id}
            )
            response.raise_for_status()
            return ExecutionResult(**response.json())
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                error={"ename": "ConnectionError", "evalue": f"Failed to reach sandbox: {e}", "traceback": []},
                execution_time_ms=0
            )

    async def close(self):
        await self.client.aclose()
