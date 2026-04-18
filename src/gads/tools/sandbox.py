import ast
import httpx
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

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
    BLACKLISTED_MODULES = {"os", "subprocess", "shutil", "socket", "requests", "urllib"}

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
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec", "open", "compile"}:
                    return f"Security Error: Use of '{node.func.id}' is not allowed."

        return None

class SandboxClient:
    """Client for interacting with the MyLocalStack Sandbox."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=70.0)

    async def execute(self, code: str, session_id: str = "default") -> ExecutionResult:
        """Validates and executes code in the sandbox."""
        
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
        try:
            response = await self.client.post(
                f"{self.base_url}/execute",
                json={"code": code, "session_id": session_id}
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
