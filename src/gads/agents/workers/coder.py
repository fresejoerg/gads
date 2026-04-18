from typing import Optional, List
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent

class CoderOutput(BaseModel):
    explanation: str = Field(description="Brief explanation of what the code does.")
    code: str = Field(description="The complete Python code block to execute.")
    libraries_used: List[str] = Field(description="List of libraries imported.")

class CoderInput(BaseModel):
    task_description: str
    available_files: List[str] = []
    previous_code: Optional[str] = None
    error_feedback: Optional[str] = None
    dataset_summary: Optional[str] = None

CODER_SYSTEM_PROMPT = """
You are an expert Data Science Python Developer. 
Your goal is to write clean, efficient, and correct Python code to solve the user's task.

STRICT DATA RULES:
1. NO HALLUCINATIONS: Do not generate mock data. 
2. DATA PROVENANCE: You MUST load data from the files listed in 'available_files'.
3. WORKING DIRECTORY: Your working directory is already set to the workspace root. Use relative paths for local files.
4. If 'error_feedback' is provided, fix the bug in your new code.
5. Always provide a complete, runnable code block for the CURRENT step.
"""

class CodeGeneratorAgent(BaseAgent[CoderInput, CoderOutput]):
    def __init__(self, model: str = "claude-sonnet-4.6"):
        super().__init__(
            name="CodeGenerator",
            model=model,
            system_prompt=CODER_SYSTEM_PROMPT,
            output_schema=CoderOutput
        )
