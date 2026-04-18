from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
import json

class CoderOutput(BaseModel):
    explanation: str = Field(description="Brief explanation of what the code does.")
    code: str = Field(description="The complete Python code block to execute.")
    libraries_used: List[str] = Field(description="List of libraries imported.")

class CoderInput(BaseModel):
    task_description: str
    available_files: List[str] = []
    authoritative_state: Dict[str, Any] = {} # Ground truth from the kernel
    previous_code: Optional[str] = None
    error_feedback: Optional[str] = None

CODER_SYSTEM_PROMPT = """
You are a precise Python Developer. 
Your goal is to write code that fulfills the user's task and NOTHING MORE.

STRICT RULES:
1. MINIMALISM: Do only what is requested. Do not add extra analysis, extra columns, or extra visualizations.
2. NO HALLUCINATIONS: Do not generate mock data. 
3. DATA PROVENANCE: You MUST use the variables and files listed in the 'RUNTIME STATE' below.
4. CONSISTENCY: If the 'RUNTIME STATE' contradicts the 'Task Description', the 'RUNTIME STATE' wins.
5. WORKING DIRECTORY: Your working directory is '/app/workspaces'.

## AUTHORITATIVE RUNTIME STATE (Source of Truth)
The following variables and data structures ALREADY EXIST in your stateful kernel memory. 
{state_summary}
"""

class CodeGeneratorAgent(BaseAgent[CoderInput, CoderOutput]):
    def __init__(self, model: str = "claude-sonnet-4.6"):
        super().__init__(
            name="CodeGenerator",
            model=model,
            system_prompt=CODER_SYSTEM_PROMPT,
            output_schema=CoderOutput
        )

    async def run(self, input_data: CoderInput) -> Any:
        state_summary = json.dumps(input_data.authoritative_state, indent=2)
        formatted_prompt = self.system_prompt.format(state_summary=state_summary)
        
        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": input_data.model_dump_json()}
        ]
        
        from gads.core.llm import get_structured_completion
        content = await get_structured_completion(
            model=self.model,
            response_model=self.output_schema,
            messages=messages
        )
        
        from gads.agents.base import AgentResponse
        return AgentResponse(content=content, model_used=self.model)
 Here is the updated code:
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
import json

class CoderOutput(BaseModel):
    explanation: str = Field(description="Brief explanation of what the code does.")
    code: str = Field(description="The complete Python code block to execute.")
    libraries_used: List[str] = Field(description="List of libraries imported.")

class CoderInput(BaseModel):
    task_description: str
    available_files: List[str] = []
    authoritative_state: Dict[str, Any] = {} # Ground truth from the kernel
    previous_code: Optional[str] = None
    error_feedback: Optional[str] = None

CODER_SYSTEM_PROMPT = """
You are a precise Python Developer. 
Your goal is to write code that fulfills the user's task and NOTHING MORE.

STRICT RULES:
1. MINIMALISM: Do only what is requested. Do not add extra analysis, extra columns, or extra visualizations.
2. NO HALLUCINATIONS: Do not generate mock data. 
3. DATA PROVENANCE: You MUST use the variables and files listed in the 'RUNTIME STATE' below.
4. CONSISTENCY: If the 'RUNTIME STATE' contradicts the 'Task Description', the 'RUNTIME STATE' wins.
5. WORKING DIRECTORY: Your working directory is '/app/workspaces'.

## AUTHORITATIVE RUNTIME STATE (Source of Truth)
The following variables and data structures ALREADY EXIST in your stateful kernel memory. 
{state_summary}
"""

class CodeGeneratorAgent(BaseAgent[CoderInput, CoderOutput]):
    def __init__(self, model: str = "claude-sonnet-4.6"):
        super().__init__(
            name="CodeGenerator",
            model=model,
            system_prompt=CODER_SYSTEM_PROMPT,
            output_schema=CoderOutput
        )

    async def run(self, input_data: CoderInput) -> Any:
        state_summary = json.dumps(input_data.authoritative_state, indent=2)
        formatted_prompt = self.system_prompt.format(state_summary=state_summary)
        
        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": input_data.model_dump_json()}
        ]
        
        from gads.core.llm import get_structured_completion
        content = await get_structured_completion(
            model=self.model,
            response_model=self.output_schema,
            messages=messages
        )
        
        from gads.agents.base import AgentResponse
        return AgentResponse(content=content, model_used=self.model)
