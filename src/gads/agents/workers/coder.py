from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.prompts import prompt_registry
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
    skills_context: Optional[str] = None
    task_id: Optional[str] = None
    postcondition_contract: Optional[Dict[str, Any]] = None
    state_summary: Optional[str] = None

class CodeGeneratorAgent(BaseAgent[CoderInput, CoderOutput]):
    def __init__(self, model: str = "claude-sonnet-4.6"):
        super().__init__(
            name="CodeGenerator",
            model=model,
            system_prompt=prompt_registry.get_prompt("CodeGenerator"),
            output_schema=CoderOutput
        )

    async def run(self, input_data: CoderInput, **kwargs) -> Any:
        # Refresh prompt from registry
        self.system_prompt = prompt_registry.get_prompt(self.name)
        
        # Use provided state summary or fallback to JSON of dict
        state_summary = input_data.state_summary or json.dumps(input_data.authoritative_state, indent=2)
        
        files_summary = ", ".join([f"'{f}'" for f in input_data.available_files]) if input_data.available_files else "None"
        contract_summary = json.dumps(input_data.postcondition_contract, indent=2) if input_data.postcondition_contract else "None. Just fulfill the task description."
        
        print(f"    [Coder] Preparing prompt. Available files: {files_summary}", flush=True)

        formatted_prompt = self.system_prompt.format(
            state_summary=state_summary,
            files_list=files_summary,
            skills_context=input_data.skills_context or "No specific skills required for this task.",
            contract_json=contract_summary
        )
        
        # Strengthen file awareness in the user message
        user_content = (
            f"TASK: {input_data.task_description}\n\n"
            f"CRITICAL: You MUST use the correct filenames. Available files are: {files_summary}\n\n"
            "Return the required FLAT JSON object."
        )

        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": user_content}
        ]
        
        from gads.core.llm import get_structured_completion
        content = await get_structured_completion(
            model=self.model,
            response_model=self.output_schema,
            messages=messages,
            **kwargs
        )
        
        from gads.agents.base import AgentResponse
        return AgentResponse(content=content, model_used=self.model)
