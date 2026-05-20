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
    file_schemas: Dict[str, Any] = {}  # column→dtype maps keyed by filename
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
        base_prompt = prompt_registry.get_prompt(self.name)
        
        # Use provided state summary or fallback to JSON of dict
        state_summary = input_data.state_summary or json.dumps(input_data.authoritative_state, indent=2)
        
        files_lines = []
        for f in input_data.available_files:
            schema_info = input_data.file_schemas.get(f, {}).get("schema")
            if schema_info:
                files_lines.append(f"'{f}' — columns: {list(schema_info.keys())}")
            else:
                files_lines.append(f"'{f}'")
        files_summary = "\n".join(files_lines) if files_lines else "None"
        contract_summary = json.dumps(input_data.postcondition_contract, indent=2) if input_data.postcondition_contract else "None. Just fulfill the task description."

        print(f"    [Coder] Preparing prompt. Available files: {', '.join(input_data.available_files)}", flush=True)

        formatted_prompt = base_prompt.format(
            state_summary=state_summary,
            files_list=files_summary,
            skills_context=input_data.skills_context or "No specific skills required for this task.",
            contract_json=contract_summary
        )
        
        # Set the prompt for the Pydantic AI agent
        self.agent._system_prompts = (formatted_prompt,)

        # Build dynamic kernel-reuse warning from live state
        kernel_warnings = []
        try:
            state_data = json.loads(state_summary) if isinstance(state_summary, str) else state_summary
            if isinstance(state_data, dict):
                for var, desc in state_data.items():
                    if isinstance(desc, str) and desc.startswith("DataFrame"):
                        kernel_warnings.append(f"`{var}`: {desc}")
        except Exception:
            pass

        user_content = f"TASK: {input_data.task_description}\n\n"
        user_content += f"CRITICAL: Use ONLY these exact filenames: {files_summary}\n\n"
        if kernel_warnings:
            user_content += (
                "KERNEL MEMORY — these DataFrames are ALREADY LOADED. "
                "Use them directly. DO NOT call pd.read_csv() or reload from disk:\n"
                + "\n".join(kernel_warnings) + "\n\n"
            )
        user_content += "Return the required FLAT JSON object."

        # Use super().run to get streaming support
        return await super().run(
            user_content,
            **kwargs
        )
