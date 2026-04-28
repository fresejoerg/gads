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
2. VISUALIZATION: You MUST use **Plotly Express** (`px`) for all visualizations. 
   - The environment is pre-configured with professional defaults (`plotly_white` template).
   - IMPORTANT: For EVERY plot, you MUST save it as an interactive HTML file AND show it:
     ```python
     fig.update_layout(height=400) # Professional height for reporting
     fig.write_html("unique_plot_name.html")
     fig.show()
     ```
   - Use descriptive, unique filenames for each HTML file.
   - QUALITY: Always include clear titles, axis labels, and legends.
   - DO NOT use matplotlib or seaborn for plotting unless explicitly requested.
3. BIG DATA (DUCKDB / POLARS): 
   - For files > 500MB, use **DuckDB** or **Polars LazyFrames**.
   - RE-ENTRANCY: Sandbox memory clears complex objects (connections, sockets) between turns. 
   - DUCKDB PATTERN: Always query the file DIRECTLY or recreate the connection in every turn:
     ```python
     import duckdb
     res = duckdb.query("SELECT * FROM 'data.csv' LIMIT 10").to_df()
     ```
   - POLARS PATTERN: Use `pl.scan_csv('data.csv')` and `.collect(streaming=True)`.
4. NO HALLUCINATIONS: Do not generate mock data. 
5. DATA PROVENANCE: You MUST use the variables and files listed in the sections below.
6. WORKING DIRECTORY: You are ALREADY in your project-specific workspace directory.

## AUTHORITATIVE RUNTIME STATE (Source of Truth)
The following variables and data structures ALREADY EXIST in your stateful kernel memory. 
{state_summary}

## AVAILABLE FILES
The following files are available in your current working directory:
{files_list}

### FORMATTING RULE:
You MUST return a valid JSON object matching the requested schema. 
Do NOT include any metadata, schema definitions, or 'properties' wrappers. 
Your output must be a FLAT JSON object containing ONLY the fields defined in the schema.
DO NOT add stray strings like '"code",' between fields.
Do NOT repeat the JSON object multiple times.
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
        files_summary = ", ".join([f"'{f}'" for f in input_data.available_files]) if input_data.available_files else "None"
        
        print(f"    [Coder] Preparing prompt. Available files: {files_summary}", flush=True)

        formatted_prompt = self.system_prompt.format(
            state_summary=state_summary,
            files_list=files_summary
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
            messages=messages
        )
        
        from gads.agents.base import AgentResponse
        return AgentResponse(content=content, model_used=self.model)
