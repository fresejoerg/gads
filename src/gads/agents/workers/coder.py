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
     fig.write_html("unique_plot_name.html")
     fig.show()
     ```
   - Use descriptive, unique filenames for each HTML file.
   - CONSOLIDATED DASHBOARD: If you generate more than one plot, you MUST create a `final_dashboard.html`. 
     - Use the following pattern for a professional vertical report:
     ```python
     import plotly.offline as pyo
     # Define your figures in a list of tuples: (figure_object, title, description)
     report_figs = [(fig1, "Figure 1: Title", "Description..."), (fig2, "Figure 2: Title", "Description...")]
     
     cards = []
     for fig, title, desc in report_figs:
         div = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')
         cards.append(f"<div class='card'><h2>{{title}}</h2><p>{{desc}}</p>{{div}}</div>")
     
     html_content = f\"\"\"
     <html>
     <head><title>GADS Research Report</title>
     <style>
       body {{{{ font-family: sans-serif; margin: 40px auto; max-width: 900px; background: #f4f7f6; color: #2c3e50; }}}}
       .card {{{{ background: white; border-radius: 12px; padding: 25px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e1e8ed; }}}}
       h1 {{{{ color: #1a2a33; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}}}
       h2 {{{{ margin-top: 0; color: #2980b9; }}}}
       p {{{{ line-height: 1.6; color: #7f8c8d; }}}}
     </style>
     </head>
     <body>
       <h1>Project Research Dashboard</h1>
       <div class='report-flow'>
         {''.join(cards)}
       </div>
     </body>
     </html>
     \"\"\"
     with open("final_dashboard.html", "w") as f:
         f.write(html_content)
     ```
   - DO NOT use matplotlib or seaborn for plotting unless explicitly requested.
3. NO HALLUCINATIONS: Do not generate mock data. 
3. DATA PROVENANCE: You MUST use the variables and files listed in the 'RUNTIME STATE' below.
4. CONSISTENCY: If the 'RUNTIME STATE' contradicts the 'Task Description', the 'RUNTIME STATE' wins.
5. WORKING DIRECTORY: You are ALREADY in your project-specific workspace directory. 
   - Files uploaded by the user are in your current directory.
   - You can read them directly using `open('filename')`, `pd.read_csv('filename')`, etc.
   - DO NOT try to "upload" files; they are already there.

## AUTHORITATIVE RUNTIME STATE (Source of Truth)
The following variables and data structures ALREADY EXIST in your stateful kernel memory. 
{state_summary}

### FORMATTING RULE:
You MUST return a valid JSON object matching the requested schema. 
Do NOT include any metadata, schema definitions, or 'properties' wrappers.
Ensure you use double quotes for keys and string values.
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
