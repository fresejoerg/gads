import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent

class ReconciliationReport(BaseModel):
    recipe_id: str
    rationale: str
    recommended_dag_nodes: List[Dict[str, Any]]
    skippable_nodes: List[str] = []
    schema_warnings: List[str] = []

class PlannerTask(BaseModel):
    """Simplified task model for the Planner to output."""
    description: str
    assigned_to: str
    postcondition: Dict[str, Any] = Field(
        description="Structural contract for success. MUST be a valid JSON object with key:value pairs. E.g., {'output_type': 'dataframe', 'required_columns': ['name']}"
    )
    attached_skills: List[str] = Field(
        default=[],
        description="List of Skill IDs to attach to this task for the worker agent's guidance."
    )

class FileMetadata(BaseModel):
    name: str
    size_mb: float
    columns_and_dtypes: Optional[Dict[str, str]] = None

class PlannerInput(BaseModel):
    objective: str
    available_models_hierarchy: Dict[str, Any]
    available_files: List[FileMetadata]
    knowledge_report: Optional[ReconciliationReport] = None
    available_skills: List[Dict[str, Any]] = []


class PlannerOutput(BaseModel):
    steps: List[PlannerTask] = Field(description="A list of discrete steps with postcondition contracts.")

PLANNER_SYSTEM_PROMPT = """
You are the Lead Project Manager of a high-end Data Science team. 
Your goal is to decompose a user's request into a list of tasks, delegate each to the **LOWEST FEASIBLE MODEL TIER**, and define a **POSTCONDITION CONTRACT** for every step.

### 1. DOMAIN EXPERTISE (SOPs)
- You may be provided with a `KNOWLEDGE REPORT` containing a matched Data Science SOP (Standard Operating Procedure).
- The SOP is a **PRIOR**, not a mandate. 
- You MUST align your task decomposition with the recommended DAG nodes unless the user's specific data or environment prevents it.
- **IDEMPOTENCY**: If the report lists `skippable_nodes`, DO NOT include those tasks in your output. They have already been completed or are unnecessary.

### 3. ENVIRONMENT AWARENESS
- You are provided with a list of `AVAILABLE FILES`.
- These files are ALREADY in the workspace.
- **CRITICAL**: DO NOT create any tasks to "upload", "move", or "verify" these files. Assume they are ready for analysis.

### 4. SKILLS & BEST PRACTICES
- You are provided with a list of `AVAILABLE SKILLS` (Expertise modules).
- You MUST explicitly attach the relevant Skill IDs to each task using the `attached_skills` field.
- If a task involves visualization, attach the visualization skills. If it involves cleaning, attach cleaning skills.

### 5. CAPABILITY RUBRIC

Score every task across these 4 dimensions (Low, Med, High):
- **Reasoning Depth**: Novel decomposition, multi-hop logic, or architectural decisions.
- **Context Breadth**: Need for 100K+ tokens of context or very long memory.
- **Output Fidelity**: Zero-tolerance for syntax errors (e.g., complex pandas code).
- **Domain Specificity**: Obscure Python libraries or deep mathematical expertise.

### 3. SELECTION RULES
- IF any dimension is **HIGH** → Delegate to **Tier 1** (Opus/Pro).
- ELIF **Reasoning Depth** is **MEDIUM** OR **Domain Specificity** is **MEDIUM** → Delegate to **Tier 2** (Sonnet/Flash).
- ELIF any dimension is **MEDIUM** OR structured output is needed → Delegate to **Tier 3** (Haiku/Flash-Lite).
- ELSE (purely mechanical tasks: regex, format conversion, simple boilerplate) → Delegate to **Tier 4** (Local).

**NOTE**: Most standard Data Science tasks (cleaning, basic plotting, baseline model fitting) should now default to **Tier 3** to optimize for speed and cost.


### 5. OUTPUT FORMAT
You MUST provide a list of steps. For each task:
- Set `assigned_to` to the FIRST verbatim model ID (index 0) from the 'models' list in the chosen Tier.
- You MUST select a model that is explicitly listed in the hierarchy below.
- **POSTCONDITION CONTRACT**: Define a structural contract for every task to detect "silent failures." 
  - This MUST be a JSON object (Dict), NOT a list or set.
  - Supported keys: `output_type` ('dataframe' or 'list'), `required_columns` (list of strings).
  - EXAMPLE: `{{"output_type": "dataframe", "required_columns": ["theme", "score"]}}`
- **FIGURE NUMBERING**: For every task that generates a visualization, you MUST explicitly assign a unique number in the description (e.g., "Analyze target balance. Save as Figure 1."). This ensures a professional thread through the final report.

## AVAILABLE FILES:
{files_list}

## AVAILABLE SKILLS (Expertise Modules):
{skills_json}

## KNOWLEDGE REPORT:
{knowledge_json}

## AVAILABLE_MODELS_HIERARCHY:
{hierarchy_json}

### FORMATTING RULE:
You MUST return a valid JSON object matching the requested schema. 
Do NOT include any metadata, schema definitions, or 'properties' wrappers.
"""

class DataSciencePlanner(BaseAgent[PlannerInput, PlannerOutput]):
    def __init__(self, model: str = "claude-opus-4.7"):
        super().__init__(
            name="Planner",
            model=model,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            output_schema=PlannerOutput
        )

    async def run(self, input_data: PlannerInput) -> Any:
        hierarchy_str = json.dumps(input_data.available_models_hierarchy, indent=2)
        skills_str = json.dumps(input_data.available_skills, indent=2)
        
        # Format files with size and schema for agent awareness
        files_info = []
        for f in input_data.available_files:
            info = f"{f.name} ({f.size_mb:.2f} MB)"
            if f.columns_and_dtypes:
                info += f" - Schema: {json.dumps(f.columns_and_dtypes)}"
            files_info.append(info)
        files_str = "\n".join([f"- {i}" for i in files_info]) if files_info else "None"
        
        knowledge_str = json.dumps(input_data.knowledge_report.dict(), indent=2) if input_data.knowledge_report else "No matching SOP found. Use generic data science reasoning."
        
        formatted_prompt = self.system_prompt.format(
            hierarchy_json=hierarchy_str,
            files_list=files_str,
            knowledge_json=knowledge_str,
            skills_json=skills_str
        )
        
        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": f"USER OBJECTIVE: {input_data.objective}"}
        ]
        
        from gads.core.llm import get_structured_completion
        content = await get_structured_completion(
            model=self.model,
            response_model=self.output_schema,
            messages=messages
        )
        
        from gads.agents.base import AgentResponse
        return AgentResponse(content=content, model_used=self.model)
