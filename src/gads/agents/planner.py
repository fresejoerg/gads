import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.prompts import prompt_registry

class ReconciliationReport(BaseModel):
    recipe_id: str
    rationale: str
    recommended_dag_nodes: List[Dict[str, Any]]
    skippable_nodes: List[str] = []
    schema_warnings: List[str] = []

class PlannerTask(BaseModel):
    """Simplified task model for the Planner to output."""
    description: str
    assigned_to: str = Field(
        description="MUST be an EXACT verbatim model ID from the AVAILABLE_MODELS_HIERARCHY (e.g. 'gemini-3.1-flash-lite-preview'). DO NOT hallucinate model names."
    )
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
    columns_and_dtypes: Optional[Dict[str, Any]] = None

class PlannerInput(BaseModel):
    objective: str
    available_models_hierarchy: Dict[str, Any]
    available_files: List[FileMetadata]
    knowledge_report: Optional[ReconciliationReport] = None
    available_skills: List[Dict[str, Any]] = []
    critique_feedback: Optional[str] = None
    previous_plan: Optional[List[str]] = None
    user_hints: Optional[Dict[str, Any]] = None


class PlannerOutput(BaseModel):
    steps: List[PlannerTask] = Field(description="A list of discrete steps with postcondition contracts.")


class DataSciencePlanner(BaseAgent[PlannerInput, PlannerOutput]):
    def __init__(self, model: str = "local_model"):
        super().__init__(
            name="Planner",
            model=model,
            system_prompt=prompt_registry.get_prompt("Planner"),
            output_schema=PlannerOutput
        )

    async def run(self, input_data: PlannerInput, stream_callback=None, **kwargs) -> Any:
        # Refresh prompt from registry in case of hot-reload
        base_prompt = prompt_registry.get_prompt(self.name)
        
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
        
        hints_str = json.dumps(input_data.user_hints, indent=2) if input_data.user_hints else "None provided."
        formatted_prompt = base_prompt.format(
            hierarchy_json=hierarchy_str,
            files_list=files_str,
            knowledge_json=knowledge_str,
            skills_json=skills_str,
            user_hints=hints_str
        )
        
        # Set the prompt for the Pydantic AI agent
        self.agent._system_prompts = (formatted_prompt,)

        user_content = f"USER OBJECTIVE: {input_data.objective}"
        if input_data.critique_feedback:
            user_content += f"\n\n--- PREVIOUS ATTEMPT REJECTED BY QA ---\nQA Feedback: {input_data.critique_feedback}"
            if input_data.previous_plan:
                user_content += f"\n\nRejected Plan:\n" + "\n".join([f"- {s}" for s in input_data.previous_plan])
            
            user_content += f"\n\nPlease generate a new plan that specifically addresses the missing requirements noted by QA."

        # Use super().run to get streaming support
        return await super().run(
            user_content,
            **kwargs
        )
