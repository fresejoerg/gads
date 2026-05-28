import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.prompts import prompt_registry
from gads.agents.planner import PlannerTask, ReconciliationReport

class PlanCritiqueInput(BaseModel):
    objective: str
    proposed_steps: List[PlannerTask]
    knowledge_report: Optional[ReconciliationReport] = None
    available_files: List[str] = []

class PlanCritiqueOutput(BaseModel):
    is_approved: bool = Field(description="True if the plan is robust and complete.")
    is_terminal_failure: bool = Field(default=False, description="True if the failure is environmental (missing files) and should HALT the workflow.")
    feedback: str = Field(description="Detailed explanation of why the plan was failed, or 'Plan approved'.")
    missing_requirements: List[str] = Field(default_factory=list, description="Specific requirements or SOP nodes that are missing.")

class PlanCritiqueAgent(BaseAgent[PlanCritiqueInput, PlanCritiqueOutput]):
    def __init__(self, model: str = "local_model"):
        super().__init__(
            name="PlanCritique",
            model=model,
            system_prompt=prompt_registry.get_prompt("PlanCritique"),
            output_schema=PlanCritiqueOutput
        )

    async def run(self, input_data: PlanCritiqueInput, stream_callback=None, **kwargs) -> Any:
        # Refresh prompt from registry
        base_prompt = prompt_registry.get_prompt(self.name)
        
        knowledge_str = json.dumps(input_data.knowledge_report.dict(), indent=2) if input_data.knowledge_report else "No specific SOP was provided."
        files_str = ", ".join([f"'{f}'" for f in input_data.available_files]) if input_data.available_files else "None (Empty Workspace)"
        
        steps_summary = []
        for i, step in enumerate(input_data.proposed_steps):
            steps_summary.append(f"Step {i+1}: {step.description} (Assigned to: {step.assigned_to})")
        steps_str = "\n".join(steps_summary)
        
        formatted_prompt = base_prompt.format(
            knowledge_json=knowledge_str,
            objective=input_data.objective,
            files_list=files_str
        )
        self.agent._system_prompts = (formatted_prompt,)

        user_content = (
            f"PROPOSED PLAN STEPS:\n{steps_str}\n\n"
            f"AVAILABLE FILES:\n{files_str}\n\n"
            "Please evaluate if these steps fully satisfy the objective and align with the SOP if applicable."
        )

        # Use super().run
        return await super().run(
            user_content,
            **kwargs
        )
