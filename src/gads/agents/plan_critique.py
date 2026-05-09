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

class PlanCritiqueOutput(BaseModel):
    is_approved: bool = Field(description="True if the plan is robust and complete.")
    feedback: str = Field(description="Detailed explanation of why the plan was failed, or 'Plan approved'.")
    missing_requirements: List[str] = Field(default_factory=list, description="Specific requirements or SOP nodes that are missing.")

class PlanCritiqueAgent(BaseAgent[PlanCritiqueInput, PlanCritiqueOutput]):
    def __init__(self, model: str = "gemini-3-flash-preview"):
        super().__init__(
            name="PlanCritique",
            model=model,
            system_prompt=prompt_registry.get_prompt("PlanCritique"),
            output_schema=PlanCritiqueOutput
        )

    async def run(self, input_data: PlanCritiqueInput, stream_callback=None, **kwargs) -> Any:
        # Refresh prompt from registry
        self.system_prompt = prompt_registry.get_prompt(self.name)
        
        knowledge_str = json.dumps(input_data.knowledge_report.dict(), indent=2) if input_data.knowledge_report else "No specific SOP was provided."
        
        steps_summary = []
        for i, step in enumerate(input_data.proposed_steps):
            steps_summary.append(f"Step {i+1}: {step.description} (Assigned to: {step.assigned_to})")
        steps_str = "\n".join(steps_summary)
        
        formatted_prompt = self.system_prompt.format(
            knowledge_json=knowledge_str,
            objective=input_data.objective
        )
        
        user_content = f"PROPOSED PLAN STEPS:\n{steps_str}\n\nPlease evaluate if these steps fully satisfy the objective and align with the SOP if applicable."

        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": user_content}
        ]
        
        from gads.core.llm import get_structured_completion
        content = await get_structured_completion(
            model=self.model,
            response_model=self.output_schema,
            messages=messages,
            stream_callback=stream_callback,
            **kwargs
        )
        
        from gads.agents.base import AgentResponse
        return AgentResponse(content=content, model_used=self.model)
