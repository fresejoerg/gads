from typing import List, Optional, Any
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.prompts import prompt_registry

class SynthesizerInput(BaseModel):
    objective: str
    context_artifacts: str # Serialized summary of what happened
    existing_narrative: Optional[str] = None
    existing_takeaways: Optional[List[str]] = None

class SynthesizerOutput(BaseModel):
    narrative: str = Field(description="The final human-friendly response or story.")
    key_takeaways: List[str] = Field(description="Bullet points of the most important findings.")

class SynthesizerAgent(BaseAgent[SynthesizerInput, SynthesizerOutput]):
    def __init__(self, model: str = "claude-opus-4.7"):
        super().__init__(
            name="Synthesizer",
            model=model,
            system_prompt=prompt_registry.get_prompt("Synthesizer"),
            output_schema=SynthesizerOutput
        )

    async def run(self, input_data: SynthesizerInput, **kwargs) -> Any:
        # Refresh prompt from registry
        self.system_prompt = prompt_registry.get_prompt(self.name)
        
        prev_state = "No previous report."
        if input_data.existing_narrative:
            prev_state = f"--- EXISTING NARRATIVE ---\n{input_data.existing_narrative}\n\n--- EXISTING TAKEAWAYS ---\n" + "\n".join([f"- {t}" for t in (input_data.existing_takeaways or [])])
            
        formatted_prompt = self.system_prompt.format(previous_state=prev_state)
        
        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": f"USER OBJECTIVE: {input_data.objective}\n\nARTIFACTS AND OUTPUTS:\n{input_data.context_artifacts}"}
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
