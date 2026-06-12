from typing import List, Optional, Any
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.prompts import prompt_registry

class SynthesizerInput(BaseModel):
    objective: str
    context_artifacts: str # Serialized summary of what happened
    existing_narrative: Optional[str] = None
    existing_takeaways: Optional[List[str]] = None

class ArtifactInsight(BaseModel):
    artifact_id: str = Field(description="The ID or filename of the artifact (e.g., 'Figure 1' or 'table_1.html').")
    contextual_text: str = Field(description="Paragraph explaining what this visualization illustrates in the context of the objective.")
    caption: str = Field(description="A concise, descriptive caption for the visualization.")

class SynthesizerOutput(BaseModel):
    narrative: str = Field(description="The final human-friendly response or story.")
    key_takeaways: List[str] = Field(description="Bullet points of the most important findings.")
    artifact_insights: List[ArtifactInsight] = Field(default_factory=list, description="Specific context and captions for each visualization generated.")

class SynthesizerAgent(BaseAgent[SynthesizerInput, SynthesizerOutput]):
    def __init__(self, model: str = "local_model"):
        super().__init__(
            name="Synthesizer",
            model=model,
            system_prompt=prompt_registry.get_prompt("Synthesizer"),
            output_schema=SynthesizerOutput
        )

    async def run(self, input_data: SynthesizerInput, **kwargs) -> Any:
        # Refresh prompt from registry
        base_prompt = prompt_registry.get_prompt(self.name)
        
        prev_state = "No previous report."
        if input_data.existing_narrative:
            prev_state = f"--- EXISTING NARRATIVE ---\n{input_data.existing_narrative}\n\n--- EXISTING TAKEAWAYS ---\n" + "\n".join([f"- {t}" for t in (input_data.existing_takeaways or [])])
            
        formatted_prompt = base_prompt.format(previous_state=prev_state)

        user_content = (
            f"USER OBJECTIVE: {input_data.objective}\n\n"
            f"ARTIFACTS AND OUTPUTS:\n{input_data.context_artifacts}"
        )

        # Use super().run
        return await super().run(
            user_content,
            system_prompt=formatted_prompt,
            **kwargs
        )
