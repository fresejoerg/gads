import json
from typing import List, Optional
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.prompts import prompt_registry


class CompletenessVerifierInput(BaseModel):
    objective: str
    completed_task_summaries: List[str]
    produced_artifact_names: List[str]
    metrics_json: Optional[dict] = None


class CompletenessVerifierOutput(BaseModel):
    is_complete: bool = Field(
        description="True if execution fully addressed all key requirements of the objective."
    )
    missing_analyses: List[str] = Field(
        default_factory=list,
        description=(
            "Specific, concrete analyses/metrics/outputs the objective requires "
            "but that were not performed. Must be non-empty when is_complete=False."
        )
    )
    verdict: str = Field(
        description="One-sentence justification for the completeness assessment."
    )


class CompletenessVerifierAgent(BaseAgent[CompletenessVerifierInput, CompletenessVerifierOutput]):
    def __init__(self, model: str = "local_model"):
        super().__init__(
            name="CompletenessVerifier",
            model=model,
            system_prompt=prompt_registry.get_prompt("CompletenessVerifier"),
            output_schema=CompletenessVerifierOutput
        )

    async def run(self, input_data: CompletenessVerifierInput, **kwargs):
        base_prompt = prompt_registry.get_prompt(self.name)
        self.agent._system_prompts = (base_prompt,)

        summaries_str = "\n".join(input_data.completed_task_summaries) or "No completed tasks."
        artifacts_str = (
            ", ".join(input_data.produced_artifact_names)
            if input_data.produced_artifact_names else "None"
        )
        metrics_str = (
            json.dumps(input_data.metrics_json, indent=2)
            if input_data.metrics_json else "Not present"
        )

        user_content = (
            f"USER OBJECTIVE:\n{input_data.objective}\n\n"
            f"COMPLETED TASK SUMMARIES:\n{summaries_str}\n\n"
            f"PRODUCED ARTIFACT FILES: {artifacts_str}\n\n"
            f"metrics.json CONTENTS:\n{metrics_str}\n\n"
            "Does the above execution fully address the user's analytical objective? "
            "List each missing analysis, metric, or comparison that was required but not performed."
        )

        return await super().run(user_content, **kwargs)
