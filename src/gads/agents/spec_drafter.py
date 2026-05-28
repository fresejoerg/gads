from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.prompts import prompt_registry


class SpecDraftInput(BaseModel):
    objective: str
    files_with_schemas: str
    available_recipe_ids: List[str]


class SpecDraft(BaseModel):
    name: str = Field(description="Short, descriptive project name (5-8 words).")
    formalized_objective: str = Field(
        description="A precise, unambiguous restatement of the user's objective with domain context."
    )
    datasets: List[str] = Field(description="Relevant dataset filenames from AVAILABLE FILES.")
    recipe_id: Optional[str] = Field(
        default=None,
        description="Exact recipe ID from the available list if a match exists, else null."
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Primary target/label column for supervised ML tasks, else null."
    )
    feature_columns: List[str] = Field(
        default_factory=list,
        description="Key feature columns from the schema most relevant to the analysis."
    )
    filters: Optional[str] = Field(
        default=None,
        description="Domain filter condition implied by the objective (e.g. 'category = electronics'), else null."
    )
    domain: Optional[str] = Field(
        default=None,
        description="Domain context (e.g. 'e-commerce', 'healthcare', 'NLP', 'time-series')."
    )


class SpecDrafterAgent(BaseAgent[SpecDraftInput, SpecDraft]):
    def __init__(self, model: str = "local_model"):
        super().__init__(
            name="SpecDrafter",
            model=model,
            system_prompt=prompt_registry.get_prompt("SpecDrafter"),
            output_schema=SpecDraft
        )

    async def run(self, input_data: SpecDraftInput, **kwargs) -> Any:
        base_prompt = prompt_registry.get_prompt(self.name)

        recipe_ids_str = ", ".join(input_data.available_recipe_ids) if input_data.available_recipe_ids else "None"
        formatted_prompt = base_prompt.format(
            available_recipe_ids=recipe_ids_str,
            files_with_schemas=input_data.files_with_schemas
        )

        self.agent._system_prompts = (formatted_prompt,)

        # Include critical context in user message so local_model path also sees it
        # (BaseAgent.run() reloads the unformatted system prompt, so structured data
        # must be in the user message to survive that reload for local models.)
        user_content = (
            f"USER OBJECTIVE: {input_data.objective}\n\n"
            f"AVAILABLE RECIPES (use exact IDs only):\n{recipe_ids_str}\n\n"
            f"AVAILABLE FILES WITH SCHEMAS:\n{input_data.files_with_schemas}\n\n"
            "Return the required FLAT JSON object."
        )

        return await super().run(user_content, **kwargs)
