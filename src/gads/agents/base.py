from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type
from pydantic import BaseModel
from gads.core.llm import get_structured_completion

TIn = TypeVar("TIn", bound=BaseModel)
TOut = TypeVar("TOut", bound=BaseModel)

class AgentResponse(BaseModel, Generic[TOut]):
    """Wrapper for agent output that includes metadata."""
    content: TOut
    model_used: str

class BaseAgent(ABC, Generic[TIn, TOut]):
    """Base class for all Data Science agents."""
    
    def __init__(self, name: str, model: str, system_prompt: str, output_schema: Type[TOut]):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.output_schema = output_schema

    async def run(self, input_data: TIn, **kwargs) -> AgentResponse[TOut]:
        """Execute the agent's logic using instructor for structured output."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_data.model_dump_json()}
        ]
        
        content = await get_structured_completion(
            model=self.model,
            response_model=self.output_schema,
            messages=messages,
            **kwargs
        )
        
        return AgentResponse(content=content, model_used=self.model)
