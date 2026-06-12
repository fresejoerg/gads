from abc import ABC
from typing import TypeVar, Generic, Type, Any, Optional
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
import os

TIn = TypeVar("TIn", bound=BaseModel)
TOut = TypeVar("TOut", bound=BaseModel)

def get_pydantic_ai_model(model_name: str) -> Any:
    """Helper to convert a model string into a Pydantic AI Model instance."""
    from gads.core.llm import LITELLM_BASE_URL, LITELLM_MASTER_KEY
    
    # Use the raw model name. Pydantic AI's OpenAIModel doesn't perform 
    # the same restrictive prefix validation that the LiteLLM SDK does.
    return OpenAIModel(
        model_name,
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_MASTER_KEY
    )

class AgentResponse(BaseModel, Generic[TOut]):
    """Wrapper for agent output that includes metadata."""
    content: TOut
    model_used: str

class BaseAgent(ABC, Generic[TIn, TOut]):
    """Base class for all Data Science agents using Pydantic AI."""
    
    def __init__(self, name: str, model: str, system_prompt: str, output_schema: Type[TOut]):
        self.name = name
        self.model_str = model
        self.model = get_pydantic_ai_model(model)
        self.system_prompt = system_prompt
        self.output_schema = output_schema
        
        # Initialize Pydantic AI Agent with minimal retries to trigger fallback faster
        self.agent = Agent(
            self.model,
            result_type=self.output_schema,
            system_prompt=self.system_prompt,
            name=self.name,
            retries=1
        )

    async def run(self, input_data: Any, **kwargs) -> AgentResponse[TOut]:
        """Execute the agent's logic with Pydantic AI and a robust manual fallback."""
        from gads.core.prompts import prompt_registry
        from gads.core.llm import get_structured_completion
        
        self.system_prompt = prompt_registry.get_prompt(self.name)
        user_prompt = input_data if isinstance(input_data, str) else input_data.model_dump_json()
        stream_callback = kwargs.pop("stream_callback", None)

        # MANDATE: Bypass Pydantic AI's tool-calling for local_model.
        # Local models frequently get stuck in infinite repetition loops when forced into Tool Calls.
        # We use our robust direct completion logic instead.
        #
        # UNIFIED COMPLETION (telemetry plan 010, Phase 2): cloud models also route
        # through the direct path so every call carries trace metadata and all models
        # share one prompt serialization. Set GADS_UNIFIED_COMPLETION=false to restore
        # the legacy Pydantic AI path for cloud models.
        unified = os.getenv("GADS_UNIFIED_COMPLETION", "true").lower() == "true"
        if self.model_str == "local_model" or unified:
            path_reason = "local_model" if self.model_str == "local_model" else "unified completion"
            print(f"  [BaseAgent] Using direct robust completion for {self.name} ({path_reason})...", flush=True)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            if self.model_str != "local_model":
                # Cloud via the direct path: the 60s default is too tight for large
                # generations (Coder); match the 300s cloud convention. local_model
                # timeout is forced to 600s inside get_structured_completion.
                kwargs.setdefault("timeout", 300.0)
            content = await get_structured_completion(
                model=self.model_str,
                response_model=self.output_schema,
                messages=messages,
                stream_callback=stream_callback,
                **kwargs
            )
            return AgentResponse(content=content, model_used=self.model_str)

        self.agent._system_prompts = (self.system_prompt,)
        raw_content = ""
        
        try:
            # TRY PRIMARY RUN (Pydantic AI) for Cloud Models
            if stream_callback:
                async with self.agent.run_stream(user_prompt, **kwargs) as result:
                    try:
                        async for message in result.stream():
                            if message:
                                chunk = str(message)
                                raw_content += chunk
                                await stream_callback(chunk)
                    except AssertionError as ae:
                        if "Expected delta with content" in str(ae):
                            print(f"  [BaseAgent] Caught Pydantic AI streaming assertion error. Finalizing...", flush=True)
                        else:
                            raise ae
                    
                    final_result = await result.get_data()
                    return AgentResponse(content=final_result, model_used=self.model_str)
            else:
                result = await self.agent.run(user_prompt, **kwargs)
                return AgentResponse(content=result.data, model_used=self.model_str)
                
        except Exception as e:
            # ROBUST FALLBACK: If Pydantic AI fails for any reason, use the proven legacy completion logic
            error_str = str(e)
            print(f"  [BaseAgent] Pydantic AI failed for {self.name}: {error_str}. Attempting robust legacy fallback...", flush=True)
            
            # Construct messages for legacy completion
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            try:
                # We call the legacy wrapper which has its own robust repair logic
                print(f"  [BaseAgent] Executing legacy fallback completion for {self.name}...", flush=True)
                content = await get_structured_completion(
                    model=self.model_str,
                    response_model=self.output_schema,
                    messages=messages,
                    stream_callback=None, # No streaming in fallback for maximum stability
                    **kwargs
                )
                print(f"  [BaseAgent] Legacy fallback SUCCEEDED for {self.name}.", flush=True)
                return AgentResponse(content=content, model_used=self.model_str)
            except Exception as fallback_error:
                print(f"  [BaseAgent] Legacy fallback also FAILED for {self.name}: {fallback_error}", flush=True)
                # If even the fallback fails, raise the ORIGINAL error to maintain escalation logic
                raise e
