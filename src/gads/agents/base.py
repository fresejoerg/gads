from abc import ABC
from typing import TypeVar, Generic, Type, Any, Optional
from pydantic import BaseModel

TIn = TypeVar("TIn", bound=BaseModel)
TOut = TypeVar("TOut", bound=BaseModel)

class AgentResponse(BaseModel, Generic[TOut]):
    """Wrapper for agent output that includes metadata."""
    content: TOut
    model_used: str

class BaseAgent(ABC, Generic[TIn, TOut]):
    """Base class for all GADS agents: one structured completion per call.

    Every model — local and cloud — goes through `core/llm.get_structured_completion`
    (instructor over the LiteLLM proxy). One path means one prompt serialization and
    one place where trace metadata (task_id / attempt / prompt_version / engine_id) is
    stamped, which the trace↔task join depends on (MLflow now; the Langfuse v2 history read
    by harvest_coder_traces.py before that).
    """
    
    def __init__(self, name: str, model: str, system_prompt: str, output_schema: Type[TOut]):
        self.name = name
        self.model_str = model
        # Callers read and assign `.model` too (executor logging/returns, server's per-task
        # model switch). It was the Pydantic AI model object until that path was removed;
        # it is now the same model string as `model_str`.
        self.model = model
        self.system_prompt = system_prompt
        self.output_schema = output_schema

    async def run(self, input_data: Any, system_prompt: Optional[str] = None, **kwargs) -> AgentResponse[TOut]:
        """Run one structured completion against the agent's output schema.

        Subclasses that resolve {placeholder} templates MUST pass the formatted
        prompt via `system_prompt` — otherwise this method re-fetches the raw
        registry template.
        """
        from gads.core.prompts import prompt_registry
        from gads.core.llm import get_structured_completion

        self.system_prompt = system_prompt if system_prompt is not None else prompt_registry.get_prompt(self.name)
        user_prompt = input_data if isinstance(input_data, str) else input_data.model_dump_json()
        stream_callback = kwargs.pop("stream_callback", None)

        print(f"  [BaseAgent] Structured completion for {self.name} ({self.model_str})...", flush=True)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        if self.model_str != "local_model":
            # The 60s default is too tight for large cloud generations (Coder); match
            # the 300s cloud convention. local_model timeout is forced to 600s inside
            # get_structured_completion.
            kwargs.setdefault("timeout", 300.0)
        try:
            content = await get_structured_completion(
                model=self.model_str,
                response_model=self.output_schema,
                messages=messages,
                stream_callback=stream_callback,
                **kwargs
            )
        except Exception as e:
            self._capture(messages, None, f"{type(e).__name__}: {e}")
            raise
        self._capture(messages, content, None)
        return AgentResponse(content=content, model_used=self.model_str)

    def _capture(self, messages, content, error):
        """Distillation capture for every stage but the Coder, which the executor records
        with its execution outcome (core/distill_capture.py). Never raises."""
        if self.name == "CodeGenerator":
            return
        try:
            from gads.core.distill_capture import record_call
            record_call(stage=self.name, model=self.model_str, messages=messages,
                        output=content.model_dump() if content is not None else None,
                        error=error)
        except Exception:
            pass
