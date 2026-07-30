from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent, AgentResponse
from gads.core.prompts import prompt_registry
import json

class CoderOutput(BaseModel):
    code: str = Field(description="The complete Python code block to execute.")
    libraries_used: List[str] = Field(default_factory=list, description="List of libraries imported.")
    explanation: Optional[str] = Field(None, description="Brief explanation of what the code does.")

class CoderInput(BaseModel):
    task_description: str
    available_files: List[str] = []
    file_schemas: Dict[str, Any] = {}  # column→dtype maps keyed by filename
    authoritative_state: Dict[str, Any] = {} # Ground truth from the kernel
    previous_code: Optional[str] = None
    error_feedback: Optional[str] = None
    common_pitfalls: Optional[str] = None  # recurring errors this recipe step hit in past runs
    skills_context: Optional[str] = None
    task_id: Optional[str] = None
    postcondition_contract: Optional[Dict[str, Any]] = None
    state_summary: Optional[str] = None

class CodeGeneratorAgent(BaseAgent[CoderInput, CoderOutput]):
    def __init__(self, model: str = "local_model"):
        super().__init__(
            name="CodeGenerator",
            model=model,
            system_prompt=prompt_registry.get_prompt("CodeGenerator"),
            output_schema=CoderOutput
        )

    async def run(self, input_data: CoderInput, **kwargs) -> Any:
        # Refresh prompt from registry
        base_prompt = prompt_registry.get_prompt(self.name)
        
        # Use provided state summary or fallback to JSON of dict
        state_summary = input_data.state_summary or json.dumps(input_data.authoritative_state, indent=2)
        
        files_lines = []
        for f in input_data.available_files:
            schema_info = input_data.file_schemas.get(f, {}).get("schema")
            if schema_info:
                files_lines.append(f"'{f}' — columns: {list(schema_info.keys())}")
            else:
                files_lines.append(f"'{f}'")
        files_summary = "\n".join(files_lines) if files_lines else "None"
        contract_summary = json.dumps(input_data.postcondition_contract, indent=2) if input_data.postcondition_contract else "None. Just fulfill the task description."

        print(f"    [Coder] Preparing prompt. Available files: {', '.join(input_data.available_files)}", flush=True)

        formatted_prompt = base_prompt.format(
            state_summary=state_summary,
            files_list=files_summary,
            skills_context=input_data.skills_context or "No specific skills required for this task.",
            contract_json=contract_summary
        )

        # Runtime-conditional guidance: only included when the workspace facts call
        # for it, so single-file tasks don't carry merge instructions they can copy
        # incorrectly.
        tabular_files = [f for f in input_data.available_files if f.endswith((".csv", ".parquet"))]
        if len(tabular_files) > 1:
            formatted_prompt += (
                "\n\n## MULTI-FILE ASSEMBLY\n"
                "Engineered features may be split across multiple files. When a task needs features "
                "from several files, merge them explicitly on the shared key column — never assume "
                "they are already in one DataFrame:\n"
                "```python\n"
                "df = df_base.merge(df_feat_a, on='id').merge(df_feat_b, on='id')\n"
                "```\n"
                "Use the EXACT column names from the schemas in AVAILABLE FILES. Do NOT invent column names."
            )

        # Build dynamic kernel-reuse warning from live state
        kernel_warnings = []
        try:
            state_data = json.loads(state_summary) if isinstance(state_summary, str) else state_summary
            if isinstance(state_data, dict):
                for var, desc in state_data.items():
                    if isinstance(desc, str) and desc.startswith("DataFrame"):
                        kernel_warnings.append(f"`{var}`: {desc}")
        except Exception:
            pass

        user_content = f"TASK: {input_data.task_description}\n\n"
        user_content += "CRITICAL: Use ONLY the exact filenames listed in AVAILABLE FILES (system prompt).\n\n"
        if kernel_warnings:
            user_content += (
                "KERNEL MEMORY — these DataFrames are ALREADY LOADED. "
                "Use them directly. DO NOT call pd.read_csv() or reload from disk:\n"
                + "\n".join(kernel_warnings) + "\n\n"
            )

        # RETRY FEEDBACK: on a re-attempt, show the model its own failed code and the
        # exact error so it can CORRECT the mistake instead of blindly re-generating.
        # (The executor supplies previous_code/error_feedback; without this block they
        # were silently discarded, so local-model retries repeated the same error.)
        # First-attempt prior from the cross-run error ledger: recurring structural errors
        # this exact recipe step has produced in the past. Surface them up front so the
        # model can avoid a known dead end on its FIRST try (not just react after failing).
        if input_data.common_pitfalls:
            user_content += (
                "## COMMON PITFALLS FOR THIS STEP (from past runs — avoid these)\n"
                "This recipe step has repeatedly failed in earlier runs with the error(s) "
                "below. Write code that does NOT hit them:\n\n"
                f"{input_data.common_pitfalls}\n\n"
            )

        if input_data.error_feedback:
            # Feed back the ERROR(S) only — not the previous code. The error is the signal
            # that nudges the model toward a different approach; re-showing its own failed
            # code tends to anchor it on the same structure. The executor accumulates every
            # prior attempt's error here (most recent last), so the model can steer clear of
            # all the dead ends it has already hit.
            user_content += (
                "## PREVIOUS ATTEMPTS FOR THIS TASK FAILED\n"
                "Your earlier code for THIS task raised the error(s) below (most recent last). "
                "Diagnose the root cause and take a genuinely DIFFERENT approach — do not repeat "
                "what has already failed:\n\n"
                f"{input_data.error_feedback}\n\n"
            )

        # RAW CODE MODE for local models: forcing a small model to escape an
        # entire Python script inside a JSON string field triggers degeneration
        # loops (observed: gemma-4-12b burning the full 8K token budget on one
        # derailed CoderOutput generation, repeatedly). Ask for plain fenced
        # code and build the schema object ourselves.
        if self.model_str == "local_model":
            from gads.core.llm import get_code_completion
            raw_system = formatted_prompt + (
                "\n\n## OUTPUT FORMAT — OVERRIDES ALL PRIOR FORMAT INSTRUCTIONS\n"
                "Return ONLY the Python code for this task, inside a single ```python fence.\n"
                "Do NOT return JSON. Do NOT write prose before or after the fence."
            )
            raw_user = user_content + "Return ONLY a single ```python code block."
            print(f"    [Coder] RAW CODE MODE (local_model) — plain fenced code, no JSON envelope.", flush=True)
            code = await get_code_completion(
                model=self.model_str,
                messages=[
                    {"role": "system", "content": raw_system},
                    {"role": "user", "content": raw_user}
                ],
                stream_callback=kwargs.pop("stream_callback", None),
            )
            return AgentResponse(
                content=CoderOutput(code=code, libraries_used=[], explanation="raw code mode (local)"),
                model_used=self.model_str
            )

        user_content += "Return the required FLAT JSON object."

        # Use super().run to get streaming support
        return await super().run(
            user_content,
            system_prompt=formatted_prompt,
            **kwargs
        )
