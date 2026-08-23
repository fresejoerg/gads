import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.prompts import prompt_registry


def _format_file_profile(f: "FileMetadata") -> str:
    """Format a rich data profile into human-readable text for the Planner prompt."""
    lines = [f"{f.name} ({f.size_mb:.2f} MB)"]
    d = f.columns_and_dtypes
    if not d or "error" in d:
        return lines[0]

    if "schema" in d:
        ncols = len(d["schema"])
        nrows = d.get("row_count")
        count_str = f" | {nrows:,} rows" if nrows else ""
        col_str = ", ".join(
            f"{k} [{v}]" for k, v in list(d["schema"].items())[:14]
        )
        lines.append(f"  Columns ({ncols}){count_str}: {col_str}")

    if d.get("null_rates"):
        nr_str = ", ".join(
            f"{k}: {v*100:.1f}%" for k, v in list(d["null_rates"].items())[:6]
        )
        lines.append(f"  Nulls: {nr_str}")

    if d.get("cardinality"):
        for col, counts in list(d["cardinality"].items())[:4]:
            top = ", ".join(
                f"{k}: {v*100:.1f}%" for k, v in list(counts.items())[:6]
            )
            lines.append(f"  {col} ({len(counts)} values): {top}")

    if d.get("imbalance_stats"):
        imbalance_alerts = []
        for col, stats in d["imbalance_stats"].items():
            ratio = stats.get("imbalance_ratio", 1.0)
            frac = stats.get("minority_class_frac", 1.0)
            if ratio > 10:
                imbalance_alerts.append(f"{col} (ratio={ratio:.1f}:1, minority={frac*100:.2f}%)")
        if imbalance_alerts:
            lines.append(f"  Class Imbalance Alerts (ratio > 10:1): " + ", ".join(imbalance_alerts))

    if d.get("numeric_stats"):
        for col, s in list(d["numeric_stats"].items())[:4]:
            lines.append(
                f"  {col}: min={s['min']:.2f}, max={s['max']:.2f}, "
                f"mean={s['mean']:.2f}, std={s['std']:.2f}"
            )

    # Excel multi-sheet
    if "excel_sheets" in d:
        lines.append(f"  Excel sheets: {d['excel_sheets']}")

    # JSON / text types
    if d.get("type") == "text":
        lines.append(f"  Text file: {d.get('words', '?')} words, {d.get('lines', '?')} lines")
    elif d.get("type") in ("records", "object", "array"):
        lines.append(f"  JSON ({d['type']}): {d.get('row_count', d.get('length', '?'))} items")

    return "\n".join(lines)

class ReconciliationReport(BaseModel):
    recipe_id: str
    rationale: str
    recommended_dag_nodes: List[Dict[str, Any]]
    invariants: List[str] = []
    skippable_nodes: List[str] = []
    schema_warnings: List[str] = []
    # True when the recipe was a WEAK Router match (confidence below the deterministic-
    # compile threshold): the Planner LLM still runs and sees this report as advisory
    # context, not a DAG to compile verbatim. False (default) preserves the original
    # deterministic-report shape used when the Planner LLM is skipped entirely.
    advisory: bool = False

class PlannerTask(BaseModel):
    """Simplified task model for the Planner to output."""
    description: str
    assigned_to: str = Field(
        description="MUST be an EXACT verbatim model ID from the AVAILABLE_MODELS_HIERARCHY (e.g. 'gemini-3.5-flash-lite'). DO NOT hallucinate model names."
    )
    postcondition: Dict[str, Any] = Field(
        description="Structural contract for success. MUST be a valid JSON object with key:value pairs. E.g., {'output_type': 'dataframe', 'required_columns': ['name']}"
    )
    attached_skills: List[str] = Field(
        default=[],
        description="List of Skill IDs to attach to this task for the worker agent's guidance."
    )

class FileMetadata(BaseModel):
    name: str
    size_mb: float
    columns_and_dtypes: Optional[Dict[str, Any]] = None

class PlannerInput(BaseModel):
    objective: str
    available_models_hierarchy: Dict[str, Any]
    available_files: List[FileMetadata]
    knowledge_report: Optional[ReconciliationReport] = None
    available_skills: List[Dict[str, Any]] = []
    critique_feedback: Optional[str] = None
    previous_plan: Optional[List[str]] = None
    user_hints: Optional[Dict[str, Any]] = None


class PlannerOutput(BaseModel):
    steps: List[PlannerTask] = Field(description="A list of discrete steps with postcondition contracts.")


class DataSciencePlanner(BaseAgent[PlannerInput, PlannerOutput]):
    def __init__(self, model: str = "local_model"):
        super().__init__(
            name="Planner",
            model=model,
            system_prompt=prompt_registry.get_prompt("Planner"),
            output_schema=PlannerOutput
        )

    async def run(self, input_data: PlannerInput, stream_callback=None, **kwargs) -> Any:
        import re
        from gads.core.registry import get_local_only
        
        # Refresh prompt from registry in case of hot-reload
        base_prompt = prompt_registry.get_prompt(self.name)
        
        if get_local_only():
            # Strip model hierarchy json
            hierarchy_str = (
                "local_only mode is active. All tiers (T1, T2, T3, T4) are collapsed. "
                "You MUST assign all tasks to the model ID: 'local_model'."
            )
            
            # Filter skills list to only those relevant to the objective or matched recipe
            objective_lower = input_data.objective.lower()
            relevant_skills = []
            
            recipe_keywords = set()
            if input_data.knowledge_report:
                recipe_id = input_data.knowledge_report.recipe_id.lower()
                recipe_keywords.update(re.findall(r"\b\w{3,}\b", recipe_id))
            
            for skill in input_data.available_skills:
                skill_id = skill.get("id", "").lower()
                triggers = skill.get("triggers", [])
                
                # Check match against objective or recipe
                match = False
                if any(t.lower() in objective_lower for t in triggers):
                    match = True
                elif skill_id in objective_lower:
                    match = True
                elif recipe_keywords and any(t.lower() in recipe_keywords for t in triggers):
                    match = True
                
                if match:
                    relevant_skills.append(skill)
            
            # If nothing matched, default to a minimal set
            if not relevant_skills:
                fallback_ids = {"visualization_best_practices", "sandbox_environment", "tabular_visualization"}
                relevant_skills = [s for s in input_data.available_skills if s.get("id") in fallback_ids]
                
            # Serialize id + description only — triggers are keyword-matcher internals,
            # not useful for skill selection, and they bloat the prompt.
            skills_str = json.dumps(
                [{"id": s["id"], "description": s.get("description", "")} for s in relevant_skills],
                indent=2
            )
            print(f"  [Planner] ✂️ Trimmed system prompt: model hierarchy stripped, skills pruned from {len(input_data.available_skills)} to {len(relevant_skills)}.", flush=True)
        else:
            hierarchy_str = json.dumps(input_data.available_models_hierarchy, indent=2)
            skills_str = json.dumps(
                [{"id": s["id"], "description": s.get("description", "")} for s in input_data.available_skills],
                indent=2
            )
        
        # Format files with rich profiles (schema, null rates, cardinality, numeric stats)
        files_str = (
            "\n\n".join(_format_file_profile(f) for f in input_data.available_files)
            if input_data.available_files else "None"
        )
        
        knowledge_str = json.dumps(input_data.knowledge_report.dict(), indent=2) if input_data.knowledge_report else "No matching SOP found. Use generic data science reasoning."
        
        hints_str = json.dumps(input_data.user_hints, indent=2) if input_data.user_hints else "None provided."
        formatted_prompt = base_prompt.format(
            hierarchy_json=hierarchy_str,
            files_list=files_str,
            knowledge_json=knowledge_str,
            skills_json=skills_str,
            user_hints=hints_str
        )
        
        user_content = f"USER OBJECTIVE: {input_data.objective}"
        if input_data.critique_feedback:
            user_content += f"\n\n--- PREVIOUS ATTEMPT REJECTED BY QA ---\nQA Feedback: {input_data.critique_feedback}"
            if input_data.previous_plan:
                user_content += f"\n\nRejected Plan:\n" + "\n".join([f"- {s}" for s in input_data.previous_plan])
            
            user_content += f"\n\nPlease generate a new plan that specifically addresses the missing requirements noted by QA."

        # Use super().run to get streaming support
        return await super().run(
            user_content,
            system_prompt=formatted_prompt,
            **kwargs
        )
