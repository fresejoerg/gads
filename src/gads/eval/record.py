"""Load one GADS execution as an immutable record the rubrics can read.

The run's record of contracts and results is Postgres + the workspace, not the MLflow
trace: the trace (tr-<project uuid hex>) carries stage spans, Scope Guard verdicts,
native invocations and per-call tokens/cost, but neither the postconditions nor what each
node captured (checked on 56dbecc4, 2026-11-11). Sources:

  task.postcondition_json   the node's contract (recipe-compiled or Planner-drafted)
  task.result_json          what the node did: metrics_captured, semantic_insights,
                            model_used, usage, artifact_files, code, stdout
  project.last_state_json   the END-OF-SESSION kernel snapshot + run metadata
                            (spec_filename, engine_id, dial, taxonomy)
  <workspace>/              metrics.json, final_dashboard.html, research_report.md,
                            workflow_execution.py, applied_recipe_<id>.md

One record = the main workflow of one project. Follow-up lane tasks (a different
instruction_id) are not part of the execution and are excluded.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from gads.core import config

# Pipeline stages: tasks assigned to these are orchestration, not execution nodes.
STAGE_AGENTS = frozenset({
    "System", "DataAnalyzer", "SpecDrafter", "Router", "DataSampler", "Planner",
    "PlanCritique", "CompletenessVerifier", "Synthesizer", "Critique",
})

# Description of the task run_agent_workflow creates first, before anything else.
INIT_TASK_PREFIX = "System is grounding session state"

WORKSPACE_ROOT = Path(config.WORKSPACE_ROOT)


@dataclass(frozen=True)
class TaskView:
    """The fields of a Task row the rubrics use, detached from the DB session."""
    id: str
    assigned_to: str
    status: str
    description: str
    created_at: str
    postcondition_json: Optional[Dict[str, Any]]
    result_json: Optional[Dict[str, Any]]
    error: Optional[str]

    @property
    def is_stage(self) -> bool:
        return self.assigned_to in STAGE_AGENTS

    @property
    def node_id(self) -> str:
        """Recipe node id when compiled from a recipe, else a slug of the task description
        (drafted plans: descriptions run to kilobytes; the slug is the error ledger's own
        stable per-node key). Display key only: pass@model attribution keeps dedupe on the
        full description, exactly as the dial ledger does."""
        rid = (self.postcondition_json or {}).get("recipe_node_id")
        if rid:
            return rid
        from gads.core.error_ledger import node_slug
        return "task:" + node_slug(self.description)


@dataclass
class Attempt:
    """One workflow attempt: a Planner task and every task after it up to the next Planner."""
    index: int                         # 1-based
    planner: Optional[TaskView]
    tasks: List[TaskView] = field(default_factory=list)

    @property
    def exec_tasks(self) -> List[TaskView]:
        return [t for t in self.tasks if not t.is_stage]

    @property
    def recipe_sections(self) -> Optional[List[Dict[str, Any]]]:
        return ((self.planner.result_json or {}).get("recipe_sections")
                if self.planner else None)

    @property
    def recipe_id(self) -> Optional[str]:
        return (self.planner.result_json or {}).get("recipe_id") if self.planner else None


@dataclass
class RunRecord:
    project_id: str
    name: str
    objective: str
    narrative: Optional[str]
    created_at: str
    updated_at: str
    last_state: Dict[str, Any]
    tasks: List[TaskView]              # main-workflow tasks, creation order
    attempts: List[Attempt]
    workspace: Path
    excluded_followup_tasks: int = 0
    main_workflow_found: bool = True   # False: tasks exist but none belong to a workflow
    executions: int = 1                # workflow launches on this project (latest is scored)

    # ---- derived ---------------------------------------------------------------
    @property
    def final_attempt(self) -> Optional[Attempt]:
        return self.attempts[-1] if self.attempts else None

    @property
    def spec_filename(self) -> Optional[str]:
        return self.last_state.get("spec_filename")

    @property
    def kernel_vars(self) -> Dict[str, Any]:
        """End-of-session kernel snapshot: variable name -> {type, value|shape|columns}.

        last_state_json mixes this snapshot with run metadata; metadata keys are the
        non-variable ones the server writes (spec_filename, dial, taxonomy, __schemas__...).
        A variable entry is a dict carrying a "type"."""
        return {k: v for k, v in self.last_state.items()
                if isinstance(v, dict) and "type" in v}

    def stage_tasks(self, agent: str) -> List[TaskView]:
        return [t for t in self.tasks if t.assigned_to == agent]

    def workspace_file(self, name: str) -> Optional[Path]:
        p = self.workspace / name
        return p if p.exists() else None

    def metrics_json(self) -> Optional[Dict[str, Any]]:
        p = self.workspace_file("metrics.json")
        if not p:
            return None
        try:
            data = json.loads(p.read_text())
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def applied_recipe_version(self) -> Optional[str]:
        """`version:` from the applied_recipe_<id>.md frontmatter the run copied into its
        workspace; the recipe as it was AT RUN TIME, not as it is now."""
        for p in sorted(self.workspace.glob("applied_recipe_*.md")):
            try:
                head = p.read_text(errors="replace").split("\n---", 1)[0]
            except Exception:
                continue
            m = re.search(r"^version:\s*['\"]?([^'\"\n]+)", head, re.M)
            if m:
                return m.group(1).strip()
        return None


def _view(t) -> TaskView:
    return TaskView(
        id=str(t.id), assigned_to=t.assigned_to or "", status=t.status or "",
        description=t.description or "", created_at=t.created_at.isoformat() if t.created_at else "",
        postcondition_json=t.postcondition_json, result_json=t.result_json, error=t.error,
    )


def partition_attempts(tasks: List[TaskView]) -> List[Attempt]:
    """Split the main workflow at each Planner task. Tasks before the first Planner
    (System init, DataAnalyzer, SpecDrafter, Router, DataSampler) belong to no attempt.

    Every replan persists a new Planner task and new execution tasks, while the earlier
    attempt's rows stay (failed, or never-run and later marked failed), so an unpartitioned
    view would score a node twice and count a superseded failure against the run."""
    attempts: List[Attempt] = []
    for t in tasks:
        if t.assigned_to == "Planner":
            attempts.append(Attempt(index=len(attempts) + 1, planner=t))
        elif attempts:
            attempts[-1].tasks.append(t)
    return attempts


def load_run(project_id: str | uuid.UUID, workspace_root: Optional[Path] = None) -> RunRecord:
    """Read one project's main workflow from Postgres + its workspace. Read-only."""
    from sqlmodel import Session, select
    from gads.core.database import engine
    from gads.core.models import Project, Task

    pid = uuid.UUID(str(project_id))
    with Session(engine) as s:
        p = s.get(Project, pid)
        if p is None:
            raise LookupError(f"no project {pid}")
        rows = s.exec(select(Task).where(Task.project_id == pid)
                      .order_by(Task.created_at)).all()
        # The main workflow's instruction is the one its System init task carries; follow-up
        # lane tasks carry their own instruction_id. Old projects with no instruction ids at
        # all compare None == None and keep every task. No init task at all means the main
        # workflow's rows are gone (e.g. projects recovered after the 2026-07-31 table loss,
        # where only follow-up tasks survived): nothing is attributed to the execution.
        # POST /projects with existing_project_id relaunches a workflow on an existing
        # project: each launch gets its own init task + instruction, and the scorecard
        # describes the LATEST execution (the file is per project).
        inits = [r for r in rows if r.assigned_to == "System"
                 and (r.description or "").startswith(INIT_TASK_PREFIX)]
        init = inits[-1] if inits else None
        main = [r for r in rows if r.instruction_id == init.instruction_id] if init else []
        views = [_view(r) for r in main]
        rec = RunRecord(
            project_id=str(pid), name=p.name or "", objective=p.objective or "",
            narrative=p.narrative,
            created_at=p.created_at.isoformat() if p.created_at else "",
            updated_at=p.updated_at.isoformat() if p.updated_at else "",
            last_state=dict(p.last_state_json or {}),
            tasks=views, attempts=partition_attempts(views),
            workspace=(workspace_root or WORKSPACE_ROOT) / str(pid),
            excluded_followup_tasks=len(rows) - len(main),
            main_workflow_found=init is not None or not rows,
            executions=len(inits),
        )
    return rec
