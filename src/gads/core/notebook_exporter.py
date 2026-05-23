import json
import uuid
import shutil
from datetime import datetime
from typing import List, Optional
from sqlmodel import Session, select
from gads.core.database import engine
from gads.core.models import Task

# Agents that produce no executable code and should be excluded from the bundle
_SYSTEM_AGENTS = {
    "System", "Router", "Planner", "SpecDrafter",
    "Synthesizer", "Critique", "PlanCritique", "Auditor"
}

_GADS_SHIM = """\
# ---------------------------------------------------------------------------
# GADS telemetry shim — enables standalone execution outside the sandbox.
# gads_emit_insight() is a no-op here; results were captured during the run.
# ---------------------------------------------------------------------------
if '_gads_insights' not in dir():
    _gads_insights = []

def gads_emit_insight(artifact, insight, evidence=""):
    pass
# ---------------------------------------------------------------------------

"""

def _sentinel_cleaned_stdout(stdout: str) -> str:
    """Strip GADS internal sentinel lines from stdout for display."""
    keep = []
    for line in stdout.splitlines():
        if not any(line.startswith(p) for p in (
            "GADS_INSIGHTS_JSON:", "GADS_FLOOR_JSON:",
            "GADS_STATE_SNAPSHOT:", "GADS_HYPOTHESIS_JSON:"
        )):
            keep.append(line)
    return "\n".join(keep).strip()


def _get_execution_tasks(project_id: uuid.UUID) -> List[Task]:
    """Return tasks that have executed code, ordered by creation time."""
    with Session(engine) as session:
        tasks = session.exec(
            select(Task)
            .where(Task.project_id == project_id)
            .order_by(Task.created_at)
        ).all()
        # Detach from session by accessing all fields
        result = []
        for t in tasks:
            if (
                t.assigned_to not in _SYSTEM_AGENTS
                and (t.result_json or {}).get("code")
                and t.status in ("completed", "failed")
            ):
                result.append(t)
        return result


def _count_bypassed(project_id: uuid.UUID) -> int:
    """Count tasks that were bypassed by the RuntimeOracle (long-running)."""
    with Session(engine) as session:
        tasks = session.exec(
            select(Task)
            .where(Task.project_id == project_id)
        ).all()
        return sum(
            1 for t in tasks
            if t.assigned_to not in _SYSTEM_AGENTS
            and "BYPASSED" in ((t.result_json or {}).get("stdout") or "")
        )


def export_python_script(project_id: uuid.UUID, workspace_dir: str) -> str:
    """Write workflow_execution.py to workspace. Returns the output path."""
    tasks = _get_execution_tasks(project_id)
    bypassed = _count_bypassed(project_id)

    lines = [
        "# GADS Workflow Execution Script",
        f"# Project:   {project_id}",
        f"# Generated: {datetime.now().isoformat()}",
        f"# Tasks with code: {len(tasks)} | Bypassed (long-running, see handover ZIPs): {bypassed}",
        "#",
        "# This script reproduces the code executed during the GADS workflow.",
        "# Run in a Python environment with project data files available.",
        "",
        _GADS_SHIM,
    ]

    for i, t in enumerate(tasks, 1):
        code = (t.result_json or {}).get("code", "").strip()
        model = (t.result_json or {}).get("model_used", t.assigned_to)
        sep = "=" * 70
        lines += [
            f"# {sep}",
            f"# TASK {i}: {t.description}",
            f"# Model: {model}  |  Status: {t.status}",
            f"# {sep}",
            code,
            "",
        ]

    out_path = f"{workspace_dir}/workflow_execution.py"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def export_notebook(project_id: uuid.UUID, workspace_dir: str) -> str:
    """Write workflow_execution.ipynb to workspace. Returns the output path."""
    tasks = _get_execution_tasks(project_id)
    bypassed = _count_bypassed(project_id)
    cells = []

    # Title cell
    header_lines = [
        f"# GADS Workflow Execution\n",
        f"**Project:** `{project_id}`  \n",
        f"**Generated:** {datetime.now().isoformat()}  \n",
        f"**Tasks with code:** {len(tasks)} | **Bypassed:** {bypassed}  \n\n",
        "Reproduces the code executed by the GADS multi-agent pipeline. "
        "The `gads_emit_insight` function is redefined as a no-op — "
        "telemetry was captured during the original run.\n"
    ]
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": header_lines
    })

    # Shim cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [_GADS_SHIM]
    })

    for i, t in enumerate(tasks, 1):
        code = (t.result_json or {}).get("code", "").strip()
        model = (t.result_json or {}).get("model_used", t.assigned_to)
        stdout = (t.result_json or {}).get("stdout", "") or ""

        desc_lines = [
            f"## Task {i}: {t.description}\n",
            f"**Model:** `{model}`  |  **Status:** `{t.status}`\n",
        ]
        if t.postcondition_json:
            ot = (t.postcondition_json or {}).get("output_type")
            if ot:
                desc_lines.append(f"**Output type:** `{ot}`\n")

        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": desc_lines
        })

        clean_stdout = _sentinel_cleaned_stdout(stdout)
        outputs = []
        if clean_stdout:
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": [line + "\n" for line in clean_stdout.splitlines()]
            })

        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": outputs,
            "source": [line + "\n" for line in code.splitlines()]
        })

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13.0"
            }
        },
        "cells": cells
    }

    out_path = f"{workspace_dir}/workflow_execution.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    return out_path


def copy_applied_recipe(
    recipe_filepath: Optional[str],
    workspace_dir: str
) -> Optional[str]:
    """Copy the applied recipe file into the workspace. Returns dest path or None."""
    if not recipe_filepath:
        return None
    import os
    if not os.path.exists(recipe_filepath):
        return None
    dest = f"{workspace_dir}/applied_recipe_{os.path.basename(recipe_filepath)}"
    shutil.copy2(recipe_filepath, dest)
    return dest
