from typing import List
from gads.core.models import Task

class HistoryRenderer:
    HISTORY_STDOUT_CAP: int = 2000

    @classmethod
    def build_coder_context(cls, tasks: List[Task], current_task_idx: int) -> str:
        """Produces the sliding-window context string for the Coder.

        Rules:
        - Failed tasks always render their full traceback and stdout regardless of window position.
        - Successful/Pending tasks in the window (first task, current task, and task before current)
          render full details (description, code, stdout).
        - Successful tasks outside the window are distilled to their orchestrator_summary.
        """
        hardened_context_parts = []
        for i, t in enumerate(tasks):
            status_str = t.status.upper()
            stdout_str = (t.result_json or {}).get("stdout", "")
            code_str = (t.result_json or {}).get("code", "")
            
            is_failed = (t.status.lower() == "failed")
            
            # Window check: 2+1 Model (First Task + Current Task + Immediately Preceding Task)
            is_first = (i == 0)
            is_recent = (i == current_task_idx or i == current_task_idx - 1)
            
            if is_failed:
                # Failed task: description + error + last 2000 chars of stdout
                err_msg = t.error or "Unknown error"
                # Cap stdout by taking the last HISTORY_STDOUT_CAP characters
                capped_stdout = stdout_str[-cls.HISTORY_STDOUT_CAP:] if len(stdout_str) > cls.HISTORY_STDOUT_CAP else stdout_str
                task_ctx = (
                    f"### TASK: {t.description}\n"
                    f"Status: {status_str}\n"
                    f"Error Traceback:\n{err_msg}\n"
                    f"Stdout Output:\n{capped_stdout}"
                )
            elif is_first or is_recent:
                # Successful or active/pending task in-window: description + code + stdout
                capped_stdout = stdout_str[:cls.HISTORY_STDOUT_CAP]
                task_ctx = (
                    f"### TASK: {t.description}\n"
                    f"Status: {status_str}\n"
                    f"Code Used:\n```python\n{code_str}\n```\n"
                    f"Stdout Output:\n{capped_stdout}"
                )
            else:
                # Successful task out-of-window: distilled metadata only
                summary = (t.result_json or {}).get("orchestrator_summary", f"{status_str}: {t.description}")
                task_ctx = f"### TASK (HISTORICAL): {t.description}\nStatus: {status_str}\nSummary: {summary}"
                
            hardened_context_parts.append(task_ctx)
            
        return "\n\n---\n\n".join(hardened_context_parts)
