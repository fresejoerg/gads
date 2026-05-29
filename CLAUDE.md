# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack & Commands

GADS is a Python 3.13 project managed by `uv`. It depends on an **external** companion stack `MyLocalStack` providing:
- LiteLLM proxy on `http://localhost:4000/v1` (model gateway)
- IPython-kernel sandbox on `http://localhost:8000`
- Shared workspace dir at `/home/joergf/projects/MyLocalStack/data/workspaces/<project_id>/` (hardcoded — see `server.WORKSPACE_ROOT`, `sandbox.list_workspace_files`, `handover.create_bundle`)

Persistence is **PostgreSQL** (not SQLite — `gads.db` is a vestigial empty file). `GADS_DATABASE_URL` is required at import time; the database module raises if it's missing.

```bash
uv sync                                                  # install deps
./start_backend.sh                                       # FastAPI on :8001 (also: PYTHONPATH=src uv run uvicorn gads.core.server:app --port 8001)
./scripts/run_streamlit.sh                               # Streamlit UI on :8003 (sources .env, sets PYTHONPATH)
curl http://localhost:8001/health                        # health check
```

No formal test suite. The `stress_test_*.py` files at the repo root are ad-hoc scripts hitting the running backend, not pytest tests. `main.py` is a legacy standalone demo of the executor (uses `core/state.Blackboard` which is otherwise unused) — don't treat it as an entry point.

## Architecture

### The Pipeline (server.py:run_agent_workflow)

The entire multi-agent orchestration lives in `src/gads/core/server.py` as one long `async def run_agent_workflow`. Each stage runs in a `while True:` retry loop that escalates the model via `get_next_model_dynamic` on failure. Stages:

1. **Router** (`agents/router.py`, T3) — classifies the objective into `task_type`/`data_modality`, picks a matching `Recipe`.
2. **Planner** (`agents/planner.py`, T2) — decomposes objective into `PlannerTask[]`, each with a `postcondition_json` contract, an `assigned_to` model from the live hierarchy, and `attached_skills`.
3. **PlanCritique** (`agents/plan_critique.py`, T2) — audits the plan before execution; can reject (re-plan with feedback) or mark `is_terminal_failure` (halt entire workflow). The outer loop allows `MAX_WORKFLOW_ATTEMPTS = 3` replans.
4. **Per-task Execution** (`core/executor.py:ExecutionManager.run_task`) — for each `Task`:
   - Probes the IPython kernel for live variables (`namespace_summary`) so the Coder sees ground truth.
   - Calls **Coder** (`agents/workers/coder.py`) to generate Python.
   - **RuntimeOracle** (`core/runtime_oracle.py`) AST-estimates runtime; >280s → bypass, generate **handover ZIP** via `HandoverManager`, continue.
   - Wraps code with `gads_emit_insight()` preamble + structural DataFrame probe postamble. Executes in sandbox.
   - On error: `ExecutionHub.escalate_task` bumps the task to the next model tier (max 2 escalations).
   - On success: scans the workspace for new files, registers `.png` as base64 plots and `.json` as interactive Plotly artifacts (via `introspection.harden_json_artifact` which strips binary data).
   - Hallucination guard: scans stdout for tokens like "mock data" / "simulating data" and fails the task even if it didn't raise.
5. **Synthesizer** (`agents/workers/synthesizer.py`, T2) — writes narrative + takeaways + per-artifact captions.
6. **Critique** (`agents/workers/critique.py`, T2) — QA pass; can reject and re-trigger the synthesis loop. Critique sees a distilled-Markdown preview of the dashboard (`core/distiller.py`), not raw HTML.
7. **Reporting** (`core/reporting.py`) — emits `final_dashboard.html` (Jinja2 template at `src/gads/templates/dashboard.html.j2`) and `research_report.md` to the workspace.

### Async Plumbing

- **Transactional Outbox**: agents write `OutboxEvent` rows; `core/bus.dispatcher_loop` polls every 500ms and broadcasts via WebSocket (`/ws`). UI subscribes with `?last_seq=N` for replay-on-reconnect.
- **Live streaming**: `LIVE_STREAMS: Dict[task_id, {reasoning, stdout}]` is a server-side dict polled by the UI via `GET /tasks/{id}/stream`. The Executor's `poll_logs_loop` polls the sandbox's `/sessions/{id}/logs` endpoint every 1s to populate stdout incrementally.
- **Watchdog**: `ExecutionHub.watchdog_loop` re-queues tasks whose `heartbeat` is older than 5 minutes.
- **Cancellation**: `POST /projects/{id}/cancel` sets `project.narrative = "[CANCELLED] …"`; the workflow polls `is_cancelled()` between stages. Cancel also force-resets the sandbox session.
- **Duplicate guard**: `ACTIVE_WORKFLOWS: set[uuid.UUID]` (in-process) prevents double-launches; supplemented by a DB query for existing `pending`/`running` tasks.

### Model Tiers & Escalation (core/registry.py)

`TIER_MAPPING` defines T1 (Opus/Pro) → T2 (Sonnet/Flash) → T3 (Haiku/Flash-Lite) → T4 (`local_model`). `get_next_model_dynamic` does **intra-tier random fallback first**, then jumps to the next tier in `TIER_ORDER = ["T3", "T2", "T1"]`. **T4 (local) never escalates to cloud** — this is a hard mandate, enforced in two places. The Planner's prompt must only emit model strings present in the runtime hierarchy fetched from LiteLLM (the orchestrator sanitizes hallucinated model names).

Toggle with `POST /config {local_only, random_routing}` or `GADS_LOCAL_ONLY=true` in `.env`. In `local_only` mode all four tiers collapse to `["local_model"]`.

### BaseAgent (agents/base.py)

All agents extend `BaseAgent[TIn, TOut]`. There are **three completion paths**:
1. `local_model` → bypasses Pydantic AI entirely (it deadlocks local models in tool-call loops) and uses `core/llm.get_structured_completion` with `instructor` directly.
2. Cloud + streaming → Pydantic AI `agent.run_stream` with a callback.
3. Pydantic AI failure → falls back to `get_structured_completion` (no streaming) before re-raising.

The `local_model` branch also injects `repetition_penalty=1.1, temperature=0.1` to prevent repetition loops.

### Knowledge & Prompt System

- **Recipes** (`src/gads/knowledge/recipes/*.md`) — YAML frontmatter (id, applies_when, requires, dag, invariants) + Markdown body with `## Rationale` heading. Loaded by `KnowledgeRegistry`. The Router matches; if matched, the Planner receives a `ReconciliationReport` as a prior.
- **Skills** (`src/gads/knowledge/skills/*.md`) — keyword-triggered expertise injected into the Coder's prompt. Planner-attached + keyword-matched skills are deduplicated and concatenated.
- **Prompts** (`core/prompts.py`) — `FACTORY_DEFAULTS` are the source of truth; user overrides are stored as files in `gads_data/prompts/` and reloaded on every agent run (hot-edit via `POST /prompts/{agent_name}`). Templates use `{placeholder}` style — when adding new placeholders, both the factory default AND every agent's `formatted_prompt = base_prompt.format(...)` call must be updated together.

### Observability

`core/llm.trace_context` is a `ContextVar` set at the top of `run_agent_workflow`. `get_structured_completion` reads it to inject Langfuse trace headers and LiteLLM metadata on every call. Stages also create explicit `langfuse_client.trace(...).span(...)` spans. **Don't forget to update `trace_context.get().update({...})` when adding a new agent stage** or its spans will hang off the wrong parent.

### Postcondition Contracts (execution_hub.py:validate_contract)

Tasks declare `postcondition_json` with `output_type`, `required_columns`, and optional `required_insights`. The hub checks for the columns in stdout OR in the live kernel state. `required_insights` checks emitted `gads_emit_insight()` payloads — but is currently a **soft fail** (logs a warning, does not fail the task) to tolerate forgetful local models.

### Project Specs (server.py:launch_from_spec)

`POST /projects/from-spec` reads a Markdown file from `specs/` with YAML frontmatter. Supported keys:

| Key | Type | Description |
|-----|------|-------------|
| `name` | str | Project display name |
| `datasets` | list[str] | Paths relative to `GADS_DATASETS_ROOT` (default `/home/joergf/datasets`) — **copied** (not symlinked) into workspace |
| `recipes` | list[str] | Recipe filenames to validate against the registry |
| `target_column` | str | Forwarded to Planner as a hint |
| `feature_columns` | list[str] | Forwarded to Planner as a hint |
| `filters` | str | Forwarded to Planner as a hint |
| `domain` | str | Forwarded to Planner as a hint |
| `recipe_id` | str | Forwarded to Planner as a hint |
| `save_model` | bool | If `true`, a deterministic post-execution hook saves the first fitted sklearn-style classifier found in the kernel (`hasattr(fit) + hasattr(predict) + hasattr(classes_)`) to `model.joblib` via joblib. Runs after all tasks complete successfully, independent of what the Planner generates. NOT forwarded to the Planner. |

Path-traversal is blocked via `Path.is_relative_to` checks; recipes are validated against the registry. The endpoint is fully transactional — failure rolls back DB and rm's the workspace.

## Conventions & Gotchas

- **Many hardcoded paths**: `WORKSPACE_ROOT = "/home/joergf/projects/MyLocalStack/data/workspaces"` and `host_path = f"/home/joergf/projects/MyLocalStack/..."` in `sandbox.list_workspace_files` are user-specific. If you move the repo, both need updating.
- **`asyncio.wait_for` is the rule** when calling the sandbox or LLM — every external call has a deliberate timeout to keep the workflow responsive. Key timeouts: Coder agent 300s, sandbox health check 5s, sandbox *execution* 720s for `local_model` / 360s for cloud (asyncio wrapper) with the sandbox body timeout set to 600s / 300s respectively. The chain must be ordered: sandbox body timeout < httpx client timeout (720s) ≤ asyncio wrapper, otherwise a transport-layer race produces an empty `ConnectionError` before the proper error surface fires.
- **`GADS_INSIGHTS_JSON:` / `GADS_FLOOR_JSON:` / `GADS_STATE_SNAPSHOT:` prefixes** — sentinel-prefixed stdout lines parsed back into structured data by the Executor and orchestrator. Don't let task code log lines starting with these strings.
- **The sliding-window context** (server.py:run_agent_workflow, "2+1 model") gives the Coder full detail for the first + last 2 tasks and only `orchestrator_summary` for the middle. When something is invisible to the Coder, suspect that distillation.
- **Cascade deletes are manual** — `DELETE /projects/{id}` walks Task/Artifact/Instruction and deletes each before deleting the Project (no FK cascade configured).
- **`core/state.Blackboard` is dead code** — only `main.py` uses it. Real state lives in the DB + `ExecutionManager.authoritative_state` + the IPython kernel.
