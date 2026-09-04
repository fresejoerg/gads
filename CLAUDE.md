# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack & Commands

**GADS = Generative-Augmented Data Science.** That is the only correct expansion —
not "General Agentic" or "Generative Agentic"; both were in the docs and have been
corrected. Generic phrases like "agentic data science" describing the *field* (or
citing others' work, e.g. CEDAR) are fine and are not expansions of the acronym.

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

Before the main loop, two one-shot stages run:
- **DataAnalyzer** (`server.py:_probe_file_schema`) — runs in a dedicated sandbox session (`probe_{project_id}`) before any agent. Profiles each CSV/Parquet file: schema, row count, null rates, cardinality for low-cardinality columns, numeric stats (min/max/mean/std). Capped at 5000 rows, 30s timeout. Results stored in `FileMetadata.columns_and_dtypes` and formatted as human-readable text for the Planner prompt via `planner.py:_format_file_profile`. Visible in the UI as a "DataAnalyzer" task. Excel/JSON/text format support is implemented but requires `openpyxl` in the sandbox container.
- **SpecDrafter** (`agents/spec_drafter.py`, T3) — formalizes the user objective into a structured spec with hints forwarded to the Planner.

1. **Router** (`agents/router.py`, T3) — classifies the objective into `task_type`/`data_modality`, picks a matching `Recipe`, and emits a `confidence` (0.0-1.0) for that match. Both labels come from **one controlled vocabulary**, `src/gads/knowledge/taxonomy.yaml` — the schema field descriptions and the prompt's vocabulary blocks are *derived* from it (`taxonomy.render_task_vocabulary`), never restated, and the emitted labels are canonicalized on the way out (`taxonomy.canonical_task` / `canonical_modality`). Labels are canonical taxonomy tasks (`classification.binary`, `regression.survival`), not the old free-text enum.
2. **Planner** (`agents/planner.py`, T2) — decomposes objective into `PlannerTask[]`, each with a `postcondition_json` contract, an `assigned_to` model from the live hierarchy, and `attached_skills`. What happens with a matched recipe depends on the Router's `confidence` against `recipe_confidence_threshold` (float, default 0.7, `POST /config {recipe_confidence_threshold}` or `GADS_RECIPE_CONFIDENCE_THRESHOLD`) — a three-way gate, computed once per routing decision as `recipe_tier` (`server.py`, `"deterministic"` / `"advisory"` / `"none"`):
   - **`confidence >= threshold`, or an explicit spec pin (`recipe_id` in spec frontmatter, or the SpecDrafter's hint when the Router found nothing) — deterministic.** The plan is compiled mechanically from the recipe DAG (`plan_is_deterministic=True`, `server.py` RECIPE PLAN COMPILER): the Planner LLM call is skipped, PlanCritique is bypassed, and each node's `produces` → `postcondition_json.required_variables` (also `required_metrics`, `fallback_native`/`fallback_call`). A pin always wins regardless of confidence — it wasn't the Router's own judgment to begin with.
   - **`confidence < threshold` — advisory.** The Planner LLM runs normally and drafts its own task list, but receives the matched recipe (rationale, DAG, invariants) as JSON context (`PlannerInput.knowledge_report`, with `ReconciliationReport.advisory=True`) — a knowledge base to draw on, adapt, or diverge from, not a script to follow. This is the delegation dial's **D2 "advised"** rung (`core/dial.py`).
   - **No recipe matched at all, or `disable_recipes: true` — none.** The Planner drafts with zero recipe input, same as always (D0/D1).
3. **PlanCritique** (`agents/plan_critique.py`, T2) — audits the plan before execution; can reject (re-plan with feedback) or mark `is_terminal_failure` (halt entire workflow). The outer loop allows `MAX_WORKFLOW_ATTEMPTS = 3` replans. Skipped for deterministic (recipe-compiled) plans.
4. **Per-task Execution** (`core/executor.py:ExecutionManager.run_task`) — for each `Task`:
   - Probes the IPython kernel for live variables (`namespace_summary`) so the Coder sees ground truth.
   - Calls **Coder** (`agents/workers/coder.py`) to generate Python.
   - **RuntimeOracle** (`core/runtime_oracle.py`) AST-estimates runtime; >280s → bypass, generate **handover ZIP** via `HandoverManager`, continue.
   - **Native-node preamble**: when the generated code references a native primitive (keyword-matched in `executor.py`), the matching `*_PREAMBLE` from `knowledge/native/` is prepended so pre-written, audited functions are defined in the kernel (see Native nodes).
   - Wraps code with `gads_emit_insight()` preamble + structural DataFrame probe postamble. Executes in sandbox.
   - **Adaptive retry loop** (replaces the old fixed 2-escalation model): up to `max_attempts=10`, but stops as soon as a failure *reason* recurs (normalized signature — distinct reasons = self-correction, keep going; a repeated reason = looping, stop). ALL prior errors (text only, never the code) are fed back to the Coder each attempt, plus a first-attempt "common pitfalls" prior mined from the cross-run **error ledger** (`core/error_ledger.py` → `research/error_ledger.jsonl`).
   - **On exhaustion — fallback** (opt-in, local only, `GADS_LOCAL_FALLBACK`): if the node declares a `fallback_native`, invoke it deterministically in the live kernel (no replan); and/or escalate the one task to a cloud model for a single attempt (`cloud`/`native_then_cloud`) — a deliberate, gated exception to "local never escalates". Fallback-completed nodes are tagged `model_used="native_fallback:…"`/`"cloud_fallback:…"`, emit a `TASK_FALLBACK` outbox event, and are counted separately in pass@model reporting.
   - On success: scans the workspace for new files, registers `.png` as base64 plots and `.json` as interactive Plotly artifacts (via `introspection.harden_json_artifact` which strips binary data).
   - Hallucination guard: scans stdout for tokens like "mock data" / "simulating data" and fails the task even if it didn't raise.
- **4b. save_model hook** — deterministic post-execution hook (see Project Specs table).
- **4c. CompletenessVerifier** (`agents/workers/completeness_verifier.py`, T2) — semantic completeness check that runs after execution and before synthesis, only when a replan is still possible (`workflow_attempt < MAX_WORKFLOW_ATTEMPTS`) and all tasks succeeded. Closes two gaps PlanCritique cannot catch: (1) `required_insights` is a soft-fail so tasks can complete without emitting interpretive work; (2) bypassed/handover tasks never executed. The verifier receives the `formalized_objective`, task `orchestrator_summary` fields, the artifact file list, and `metrics.json` contents (if present) as ground-truth scalar evidence. If gaps are found **and `missing_analyses` is non-empty**, feeds them as `critique_feedback` for a replan via `continue`. Fail-open: any verifier exception falls through to synthesis without blocking the workflow.
5. **Synthesizer** (`agents/workers/synthesizer.py`, T2) — writes narrative + takeaways + per-artifact captions.
6. **Critique** (`agents/workers/critique.py`, T2) — QA pass; can reject and re-trigger the synthesis loop. Critique sees a distilled-Markdown preview of the dashboard (`core/distiller.py`), not raw HTML.
7. **Reporting** (`core/reporting.py`) — emits `final_dashboard.html` (Jinja2 template at `src/gads/templates/dashboard.html.j2`) and `research_report.md` to the workspace. **The recipe DAG composes the report**: `core/report_sections.py` builds one section per DAG node, in execution order — including reasoning/audit nodes that draw nothing and nodes that never ran (rendered as explicit gaps). Each section carries that node's captured metrics, `gads_emit_insight()` payloads, files, stdout and charts; artifacts are filed by the `task_id`/`node_id` stamped on them at creation (`server._artifact_origin`), falling back to filenames in `orchestrator_summary` for older projects, and anything unattributable renders in a trailing section rather than being dropped. The skeleton is persisted by the RecipeCompiler onto the Planner task (`result_json.recipe_sections`) and tasks are linked back via `postcondition_json.recipe_node_id`, so `rebuild_dashboard` reproduces the same structure from the DB alone. A drafted (non-recipe) plan degrades to one section per executed task. The Synthesizer supplies optional per-section prose (`section_notes`); the structure never depends on it.

**Replan-on-failure & resume-from-failed-node**: an execution failure (or a CompletenessVerifier gap) triggers a replan — `continue` back to Planning, up to `MAX_WORKFLOW_ATTEMPTS = 3` (without this an execution failure would fall straight through to synthesis). For a **deterministic recipe plan** the sandbox kernel is *preserved* across the replan (`_cleanup_stale_sessions(reset_current=False)` when `plan_is_deterministic and workflow_attempt>1`), so `run_task`'s resume path skips any node that completed in a prior attempt whose declared `produces` variables are verifiably live in the kernel — re-running only the failed node + its downstream instead of the whole DAG (`result_json.resumed_from_prior_attempt=True`). Drafted (non-deterministic) plans always reset the kernel. Resume is what makes replan-on-failure cheap enough to enable.

### Async Plumbing

- **Transactional Outbox**: agents write `OutboxEvent` rows; `core/bus.dispatcher_loop` polls every 500ms and broadcasts via WebSocket (`/ws`). UI subscribes with `?last_seq=N` for replay-on-reconnect.
- **Live streaming**: `LIVE_STREAMS: Dict[task_id, {reasoning, stdout}]` is a server-side dict polled by the UI via `GET /tasks/{id}/stream`. The Executor's `poll_logs_loop` polls the sandbox's `/sessions/{id}/logs` endpoint every 1s to populate stdout incrementally.
- **Watchdog**: `ExecutionHub.watchdog_loop` re-queues tasks whose `heartbeat` is older than 5 minutes.
- **Cancellation**: `POST /projects/{id}/cancel` sets `project.narrative = "[CANCELLED] …"`; the workflow polls `is_cancelled()` between stages. Cancel also force-resets the sandbox session.
- **Duplicate guard**: `ACTIVE_WORKFLOWS: set[uuid.UUID]` (in-process) prevents double-launches; supplemented by a DB query for existing `pending`/`running` tasks.

### Model Tiers & Escalation (core/registry.py)

`TIER_MAPPING` defines T1 (Opus/Pro) → T2 (Sonnet/Flash) → T3 (Haiku/Flash-Lite) → T4 (`local_model`). `get_next_model_dynamic` does **intra-tier random fallback first**, then jumps to the next tier in `TIER_ORDER = ["T3", "T2", "T1"]`. **T4 (local) never escalates to cloud** — this is a hard mandate, enforced in two places. The Planner's prompt must only emit model strings present in the runtime hierarchy fetched from LiteLLM (the orchestrator sanitizes hallucinated model names).

**Run mode** (`registry.py`, `GADS_RUN_MODE` / `POST /config {run_mode}` / Streamlit switch): `research` (default) attempts every node with the model and treats a declared native as a post-exhaustion rescue only — this is what makes `pass_at_model` meaningful. `production` invokes a node's native directly and never asks the model, except on nodes marked `model_required`. Native-by-policy nodes are tagged `native_primary:…`, counted apart from both `model_pass` and `fallback_pass`, and **excluded from the pass@model denominator** (`attempted_nodes`) — a node the model was never asked to do is not evidence about the model.

**Routing modes** (`registry.py`, `resolve_stage_model` is the single choke point): `cloud` (tiered + escalation ladder), `local` (all tiers collapse to `["local_model"]`, no escalation), `hybrid` (plan construction — SpecDrafter/Router/Planner/PlanCritique — and report writing — Synthesizer/Critique — on cloud tiers; execution tasks + CompletenessVerifier on `local_model`), `cloud_pinned` (one operator-chosen cloud model for every stage, **no escalation ladder**). Set via `POST /config {routing_mode, pinned_model, random_routing}` (legacy `{local_only}` still maps to local/cloud and cannot clobber a mode when absent) or `GADS_ROUTING_MODE`/`GADS_PINNED_MODEL` in `.env` (fallback: `GADS_LOCAL_ONLY`). Runtime config is in-process only — a backend restart reverts to `.env`.

**Local fallback** (`GADS_LOCAL_FALLBACK`, or `POST /config {local_fallback}`): `none` (default) | `native` | `cloud` | `native_then_cloud` — what happens when a local task exhausts all retries (see Execution). `cloud`/`native_then_cloud` are the deliberate, opt-in, *post-exhaustion only* exception to the never-escalate mandate; the cloud model is resolved via `get_model_hierarchy(force_cloud=True)` (which builds real cloud tiers even in local mode, since the normal local hierarchy collapses to `local_model`). Default `none` keeps the local capability boundary visible.

### BaseAgent (agents/base.py)

All agents extend `BaseAgent[TIn, TOut]`. There are **three completion paths**:
1. `local_model` → bypasses Pydantic AI entirely (it deadlocks local models in tool-call loops) and uses `core/llm.get_structured_completion` with `instructor` directly.
2. Cloud + streaming → Pydantic AI `agent.run_stream` with a callback.
3. Pydantic AI failure → falls back to `get_structured_completion` (no streaming) before re-raising.

The `local_model` branch also injects `repetition_penalty=1.1, temperature=0.1` to prevent repetition loops.

### Knowledge & Prompt System

- **Recipes** (`src/gads/knowledge/recipes/*.md`) — YAML frontmatter (id, applies_when, requires, dag, invariants) + Markdown body with `## Rationale` heading. Loaded by `KnowledgeRegistry`. The Router matches; the Planner always receives a `ReconciliationReport` as a prior when a recipe matched, but whether the `dag` is compiled deterministically or passed as advisory context depends on the Router's confidence — see the Planner section above. Per-node fields (`RecipeTask`): `intent`, `worker_tier`, `depends_on`, `produces`, `required_metrics`, `attached_skills`, `fallback_native`/`fallback_call` (the node's native safety net for the local fallback), `report` (`{title, summary, collapsed}` — presentation only; the node gets a dashboard section either way), and `model_required` (production mode must not substitute the native — for nodes whose deliverable is the model's reasoning rather than a computation).
- **Controlled vocabulary** (`src/gads/knowledge/taxonomy.yaml`) — the single source of truth for `task_type` / `data_modality`, shared by spec `taxonomy:` blocks, recipe `applies_when` and the Router. `core/taxonomy.py` canonicalizes any of the three (`canonical_task("cox_regression") -> "regression.survival"`) and `tasks_overlap` makes a bare family cover its subtypes, so the coverage oracle (`KnowledgeRegistry.find_matches`) matches on meaning rather than spelling. It also declares `training_task_families` (drives the SampleBudget row cap). A recipe may set `applies_when.pin_only: true` — a research instrument (delegation-dial arm, AAH grounding rung) that is withheld from the agent recipe catalogue and from the oracle, reachable only by a spec `recipe_id` pin. Guard: `PYTHONPATH=src uv run python scripts/test_vocabulary.py` (no LLM calls); routing quality: `scripts/eval_routing.py`.
- **Skills** (`src/gads/knowledge/skills/*.md`) — keyword-triggered expertise injected into the Coder's prompt. Planner-attached + keyword-matched skills are deduplicated and concatenated. An embedding index (`core/skill_semantics.py`) supplements keyword matching for tasks with no curated `attached_skills`.
- **Native nodes** (`src/gads/knowledge/native/*.py`) — pre-written, audited Python functions injected into the sandbox as a preamble for high-stakes / single-right-answer steps where LLM codegen reliably fails: AutoGluon fit/predict, DoWhy & Bayesian ATE, implicit-CF recommenders, the skore `gads_audit_model` methodological gate, and survival (`gads_make_surv_target`, `gads_evaluate_survival`, `gads_cox_ph_report`; plus fallback-only `gads_kaplan_meier`, `gads_plot_survival_curves`). Registered in `native/__init__.py` (`NATIVE_REGISTRY`, `NATIVE_SOURCE`, `*_PREAMBLE`); functions are annotation-free and self-contained (imports inside) so their source injects verbatim via `inspect.getsource`. Keyword-triggered in `executor.py`. **Design rule:** nativize invariant/correctness/guarantee operations (loses ~no capability); keep genuinely variable work (plotting, feature engineering, model choice) model-generated with the native only as an opt-in `fallback_native`, so model capability stays measured (see approach_docs/019).
- **Prompts** (`core/prompts.py`) — `FACTORY_DEFAULTS` are the source of truth; user overrides are stored as files in `gads_data/prompts/` and reloaded on every agent run (hot-edit via `POST /prompts/{agent_name}`). Templates use `{placeholder}` style — when adding new placeholders, both the factory default AND every agent's `formatted_prompt = base_prompt.format(...)` call must be updated together.

### Observability

`core/llm.trace_context` is a `ContextVar` set at the top of `run_agent_workflow`. `get_structured_completion` reads it to inject Langfuse trace headers and LiteLLM metadata on every call. Stages also create explicit `langfuse_client.trace(...).span(...)` spans. **Don't forget to update `trace_context.get().update({...})` when adding a new agent stage** or its spans will hang off the wrong parent.

**Delegation dial & pass@model** (`core/dial.py`): each completed run appends a record to `research/dial_ledger.jsonl` — the delegation rung (D0…D5) × routing outcome, plus `pass_at_model` (`model_pass`/`exec_nodes`, the fraction the assigned model did itself) vs `fallback_pass` (`native_fallback`/`cloud_fallback`, nodes a fallback rescued). The two are kept **separate and never collapsed** so a fallback-assisted pass cannot masquerade as a model pass — this is what keeps the efficiency-boundary measurement honest.

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
| `sample_rows` | int | Hard sandbox budget constraint — caps the maximum number of rows processed in ML training/analysis tasks to prevent execution timeouts |
| `save_model` | bool | If `true`, a deterministic post-execution hook saves the first fitted sklearn-style classifier found in the kernel (`hasattr(fit) + hasattr(predict) + hasattr(classes_)`) to `model.joblib` via joblib. Runs after all tasks complete successfully, independent of what the Planner generates. NOT forwarded to the Planner. |
| `disable_recipes` | bool | If `true`, forces the drafted-plan lane: the spec-pin fast path and Router recipe matching are both skipped, so the plan is LLM-drafted (used by delegation-dial D0/D1 specs). Also settable per launch via the request body. |

Path-traversal is blocked via `Path.is_relative_to` checks; recipes are validated against the registry. The endpoint is fully transactional — failure rolls back DB and rm's the workspace.

## Conventions & Gotchas

- **Many hardcoded paths**: `WORKSPACE_ROOT = "/home/joergf/projects/MyLocalStack/data/workspaces"` and `host_path = f"/home/joergf/projects/MyLocalStack/..."` in `sandbox.list_workspace_files` are user-specific. If you move the repo, both need updating.
- **`asyncio.wait_for` is the rule** when calling the sandbox or LLM — every external call has a deliberate timeout to keep the workflow responsive. Key timeouts: Coder agent 300s, sandbox health check 5s, sandbox *execution* 720s for `local_model` / 360s for cloud (asyncio wrapper) with the sandbox body timeout set to 600s / 300s respectively. The chain must be ordered: sandbox body timeout < httpx client timeout (720s) ≤ asyncio wrapper, otherwise a transport-layer race produces an empty `ConnectionError` before the proper error surface fires.
- **`GADS_INSIGHTS_JSON:` / `GADS_FLOOR_JSON:` / `GADS_STATE_SNAPSHOT:` prefixes** — sentinel-prefixed stdout lines parsed back into structured data by the Executor and orchestrator. Don't let task code log lines starting with these strings.
- **The sliding-window context** (server.py:run_agent_workflow, "2+1 model") gives the Coder full detail for the first + last 2 tasks and only `orchestrator_summary` for the middle. When something is invisible to the Coder, suspect that distillation.
- **Cascade deletes are manual** — `DELETE /projects/{id}` walks Task/Artifact/Instruction and deletes each before deleting the Project (no FK cascade configured).
- **`core/state.Blackboard` is dead code** — only `main.py` uses it. Real state lives in the DB + `ExecutionManager.authoritative_state` + the IPython kernel.
- **Inbox Collaboration & GOD Tasks**: At the start of every session, ALWAYS check if the background monitor task (`scripts/monitor_inbox.py`) is running. If not, immediately start it. When a task message arrives in `agent_inbox.jsonl` from `"from": "GOD"`, do not simply write an acknowledgment and stop. You must immediately parse the task, take ownership of it, and proactively execute/implement the required work, communicating updates and coordinating with the other agent (`Deepfrese`) via the inbox as needed to advance the work.


