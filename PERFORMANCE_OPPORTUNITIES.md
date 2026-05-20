# GADS Performance Opportunities — Local-LLM-First Lens

This document audits the current state of GADS (commit `c9369db` on `feature/semantic-telemetry`) through one specific lens: **a local-LLM-first system executing data-science workflows guided by recipes/playbooks**.

The framing assumption is that in `local_only` mode (`GADS_LOCAL_ONLY=true`), an LLM round-trip is roughly **5–30× more expensive** than a cloud round-trip — both in wall-clock time and in failure rate. So the largest performance wins come from, in order:

1. **Shrinking what the LLM has to *think about*** — let recipes/playbooks pre-populate plan drafts so the Planner adapts rather than authors; let skill content pre-fill code scaffolding so the Coder slot-fills rather than writes from scratch.
2. **Shrinking prompts themselves** (every kilobyte costs tokens that the local KV-cache has to chew through).
3. **Caching and reusing prefixes** (LM Studio reuses the prefix KV-cache; identical system prompts are nearly free on the second call).
4. **Failing fast** (a 2-retry inner loop on a 60-second local generation = 3 minutes of wasted compute).
5. **Reducing sandbox round-trips** (cheap individually, but happen 3× per task).
6. **Avoiding LLM calls that aren't actually doing work** (rubber-stamp critiques, regenerated trivial plans) — but never replacing LLM judgment with hard-coded paths.

> **Framing principle — recipes guide, they do not override.** Recipes are *priors*, not deterministic DAG runners. The Planner LLM stays in the loop on every workflow, including recipe-matched ones, because it's the only thing that can adapt the recipe's abstract steps to the user's actual files, columns, and intent. The performance opportunities below aim to make the LLM's job *smaller and more focused*, not to remove it. Equally importantly: every optimization that doesn't depend on a recipe match (prompt slimming, schema surfacing, retry gating, short-circuiting trivial Critique) helps no-recipe workflows just as much, which is the common case.

Items are tagged with rough impact (`H`/`M`/`L`) and the source location.

---

## 1. Recipes do not yet earn their keep

This is the highest-leverage area in the codebase. Recipes are well-modeled (`core/knowledge.py:Recipe`/`RecipeTask` with `intent`, `worker_tier`, `produces`, `postconditions`, `skippable_if`) — but the runtime barely uses any of that structure.

### 1.1 `H` — Pre-populate the Planner with a recipe-derived draft (LLM still authors the final plan)

**Where:** `server.py:539-602` (planner loop), `agents/planner.py:60-78` (prompt assembly), `core/prompts.py:12-81` (Planner system prompt).

When the Router matches a recipe (`server.py:455-468`), the `ReconciliationReport` is built with the full `recommended_dag_nodes` and dropped into the Planner's prompt as raw JSON. The Planner is then asked to "align with recommended DAG nodes unless data prevents it" — meaning it has to *both* read the structured DAG *and* re-emit a parallel `PlannerTask[]` from scratch. That's a heavy reasoning task for a local model: parse the JSON, interpret the rationale, decide what to keep/drop, and rewrite each step in `PlannerTask` shape.

The opportunity is to do the mechanical translation **before** the Planner runs, then hand the LLM the *adapted* draft to revise:

- Server-side, build a `PlannerTask[]` skeleton from `RecipeTask` entries (description = intent, assigned_to from `worker_tier` + `TIER_MAPPING`, attached_skills via a recipe→skill map, postcondition pre-filled from `RecipeTask.postconditions`).
- Pass this skeleton to the Planner as a "draft plan to adapt" in the user message, *not* as raw recipe JSON to interpret.
- The Planner's job becomes a much smaller slot-filling task: substitute the actual file/column/threshold values from the objective, drop steps the user explicitly excluded, add steps the user explicitly requested, finalize the contracts.

The Planner LLM is still called on every workflow — and is still the authority on the final plan. But its prompt shrinks (no need to embed full recipe rationale + DAG JSON), its reasoning load shrinks (adapt vs author), and its output structure is half-pre-decided. Local models are reliable at slot-fill; they are unreliable at multi-step decomposition.

**For no-recipe workflows nothing changes** — the Planner runs as today, decomposing the objective from scratch. The optimization is purely additive on the recipe-match path.

Pairs naturally with 1.2 below (using `RecipeTask.postconditions` to pre-fill stronger default contracts) and 2.1 below (trimming the Planner's system prompt now that the draft carries most of the structure).

### 1.2 `H` — RecipeTask `postconditions` are dropped on the floor

**Where:** `agents/planner.py:7-12`, `core/execution_hub.py:30-88`.

`RecipeTask.postconditions` is a list of executable-looking Python assertions (e.g. `"abs(y_train.mean() - y_test.mean()) < 0.05"`). The Planner's `PlannerTask.postcondition` is instead a much weaker `{output_type, required_columns}` JSON dict, and the LLM is asked to invent that contract from scratch on every plan.

These are two parallel postcondition systems that don't talk. The recipe-authored postconditions are stronger and free. Two complementary uses:
- **As stronger default contracts for the LLM to refine:** when a recipe matches, translate the assertions into `{output_type, required_columns}` shape and pre-fill them into the draft `PlannerTask` (see 1.1). The Planner is still free to revise or extend them based on the user's objective — but it starts from a high-quality default instead of inventing one.
- **As runtime invariants regardless of recipe match:** evaluate the raw assertions server-side in the sandbox after each task (`ExecutionHub.validate_contract`), turning them into actual correctness checks. The recipe author wrote `"abs(y_train.mean() - y_test.mean()) < 0.05"` for a reason — running it as an assertion is more reliable than asking an LLM to invent a column-name contract that approximates it.

Net effect: the Planner's job on contracts shrinks (defaults are stronger), and validation gets real teeth — without preventing the LLM from overriding when the data demands it.

### 1.3 `M` — Recipe `skippable_if` is also dropped

**Where:** `core/knowledge.py:14`, `agents/planner.py:11`.

`RecipeTask.skippable_if` carries strings like `"blackboard.has_fact('target_distribution_known')"`. `ReconciliationReport.skippable_nodes` is hard-coded to `[]` (`server.py:461`). The orchestrator never evaluates these predicates against `executor.authoritative_state`, so every recipe always runs all DAG nodes regardless of prior work.

For the multi-turn "Durable Project Memory" feature advertised in `USER_GUIDE.md` section 6, this means follow-up instructions in an existing project re-do already-completed steps. Implementing `skippable_if` evaluation (even a simple stringmatch against `authoritative_state` keys) directly cuts follow-up workflow length.

### 1.4 `M` — Recipes are matched 1-of-1, not composed

**Where:** `agents/router.py:32-41`, `server.py:455-468`.

The Router emits exactly one `matched_recipe_id`. Real DS objectives often blend two — e.g. "do thematic analysis *and* train a classifier on the themes" needs both `thematic_analysis_unstructured` and `binary_classification_tabular`. Currently the Router has to pick one, and the rest is left to the Planner LLM to invent unaided.

A multi-recipe matcher (return a ranked list with confidence) would let the Planner adapt from multiple priors at once — same LLM call, but with a richer skeleton to revise. Composing recipes is still LLM-driven; the change is just giving the Planner more relevant priors to choose from.

### 1.5 `M` — Skill selection is keyword-based and lossy

**Where:** `core/knowledge.py:179-187`.

`find_skills` is a substring match on the description text. The `visualization` skill triggers on `["plot", "chart", "visualize", "visualization"]` — so "Save Figure 2 as a histogram" matches none of those keywords. The Planner *can* attach skills explicitly via `attached_skills`, but in practice local models rarely populate that field reliably.

Cheap wins:
- Expand triggers (add `"figure"`, `"histogram"`, `"scatter"`, `"bar"`, `"line"` etc.).
- Match on intent tags (`task_type`, `data_modality`) in addition to text — once those are persisted on the task.

---

## 2. Prompt-bloat tax (local models pay it on every call)

### 2.1 `H` — Planner prompt always includes the full hierarchy JSON + all skill descriptions

**Where:** `core/prompts.py:75-77`, `agents/planner.py:60-78`.

The Planner system prompt is rendered fresh each call and contains:
- `AVAILABLE_MODELS_HIERARCHY` as full JSON (~500–1500 tokens depending on tier population)
- `AVAILABLE SKILLS` as full JSON (every skill's `id` + `triggers` list)
- `KNOWLEDGE REPORT` as full JSON

In `local_only` mode the hierarchy is `["local_model"] × 4 tiers` (`core/registry.py:55-60`). Inlining 1500 tokens of T1/T2/T3 model lists for a system that can only pick "local_model" is pure overhead. A two-line `local_only` branch in the Planner agent could strip this entirely.

Even in cloud mode, the hierarchy could be shrunk to model IDs only (drop the `description` field — it's never referenced).

### 2.2 `H` — CodeGenerator system prompt carries a 25-line Plotly bdata example to every task

**Where:** `core/prompts.py:117-138` (the `MANDATORY BROWSER COMPATIBILITY LOOP` block).

This block is in every Coder call regardless of whether the task touches Plotly. For non-plot tasks (data loading, splitting, model training) it's ~300 tokens of dead weight. Worse: the workaround is **also redundant** because `core/introspection.py:harden_json_artifact` already decodes bdata server-side after the fact (`server.py:945-946`).

Three options, ordered by ambition:
- **Quick:** Delete the block from the system prompt and rely on `harden_json_artifact` as the safety net.
- **Better:** Replace the block with a single line: `"To save a plot: use gads_save_plot(fig, 'name.json')"` and add `gads_save_plot` as a helper preloaded by the sandbox preamble (next to `gads_emit_insight`).
- **Best:** Conditionally append the Plotly skill content only when a visualization skill is attached.

The "best" path is the closest to what skills were designed for — skill content already moves through `skills_context` (`agents/workers/coder.py:47`), so the visualization specifics should live in `knowledge/skills/visualization.md` (currently a 4-line stub) and only be injected when needed.

### 2.3 `M` — Coder prompt restates `files_list` twice

**Where:** `agents/workers/coder.py:42-58`.

Files appear in both the formatted system prompt (`{files_list}`) and the user message (`"CRITICAL: You MUST use the correct filenames. Available files are: {files_summary}"`). Belt-and-suspenders against the local model ignoring the system prompt, but for short objectives this can easily double the prompt size. Pick one.

### 2.4 `M` — The Coder doesn't get column schemas

**Where:** `server.py:519-525` (schemas are extracted) → `server.py:529-537` (persisted) → never re-read by the Coder.

`_probe_file_schema` extracts `{schema: {col: dtype}, sample: [...]}` for every CSV/parquet and stuffs it into `project.last_state_json["__schemas__"]`. The Planner reads this via `FileMetadata.columns_and_dtypes`. The Coder does *not* — it only sees filenames and (separately) the kernel namespace, which is empty before `df` is loaded.

Result: on the *first* task that loads a file, the Coder is guessing column names from the objective text. This is a major source of "column not found" errors that trigger expensive escalations.

Wire the schema cache into `CoderInput` and into the prompt under `AUTHORITATIVE RUNTIME STATE`. Negligible token cost (schema for a normal CSV is 30 tokens) for a big drop in retry rate.

### 2.5 `L` — Two system prompts on every agent: factory default load + override re-read on every call

**Where:** `agents/base.py:50-52`, `core/prompts.py:298-308`.

The `PromptRegistry` already caches all prompts in memory. But every `BaseAgent.run` re-fetches and re-formats. Cost is small but it forecloses on a future optimization: **content-addressed prompt prefixes**. If a stable system-prompt hash were sent to LiteLLM with `extra_body={"cache_id": ...}`, the LM Studio backend could be persuaded to reuse the KV cache. Out of scope for now, but worth keeping in mind when reorganizing prompts.

---

## 3. Inner loops that multiply local-LLM cost

### 3.1 `H` — `local_only` mode still attempts up to 2 inner Coder retries

**Where:** `core/executor.py:42-43, 277-278`.

`max_retries = 2` is hardcoded. On failure, `ExecutionHub.escalate_task` is supposed to bump the model — but in `local_only` mode `get_next_model_dynamic` returns `None` (`core/registry.py:99-100`). So the inner Coder loop just retries the *same* local model with *more* prompt context (the prior error appended). For a 60-second-per-token local model, this is up to **3× the latency for the same task** with negligible improvement.

Suggestion: in `local_only` mode, gate retries on error category — retry only for transient errors (`ConnectionError`, `TimeoutError`, certain `JSONDecodeError`s), fail fast for `KeyError`/`ValueError`/`AttributeError` (those are model confusion that won't fix itself with one more retry).

### 3.2 `H` — JSON repair path costs a *second* LLM call

**Where:** `core/llm.py:131-155` (streaming branch) and `core/llm.py:170-206` (non-streaming branch).

When the local model emits malformed JSON, the system constructs a new conversation (`messages + assistant_attempt + user_correction`) and **calls the LLM again** via `instructor`. For a `qwen3-coder`/`kimi`-class local model this is another 30–60 seconds.

Cheaper alternatives, ordered:
- **Heuristic repair first** (try adding missing closing braces/brackets, strip code fences, balance quotes) — works ~70% of the time on local model output.
- **Constrained decoding via the LiteLLM proxy** — LM Studio supports JSON-mode/grammar constraints in newer builds; turning that on at the proxy level removes the entire repair pathway.
- **Schema-guided decoding via instructor's TOOLS mode** — but note the comment in `agents/base.py:60` that this was intentionally abandoned for local models. Worth re-testing on current LM Studio versions, as the deadlock was specific to older `pydantic-ai`.

### 3.3 `H` — Synthesizer + Critique run sequentially every workflow attempt, even for trivial objectives

**Where:** `server.py:967-1112`.

For an objective like "count the rows in sales.csv", the system runs:
1. Router (T3 LLM call)
2. Planner (T2 LLM call)
3. PlanCritique (T2 LLM call)
4. Coder + execution (T3 LLM call)
5. Synthesizer (T2 LLM call)
6. Critique (T2 LLM call)
7. Reporting (deterministic)

That's **6 LLM calls** for a `wc -l`. Five of them are local-model calls if `local_only` is on.

The PlanCritique prompt itself acknowledges this (`core/prompts.py:253` — "Objectives like 'Calculate 1+1', 'Print hello', or 'Count rows' are 100% valid"). It's already a rubber-stamp for trivial cases. A short-circuit at the Router stage — "if `task_type == 'trivial'` or task count == 1, skip PlanCritique and Critique" — would cut 30–50% of LLM calls on simple workflows.

### 3.4 `M` — PlanCritique is duplicative with Critique

**Where:** `agents/plan_critique.py`, `agents/workers/critique.py`.

Both agents are "is this output acceptable?" with re-plan/re-synthesize loops. PlanCritique evaluates the plan before execution; Critique evaluates the synthesis after. The unique value of PlanCritique is catching `is_terminal_failure` (missing files) before sandbox spinup, which is real and worth keeping — but the rest of its rejection logic ("Match Objective" rule) overlaps with what Critique does later.

Could be merged into a single "Auditor" with a `phase` parameter and one prompt, saving one round of prompt-warmup overhead.

### 3.5 `M` — `MAX_WORKFLOW_ATTEMPTS = 3` for local-only is too generous

**Where:** `server.py:500`.

Three full Planner→Execute→Synth→Critique replans for a local model = potentially 12+ LLM calls × 60s = 12 minutes of compute before giving up. In `local_only` mode, replanning rarely helps (the model isn't getting smarter on attempt 3). Drop to 1 for local-only mode, or make it an env var.

---

## 4. Sandbox round-trips: cheap individually, but they add up

Every task currently does 3 sandbox executions before the user's code even runs:

| Round-trip | Where | Purpose |
|---|---|---|
| 1. `_probe_file_schema` | `server.py:1433-1485` | DuckDB DESCRIBE per CSV (only at plan time) |
| 2. `snapshot_code` | `server.py:787-808` | `GADS_STATE_SNAPSHOT:` introspection (every task) |
| 3. `wrapped_code` | `executor.py:206` | the actual user code with telemetry hooks |
| 4. `structural_probe_code` | `executor.py:221-244` | `GADS_FLOOR_JSON:` DataFrame stats |

### 4.1 `M` — Probes 2 and 4 cover overlapping ground

Both walk `globals()` and emit `{name, columns, shape}` for DataFrames. The pre-probe is for the Coder's prompt; the post-probe is for telemetry. They could be unified into a single probe at end-of-task that updates `ExecutionManager.authoritative_state` in one shot. The pre-task probe is then served from `authoritative_state` (already kept in memory) — zero sandbox round-trips.

Note `executor.py:222` has a Python syntax error in the probed code (`import pandas as pd as _pd`) — the `as` keyword can't be chained. That probe is currently a no-op when the import fires; it silently fails inside the `try/except` at `executor.py:238-245` and structural insights are never emitted. Fixing this is itself a tiny win, but the larger move is to merge the probes.

### 4.2 `M` — `_probe_file_schema` shells out to the sandbox to run DuckDB

**Where:** `server.py:1433-1485`.

DuckDB is just a Python package. The probe could run **in-process** in the server (the server already imports DuckDB indirectly via plotly/etc.). That cuts a network round-trip and avoids polluting the per-project kernel. The "dedicated session" (`probe_{project_id}`) is created and torn down per probe, which adds non-trivial latency on each one.

### 4.3 `L` — `poll_logs_loop` polls every 1.0 s

**Where:** `core/executor.py:70`.

For fast tasks (sub-3s), this can miss the entire output window. For slow tasks it's fine. The poll cadence is fine in absolute terms but the polling task is unconditional — it runs even when no `stdout_callback` is provided (it short-circuits at line 64, but the task is still created and torn down). Cheap, but symbolic of "always-on" patterns that could be conditional.

---

## 5. Architectural directions worth a separate roadmap entry

These are bigger swings, not quick wins, but they're the natural endpoints of the above.

### 5.1 Code-snippet library (the missing "skill body")

The current skills (`knowledge/skills/*.md`) are *prose advice* injected into the Coder prompt. A local model still has to translate the prose into code on every task. A natural next step: each skill carries one or more **vetted code snippets** that the Coder can adapt rather than regenerate from prose.

This is the local-LLM equivalent of starter-template completion — the model is much better at filling in `{COLUMN}`/`{FILE}` slots in a vetted template than at writing the surrounding scaffolding from scratch. The Coder LLM still chooses *which* template applies, still adapts it to the task description, and still owns the final code — but its baseline reasoning load drops significantly.

Concrete: `visualization.md` could carry a `plotly_bar_chart_template.py` snippet, `large_dataset_handling.md` could carry the `duckdb.query("SELECT ... FROM '{file}' USING SAMPLE 100000 ROWS")` template, etc. For tasks where no skill matches, the Coder operates exactly as today — writing from scratch with general prompt guidance. So this helps recipe-less and skill-less workflows degrade gracefully rather than penalizing them.

### 5.2 Two-phase Coder: "draft small, expand on success"

The Coder currently emits one big code block. For local models, a tighter loop would be:
1. Phase A: emit only the *imports + critical computation* (small output, fast).
2. Server runs Phase A, captures dtypes/columns from the resulting kernel state.
3. Phase B: emit the visualization/save code with ground-truth column names already in the prompt.

This trades one LLM call for two smaller ones, which on a local model with quadratic attention cost is often a net win — and the second call is much less likely to fail (no column-name guessing).

### 5.3 Distinguish "model" from "kernel" state

`ExecutionManager.authoritative_state` (`core/executor.py:24`) and the in-kernel `globals()` are kept loosely in sync via the `GADS_STATE_SNAPSHOT:` parsing. Right now `authoritative_state` is rebuilt from kernel scratch every task. For a local-first system, treating it as the *primary* state (durable, indexed, queryable in-process) and the kernel as a *cache* opens up:
- Skipping the pre-task snapshot probe entirely (4.1 above) by trusting `authoritative_state`.
- Surfacing column schemas to the Coder without a probe (2.4 above).
- Implementing `skippable_if` against a queryable state object (1.3 above).

### 5.4 Per-recipe "fast-path" worker tiers

`TIER_MAPPING` is global. But recipes know what they need: a `anomaly_detection_tabular` recipe could declare "all steps T3-capable" (most of it is `IsolationForest(...)` boilerplate), while `thematic_analysis_unstructured` legitimately needs T2 for embedding/clustering judgment calls. `Recipe.dag[i].worker_tier` is already declared in the YAML — so the recipe-derived draft (1.1) can carry those tier assignments as the *starting point*, and the Planner LLM can still override on a per-task basis when the user's specific objective demands it (e.g. an unusually complex dataset that warrants escalation). The Planner's capability rubric (`prompts.py:37-51`) remains the authoritative path for no-recipe workflows.

---

## 6. Small, cheap fixes worth doing while you're in there

- **`executor.py:222`** — `import pandas as pd as _pd` is a `SyntaxError` (the structural probe is silently failing — see 4.1). Fix to `import pandas as _pd`.
- **`server.py:282-294`** — `cancel_project` instantiates a fresh `ExecutionManager()` just to reach `.sandbox.reset_session`. Either thread the executor through, or make `SandboxClient` directly instantiable.
- **`server.py:396`** — `exec_mgr = ExecutionManager()` is constructed for a single health check; the `executor` instance from line 383 is right there.
- **`llm.py:71-76`** — `langfuse_client.flush()` is called *on every LLM call*. The Langfuse client batches internally; forcing a flush per call defeats batching and adds 50–200ms per call. Move to one flush per workflow stage or on workflow exit.
- **`server.py:894`** — `_get_recursive_files(workspace_dir)` runs `os.walk` after every task. For projects with handover ZIPs (often 10–100 MB), this is fine, but is called twice (once for `new_files_after`, once for `files_after`). Cache the result.
- **`core/registry.py:62-86`** — `get_model_hierarchy()` makes an HTTP call to LiteLLM `/models` *on every workflow start*. Cache with a short TTL (60s) — the model list rarely changes between workflows.
- **`prompts.py:54-57`** — section headers are numbered `### 1, ### 2, ### 4, ### 5, ### 3, ### 5` (yes, two 5's, skipping 3). Local models are sensitive to structural cues; clean numbering can measurably improve plan quality.

---

## Suggested ordering

If only a few of these can be done now, the order with the biggest ratio of (impact / effort) for a local-LLM-first deployment is:

1. **2.2** — Strip the Plotly bdata block out of the CodeGenerator system prompt. (1 hour, immediate ~300 token / task saving. Helps every workflow, recipe or not.)
2. **2.4** — Surface schemas to the Coder. (2 hours, big drop in column-name retries. Helps every workflow.)
3. **3.1** — Gate inner retries on error category in `local_only` mode. (2 hours, kills wasted retry loops. Helps every workflow.)
4. **3.3** — Short-circuit PlanCritique/Critique for trivial workflows. (3 hours, cuts ~30% of LLM calls on simple objectives. Helps every workflow.)
5. **4.1 / 6** — Fix the structural-probe `SyntaxError` and merge the pre/post probes. (Half a day, cleaner per-task latency. Helps every workflow.)
6. **3.2** — Heuristic-first JSON repair + investigate proxy-level JSON-mode. (1–2 days, removes the single largest local-model latency spike when output is malformed. Helps every workflow.)
7. **1.1 + 1.2** — Recipe-derived draft plans with pre-filled contracts, handed to the Planner LLM as a starting point to adapt. (1–2 days, biggest reduction in Planner reasoning load when a recipe matches — no effect on no-recipe workflows, which keep running the full Planner.)

Note that items 1–6 all help the **no-recipe case** as much as the recipe-match case. The recipe-specific item is deliberately last in the ordering: it has a big upside on a subset of workflows, but the cross-cutting items above pay off on every workflow regardless of whether a recipe was matched.
