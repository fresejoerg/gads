# Run scorecard: data dictionary

Field-by-field definition of the eval object in `research/evals/<project_id>.json`
(`"schema": "gads.run_eval"`, `rubric_version` 1.0.0). Produced by `src/gads/eval/`; see
[README.md](README.md) for how scorecards are generated and read.

**Conventions.**
- *Status* always means one of `pass`, `fail` or `not_evaluable`. `not_evaluable` means the
  record lacks the evidence (an older run predating a field, a node that never got far
  enough, a run still in progress). It never counts as a pass or a fail.
- The *main workflow* is the scored execution: the tasks carrying the instruction id of the
  project's latest workflow launch. Follow-up lane tasks are excluded.
- An *attempt* is one Planner task plus every task after it up to the next Planner task. A
  replan starts a new attempt. Only the *final attempt* is scored; earlier ones are history.
- A *node* is one execution task (not a pipeline stage such as Router, Planner, Synthesizer).
- `null` means the source did not record the value, not that it is zero or false.

---

## Top level

| Field | Type | Definition |
|---|---|---|
| `schema` | string | Always `"gads.run_eval"`. Identifies the document type. |
| `rubric_version` | string | Version of the rubric set that produced the card (`scorecard.RUBRIC_VERSION`). Bumped whenever a rubric's meaning changes. Cards with different versions came from different instruments and must not be compared directly. |
| `project_id` | string (UUID) | The GADS project. Also the file name, and the MLflow trace id (`tr-<uuid without dashes>`). |
| `name` | string | Project name as stored on the project row. |
| `verdict` | string | Headline result, decided by the **critical** checks only. One of the values below. |
| `critical` | object | Counts of critical checks by status: `{pass, fail, not_evaluable}`. |
| `dimensions` | object | Counts of **all** checks by status, per dimension. See [Dimensions](#dimensions). |
| `benchmark` | object \| null | Benchmark layer summary, when the run's spec is a benchmark spec. See [`benchmark`](#benchmark). |
| `run` | object | What happened to the run as a whole. See [`run`](#run). |
| `fingerprint` | object | The harness configuration the run executed under. See [`fingerprint`](#fingerprint). |
| `attribution` | object | pass@model accounting. See [`attribution`](#attribution). |
| `usage` | object | Tokens and cost. See [`usage`](#usage). |
| `nodes` | array | One entry per final-attempt node. See [`nodes[]`](#nodes). |
| `prior_attempts` | array | Superseded attempts. See [`prior_attempts[]`](#prior_attempts). |
| `llm_judgments` | object | LLM verdicts recorded for context only, **never** part of the verdict. See [`llm_judgments`](#llm_judgments). |
| `checks` | array | Every individual check. See [`checks[]`](#checks) and the [check catalogue](#check-catalogue). |
| `scored` | object | When and by which code the card was produced. See [`scored`](#scored). |

### `verdict` values

| Value | Meaning |
|---|---|
| `pass` | Every critical check passed. |
| `fail` | At least one critical check failed. |
| `inconclusive` | No critical check failed, but at least one could not be evaluated. |
| `not_evaluable` | No verdict is possible: `run.terminal_state` is `in_progress`, `never_started` or `record_missing`. |

The critical checks are `run.completed`, `run.has_plan`, and for every final-attempt node
`node.executed`, `node.metric.*` and `node.variable.*`.

### Dimensions

Each key of `dimensions` holds `{pass, fail, not_evaluable}` counts. A dimension appears
only if at least one check belongs to it.

| Dimension | What its checks measure |
|---|---|
| `execution` | The run as a process: did it reach final reporting, was there a plan, were all recipe nodes planned. |
| `contract` | Whether each node met its contract (`postcondition_json`): executed, required metrics, variables, insights, columns. |
| `integrity` | Signs of faked or inconsistent results: hallucination phrases in stdout, node metrics vs `metrics.json`. |
| `deliverables` | Files final reporting must leave in the workspace. |
| `benchmark` | The `score_benchmark.py` checks, present only with a benchmark layer. |

---

## `benchmark`

`null` unless the spec the run was launched from is the `spec_file` of exactly one
`research/benchmarks/*/expected.json`, and the run is not in progress.

| Field | Type | Definition |
|---|---|---|
| `benchmark_id` | string | The benchmark's id from its `expected.json` (e.g. `air_passengers_v1`). |
| `verdict` | string | `pass` iff every benchmark check passed, else `fail`. Independent of the top-level `verdict`. |
| `passed` | integer | Number of benchmark checks that passed. |
| `total` | integer | Number of benchmark checks run. |

---

## `run`

| Field | Type | Definition |
|---|---|---|
| `terminal_state` | string | How the main workflow ended. One of the values below. |
| `void_suspected` | boolean | `true` when `terminal_state` is `abandoned`: the process stopped mid-run, a harness failure rather than evidence about the model. The verdict is still `fail`; analyses of model capability should filter these out. |
| `workflow_attempts` | integer | Number of attempts (Planner tasks) in the main workflow. `0` means the run never produced a plan. |
| `recipe_id` | string \| null | Recipe recorded by the final attempt's Planner task; `null` for drafted plans with no recipe. |
| `plan` | string \| null | `recipe_compiled` (final plan compiled mechanically from a recipe DAG), `planner_drafted` (written by the Planner LLM, including advisory D2 runs), or `null` (no plan). |
| `exec_nodes` | integer | Number of nodes in the final attempt (length of `nodes`). |
| `created_at` | string (ISO 8601, local time) | When the project row was created. |
| `excluded_followup_tasks` | integer | Tasks on the project that are not part of the scored execution: follow-up lane tasks and, on a relaunched project, the earlier launches' tasks. |
| `executions_on_project` | integer | How many times a workflow was launched on this project. Above 1 means it was relaunched; only the latest launch is scored. |

### `terminal_state` values

Evaluated in this order; the first match wins.

| Value | Condition |
|---|---|
| `record_missing` | Tasks exist but none belongs to a workflow launch (no System init task). Seen on projects recovered after the 2026-07-31 table loss. |
| `cancelled` | The project narrative starts with `[CANCELLED]` (user cancel). |
| `halted` | The project narrative starts with `[HALTED]` (terminal plan rejection or fatal error). |
| `completed` | The final reporting task completed. Says nothing about quality, and is not the Critique's approval. |
| `in_progress` | Some task is still `pending` or `running`. |
| `never_started` | The project has no tasks at all. |
| `abandoned` | All tasks finished, but there was no reporting and no halt or cancel record: the backend stopped mid-run. |

---

## `fingerprint`

The configuration the run executed under, as far as the run recorded it. This is run-time
state, not the state at scoring time (that is `scored`).

| Field | Type | Definition |
|---|---|---|
| `spec` | string \| null | Spec file the run was launched from (`last_state_json.spec_filename`); `null` for runs launched from a free-text objective. |
| `recipe_id` | string \| null | Same as `run.recipe_id`. |
| `recipe_version` | string \| null | `version:` from the `applied_recipe_<id>.md` the run copied into its workspace, so it is the recipe **as it was at run time**. `null` if the workspace holds no such file. |
| `engine_id` | string \| null | Which weights served `local_model` (e.g. `google/gemma-4-12b@base`); `unknown` if it could not be determined; `null` if never stamped (cloud-only runs, older runs). |
| `routing_mode` | string \| null | `cloud`, `local`, `hybrid` or `cloud_pinned`. Taken from the dial ledger, which only has rows for runs that reached the end of the pipeline, so `null` for most failed runs. |
| `run_mode` | string \| null | `research` or `production`. Same source and caveat as `routing_mode`. |
| `rung` | string \| null | Delegation-dial rung `D0`…`D5` of the project (the lowest rung over its tasks). |

---

## `attribution`

pass@model accounting, computed by `dial.node_attribution`, the same function that writes
`research/dial_ledger.jsonl`. Unlike the checks, it is computed over **all attempts** of the
main workflow, deduplicated per node: the last completed instance of a node wins.

| Field | Type | Definition |
|---|---|---|
| `exec_nodes` | integer | Distinct nodes that completed, in any attempt. |
| `attempted_nodes` | integer | Nodes the assigned model was actually asked to do: `exec_nodes − native_primary + failed_nodes`. The pass@model denominator. |
| `failed_nodes` | integer | Nodes that ran, exhausted their retries and never completed in any attempt. Nodes that never ran are not counted; the model was never asked to do them. |
| `native_primary` | integer | Nodes run by a native function by policy (production mode). Excluded from the denominator: the model was not asked. |
| `model_pass` | integer | Completed nodes the assigned model did itself (no fallback, not native by policy). |
| `fallback_pass` | integer | Completed nodes a fallback rescued: `native_fallback + cloud_fallback`. Never counted as a model pass. |
| `native_fallback` | integer | Nodes completed by a node's declared native fallback after the model exhausted its retries. |
| `cloud_fallback` | integer | Nodes completed by a single cloud-model attempt after a local model exhausted its retries. |
| `pass_at_model` | number \| null | `model_pass / attempted_nodes`, rounded to 3 decimals; `null` when nothing was attempted. |

---

## `usage`

Tokens and cost over every task of the main workflow, all attempts and all stages (Router,
Planner, Synthesizer, …), not only nodes. A node resumed from a prior attempt copies that
attempt's result, including its usage, so resumed nodes are skipped to avoid double counting.

| Field | Type | Definition |
|---|---|---|
| `calls` | integer | Model calls recorded across all tasks. |
| `prompt_tokens` | integer | Input tokens across those calls. |
| `completion_tokens` | integer | Output tokens across those calls (includes reasoning tokens where the provider reports them as output). |
| `cost_usd` | number | Summed cost of the priced calls, in USD. A **lower bound** whenever `cost_complete` is `false`. |
| `unpriced_calls` | integer | Calls with no known price (the local model, or a provider the gateway could not price). |
| `cost_complete` | boolean | `true` iff `unpriced_calls == 0`, i.e. `cost_usd` is the full cost. A `false` with `cost_usd: 0.0` means "cost unknown", not "free". |
| `models` | array of strings | Distinct model names that served any call, sorted. |

---

## `nodes[]`

One entry per node of the final attempt, in creation order.

| Field | Type | Definition |
|---|---|---|
| `node` | string | Node key: the recipe node id for recipe-compiled plans (e.g. `train_forecast_model`); for drafted plans, `task:` plus a slug of the first 60 characters of the task description. Matches `checks[].node`. |
| `task_id` | string (UUID) | The task row. |
| `status` | string | The task's final status: `completed`, `failed`, `bypassed`, `pending` or `running`. |
| `contract_source` | string | Who wrote the node's contract: `recipe` (curated, from the recipe DAG), `planner_drafted` (written by the Planner LLM, weaker evidence when passed), or `none` (no contract). |
| `assigned_to` | string | Model the task was assigned to (e.g. `local_model`, `gemini-3.8-flash`). |
| `model_used` | string \| null | What actually completed the node, as recorded by the executor. A plain model name, or a prefixed form such as `native_fallback:gads_plot_forecasts`. `null` if the node never ran. |
| `invocation` | string | `model_used` classified: `model`, `native_fallback`, `cloud_fallback`, `native_primary`, or `none` (never ran). |
| `resumed` | boolean | `true` if the node was skipped on this attempt because it completed in a prior attempt and its outputs were still live in the kernel (recipe plans only). |
| `generations` | integer \| null | Model calls attributed to this node, cumulative across its retries; mostly Coder generations. `1` means first-try success. |
| `cost_usd` | number \| null | Cost of those calls; `null` or `0.0` when unpriced. |
| `checks` | object | Counts of this node's checks by status: `{pass, fail, not_evaluable}`. |

---

## `prior_attempts[]`

One entry per superseded attempt (every attempt but the last). History only, not scored.

| Field | Type | Definition |
|---|---|---|
| `attempt` | integer | Attempt number, starting at 1. |
| `exec_nodes` | integer | Nodes planned in that attempt. |
| `completed` | integer | Of those, how many completed. |
| `failed` | array | Nodes that ran and failed in that attempt (nodes that never ran are omitted). Each entry is `{node, error}`: the node key and the first 200 characters of the error. |

---

## `llm_judgments`

LLM verdicts, recorded so a reader can compare them with the code-only verdict. None of
them affects `verdict`, any check, or any count.

| Field | Type | Definition |
|---|---|---|
| `critique_attempts` | integer | Critique tasks in the main workflow (one per synthesis-review round, across attempts). |
| `completeness_verifier_ran` | boolean | Whether the CompletenessVerifier ran at least once. |
| `dial_ledger_outcome` | string \| null | `outcome` of the run's dial-ledger row: `pass` requires the Critique's approval **and** zero failed tasks across all attempts, superseded ones included. It can therefore disagree with `verdict`, which scores the final attempt only. `null` when the run has no ledger row (it did not reach the end of the pipeline). |

---

## `checks[]`

| Field | Type | Definition |
|---|---|---|
| `id` | string | Check identifier, `<scope>.<rubric>[.<subject>]`. Subjects are user data (metric, variable, column or file names, benchmark check names) and may contain dots, spaces or regex characters, so split on the first two dots only. |
| `dimension` | string | One of the [dimensions](#dimensions). |
| `status` | string | `pass`, `fail` or `not_evaluable`. |
| `detail` | string | Human-readable reason: the observed value on a pass, the cause on a fail or not_evaluable. Free text, not for parsing. |
| `critical` | boolean | Whether this check decides the verdict. |
| `node` | string | Node key (as in `nodes[].node`). Absent on run-level and benchmark checks. |
| `evidence` | string | Present when the evidence source is not the node's own result. Currently only `end_of_session_kernel`: the project's last kernel snapshot, which accumulates every name bound during the run and omits callables. |

## Check catalogue

| `id` | Dim. | Crit. | Passes when | `not_evaluable` when |
|---|---|---|---|---|
| `run.completed` | execution | yes | `terminal_state` is `completed`. Fails for `halted`, `cancelled`, `abandoned`. | `in_progress`, `never_started`, `record_missing` |
| `run.has_plan` | execution | yes | The final attempt has at least one node. Fails when no Planner task exists. | `never_started`, `record_missing` |
| `run.recipe_nodes_planned` | execution | no | Every node of the compiled recipe has a task in the final attempt. | Plan not compiled from a recipe |
| `run.file.final_dashboard.html` | deliverables | no | File exists in the workspace. | Run did not complete |
| `run.file.research_report.md` | deliverables | no | File exists. | Run did not complete |
| `run.file.workflow_execution.py` | deliverables | no | File exists. | Run did not complete |
| `run.file.metrics.json` | deliverables | no | File exists and parses. Only checked when some final-attempt contract declares `required_metrics`. | (not emitted otherwise) |
| `run.metrics_consistent` | integrity | no | Every numeric metric a final-attempt node captured that also appears in `metrics.json` has the same value (relative tolerance 1e-9). | No captured metric appears in `metrics.json` |
| `node.executed` | contract | yes | Node `completed` (detail says if it was resumed). Fails if `failed`, never ran, or `bypassed` (handover bundle instead of execution). | Run in progress and node pending/running |
| `node.metric.<m>` | contract | yes | `<m>` is in the node's `metrics_captured` and is a finite number (NaN, ±inf, booleans and non-numeric strings fail). | Node did not complete; or the result predates metric capture |
| `node.variable.<v>` | contract | yes | `<v>` is in the end-of-session kernel snapshot. Fails when it never appeared in any snapshot this run. | Node did not complete; no snapshot; or the final attempt is a replanned **drafted** plan, whose kernel was reset while the snapshot still holds earlier attempts' names (presence not attributable) |
| `node.insight.<i>` | contract | no | `<i>` occurs (case-insensitive substring) in an emitted insight's `artifact` or `insight` text, the same match the runtime uses. At run time this is only a warning. | Node did not complete; result has no insights |
| `node.column.<c>` | contract | no | Some DataFrame in the end-of-session kernel has a column whose name equals `<c>` after lowercasing and stripping non-alphanumerics. Stricter than the runtime's LLM column check, deliberately. Only for `output_type: dataframe` contracts. | Node did not complete; no DataFrame in the snapshot |
| `node.no_hallucination_phrases` | integrity | no | The node's stdout contains none of the phrases that signal faked or skipped data (`stdout_clean.HALLUCINATION_TOKENS`, e.g. "mock data", "simulating data"). | (emitted only for completed nodes) |
| `benchmark.<section>.<name>` | benchmark | no | The corresponding `score_benchmark.py` check passed. `<section>` is `reproducibility` (metric values vs `expected.json`), `completeness` (required artifacts) or `methodology` (required/forbidden code patterns in `workflow_execution.py`). | (never; benchmark checks are pass/fail) |

---

## `scored`

| Field | Type | Definition |
|---|---|---|
| `at` | string (ISO 8601, UTC) | When the card was written. |
| `commit` | string \| null | Short git hash of the GADS checkout **that scored the run** (not the one that executed it), with `-dirty` appended when tracked files had uncommitted changes. `null` if git was unavailable. |

A re-score that would change nothing except `scored` leaves the file untouched, so these two
fields record the last time the card's content changed.
