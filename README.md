# GADS — General Agentic Data Science

> **Local-first agentic data science for teams whose data can't go to the cloud** — a benchmark-proven, expanding library of DS workflows tuned to run reliably on small local models, so you get real mileage on your own hardware.

GADS is a **contract-driven, multi-agent system for autonomous data science**. Point it at one or more datasets, give it a one-line objective — *"estimate the causal effect of transaction amount on fraud"* — and it runs the full workflow (framing → planning → code generation → sandboxed execution → verification → synthesis → reporting) and hands back an interactive dashboard, a research report, an exportable notebook, and machine-readable metrics.

It is also a **research instrument**. GADS is built to answer one question: *where is the efficiency boundary between agentic scaffolding and raw model capability?* — how far down the ladder (from frontier cloud models to a single 12B local model) the executing LLM can be pushed before workflow reliability collapses, and how much of that collapse deterministic structure can buy back. Every run is graded on **reproducibility** and **methodological appropriateness** and logged to an evidence ledger.

> 📖 For how to write effective objectives, see the **[User Guide](USER_GUIDE.md)**. For contributor/architecture notes, see **[CLAUDE.md](CLAUDE.md)**.

---

## Core philosophy: *prefer a lookup to a generation*

GADS's most distinctive trait is an unusually deep investment in **not** calling the LLM. Any decision that can be made deterministically is taken away from the model:

- **Spec-pinned recipes** skip the Router LLM entirely.
- **Matched recipes compile straight into task plans** — skipping both the Planner and the Plan Critic.
- **Native kernel functions** replace hand-generated statistics code.
- A **regex sanitizer** repairs the model's known failure modes before execution.

Where a typical agent framework asks *"how do we prompt better?"*, GADS asks *"can this decision be a lookup instead?"* Each removed call is a removed failure mode — on local models, often a fatal one. This is what lets a small local model complete workflows that would otherwise only run on frontier models.

**Gemini-first, local-focused.** GADS runs on **local LLMs** (via LM Studio) and **Gemini** as primary brains, with Claude, GPT, and Kimi tiers available. A hard **local-isolation mandate** keeps the boundary honest: `local_model` never escalates to a cloud model, and a cloud run never silently degrades onto the local model — it fails loudly instead.

---

## The pipeline

The entire orchestration is one auditable coroutine (`run_agent_workflow` in `core/server.py`). Stages:

1. **DataAnalyzer** *(deterministic)* — profiles every file in an isolated probe session: schema, dtypes, row counts, null rates, cardinality, numeric stats. Feeds measured schemas into every downstream prompt so a small model can't invent column names.
2. **SpecDrafter** — formalizes the objective into `workflow_spec.md`. A `recipe_id` pinned here is a launch-validated **hard pin** that overrides the Router.
3. **Router** — classifies task type / data modality and matches a recipe. **Skipped entirely** when a spec pins a valid recipe.
4. **Planner** — decomposes the objective into contracted tasks with per-task model assignments. **Bypassed** when a recipe matches: the recipe DAG *is* the plan.
5. **Plan Critique** — audits the plan before execution; can reject-and-replan or halt. (Auto-approved for compiled recipe plans.)
6. **Task execution** — per task: assemble context (a sliding-window history + a live kernel snapshot), generate code, sanitize it, estimate runtime (bypass + handover bundle if > 280 s), execute in a persistent sandbox, validate against the postcondition contract, register artifacts.
7. **CompletenessVerifier** *(fail-open)* — semantic coverage check that can trigger a replan for named gaps.
8. **Synthesizer → Critique** — narrative, takeaways, and per-artifact captions, QA-checked against a distilled view of the dashboard.
9. **Reporting** — emits `final_dashboard.html`, `research_report.md`, a `.py`/`.ipynb` code bundle, the applied recipe, and a delegation-dial ledger entry.

Failure is handled by **three nested retry ladders**: the Coder retries with error feedback (≤2), then the task escalates to a stronger model (≤2), then the whole workflow replans with failure context (≤3 attempts).

---

## The delegation dial (research object)

GADS measures capability along a **delegation dial** — how much of the *method* is fixed by scaffolding versus left to the model:

| Rung | Realization |
|------|-------------|
| **D0** | Full delegation — LLM-drafted plan from a bare objective |
| **D1** | Framed drafted — LLM-drafted plan with spec hints |
| **D3** | Directed — plan compiled from a recipe; model writes the code |
| **D4** | Curated skill — worked code patterns injected into the Coder |
| **D5** | Mechanized — a deterministic native kernel function does the work |

The **project rung is the minimum over its task rungs**, and `D*(engine)` is the lowest rung an engine can hold within tolerance. Crossed against the engine (local 12B ↔ cloud tiers), this produces a **rung × engine** pass/fail grid — the central evidence base, accumulated in `research/dial_ledger.jsonl` with benchmarks under `research/benchmarks/` and findings in `research/JOURNAL.md`.

---

## Model tiers & routing

Models are accessed only through the LiteLLM proxy, so *"which model"* is pure configuration. The tier hierarchy (`core/registry.py`, latest generation per provider):

| Tier | Role | Gemini | Anthropic | OpenAI | Kimi |
|------|------|--------|-----------|--------|------|
| **T1** | Architect | gemini-3.1-pro-preview | claude-opus-4.8, claude-fable-5 | gpt-5.6-sol | kimi-k3 |
| **T2** | Coder | gemini-3.6-flash | claude-sonnet-5 | gpt-5.6-terra | kimi-k2.7-code |
| **T3** | Worker | gemini-3.5-flash-lite | claude-haiku-4.5 | gpt-5.6-luna | kimi-k2.7-code-highspeed |
| **T4** | Local | — | — | — | `local_model` (isolated; never escalates to cloud) |

Escalation climbs `T3 → T2 → T1` (intra-tier fallback first). **Routing modes** (`POST /config` or `.env`): `cloud` (tiered + escalation), `local` (everything on `local_model`), `hybrid` (plan/report on cloud, execution local), `cloud_pinned` (one operator-chosen cloud model, no ladder).

---

## Architecture

```
src/gads/
  core/        orchestrator (server), executor, execution hub, model registry,
               delegation-dial scorer, runtime oracle, reporting, handover bundles,
               outbox bus, LLM connectors, database models
  agents/      Router, SpecDrafter, Planner, PlanCritique, and worker agents
               (Coder, CompletenessVerifier, Synthesizer, Critique)
  knowledge/   recipes/  — YAML-frontmatter Markdown SOPs (compiled into plans)
               skills/   — keyword/embedding-triggered expertise for the Coder
               native/   — deterministic kernel functions (D5 nodes)
  templates/   Jinja2 dashboard template
  tools/       sandbox client and validators
  ui/          Streamlit control center (real-time state polling)
research/      JOURNAL.md, dial_ledger.jsonl, benchmarks/  (the research instrument)
specs/         launchable project specs (POST /projects/from-spec)
```

Durable state lives in **PostgreSQL** (via SQLModel); UI events are written to a transactional outbox in the same commit as the state they describe and streamed over WebSocket with replay-on-reconnect.

### Stateful Python sandbox

A Docker-isolated **IPython kernel** (one persistent session per project) keeps variable state across agent turns. Fixed package set (no runtime `pip install`):

- **Data:** pandas, numpy, polars, pyarrow, duckdb
- **ML:** scikit-learn, torch, lightgbm, xgboost, shap, joblib
- **AutoML:** AutoGluon (tabular + timeseries)
- **Causal:** dowhy, econml, causalml, causallearn, linearmodels, statsmodels, pymc, arviz, bambi, pycausalimpact, pgmpy
- **NLP:** sentence-transformers, nltk, textblob
- **Viz:** matplotlib, seaborn, plotly, kaleido, networkx

---

## Getting started

### 1. Prerequisites
- **[MyLocalStack](https://codeberg.org/deepfrese/MyLocalStack)** running — LiteLLM proxy on `:4000`, IPython sandbox on `:8000`, shared workspace directory.
- **[uv](https://docs.astral.sh/uv/)** package manager.
- A local model server (e.g. LM Studio) for local/hybrid modes, and/or cloud API keys.

### 2. Install
```bash
git clone https://codeberg.org/deepfrese/GADS.git
cd GADS
uv sync
```

### 3. Configure
Copy `.env.example` to `.env` and set `GADS_DATABASE_URL` (required) and any API keys. Choose a routing mode:

```env
# Run entirely on local models:
GADS_ROUTING_MODE=local        # (legacy alias: GADS_LOCAL_ONLY=true)
# Or: cloud | hybrid | cloud_pinned  (with GADS_PINNED_MODEL for the last)
```

### 4. Run
```bash
./start_backend.sh              # FastAPI backend on :8001
./scripts/run_streamlit.sh      # GADS Control Center on :8003
curl http://localhost:8001/health
```

The **GADS Control Center** is a 3-panel Streamlit IDE with a persistent project archive, real-time task tracking, live reasoning/stdout streams, and hot-editable agent prompts.
