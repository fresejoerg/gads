# GADS Next Steps Plan
**Status: Phase 1 COMPLETE — SPRINT-2 + SPRINT-3 implemented by Deepfrese; Systemd Services & UI Recipe-Disable Toggle configured by Antigravity (2026-06-10)**
**Authors: Deepfrese + Antigravity**

---

## Context: What Was Completed This Session

| Ticket / Task | What Was Done |
|---------------|---------------|
| SPRINT-1 · History Renderer | `src/gads/core/history_renderer.py` — failed tasks now always render full error + stdout tail regardless of sliding-window position. Committed `c071da5`. |
| SPRINT-7 · Class Imbalance Hardening | DataAnalyzer computes `imbalance_ratio`/`minority_class_frac`; stratified SampleGuard; `gads_calibrate_threshold()` native function; PR AUC secondary metric; `class_weight='balanced'` in binary recipe; `f1_macro` for multiclass. Committed `aac7eb8`. |
| AutoGluon recipe hardening | Sanitizers for 10+ local-model failure modes (def main wrapper, df_clean undefined, hallucinated imports, predict_proba indexing, etc.). Namespace snapshot now captures scalars (`problem_type`, `target_col`, `naive_baseline`). |
| Systemd Services Configuration | Configured `gads-backend.service`, `gads-ui.service`, and `gads-monitor.service` as user-level systemd units to ensure robustness, persistence, and auto-restart capability. |
| UI Disable-Recipes Toggle | Added a toggle flag `disable_recipes` inside Streamlit and FastAPI to allow runs with only the spec and sandbox environment setup, facilitating side-by-side comparison. |

---

## Remaining Active Sprint Tickets

| Priority | Ticket | Effort | Blocker |
|----------|--------|--------|---------|
| 1 | ~~**SPRINT-2** · Recipe-Guided Metric Propagation~~ | ✅ Done | Commit 0053b1e |
| 2 | **SPRINT-6** · UI Fast Mode + Visible DataSampler | S–M | Antigravity |
| 3 | ~~**Causal Stack Install**~~ · Sandbox + 4 recipes + 3 skills | ~~M~~ | **Already done** ✅ |
| 4 | ~~**SPRINT-3** · Native Causal Nodes~~ | ✅ Done | Commit 1b96667 |
| 5 | **SPRINT-4** · Trace Distillation | M | Needs successful runs from SPRINT-3 |

Plus lower-priority items agreed during collaboration:
- Planner prompt trimming in `local_only` mode (Antigravity)
- Schema feeding to Coder prompt (Antigravity)

---

## Proposed Work Plan

### Phase 1 — Quick Wins (parallelisable, ~2 days total)

**Deepfrese: SPRINT-2 · Recipe-Guided Metric Propagation**

The Planner generates `required_metrics` names from scratch and local models hallucinate them (`conf_interval_lower` instead of `ate`, etc.). The recipe DAG nodes already declare the correct names — we just need to thread them into the Planner.

- Extract `recipe_metrics_map: Dict[str, List[str]]` from matched recipe after Router
- Pass as `PlannerInput.recipe_metrics_hints` (new optional field)
- Update Planner prompt: "use these exact metric names, do not rename"
- Post-process Planner output: overwrite any task's `required_metrics` with recipe's authoritative list if description keyword-matches a DAG node (belt-and-suspenders)
- Files: `src/gads/agents/planner.py`, `src/gads/core/server.py`
- Verification: run `causal_fraud_amount.md` spec, confirm tasks get `required_metrics: [ate]` not invented names

**Antigravity: SPRINT-6 · UI Fast Mode + Visible DataSampler**

Infrastructure (Recipe Enforcer, SampleBudgetAdvisor) is already done. This is purely the UI surface:

- "Fast mode" checkbox in project creation form → appends `sample_rows: 50000` to spec hints
- "DataSampler" visible task card in workflow when SampleBudgetAdvisor fires (like the existing DataAnalyzer card), showing: original rows, sample size, reason
- Document `sample_rows` in CLAUDE.md Project Specs table
- Files: `src/gads/ui/streamlit_app.py`, `src/gads/core/server.py` (DataSampler task creation), `CLAUDE.md`

---

### Phase 2 — Native Causal Nodes (~3–5 days)

> **Note:** Causal stack installation is already complete — all 4 recipes, 3 skills, and sandbox requirements (`dowhy`, `econml`, `causalml`, `causal-learn`, `linearmodels`) are already in the codebase. SPRINT-3 is unblocked and can start immediately after Phase 1.

**Deepfrese: SPRINT-3 · Native Causal Nodes**

Local model cannot reliably write DoWhy 4-step from scratch. Replace with deterministic native wrappers.

- `src/gads/knowledge/native/causal.py`:
  - `gads_causal_estimate_ate(df, treatment_col, outcome_col, confounder_cols, method="auto", max_rows=20000) -> dict` — full DoWhy 4-step, returns `{ate, placebo_new_effect, subset_new_effect}`
  - `gads_causal_bayesian_ate(df, treatment_col, outcome_col, confounder_cols, max_rows=5000) -> dict` — Bambi MCMC with `chains=1, cores=1, draws=500`
- Inject via `executor.py` preamble when causal keywords detected
- Sanitizer: if `CausalModel(` appears without `gads_causal_estimate_ate`, inject redirect comment
- Update `causal_inference_dowhy.md` skill: native-node call pattern at top, manual fallback below
- Verification: run `causal_fraud_amount.md`, confirm `metrics.json` has `ate`, `placebo_new_effect`, `subset_new_effect`

---

### Phase 3 — Flywheel (~2 days, after Phase 2 produces reliable runs)

**Antigravity: SPRINT-4 · Trace Distillation**

Offline CLI script to mine successful Langfuse runs and propose recipe invariants.

- `scripts/distill_traces.py`: query Langfuse for traces where Synthesizer ran; extract code from `result_json.code`; keyword-match against recipe `intent` fields; output `proposed_invariants.md`
- Detects: subsample patterns, `class_weight='balanced'`, `random_state=42`
- Human review gate: proposed invariants are never auto-committed
- Verification: run against successful DoWhy run (`d06f005b-...`), confirms "subsample df to 20K before CausalModel"

---

### Phase 4 — Polish (can be parallelised with Phase 2–3)

**Antigravity: Planner Prompt Trimming**
In `local_only` mode, strip the 1500+ token model hierarchy JSON and unused skill/knowledge lists from the Planner system prompt.

**Antigravity: Schema Feeding to Coder**
Wire column schemas from DataAnalyzer file profile into `CoderInput` so local model doesn't guess column names on first load.

---

## Summary Table

| Phase | Item | Owner | Effort | Can Start |
|-------|------|-------|--------|-----------|
| 1 | SPRINT-2: Recipe metric propagation | Deepfrese | S–M | Now |
| 1 | SPRINT-6: UI Fast mode + DataSampler | Antigravity | S–M | Now |
| 2a | ~~Causal stack installation~~ | — | ✅ Already done | — |
| 2b | SPRINT-3: Native causal nodes | Deepfrese | L | After Phase 1 |
| 3 | SPRINT-4: Trace distillation | Antigravity | M | After 2b |
| 4 | Planner prompt trimming | Antigravity | S | Anytime |
| 4 | Schema feeding to Coder | Antigravity | S | Anytime |

---

*This plan is pending GOD approval. No code changes will be made until approval is received.*
*Both agents have reviewed and agreed. Ready for GOD's approval.*

---

## Antigravity's Review Notes

1. **Causal Stack Verification (Phase 2a - Done)**: 
   * I ran a verification script in the active sandbox and confirmed that all causal stack libraries (`dowhy`, `econml`, `causalml`, `causal-learn`, `pymc`, `bambi`, `arviz`, etc.) are already installed and import successfully.
   * The 4 causal recipes (`causal_effect_estimation_dowhy.md`, `heterogeneous_treatment_effects.md`, `causal_discovery.md`, `iv_panel_econometrics.md`) and 3 causal skills (`causal_inference_dowhy.md`, `causal_ml_econml.md`, `causal_discovery_skill.md`) already exist in the codebase.
   * `prompts.py` already includes the Router guideline for causal inference.
   * **Conclusion**: Phase 2a is fully completed. Deepfrese can start SPRINT-3 (Phase 2b - Native Causal Nodes) immediately upon GOD's approval without waiting for any installation steps.

2. **Phase 1 Assignment (SPRINT-6)**:
   * I am ready to start on **SPRINT-6** (UI Fast mode toggle + DataSampler visible task) as soon as GOD approves the plan.

3. **General Alignment**:
   * I fully endorse the proposed timeline and parallel execution path. Let's submit this plan to GOD for approval.
