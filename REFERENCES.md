# REFERENCES

Papers, libraries, and external sources that informed the design and implementation of GADS.

---

## Papers

**[1] CEDAR: Context Engineering for Agentic Data Science**
Rishiraj Saha Roy, Chris Hinze, Luzian Hahn, Fabian Kuech
*Proceedings of ECIR 2026 · Submitted January 10, 2026*
arXiv:2601.06606 · https://arxiv.org/abs/2601.06606

Informed:
- History rendering strategy: strip failed code from context, preserve only successful code + output heads + error tracebacks (see `core/history_renderer` plan)
- Output bundle design: `.py` script + `.ipynb` notebook export as default workflow outputs (`core/notebook_exporter.py`)
- Principle that data stays local and only aggregate statistics enter LLM prompts (already applied in GADS schema probe and file_schemas design)

---

## Videos

**[V1] Thomas Wiecki (PyMC Labs / Decision.AI) — "The Future of Agentic Data Science"**
*Vanishing Gradients podcast, hosted by Jure Leskovec*
https://www.youtube.com/watch?v=fm5fd-OlYkM

Informed / potentially informs:

- **Lazy skill loading to prevent context pollution**: The host describes a pattern where the harness only puts skill *names and descriptions* into context initially; the full skill body is loaded on demand when the agent identifies it's needed. GADS currently concatenates all matched skills into the Coder prompt upfront. Adopting this two-phase load (index first, full content on demand) would directly address the "context pollution" issue already flagged in CLAUDE.md.

- **Skill over-specificity is a known failure mode**: LLMs tend to write skills that are too specific to the example they were shown ("do this with Thomas's video") rather than general and portable. Worth watching for in GADS's SpecDrafter and in any auto-generated recipe/skill output.

- **Eval loop as a first-class system component**: Thomas frames the four components of a good agentic data science system as: harness, skills, orchestration, and *observability + evals*. The eval component he describes is a Bayesian A/B test comparing end-to-end outcomes with vs. without custom skills. GADS has the first three but no benchmarking loop — this is the missing piece for measuring whether recipes/skills actually improve results.

- **Programmatic self-validation is stronger than LLM evaluation**: The Bayesian/causal workflow has inbuilt validation (PyMC sampler failures, posterior predictive checks) that catches wrong models *automatically*, without an LLM judge. GADS's postcondition contracts rely heavily on LLM semantic validation; where possible, leaning on library-level failures (e.g. sklearn `fit` raising, statsmodels convergence warnings emitted to stdout) would be more reliable.

- **Garden of forking paths / multi-path analysis**: Thomas's Decision Lab runs the full analytical workflow across many parallel paths (different outlier removal strategies, different model families), then consolidates at reporting. GADS runs a single sequential path per run. A "multi-path EDA" recipe could encode this pattern for tasks where analytical choices are ambiguous.

- **Skills encode judgment checkpoints, not just domain facts**: The framing is judgment (human) → intelligence (LLM pattern matching) → procedural (code). Skills should specify *where in the workflow* the agent should pause and ask the human, not just what domain knowledge to apply. This could inform how GADS recipes define `invariants` and how the Planner writes `postcondition_json` for open-ended EDA tasks.

---

## Libraries & Tools

**[L1] skore (probabl.ai)** — https://docs.skore.probabl.ai · https://blog.probabl.ai/data-science-agents-you-can-trust

Directly informed the **`gads_audit_model` native node**: skore's `EstimatorReport` automated checks (leakage / over-under-fit / class imbalance / worse-than-baseline / low-value features) are wrapped as GADS's methodological-soundness gate — the signal for *methodological appropriateness*, the axis on which agentic DS most often fails silently. The "data science agents you can trust" post framed the case for turning those judgment calls into measurable checks.

**[L2] lifelines & scikit-survival** — https://lifelines.readthedocs.io · https://scikit-survival.readthedocs.io

The two survival-analysis engines behind the `survival_analysis.*` recipes and native nodes: lifelines for interpretable inference (Kaplan-Meier, log-rank, Cox PH + the proportional-hazards assumption test) and scikit-survival for ML prediction (Random Survival Forest + censoring-aware metrics: IPCW C-index, time-dependent AUC, Integrated Brier Score).

**[L3] probabl-ai/skills — Data Science Skills for AI agents** — https://github.com/probabl-ai/skills · https://blog.probabl.ai/teaching-agents-data-science-skills

BSD-3-Clause, 14 skills (2026-08-27), from the same company as skore [L1]. Reviewed
2026-08-27; **nothing adopted yet** — this entry records what was learned, not a decision.

Their stated aim — *"agents need encoded best practices that steer capable tools toward
statistically sound methodology"* — is close to verbatim the methodological-appropriateness
thesis behind `gads_audit_model` and the model-selection recipe. Independent convergence
from the people who maintain scikit-learn is meaningful external validation of the bet.

**The sharpest observation: their "skills" map onto GADS *recipes*, not GADS skills.**
`build-ml-pipeline -> evaluate-ml-pipeline -> audit-ml-pipeline -> iterate-*` is a workflow
with explicit handoffs, file-layout conventions (`experiments/NN_*.py`, `audit/NN_*.py`,
`journal/NN_*.md` aligned 1:1), bundled executable scripts, and per-skill TRIGGER/SKIP
routing. That is a DAG with contracts — i.e. what a GADS recipe is. GADS skills are the
narrower thing: prose expertise injected into one Coder prompt.

They use one mechanism where GADS uses two, almost certainly because Claude Code has no
recipe/DAG concept and skills are its only extension point. GADS's substrate is richer
here; the comparison is a point in its favour, not a gap.

**Three things genuinely worth stealing:**

1. **`skrub` DataOps as a structural leakage guard.** `build-ml-pipeline` declares the
   pipeline as a skrub graph that "stops at the declared object — no fit, split, tuning,
   or persistence". GADS enforces the same split-before-fit ordering *imperatively* inside
   `gads_apply_transformations`; a declarative graph makes the violation unrepresentable
   rather than merely audited. skrub is not in the sandbox. This is the strongest single
   idea in the repo.
2. **"Stops at ..." as an explicit per-step boundary.** Every skill declares its own hard
   stop ("stops at what does the report say"). More legible than a `produces` list, and it
   states what a node must NOT do — which is exactly where the recipe-compiled nodes have
   been vague.
3. **`iterate-from-skore` vs `iterate-from-user` as separate capabilities** — iteration
   split by who initiated it. GADS's replan-on-failure is orchestrator logic with no
   described counterpart, and the distinction is real (a metric-triggered retry and a
   user-triggered change of direction want different behaviour).

**Format is only partly compatible.** Theirs is the Anthropic Agent Skills format
(`name:` + `description:`); GADS uses `id:` + `description:` + a `triggers:` keyword LIST.
Probabl embed their triggers as prose inside `description`, so GADS's keyword matcher
cannot consume them directly. They would, however, work through the embedding index
(`core/skill_semantics.py`), which matches on description text — so a shim is plausible
without touching the matcher.

**Overlap with the existing 24 is smaller than it looks.** Theirs is a deep vertical on
sklearn/skrub/skore experiment hygiene; GADS's breadth is methodological (causal, survival,
recommendation, ranking, forecasting) with `model_audit` / `model_selection_tabular` /
`supervised_modeling` the only real intersection. Against the 028 gap list they do not
touch the big holes (`data_preparation.cleaning`, `analytics.ab_test`, explainability,
fairness). Complementary rather than duplicative.

**Not yet checked:** whether the bundled `scripts/` would run under the sandbox's AST
validator (they use IPython `InteractiveShell.run_cell`, and `subprocess`/`urllib` are
blacklisted), and whether skrub fits the 3G memory cap.
