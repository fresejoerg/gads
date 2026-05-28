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
