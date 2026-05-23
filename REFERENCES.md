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
