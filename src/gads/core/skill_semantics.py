"""Semantic skill retrieval: embedding-based matching of task descriptions to skills.

Complements the keyword-trigger matcher in `knowledge.py`. Keyword triggers are precise
but brittle — a task phrased as "quantify how the campaign changed revenue" never says
"causal impact", so the right skill is never loaded. This module embeds each skill's
metadata and ranks skills by cosine similarity to the task intent.

Design constraints (see research/JOURNAL.md 2026-07-13 entry on layering):
- **Local + deterministic**: fastembed (ONNX, CPU) running all-MiniLM-L6-v2 — the same
  model family the sandbox uses for `local_text_embedding`. Same input → same vector;
  no network after the one-time model download; no torch in the backend venv.
- **Graceful degradation**: if fastembed (or its model) is unavailable, every call
  returns [] and the system behaves exactly as before (keyword-only). The import is
  lazy so backend startup never pays the model-load cost.
- **Reproducibility surface**: callers gate semantic discovery to tasks WITHOUT curated
  `attached_skills`, so compiled benchmark prompts stay byte-stable. The embedding
  model+version is pinned via MODEL_NAME and logged with every match.
"""
from __future__ import annotations

import os
import re
import threading
from typing import List, Optional, Tuple

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Tunables (env-overridable). Calibrated 2026-07-13 on the 15-skill library: absolute
# cosine values are NOT comparable across queries with this model (a true match can
# score 0.27 on one query and 0.5 on another), so selection combines a per-query
# z-score (is this skill an outlier among all skills for THIS query?) with an absolute
# floor (kills low-signal queries where even the outlier is noise). Technical
# Planner-style descriptions select cleanly; colloquial paraphrases may return [] —
# that is the safe failure (keyword matching still applies).
DEFAULT_FLOOR = float(os.getenv("GADS_SEMANTIC_SKILL_THRESHOLD", "0.25"))
DEFAULT_ZMIN = float(os.getenv("GADS_SEMANTIC_SKILL_ZMIN", "2.0"))
DEFAULT_TOP_K = int(os.getenv("GADS_SEMANTIC_SKILL_TOP_K", "3"))
ENABLED = os.getenv("GADS_SEMANTIC_SKILLS", "1") not in ("0", "false", "no")

_QUERY_CHAR_LIMIT = 1500


def _skill_card(skill) -> str:
    """Text embedded per skill: id, description, and triggers.

    Deliberately lean — including section headings measurably degraded ranking
    (keyword-dense soup dilutes the vector; calibration 2026-07-13 went from
    top-3 containment 2/5 to 5/5 by dropping them).
    """
    return f"{skill.id.replace('_', ' ')}: {skill.description} Keywords: {', '.join(skill.triggers)}"


def strip_recipe_boilerplate(task_description: str) -> str:
    """Reduce a task description to its intent for embedding.

    The [RECIPE INVARIANTS ...] and [SPEC HINTS ...] blocks are appended to every task
    of a compiled plan — identical across nodes, so they only blur the query vector.
    """
    text = re.sub(r"\[RECIPE INVARIANTS[^\]]*\]", "", task_description, flags=re.DOTALL)
    text = re.sub(r"\[SPEC HINTS:[^\]]*\]", "", text, flags=re.DOTALL)
    return text.strip()[:_QUERY_CHAR_LIMIT]


class SkillEmbeddingIndex:
    """Lazily built embedding index over a skill collection.

    Thread-safe lazy init: the first semantic query pays the model load (~2-4s);
    subsequent queries are milliseconds. `rebuild()` must be called whenever the
    skill collection changes (KnowledgeRegistry.load_skills does this).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._model_failed = False
        self._skill_ids: List[str] = []
        self._vectors = None  # np.ndarray [n_skills, dim], L2-normalized
        self._pending_skills: Optional[list] = None

    # ——— lifecycle ———
    def rebuild(self, skills: list) -> None:
        """Schedule (re)indexing of `skills`. Cheap — actual embedding is deferred."""
        with self._lock:
            self._pending_skills = list(skills)
            self._skill_ids = []
            self._vectors = None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._model_failed or not ENABLED:
            return False
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=MODEL_NAME)
            return True
        except Exception as e:
            self._model_failed = True
            print(f"  [SkillSemantics] disabled — embedding model unavailable: {e}", flush=True)
            return False

    def _ensure_index(self) -> bool:
        import numpy as np
        with self._lock:
            if self._vectors is not None and self._pending_skills is None:
                return True
            if not self._ensure_model():
                return False
            skills = self._pending_skills or []
            if not skills:
                return False
            cards = [_skill_card(s) for s in skills]
            vecs = np.array(list(self._model.embed(cards)), dtype=np.float32)
            vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
            self._skill_ids = [s.id for s in skills]
            self._vectors = vecs
            self._pending_skills = None
            print(f"  [SkillSemantics] indexed {len(skills)} skills with {MODEL_NAME}", flush=True)
            return True

    # ——— query ———
    def find(
        self,
        task_description: str,
        top_k: int = DEFAULT_TOP_K,
        floor: float = DEFAULT_FLOOR,
        z_min: float = DEFAULT_ZMIN,
    ) -> List[Tuple[str, float]]:
        """Return [(skill_id, cosine_similarity)] for outlier matches, best first.

        A skill is selected when it is BOTH a per-query outlier (z-score over the
        full skill-score distribution >= z_min) AND above the absolute floor.
        Returns [] when semantic matching is disabled or unavailable — callers can
        always union the result with keyword matches without special-casing.
        """
        if not ENABLED or not task_description:
            return []
        if not self._ensure_index():
            return []
        import numpy as np
        query = strip_recipe_boilerplate(task_description)
        q = np.array(list(self._model.embed([query])), dtype=np.float32)[0]
        q /= (np.linalg.norm(q) + 1e-12)
        sims = self._vectors @ q
        z = (sims - sims.mean()) / (sims.std() + 1e-9)
        order = np.argsort(-sims)[:top_k]
        return [
            (self._skill_ids[i], float(sims[i]))
            for i in order
            if z[i] >= z_min and sims[i] >= floor
        ]
