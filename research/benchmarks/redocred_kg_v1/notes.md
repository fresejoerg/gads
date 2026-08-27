# redocred_kg_v1 — knowledge-graph extraction against gold triplets

**Established:** 2026-08-27. Corpus staged by `scripts/stage_redocred.py` into
`$GADS_DATASETS_ROOT/redocred`; licence and ceiling in that directory's `PROVENANCE.md`.

## Source

**Re-DocRED** (MIT, tonytan48) re-labelling **DocRED** (MIT, THUNLP), Wikipedia-derived.
Re-DocRED specifically: the original has a documented false-negative problem, and scoring
against it would count correct extractions as hallucinations.

100 documents, 3,655 gold triplets, 6 entity types, 93 relation types (Wikidata
properties, labels fetched from the Wikidata API). Median 188 words/doc, ~19 entities and
~32 gold triplets per document.

Chosen over Amazon reviews, which were considered first: reviews carry near-zero named
entities ("these things", "I", "a prime member"), no entity-to-entity relations, and no
gold. They remain the right corpus for scale/robustness and for the CF and sentiment work
— just not for measuring extraction quality.

## Ceiling — read before scoring

**~97.6% of gold mentions occur verbatim in the reconstructed text.** DocRED ships
tokenised sentences and `text` is a space-join, so `Wilfried " Willi " Schneider` does not
always match the gold `name`. The extraction natives DROP anything not found verbatim
(spans are located, never trusted), so 97.6% is the benchmark's ceiling, not extractor
error.

## First run — 2026-08-27, local_model, 5 documents

Smoke run, not a reference run. 5 docs / 6 chunks, user-supplied ontology (all 93
relations), `local_model` via the sandbox-scoped LiteLLM key.

| | |
|---|---|
| mentions extracted | 120 (2 dropped not-in-text, 0 off-ontology) |
| **span integrity** | **120/120 spans quote the source exactly** |
| entity recall | **0.765** (65 of 85 gold entities) |
| triplets emitted | 31 — after dropping **15 unsupported-by-text** and 1 off-ontology |
| triplet precision | **0.258** |
| triplet recall | **0.041** |
| failed LLM calls | 0 |

### What this says

**The plumbing works.** Zero failed calls with a 93-relation prompt; every span verifiable
against the source; the ontology normalised 4 awkward Wikidata labels
(`HAS_PART(S)` → `HAS_PART_S`) without collisions.

**The not-in-text guard is doing heavy lifting.** 15 of 46 candidate triplets — roughly a
third — named a head or tail that does not occur in the chunk. Without that check those
are silent hallucinations presented as facts with provenance attached. This is the single
strongest justification for the "locate, never trust" rule.

**The local model is a weak relation extractor at this ontology size.** Precision 0.258
means three of four emitted triplets are wrong. The failure has a clear shape: it anchors
on the document subject and attaches everything to it — every triplet in doc `0001` has
head `Ross Patterson Alger`, including `CAPITAL_OF Calgary` (he was its mayor) and
`MEMBER_OF_SPORTS_TEAM Royal Canadian Air Force`. Relation *type* selection is the weak
step, not entity spotting: entity recall 0.765 against triplet precision 0.258.

### Caveats on these numbers

- **5 documents.** Indicative only.
- **Exact surface-form matching, no resolution applied before scoring.** `gads_resolve_entities`
  was not run, so alias variants count as misses. Real recall is higher than 0.041.
- **Gold is document-level and includes inferred/cross-sentence facts** (~39 per doc);
  chunk-local extraction cannot reach many of them by construction. Recall against this
  gold is a lower bound on a harder task than the extractor attempts.

Precision is the trustworthy number here, and 0.258 is the honest headline.

## Next

- Reference runs proper: 100 docs, cloud + local, per the 026 protocol.
- Score after entity resolution, and relation-type-only accuracy on gold entity pairs, to
  separate "found the pair" from "named the relation".
- The obvious hypothesis to test: a smaller ontology (10-15 relations) should lift
  precision sharply if type selection is the bottleneck.
