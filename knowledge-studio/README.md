# GADS Knowledge Studio

A dedicated SPA for viewing, editing, and organizing the system's IP — the **recipes**,
**skills**, and **native nodes**. Decoupled from the workflow runner: its own entry point,
no shared state, talking only to the read/write **Knowledge API** on the GADS backend
(`approach_docs/017`).

## Run

```bash
./start_backend.sh          # GADS backend on :8001 (serves the Knowledge API)
./scripts/run_studio.sh     # Studio on http://localhost:5173  (installs deps on first run)
```

The Vite dev server proxies the API paths (`/knowledge`, `/recipes`, `/skills`, `/native`)
to the backend, so the browser makes same-origin calls. Point the proxy elsewhere with
`GADS_API_URL=http://host:port ./scripts/run_studio.sh`.

## Stack

Vite + React + TypeScript · **Monaco** editor · **React Flow** for the DAG diagram.
`npm run build` emits a static bundle in `dist/`.

Monaco is **bundled locally** (not loaded from a CDN) and **code-split** — it ships as a
separate chunk that only loads when the Edit/Source tab opens, so the Library / Overview /
Coverage views stay lean (~98 KB gzip) and the tool works fully air-gapped. See
`src/monaco-setup.ts`.

## What's here (Phase 1)

- **Library** — browse/filter/search all items by type; provenance badges (shipped /
  overlay / overridden).
- **Item · Overview** — parsed frontmatter as first-class UI: recipe `applies_when`/
  `requires`/`author` cards, the **DAG as a flow diagram** (D5 nodes ringed, attached
  skills chipped), invariants checklist, rationale; skill triggers + content.
- **Item · Edit** — Monaco with **live server-side validation** (errors block save,
  warnings don't); writes go to the git-ignored overlay (`gads_data/knowledge/`), never
  the shipped IP. Native modules are editable too (Python; AST + hazard validation), but
  carry a caveat: overlay native edits are **not yet loaded by the executor** — a running
  workflow uses the shipped version until the dynamic-load contract lands.
- **Item · Evidence** (recipes) — per-engine benchmark pass rate from the dial ledger.
- **Item · Impact** — specs/recipes referencing the item (rename/deprecate guard).
- **Coverage** — task_type × delegation-rung matrix + orphan analysis.
- **Organize** — overlay-override management (list every overridden/overlay item and
  **reset it to the shipped baseline** in one click) plus a **git history + diff surface**:
  browse an item's commit log, view what any single commit changed, or diff two revisions
  (or a revision vs the working tree). Diffs track the *shipped* file only — the overlay is
  git-ignored, so uncommitted overlay edits are managed via Reset, not shown as a git diff.

## Known follow-ups

- **Native dynamic-load contract** — make the executor resolve overlay native modules
  (declare `id`/`preamble`/`triggers` per module; load overlay before shipped) so edits
  saved here actually take effect at runtime. Overlay write + validation already ship;
  this is the remaining execution-path half.
- **Organize · P2** — a `maturity` field (`draft | battle-tested | deprecated`) and
  recipe-pack export/import. Both need backend schema/endpoint work first (the current
  Organize view is scoped to what the read/write API already backs: provenance, reset,
  history, diff).
