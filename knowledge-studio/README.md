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

## What's here (Phase 1)

- **Library** — browse/filter/search all items by type; provenance badges (shipped /
  overlay / overridden).
- **Item · Overview** — parsed frontmatter as first-class UI: recipe `applies_when`/
  `requires`/`author` cards, the **DAG as a flow diagram** (D5 nodes ringed, attached
  skills chipped), invariants checklist, rationale; skill triggers + content.
- **Item · Edit** — Monaco with **live server-side validation** (errors block save,
  warnings don't); writes go to the git-ignored overlay (`gads_data/knowledge/`), never
  the shipped IP. Native modules are read-only pending the native-editing increment.
- **Item · Evidence** (recipes) — per-engine benchmark pass rate from the dial ledger.
- **Item · Impact** — specs/recipes referencing the item (rename/deprecate guard).
- **Coverage** — task_type × delegation-rung matrix + orphan analysis.

## Known follow-ups

- **Monaco is loaded from a CDN** by `@monaco-editor/react`'s default loader. For a
  strictly offline/air-gapped deployment, bundle `monaco-editor` locally (e.g.
  `vite-plugin-monaco-editor`) — tracked for Phase 2.
- Native-node editing (overlay write + AST/hazard guardrails + dynamic executor load).
- Organize view: maturity field, git history/diff surface, recipe-pack export/import.
