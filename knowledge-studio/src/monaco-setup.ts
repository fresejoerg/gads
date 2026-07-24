// Local-first Monaco: @monaco-editor/react loads the editor from a CDN by default,
// which breaks in an air-gapped deployment — the whole point of GADS. Point its loader
// at the bundled `monaco-editor` package instead, and serve the editor web-worker from
// the local build (Vite ?worker import) rather than a remote CDN.
//
// We edit Markdown (recipes/skills) and Python (native) — both are Monarch tokenizers
// with no dedicated language worker, so the base editor.worker is all that's needed.
import { loader } from "@monaco-editor/react";
// Slim import: the core editor API plus only the two Monarch languages we edit —
// avoids bundling every basic language (abap, solidity, sql, …) and the json/ts/css
// language services we never use.
import * as monaco from "monaco-editor/esm/vs/editor/editor.api";
import "monaco-editor/esm/vs/basic-languages/markdown/markdown.contribution";
import "monaco-editor/esm/vs/basic-languages/python/python.contribution";
import EditorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";

self.MonacoEnvironment = {
  getWorker() {
    return new EditorWorker();
  },
};

loader.config({ monaco });
