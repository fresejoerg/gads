import type { ListItem, ItemDetail, ValidationResult, Evidence, Impact, History, Diff } from "./types";

// Vite proxies these paths to the GADS backend (see vite.config.ts), so calls are
// same-origin in dev. In a static deployment, set VITE_API_BASE at build time.
const BASE = (import.meta.env.VITE_API_BASE as string) || "";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  return r.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json() as Promise<T>;
}

// item type -> save path segment
const SAVE_PATH: Record<string, string> = { recipes: "recipes", skills: "skills", native: "native" };

export const api = {
  items: () => get<ListItem[]>("/knowledge/items"),
  detail: (type: string, id: string) =>
    get<ItemDetail>(`/knowledge/${type}/${encodeURIComponent(id)}`),
  graph: () => get<GraphResp>("/knowledge/graph"),
  coverage: () => get<CoverageResp>("/knowledge/coverage"),
  evidence: (id: string) =>
    get<Evidence>(`/knowledge/recipe/${encodeURIComponent(id)}/evidence`),
  impact: (type: string, id: string) =>
    get<Impact>(`/knowledge/${type}/${encodeURIComponent(id)}/impact`),
  validate: (type: string, content: string, filename?: string) =>
    post<ValidationResult>("/knowledge/validate", { type, content, filename }),
  save: (type: "recipes" | "skills" | "native", filename: string, content: string) =>
    post<{ status: string }>(`/${SAVE_PATH[type]}/${filename}`, { content }),
  history: (type: string, id: string, limit = 50) =>
    get<History>(`/knowledge/${type}/${encodeURIComponent(id)}/history?limit=${limit}`),
  diff: (type: string, id: string, fromRef?: string, toRef?: string) => {
    const q = new URLSearchParams();
    if (fromRef) q.set("from_ref", fromRef);
    if (toRef) q.set("to_ref", toRef);
    const qs = q.toString();
    return get<Diff>(`/knowledge/${type}/${encodeURIComponent(id)}/diff${qs ? `?${qs}` : ""}`);
  },
  reset: (type: "recipes" | "skills" | "native", filename: string) =>
    post<{ status: string; provenance: string }>(`/knowledge/${type}/${filename}/reset`, {}),
};

export interface GraphResp {
  nodes: Array<{ id: string; type: string; provenance?: string; triggers?: string[] }>;
  edges: Array<{ source: string; target: string; type: string; kind: string; dangling?: boolean }>;
  counts: Record<string, number>;
}

export interface CoverageResp {
  matrix: Record<string, Record<string, string[]>>;
  rungs: string[];
  recipe_rungs: Record<string, string>;
  task_types: string[];
  orphans: {
    recipes_without_task_type: string[];
    skills_keyword_only: string[];
    native_unreferenced: string[];
  };
}
