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
  taxonomyCoverage: () => get<TaxonomyCoverage>("/taxonomy/coverage"),
  taxonomySpecs: () => get<SpecTag[]>("/taxonomy/specs"),
  taxonomyRuns: (limit = 100) => get<RunTag[]>(`/taxonomy/runs?limit=${limit}`),
  taxonomyRecipes: () => get<RecipeCoverage>("/taxonomy/recipes"),
};

export interface RecipeCoverage {
  intents: string[];
  task_families: string[];
  matrix: Record<string, Record<string, string[]>>;
  total_recipes: number;
  intent_distribution: Record<string, number>;
  family_distribution: Record<string, number>;
  modality_distribution: Record<string, number>;
  unmapped: Array<{ id: string; declared: string[] }>;
  populated_cells: Array<[string, string]>;
}

export interface RunTag {
  project_id: string;
  name: string;
  created_at: string | null;
  source: string | null;
  taxonomy: {
    intent: string;
    task: string[];
    modality: string[];
    domain: string;
    domain_detail?: string;
    deliverable: string[];
    validation?: string[];
  };
}

export interface TaxonomyCoverage {
  intents: string[];
  task_families: string[];
  matrix: Record<string, Record<string, string[]>>;
  distinct_projects: number;
  total_specs: number;
  intent_distribution: Record<string, number>;
  modality_distribution: Record<string, number>;
  domain_distribution: Record<string, number>;
  populated_cells: Array<[string, string]>;
}

export interface SpecTag {
  spec: string;
  name: string | null;
  recipe_id: string | null;
  rung: string | null;
  tagged: boolean;
  intent: string | null;
  task: string[];
  modality: string[];
  domain: string | null;
  deliverable: string[];
  validation: string[];
}

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
