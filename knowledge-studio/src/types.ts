export type ItemType = "recipe" | "skill" | "native";
export type Provenance = "shipped" | "overlay" | "overridden";

export interface ListItem {
  type: ItemType;
  id: string;
  filename: string;
  provenance?: Provenance;
  // recipe
  version?: string;
  task_type?: string[];
  data_modality?: string[];
  capabilities?: string[];
  // skill
  description?: string;
  triggers?: string[];
}

export interface RecipeTask {
  id: string;
  intent: string;
  worker_tier: string;
  depends_on: string[];
  produces: string[];
  postconditions: string[];
  required_metrics: string[];
  attached_skills: string[];
  skippable_if: string | null;
  rationale_required: boolean;
}

export interface ParsedRecipe {
  id: string;
  version: string;
  author: string;
  applies_when: Record<string, unknown>;
  requires: Record<string, unknown>;
  dag: RecipeTask[];
  invariants: string[];
  rationale: string;
}

export interface ParsedSkill {
  id: string;
  triggers: string[];
  description: string;
  content: string;
}

export interface ItemDetail {
  type: "recipes" | "skills" | "native";
  id: string;
  filename: string;
  provenance: Provenance;
  editable: boolean;
  runtime_pending?: boolean; // native: saved to overlay but executor still uses shipped
  raw: string;
  parsed: ParsedRecipe | ParsedSkill | null;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export interface Evidence {
  id: string;
  total_runs: number;
  overall_pass_rate: number | null;
  rungs_observed: string[];
  by_engine: Record<string, { runs: number; pass: number; fail: number; pass_rate: number | null }>;
  recent: Array<{ ts: string; engine: string; rung: string; outcome: string; spec: string; project_id: string }>;
}

export interface Impact {
  item_type: string;
  id: string;
  referenced_by: {
    specs: string[];
    recipes: Array<{ recipe: string; nodes?: string[]; uses?: unknown[] }>;
    ledger_runs?: number;
    total: number;
  };
}
