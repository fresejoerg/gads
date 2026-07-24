import { useMemo } from "react";
import ReactFlow, { Background, Controls, MarkerType, Position, type Node, type Edge } from "reactflow";
import "reactflow/dist/style.css";
import type { RecipeTask } from "../types";

// Deterministic layered layout: rank = longest dependency depth from a root, so the
// recipe DAG reads left→right in execution order without a layout engine dependency.
function layer(tasks: RecipeTask[]): Map<string, number> {
  const byId = new Map(tasks.map((t) => [t.id, t]));
  const memo = new Map<string, number>();
  const visiting = new Set<string>();
  function depth(id: string): number {
    if (memo.has(id)) return memo.get(id)!;
    if (visiting.has(id)) return 0; // cycle guard
    visiting.add(id);
    const t = byId.get(id);
    const deps = (t?.depends_on || []).filter((d) => byId.has(d));
    const d = deps.length ? 1 + Math.max(...deps.map(depth)) : 0;
    visiting.delete(id);
    memo.set(id, d);
    return d;
  }
  tasks.forEach((t) => depth(t.id));
  return memo;
}

function rung(t: RecipeTask): "d3" | "d4" | "d5" {
  if (t.intent && t.intent.includes("gads_causal_estimate_ate")) return "d5";
  const curated = (t.attached_skills || []).filter((s) => s !== "sandbox_environment");
  return curated.length ? "d4" : "d3";
}

export function DagDiagram({ tasks }: { tasks: RecipeTask[] }) {
  const { nodes, edges } = useMemo(() => {
    const depths = layer(tasks);
    const perLayer = new Map<number, number>();
    const nodes: Node[] = tasks.map((t) => {
      const d = depths.get(t.id) ?? 0;
      const row = perLayer.get(d) ?? 0;
      perLayer.set(d, row + 1);
      const r = rung(t);
      const skills = (t.attached_skills || []).filter((s) => s !== "sandbox_environment");
      return {
        id: t.id,
        position: { x: d * 250, y: row * 128 },
        data: {
          label: (
            <div className={`dag-node ${r}`}>
              <div className="n-id">{t.id}</div>
              <div className="n-intent">{t.intent}</div>
              <div className="n-tags">
                <span className="n-tag tier">{t.worker_tier}</span>
                {r === "d5" && <span className="n-tag native">native</span>}
                {skills.map((s) => (
                  <span className="n-tag skill" key={s}>
                    {s}
                  </span>
                ))}
              </div>
            </div>
          ),
        },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        style: { background: "transparent", border: "none", padding: 0, width: "auto" },
      };
    });

    const ids = new Set(tasks.map((t) => t.id));
    const edges: Edge[] = [];
    tasks.forEach((t) =>
      (t.depends_on || [])
        .filter((d) => ids.has(d))
        .forEach((d) =>
          edges.push({
            id: `${d}->${t.id}`,
            source: d,
            target: t.id,
            markerEnd: { type: MarkerType.ArrowClosed, color: "#3a4351" },
            style: { stroke: "#3a4351", strokeWidth: 1.5 },
          })
        )
    );
    return { nodes, edges };
  }, [tasks]);

  return (
    <div className="dag-wrap">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        nodesDraggable={false}
        nodesConnectable={false}
        proOptions={{ hideAttribution: true }}
        minZoom={0.3}
      >
        <Background color="#1c2230" gap={20} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
