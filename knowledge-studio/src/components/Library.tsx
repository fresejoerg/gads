import { useMemo, useState } from "react";
import type { ListItem, ItemType } from "../types";
import { ItemDetail } from "./ItemDetail";

const TYPES: Array<{ key: ItemType | "all"; label: string }> = [
  { key: "all", label: "All" },
  { key: "recipe", label: "Recipes" },
  { key: "skill", label: "Skills" },
  { key: "native", label: "Native" },
];

export function Library({ items, onSaved }: { items: ListItem[]; onSaved: () => void }) {
  const [filter, setFilter] = useState<ItemType | "all">("all");
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<{ type: string; id: string } | null>(null);

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return items.filter((it) => {
      if (filter !== "all" && it.type !== filter) return false;
      if (!needle) return true;
      const hay = [it.id, it.description, ...(it.task_type || []), ...(it.triggers || [])]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return hay.includes(needle);
    });
  }, [items, filter, q]);

  const groups = useMemo(() => {
    const g: Record<string, ListItem[]> = {};
    for (const it of filtered) (g[it.type] ||= []).push(it);
    return g;
  }, [filtered]);

  const order: ItemType[] = ["recipe", "skill", "native"];

  return (
    <>
      <div className="sidebar">
        <div className="search">
          <input
            type="text"
            placeholder="Search id, trigger, task type…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
          <div className="typefilter">
            {TYPES.map((t) => (
              <button key={t.key} className={filter === t.key ? "on" : ""} onClick={() => setFilter(t.key)}>
                {t.label}
              </button>
            ))}
          </div>
        </div>
        <div className="list">
          {filtered.length === 0 && <div className="list-empty">No items match.</div>}
          {order.map((type) =>
            groups[type]?.length ? (
              <div key={type}>
                <div className="group-label">
                  {type}s · {groups[type].length}
                </div>
                {groups[type].map((it) => (
                  <div
                    key={`${it.type}:${it.id}`}
                    className={`list-item ${sel?.id === it.id && sel?.type === it.type ? "sel" : ""}`}
                    onClick={() => setSel({ type: it.type, id: it.id })}
                  >
                    <div className="li-id">{it.id}</div>
                    <div className="li-meta">
                      {it.provenance && <span className={`badge ${it.provenance}`}>{it.provenance}</span>}
                      {it.version && <span>v{it.version}</span>}
                      {it.type === "skill" && it.triggers && <span>{it.triggers.length} triggers</span>}
                    </div>
                  </div>
                ))}
              </div>
            ) : null
          )}
        </div>
      </div>
      <div className="main">
        {sel ? (
          <ItemDetail key={`${sel.type}:${sel.id}`} type={sel.type} id={sel.id} onSaved={onSaved} />
        ) : (
          <div className="placeholder">Select an item from the library</div>
        )}
      </div>
    </>
  );
}
