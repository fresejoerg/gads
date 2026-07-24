import { useEffect, useState } from "react";
import { api, type CoverageResp } from "../api";

export function Coverage() {
  const [cov, setCov] = useState<CoverageResp | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.coverage().then(setCov).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div className="error-banner">{err}</div>;
  if (!cov) return <div className="loading">Loading coverage…</div>;

  return (
    <div className="main">
      <div className="detail">
        <div className="detail-head">
          <h1>Coverage</h1>
        </div>
        <div className="detail-sub">
          Recipe library across task type × delegation rung — where the battle-tested IP is thick, and where it's thin.
        </div>

        <div className="card">
          <h3>task_type × rung</h3>
          <div className="cov-grid">
            <table className="cov">
              <thead>
                <tr>
                  <th>task_type</th>
                  {cov.rungs.map((r) => (
                    <th key={r}>{r}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cov.task_types.map((tt) => (
                  <tr key={tt}>
                    <td className="tt">{tt}</td>
                    {cov.rungs.map((r) => {
                      const recipes = cov.matrix[tt]?.[r] || [];
                      return (
                        <td key={r} title={recipes.join("\n")}>
                          <span className={`cnt ${recipes.length ? "has" : "none"}`}>
                            {recipes.length || "·"}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <h3>Orphans</h3>
          <OrphanRow label="Skills reachable only by keyword (not attached to any recipe)" items={cov.orphans.skills_keyword_only} />
          <OrphanRow label="Native modules no recipe mechanizes" items={cov.orphans.native_unreferenced} />
          <OrphanRow label="Recipes without a declared task_type" items={cov.orphans.recipes_without_task_type} />
        </div>
      </div>
    </div>
  );
}

function OrphanRow({ label, items }: { label: string; items: string[] }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ color: "var(--text-dim)", fontSize: 12.5, marginBottom: 5 }}>
        {label} · {items.length}
      </div>
      <div>
        {items.length === 0 ? (
          <span style={{ color: "var(--text-faint)", fontSize: 12.5 }}>none</span>
        ) : (
          items.map((s) => (
            <span className="chip" key={s}>
              {s}
            </span>
          ))
        )}
      </div>
    </div>
  );
}
