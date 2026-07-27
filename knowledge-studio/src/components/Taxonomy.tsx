import { useEffect, useState } from "react";
import { api, type TaxonomyCoverage, type SpecTag, type RunTag } from "../api";

// Short human labels for the intent axis (facet A, approach_docs/018 §3).
const INTENT_LABEL: Record<string, string> = {
  descriptive: "Descriptive",
  diagnostic: "Diagnostic",
  predictive: "Predictive",
  causal: "Causal",
  prescriptive: "Prescriptive",
  generative: "Generative",
  structure_discovery: "Structure discovery",
};

export function Taxonomy() {
  const [cov, setCov] = useState<TaxonomyCoverage | null>(null);
  const [specs, setSpecs] = useState<SpecTag[]>([]);
  const [runs, setRuns] = useState<RunTag[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [cell, setCell] = useState<{ intent: string; family: string; specs: string[] } | null>(null);

  useEffect(() => {
    Promise.all([api.taxonomyCoverage(), api.taxonomySpecs(), api.taxonomyRuns()])
      .then(([c, s, r]) => {
        setCov(c);
        setSpecs(s);
        setRuns(r);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div className="error-banner">{err}</div>;
  if (!cov) return <div className="loading">Loading taxonomy coverage…</div>;

  const untagged = specs.filter((s) => !s.tagged);

  return (
    <div className="main">
      <div className="detail">
        <div className="detail-head">
          <h1>Taxonomy coverage</h1>
        </div>
        <div className="detail-sub">
          Configured specs across the data-science project taxonomy (approach_docs/018).
          <strong> {cov.distinct_projects}</strong> distinct project types over {cov.total_specs} specs
          (dial variants folded). Empty cells are the gaps.
        </div>

        <div className="card">
          <h3>Intent × task family</h3>
          <div className="cov-grid">
            <table className="cov taxo">
              <thead>
                <tr>
                  <th>intent \ task</th>
                  {cov.task_families.map((f) => (
                    <th key={f} className="rot">
                      <span>{f}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cov.intents.map((intent) => (
                  <tr key={intent}>
                    <td className="tt">{INTENT_LABEL[intent] || intent}</td>
                    {cov.task_families.map((f) => {
                      const cellSpecs = cov.matrix[intent]?.[f] || [];
                      const has = cellSpecs.length > 0;
                      return (
                        <td
                          key={f}
                          title={has ? cellSpecs.join("\n") : "no spec"}
                          onClick={() => has && setCell({ intent, family: f, specs: cellSpecs })}
                          style={{ cursor: has ? "pointer" : "default" }}
                        >
                          <span className={`cnt ${has ? "has" : "none"}`}>{cellSpecs.length || "·"}</span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="rung-note">
            Coverage sits in just <strong>{cov.populated_cells.length}</strong> of{" "}
            {cov.intents.length * cov.task_families.length} cells. Every "·" is a project type the lab
            has no spec for — high-value empties include <em>predictive · forecasting</em>,{" "}
            <em>structure_discovery · clustering</em>, and <em>predictive · anomaly_detection</em>
            {" "}(recipes exist, no spec exercises them).
          </div>
          {cell && (
            <div className="cell-drill">
              <div className="cell-drill-head">
                {INTENT_LABEL[cell.intent] || cell.intent} · {cell.family} — {cell.specs.length} spec
                {cell.specs.length === 1 ? "" : "s"}
                <button className="btn sm" onClick={() => setCell(null)}>
                  close
                </button>
              </div>
              <div>
                {cell.specs.map((s) => (
                  <span className="chip" key={s}>
                    {s}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="dist-row">
          <Dist title="By intent" data={cov.intent_distribution} />
          <Dist title="By modality" data={cov.modality_distribution} />
          <Dist title="By domain" data={cov.domain_distribution} />
        </div>

        {untagged.length > 0 && (
          <div className="card">
            <h3>Untagged specs · {untagged.length}</h3>
            <div>
              {untagged.map((s) => (
                <span className="chip" key={s.spec}>
                  {s.spec}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="card">
          <h3>Classified runs · {runs.length}</h3>
          <div className="detail-sub" style={{ margin: "0 0 10px" }}>
            Every launched run is classified — spec-launched runs keep their block, ad-hoc runs
            are derived from the Router's intent + a domain hint (<code>source</code> shows which).
          </div>
          {runs.length === 0 ? (
            <div style={{ color: "var(--text-faint)", fontSize: 12.5 }}>
              No classified runs yet — launch a project and it will appear here.
            </div>
          ) : (
            <table className="tbl">
              <thead>
                <tr>
                  <th>When</th>
                  <th>Run</th>
                  <th>Intent</th>
                  <th>Task</th>
                  <th>Modality</th>
                  <th>Domain</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.project_id}>
                    <td>{r.created_at?.slice(0, 10)}</td>
                    <td>{r.name}</td>
                    <td>{INTENT_LABEL[r.taxonomy.intent] || r.taxonomy.intent}</td>
                    <td>{r.taxonomy.task.join(", ")}</td>
                    <td>{r.taxonomy.modality.join(", ")}</td>
                    <td>{r.taxonomy.domain}</td>
                    <td>
                      <span className={`badge ${r.source === "spec" ? "shipped" : "overlay"}`}>{r.source || "?"}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

function Dist({ title, data }: { title: string; data: Record<string, number> }) {
  const rows = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const max = rows.length ? Math.max(...rows.map((r) => r[1])) : 1;
  return (
    <div className="card dist-card">
      <h3>{title}</h3>
      {rows.length === 0 && <div style={{ color: "var(--text-faint)", fontSize: 12.5 }}>none</div>}
      {rows.map(([k, v]) => (
        <div className="dist-line" key={k}>
          <span className="dist-key">{k}</span>
          <span className="bar" style={{ flex: 1 }}>
            <span style={{ width: `${(v / max) * 100}%` }} />
          </span>
          <span className="dist-val">{v}</span>
        </div>
      ))}
    </div>
  );
}
