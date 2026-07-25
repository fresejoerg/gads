import { lazy, Suspense, useEffect, useState } from "react";
import { api } from "../api";
import type { ItemDetail as Detail, ParsedRecipe, ParsedSkill, Evidence, Impact } from "../types";
import { DagDiagram } from "./DagDiagram";

// Monaco is heavy and only the Edit/Source tab needs it — load it on demand so the
// Library / Overview / Coverage views stay lean.
const Editor = lazy(() => import("./Editor").then((m) => ({ default: m.Editor })));

type Sub = "overview" | "edit" | "evidence" | "impact";

export function ItemDetail({ type, id, onSaved }: { type: string; id: string; onSaved: () => void }) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [sub, setSub] = useState<Sub>("overview");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setDetail(null);
    setErr(null);
    api.detail(type, id).then(setDetail).catch((e) => setErr(String(e)));
  }, [type, id]);

  if (err) return <div className="error-banner">{err}</div>;
  if (!detail) return <div className="loading">Loading {id}…</div>;

  const isRecipe = detail.type === "recipes";
  const subs: Array<[Sub, string]> = [
    ["overview", "Overview"],
    ["edit", detail.editable ? "Edit" : "Source"],
    ...(isRecipe ? ([["evidence", "Evidence"], ["impact", "Impact"]] as Array<[Sub, string]>) : ([["impact", "Impact"]] as Array<[Sub, string]>)),
  ];

  const singular = detail.type === "recipes" ? "recipe" : detail.type === "skills" ? "skill" : "native";

  return (
    <div className="detail">
      <div className="detail-head">
        <span className={`badge ${singular}`}>{singular}</span>
        <h1>{detail.id}</h1>
        <span className={`badge ${detail.provenance}`}>{detail.provenance}</span>
      </div>
      <div className="detail-sub">{detail.filename}</div>

      <div className="subtabs">
        {subs.map(([k, label]) => (
          <button key={k} className={`subtab ${sub === k ? "active" : ""}`} onClick={() => setSub(k)}>
            {label}
          </button>
        ))}
      </div>

      {sub === "overview" && <Overview detail={detail} />}
      {sub === "edit" && (
        <Suspense fallback={<div className="loading">Loading editor…</div>}>
          <Editor
            type={detail.type}
            filename={detail.filename}
            initial={detail.raw}
            editable={detail.editable}
            runtimePending={detail.runtime_pending}
            onSaved={() => {
              api.detail(type, id).then(setDetail);
              onSaved();
            }}
          />
        </Suspense>
      )}
      {sub === "evidence" && isRecipe && <EvidencePanel id={id} />}
      {sub === "impact" && <ImpactPanel type={singular} id={id} />}
    </div>
  );
}

function Overview({ detail }: { detail: Detail }) {
  if (detail.type === "recipes" && detail.parsed) {
    const r = detail.parsed as ParsedRecipe;
    return (
      <>
        <div className="card">
          <h3>Applies when</h3>
          <div className="kv">
            {Object.entries(r.applies_when || {}).map(([k, v]) => (
              <FragmentKV key={k} k={k} v={v} />
            ))}
            {Object.keys(r.requires || {}).length > 0 && <div className="k">requires</div>}
            {Object.keys(r.requires || {}).length > 0 && (
              <div className="v">
                {Object.entries(r.requires).map(([k, v]) => `${k}: ${fmt(v)}`).join("  ·  ")}
              </div>
            )}
            <div className="k">author</div>
            <div className="v">{r.author} · v{r.version}</div>
          </div>
        </div>

        <div className="card">
          <h3>DAG · {r.dag.length} tasks</h3>
          <DagDiagram tasks={r.dag} />
        </div>

        {r.invariants?.length > 0 && (
          <div className="card">
            <h3>Invariants</h3>
            <ul className="invariants">
              {r.invariants.map((inv, i) => (
                <li key={i}>{inv}</li>
              ))}
            </ul>
          </div>
        )}

        {r.rationale && (
          <div className="card">
            <h3>Rationale</h3>
            <div className="rationale">{r.rationale}</div>
          </div>
        )}
      </>
    );
  }

  if (detail.type === "skills" && detail.parsed) {
    const s = detail.parsed as ParsedSkill;
    return (
      <>
        <div className="card">
          <h3>Description</h3>
          <div className="rationale">{s.description || "—"}</div>
        </div>
        <div className="card">
          <h3>Triggers · {s.triggers.length}</h3>
          <div>
            {s.triggers.map((t) => (
              <span className="chip" key={t}>
                {t}
              </span>
            ))}
          </div>
        </div>
        <div className="card">
          <h3>Content</h3>
          <pre className="rationale" style={{ fontFamily: "var(--mono)", fontSize: 12.5 }}>
            {s.content}
          </pre>
        </div>
      </>
    );
  }

  // native — no parsed frontmatter
  return (
    <div className="card">
      <h3>Native module</h3>
      <pre className="rationale" style={{ fontFamily: "var(--mono)", fontSize: 12.5 }}>
        {detail.raw.slice(0, 4000)}
        {detail.raw.length > 4000 ? "\n… (truncated — see Source tab)" : ""}
      </pre>
    </div>
  );
}

function FragmentKV({ k, v }: { k: string; v: unknown }) {
  return (
    <>
      <div className="k">{k}</div>
      <div className="v">{fmt(v)}</div>
    </>
  );
}

function fmt(v: unknown): string {
  if (Array.isArray(v)) return v.join(", ");
  if (v && typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function EvidencePanel({ id }: { id: string }) {
  const [ev, setEv] = useState<Evidence | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.evidence(id).then(setEv).catch((e) => setErr(String(e)));
  }, [id]);
  if (err) return <div className="error-banner">{err}</div>;
  if (!ev) return <div className="loading">Loading evidence…</div>;
  if (ev.total_runs === 0)
    return <div className="card">No benchmark runs recorded for this recipe in the dial ledger yet.</div>;

  return (
    <>
      <div className="metric-row">
        <div>
          <div className="k">Overall pass rate</div>
          <div className="metric-big">{ev.overall_pass_rate != null ? `${Math.round(ev.overall_pass_rate * 100)}%` : "—"}</div>
        </div>
        <div>
          <div className="k">Total runs</div>
          <div className="metric-big">{ev.total_runs}</div>
        </div>
        <div>
          <div className="k">Rungs observed</div>
          <div className="metric-big">{ev.rungs_observed.join(" ") || "—"}</div>
        </div>
      </div>

      <div className="card">
        <h3>By engine</h3>
        <table className="tbl">
          <thead>
            <tr>
              <th>Engine</th>
              <th>Runs</th>
              <th>Pass</th>
              <th>Fail</th>
              <th>Pass rate</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(ev.by_engine).map(([eng, b]) => (
              <tr key={eng}>
                <td>{eng}</td>
                <td>{b.runs}</td>
                <td style={{ color: "var(--green)" }}>{b.pass}</td>
                <td style={{ color: b.fail ? "var(--red)" : "var(--text-faint)" }}>{b.fail}</td>
                <td>
                  <span className="bar" style={{ width: 80 }}>
                    <span style={{ width: `${(b.pass_rate || 0) * 100}%` }} />
                  </span>{" "}
                  {b.pass_rate != null ? `${Math.round(b.pass_rate * 100)}%` : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>Recent runs</h3>
        <table className="tbl">
          <thead>
            <tr>
              <th>When</th>
              <th>Engine</th>
              <th>Rung</th>
              <th>Outcome</th>
              <th>Spec</th>
            </tr>
          </thead>
          <tbody>
            {ev.recent.map((r, i) => (
              <tr key={i}>
                <td>{r.ts?.slice(0, 10)}</td>
                <td>{r.engine}</td>
                <td>{r.rung}</td>
                <td style={{ color: r.outcome === "pass" ? "var(--green)" : "var(--red)" }}>{r.outcome}</td>
                <td>{r.spec}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function ImpactPanel({ type, id }: { type: string; id: string }) {
  const [imp, setImp] = useState<Impact | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.impact(type, id).then(setImp).catch((e) => setErr(String(e)));
  }, [type, id]);
  if (err) return <div className="error-banner">{err}</div>;
  if (!imp) return <div className="loading">Loading impact…</div>;
  const rb = imp.referenced_by;

  return (
    <>
      <div className="card">
        <h3>Referenced by · {rb.total} direct</h3>
        {rb.total === 0 && <div style={{ color: "var(--text-dim)" }}>Nothing references this item — safe to rename or deprecate.</div>}
        {rb.specs.length > 0 && (
          <>
            <div style={{ color: "var(--text-dim)", margin: "6px 0" }}>Specs that pin it:</div>
            <div>
              {rb.specs.map((s) => (
                <span className="chip" key={s}>
                  {s}
                </span>
              ))}
            </div>
          </>
        )}
        {rb.recipes.length > 0 && (
          <>
            <div style={{ color: "var(--text-dim)", margin: "10px 0 6px" }}>Recipes that use it:</div>
            <div>
              {rb.recipes.map((r) => (
                <span className="chip" key={r.recipe}>
                  {r.recipe}
                  {r.nodes ? ` (${r.nodes.join(", ")})` : ""}
                </span>
              ))}
            </div>
          </>
        )}
      </div>
      {typeof rb.ledger_runs === "number" && (
        <div className="card">
          <h3>Ledger</h3>
          <div className="kv">
            <div className="k">Benchmark runs</div>
            <div className="v">{rb.ledger_runs}</div>
          </div>
        </div>
      )}
    </>
  );
}
