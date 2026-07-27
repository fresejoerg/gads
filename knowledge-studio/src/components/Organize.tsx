import { useEffect, useMemo, useState } from "react";
import type { ListItem, ItemType, History, Diff } from "../types";
import { api } from "../api";

// item.type is singular; the reset endpoint wants the plural path segment.
const PLURAL: Record<ItemType, "recipes" | "skills" | "native"> = {
  recipe: "recipes",
  skill: "skills",
  native: "native",
};

export function Organize({ items, onChanged }: { items: ListItem[]; onChanged: () => void }) {
  const [sel, setSel] = useState<{ type: ItemType; id: string; filename: string } | null>(null);

  const overrides = useMemo(
    () => items.filter((it) => it.provenance === "overlay" || it.provenance === "overridden"),
    [items]
  );

  const order: ItemType[] = ["recipe", "skill", "native"];
  const groups = useMemo(() => {
    const g: Record<string, ListItem[]> = {};
    for (const it of items) (g[it.type] ||= []).push(it);
    return g;
  }, [items]);

  return (
    <>
      <div className="sidebar">
        <div className="org-overrides">
          <div className="group-label">overlay edits · {overrides.length}</div>
          {overrides.length === 0 ? (
            <div className="list-empty" style={{ padding: "8px 12px" }}>
              No overlay edits — the library is entirely shipped baseline.
            </div>
          ) : (
            overrides.map((it) => (
              <OverrideRow
                key={`${it.type}:${it.id}`}
                item={it}
                onReset={onChanged}
                onOpen={() => setSel({ type: it.type, id: it.id, filename: it.filename })}
              />
            ))
          )}
        </div>

        <div className="list">
          <div className="group-label" style={{ marginTop: 6 }}>
            history — pick any item
          </div>
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
                    onClick={() => setSel({ type: it.type, id: it.id, filename: it.filename })}
                  >
                    <div className="li-id">{it.id}</div>
                    <div className="li-meta">
                      {it.provenance && <span className={`badge ${it.provenance}`}>{it.provenance}</span>}
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
          <HistoryPanel key={`${sel.type}:${sel.id}`} type={sel.type} id={sel.id} />
        ) : (
          <div className="placeholder">
            Select an item to see its git history, diff revisions, and manage overlay edits.
          </div>
        )}
      </div>
    </>
  );
}

function OverrideRow({ item, onReset, onOpen }: { item: ListItem; onReset: () => void; onOpen: () => void }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function reset() {
    if (!window.confirm(`Discard overlay edits and revert "${item.id}" to the shipped version?`)) return;
    setBusy(true);
    setErr(null);
    try {
      await api.reset(PLURAL[item.type], item.filename);
      onReset();
    } catch (e) {
      setErr(String(e));
      setBusy(false);
    }
  }

  return (
    <div className="override-row">
      <div className="or-main" onClick={onOpen}>
        <span className={`badge ${item.type}`}>{item.type}</span>
        <span className="or-id">{item.id}</span>
        <span className={`badge ${item.provenance}`}>{item.provenance}</span>
      </div>
      <button className="btn danger sm" onClick={reset} disabled={busy}>
        {busy ? "…" : "Reset to shipped"}
      </button>
      {err && <div className="vline err">{err}</div>}
    </div>
  );
}

const WORKING = "__working__";
const NONE = "";

function HistoryPanel({ type, id }: { type: ItemType; id: string }) {
  const [hist, setHist] = useState<History | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [fromRef, setFromRef] = useState<string>(NONE);
  const [toRef, setToRef] = useState<string>(WORKING);
  const [diff, setDiff] = useState<Diff | null>(null);
  const [diffErr, setDiffErr] = useState<string | null>(null);
  const [loadingDiff, setLoadingDiff] = useState(false);

  useEffect(() => {
    setHist(null);
    setErr(null);
    setDiff(null);
    setFromRef(NONE);
    setToRef(WORKING);
    api.history(type, id).then(setHist).catch((e) => setErr(String(e)));
  }, [type, id]);

  async function runDiff() {
    setLoadingDiff(true);
    setDiff(null);
    setDiffErr(null);
    try {
      // Endpoint semantics: from+to → range; from only → that commit's change; neither → working tree vs HEAD.
      const f = fromRef === NONE ? undefined : fromRef;
      const t = toRef === WORKING ? undefined : toRef;
      const d = await api.diff(type, id, f, t);
      setDiff(d);
    } catch (e) {
      setDiffErr(String(e));
    } finally {
      setLoadingDiff(false);
    }
  }

  function showCommit(sha: string) {
    setFromRef(sha);
    setToRef(WORKING);
    api
      .diff(type, id, sha, undefined)
      .then(setDiff)
      .catch((e) => setDiffErr(String(e)));
  }

  if (err) return <div className="error-banner">{err}</div>;
  if (!hist) return <div className="loading">Loading history for {id}…</div>;

  return (
    <div className="detail">
      <div className="detail-head">
        <span className={`badge ${type}`}>{type}</span>
        <h1>{id}</h1>
      </div>
      <div className="detail-sub">
        {hist.file}
        {hist.has_overlay_edits && <span className="badge overridden" style={{ marginLeft: 8 }}>has overlay edits</span>}
      </div>

      {hist.has_overlay_edits && (
        <div className="readonly-note" style={{ marginTop: 10 }}>
          This item has overlay edits in <code>gads_data/knowledge/</code>. Git history and diffs below
          track the <strong>shipped</strong> file only — the overlay is git-ignored, so your uncommitted
          overlay changes do not appear here. Use <em>Reset to shipped</em> in the sidebar to discard them.
        </div>
      )}

      <div className="card">
        <h3>Compare revisions</h3>
        <div className="diff-controls">
          <label>
            from
            <select value={fromRef} onChange={(e) => setFromRef(e.target.value)}>
              <option value={NONE}>— (HEAD)</option>
              {hist.commits.map((c) => (
                <option key={c.sha} value={c.sha}>
                  {c.short} · {c.subject.slice(0, 48)}
                </option>
              ))}
            </select>
          </label>
          <label>
            to
            <select value={toRef} onChange={(e) => setToRef(e.target.value)}>
              <option value={WORKING}>working tree</option>
              {hist.commits.map((c) => (
                <option key={c.sha} value={c.sha}>
                  {c.short} · {c.subject.slice(0, 48)}
                </option>
              ))}
            </select>
          </label>
          <button className="btn primary" onClick={runDiff} disabled={loadingDiff}>
            {loadingDiff ? "Diffing…" : "Diff"}
          </button>
        </div>
        {diffErr && <div className="vline err">{diffErr}</div>}
        {diff && <DiffView text={diff.diff} />}
      </div>

      <div className="card">
        <h3>Commits · {hist.commits.length}</h3>
        {!hist.tracked && <div style={{ color: "var(--text-dim)" }}>No git history — this file is not committed.</div>}
        <div className="commit-list">
          {hist.commits.map((c) => (
            <div className="commit-row" key={c.sha} onClick={() => showCommit(c.sha)} title="Show what this commit changed">
              <span className="commit-sha">{c.short}</span>
              <span className="commit-subject">{c.subject}</span>
              <span className="commit-meta">
                {c.author} · {c.date.slice(0, 10)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DiffView({ text }: { text: string }) {
  if (!text.trim()) return <div style={{ color: "var(--text-dim)", marginTop: 8 }}>No changes.</div>;
  const lines = text.split("\n");
  return (
    <pre className="diff-view">
      {lines.map((ln, i) => {
        let cls = "d-ctx";
        if (ln.startsWith("+++") || ln.startsWith("---")) cls = "d-file";
        else if (ln.startsWith("@@")) cls = "d-hunk";
        else if (ln.startsWith("+")) cls = "d-add";
        else if (ln.startsWith("-")) cls = "d-del";
        else if (ln.startsWith("diff ") || ln.startsWith("index ")) cls = "d-file";
        return (
          <div className={`d-line ${cls}`} key={i}>
            {ln || " "}
          </div>
        );
      })}
    </pre>
  );
}
