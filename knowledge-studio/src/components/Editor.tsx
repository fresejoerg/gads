import { useEffect, useRef, useState } from "react";
import MonacoEditor from "@monaco-editor/react";
import "../monaco-setup"; // pulls Monaco into THIS lazy chunk, not the initial bundle
import { api } from "../api";
import type { ValidationResult } from "../types";

// singular type used by the /knowledge/validate contract
const VALIDATE_TYPE: Record<string, string> = { recipes: "recipe", skills: "skill", native: "native" };

export function Editor({
  type,
  filename,
  initial,
  editable,
  runtimePending,
  onSaved,
}: {
  type: "recipes" | "skills" | "native";
  filename: string;
  initial: string;
  editable: boolean;
  runtimePending?: boolean;
  onSaved: () => void;
}) {
  const [content, setContent] = useState(initial);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [saving, setSaving] = useState(false);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);
  const debounce = useRef<number | undefined>(undefined);

  const dirty = content !== initial;
  const language = type === "native" ? "python" : "markdown";

  // Live validation, debounced.
  useEffect(() => {
    window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => {
      api
        .validate(VALIDATE_TYPE[type], content, filename)
        .then(setValidation)
        .catch(() => setValidation(null));
    }, 500);
    return () => window.clearTimeout(debounce.current);
  }, [content, type, filename]);

  async function save() {
    setSaving(true);
    setSavedMsg(null);
    try {
      await api.save(type, filename, content);
      setSavedMsg(runtimePending ? "Saved to overlay ✓ (not yet loaded by the executor)" : "Saved to overlay ✓");
      onSaved();
    } catch (e) {
      setSavedMsg(`Save failed: ${e}`);
    } finally {
      setSaving(false);
    }
  }

  const blockingErrors = validation?.errors?.length ?? 0;

  return (
    <>
      {type === "native" && (
        <div className="readonly-note">
          ⚠ Native modules are <strong>Python injected into sandbox-executed code</strong>. Edits are
          validated (syntax&nbsp;+ hazard scan) and saved to the git-ignored overlay — but the executor
          does <strong>not yet</strong> load overlay native modules, so a running workflow still uses the
          <strong> shipped</strong> version until the dynamic-load contract lands.
        </div>
      )}
      <div className="editor-bar">
        <button className="btn primary" onClick={save} disabled={!editable || !dirty || saving || blockingErrors > 0}>
          {saving ? "Saving…" : "Save to overlay"}
        </button>
        <button className="btn" onClick={() => setContent(initial)} disabled={!dirty}>
          Revert edits
        </button>
        {dirty && <span style={{ color: "var(--amber)", fontSize: 12.5 }}>unsaved changes</span>}
        {savedMsg && <span style={{ color: "var(--green)", fontSize: 12.5 }}>{savedMsg}</span>}
      </div>

      <div className="monaco-host">
        <MonacoEditor
          height="460px"
          theme="vs-dark"
          language={language}
          value={content}
          onChange={(v) => setContent(v ?? "")}
          options={{
            readOnly: !editable,
            minimap: { enabled: false },
            fontSize: 13,
            fontFamily: "var(--mono)",
            wordWrap: "on",
            scrollBeyondLastLine: false,
            padding: { top: 10 },
          }}
        />
      </div>

      <div className="validation">
        {validation && blockingErrors === 0 && (validation.warnings?.length ?? 0) === 0 && (
          <div className="vline ok">✓ Valid — no errors or warnings</div>
        )}
        {validation?.errors?.map((e, i) => (
          <div className="vline err" key={`e${i}`}>
            ✕ {e}
          </div>
        ))}
        {validation?.warnings?.map((w, i) => (
          <div className="vline warn" key={`w${i}`}>
            ⚠ {w}
          </div>
        ))}
      </div>
    </>
  );
}
