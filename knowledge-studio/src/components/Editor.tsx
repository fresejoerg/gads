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
  onSaved,
}: {
  type: "recipes" | "skills" | "native";
  filename: string;
  initial: string;
  editable: boolean;
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
    if (type === "native") return; // native write lands in a later increment
    setSaving(true);
    setSavedMsg(null);
    try {
      await api.save(type, filename, content);
      setSavedMsg("Saved to overlay ✓");
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
      {!editable && (
        <div className="readonly-note">
          Native modules run Python inside the sandbox — editing them is not yet enabled in the Studio (a
          later increment adds it with AST/hazard guardrails). Shown read-only.
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
