import { useEffect, useState } from "react";
import { api } from "./api";
import type { ListItem } from "./types";
import { Library } from "./components/Library";
import { Coverage } from "./components/Coverage";
import { Organize } from "./components/Organize";

type Tab = "library" | "coverage" | "organize";

export default function App() {
  const [tab, setTab] = useState<Tab>("library");
  const [items, setItems] = useState<ListItem[]>([]);
  const [health, setHealth] = useState<"ok" | "down" | "unknown">("unknown");
  const [error, setError] = useState<string | null>(null);

  async function reload() {
    try {
      const data = await api.items();
      setItems(data);
      setHealth("ok");
      setError(null);
    } catch (e) {
      setHealth("down");
      setError(String(e));
    }
  }

  useEffect(() => {
    reload();
  }, []);

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">
          GADS <span>Knowledge Studio</span>
        </div>
        <div className="tabs">
          <button className={`tab ${tab === "library" ? "active" : ""}`} onClick={() => setTab("library")}>
            Library
          </button>
          <button className={`tab ${tab === "coverage" ? "active" : ""}`} onClick={() => setTab("coverage")}>
            Coverage
          </button>
          <button className={`tab ${tab === "organize" ? "active" : ""}`} onClick={() => setTab("organize")}>
            Organize
          </button>
        </div>
        <div className="spacer" />
        <div className="status">
          <span className={`dot ${health === "ok" ? "ok" : health === "down" ? "down" : ""}`} />
          {health === "ok" ? `${items.length} items` : health === "down" ? "backend offline" : "…"}
        </div>
      </div>

      {error && health === "down" && (
        <div className="error-banner">
          Cannot reach the GADS backend. Start it with <code>./start_backend.sh</code> (expected on :8001). — {error}
        </div>
      )}

      <div className="body">
        {tab === "library" && <Library items={items} onSaved={reload} />}
        {tab === "coverage" && <Coverage />}
        {tab === "organize" && <Organize items={items} onChanged={reload} />}
      </div>
    </div>
  );
}
