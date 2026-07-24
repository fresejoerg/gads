import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The Studio talks to the GADS backend Knowledge API. In dev we proxy the API
// paths to the backend (:8001) so the browser makes same-origin calls; override
// the target with GADS_API_URL if the backend runs elsewhere.
const API = process.env.GADS_API_URL || "http://localhost:8001";
const API_PREFIXES = ["/knowledge", "/recipes", "/skills", "/native", "/health"];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      API_PREFIXES.map((p) => [p, { target: API, changeOrigin: true }])
    ),
  },
});
