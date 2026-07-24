#!/usr/bin/env bash
# Launch the GADS Knowledge Studio SPA (Vite dev server on :5173).
# It proxies the Knowledge API to the backend (:8001) — start ./start_backend.sh first.
set -euo pipefail
cd "$(dirname "$0")/../knowledge-studio"

if [ ! -d node_modules ]; then
  echo "[studio] installing dependencies…"
  npm install
fi

# Point the API proxy elsewhere with: GADS_API_URL=http://host:port ./scripts/run_studio.sh
exec npm run dev
