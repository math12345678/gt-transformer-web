#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
VENV_PY="${ROOT}/.venv/bin/python"
PORT="${PORT:-8050}"

echo "Starting gt-transformer live inference web interface on http://127.0.0.1:${PORT} ..."
cd "${HERE}"
exec "${VENV_PY}" -m uvicorn app:app --host 127.0.0.1 --port "${PORT}"
