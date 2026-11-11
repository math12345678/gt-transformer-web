#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

PYTHON_BIN=""

if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
    PYTHON_BIN="${PYTHON}"
elif [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="${PYTHON_BIN}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${HERE}/.venv/bin/python" ]]; then
    PYTHON_BIN="${HERE}/.venv/bin/python"
elif [[ -x "${HERE}/venv/bin/python" ]]; then
    PYTHON_BIN="${HERE}/venv/bin/python"
elif [[ -x "${HERE}/../.venv/bin/python" ]]; then
    PYTHON_BIN="${HERE}/../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "Error: Could not locate a Python interpreter." >&2
    echo "Please create a virtual environment or install dependencies:" >&2
    echo "  python3 -m venv .venv" >&2
    echo "  source .venv/bin/activate" >&2
    echo "  pip install -r requirements.txt" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c "import uvicorn, fastapi" >/dev/null 2>&1; then
    echo "Error: Python interpreter '${PYTHON_BIN}' is missing required packages (uvicorn, fastapi)." >&2
    echo "Please install dependencies via:" >&2
    echo "  ${PYTHON_BIN} -m pip install -r requirements.txt" >&2
    exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8050}"

echo "Starting gt-transformer live inference web interface on http://${HOST}:${PORT} ..."
echo "Using Python: ${PYTHON_BIN}"
exec "${PYTHON_BIN}" -m uvicorn app:app --host "${HOST}" --port "${PORT}" "$@"
