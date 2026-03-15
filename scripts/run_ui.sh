#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Prefer Python environments where streamlit is installed.
PY_CANDIDATES=(
  "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
  "/usr/local/bin/python3"
  "$(command -v python3)"
)

for PY in "${PY_CANDIDATES[@]}"; do
  if [[ -n "${PY}" ]] && [[ -x "${PY}" ]]; then
    if "${PY}" -c "import streamlit" >/dev/null 2>&1; then
      echo "Using Python: ${PY}"
      exec "${PY}" -m streamlit run src/epoch_ui/app.py "$@"
    fi
  fi
done

echo "Error: streamlit is not importable in available python3 interpreters." >&2
echo "Install with: python3 -m pip install streamlit plotly" >&2
exit 1
