#!/usr/bin/env bash
# ==============================================================================
# DT049 — Local Streamlit Product Dashboard Launcher
# ==============================================================================
# Resolves repository root safely, verifies virtual environment and app,
# and launches the dashboard strictly bound to local loopback (127.0.0.1:8501).
# ==============================================================================

set -euo pipefail

# 1. Resolve repository root safely
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# 2. Check virtual environment
VENV_DIR="${REPO_ROOT}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: Virtual environment not found at ${VENV_DIR}" >&2
    exit 1
fi

# Activate virtual environment
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# 3. Verify Streamlit binary and demo app
if ! command -v streamlit &>/dev/null; then
    echo "ERROR: streamlit binary not found in activated .venv" >&2
    exit 1
fi

APP_PATH="${REPO_ROOT}/demo/app.py"
if [[ ! -f "${APP_PATH}" ]]; then
    echo "ERROR: Dashboard script not found at ${APP_PATH}" >&2
    exit 1
fi

# 4. Display startup banner
echo "=================================================="
echo "DT049 — Multivariate Anomaly Detection on HAI 21.03"
echo "=================================================="
echo "Repository Root : ${REPO_ROOT}"
echo "Python Runtime  : $(python --version)"
echo "Streamlit Version: $(streamlit --version)"
echo ""
echo "DT049 dashboard:"
echo "http://127.0.0.1:8501"
echo "=================================================="
echo "Launching local Streamlit server on 127.0.0.1:8501..."
echo "Press Ctrl+C to terminate."
echo ""

# 5. Launch strictly on 127.0.0.1:8501 without opening external browsers
exec streamlit run "${APP_PATH}" \
    --server.address 127.0.0.1 \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false
