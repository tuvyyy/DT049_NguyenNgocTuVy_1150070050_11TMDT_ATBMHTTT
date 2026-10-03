#!/usr/bin/env bash
# ==============================================================================
# DT049 — Local Streamlit Product Dashboard Stop Script
# ==============================================================================
# Safely terminates ONLY the DT049 Streamlit process running demo/app.py.
# Does not terminate unrelated Python or Streamlit processes.
# ==============================================================================

set -euo pipefail

# Find PIDs matching streamlit running demo/app.py
PIDS=$(pgrep -f "streamlit.*demo/app\.py" || true)

if [[ -z "${PIDS}" ]]; then
    echo "No active DT049 dashboard process found running demo/app.py on 127.0.0.1:8501."
    exit 0
fi

echo "Found active DT049 dashboard process(es): ${PIDS}"
for PID in ${PIDS}; do
    echo "Terminating PID ${PID}..."
    kill -15 "${PID}" 2>/dev/null || true
done

# Wait up to 3 seconds for graceful shutdown
sleep 2

# Verify termination, force kill if still running
STILL_RUNNING=$(pgrep -f "streamlit.*demo/app\.py" || true)
if [[ -n "${STILL_RUNNING}" ]]; then
    echo "Force terminating remaining PID(s): ${STILL_RUNNING}..."
    kill -9 ${STILL_RUNNING} 2>/dev/null || true
fi

echo "DT049 dashboard successfully stopped."
