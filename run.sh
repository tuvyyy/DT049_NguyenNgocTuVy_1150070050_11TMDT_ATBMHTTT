#!/usr/bin/env bash
# DT049 Quick Launcher
set -e
cd "$(dirname "$0")"
./demo/stop_dashboard.sh >/dev/null 2>&1 || true
exec .venv/bin/streamlit run demo/app.py --server.address 0.0.0.0 --server.port 8501
