"""Streamlit launcher. Frontend: fe/. Backend: be/."""

from pathlib import Path
import sys

DEMO_ROOT = Path(__file__).resolve().parent
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from fe.app import run_app

run_app()
