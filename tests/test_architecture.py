"""Keep model execution out of FE and Streamlit out of backend services."""

import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
from be import artifacts


def imports(path):
    result = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module.split(".")[0])
    return result


class TestArchitecture(unittest.TestCase):
    def test_backend_and_frontend_have_separate_dependencies(self):
        for file in (ROOT / "demo/be").glob("*.py"):
            self.assertNotIn("streamlit", imports(file), str(file))
        for file in (ROOT / "demo/fe").rglob("*.py"):
            self.assertFalse(imports(file) & {"torch", "sklearn"}, str(file))
        self.assertNotIn("streamlit", imports(ROOT / "demo/app.py"))

    def test_artifact_paths_still_use_project_root(self):
        self.assertEqual(artifacts.REPO_ROOT, ROOT)
        self.assertTrue(artifacts.check_artifact_health()["is_healthy"])
        table = artifacts.read_csv("results/metrics/final_test_summary.csv")
        self.assertEqual(int(table.set_index("model_id").loc["gru_seed42", "tp_detected"]), 40)
        with self.assertRaises(ValueError):
            artifacts.metric_download("results/metrics/../../README.md")


if __name__ == "__main__":
    unittest.main(verbosity=2)
