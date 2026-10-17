"""Archive integrity and reversible restoration checks; no model execution."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dnn_restore", ROOT / "backupcuoiki/khoi_phuc_dnn.py")
restore_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore_module)


class TestMidtermLayout(unittest.TestCase):
    def test_all_archived_files_keep_original_content(self):
        manifest = json.loads((ROOT / "backupcuoiki/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["files"]), 60)
        for entry in manifest["files"]:
            archive = ROOT / entry["archive"]
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), entry["sha256"], str(archive))
            if entry["group"] == "dnn":
                restored = ROOT / entry["source"]
                if restored.exists():
                    self.assertEqual(hashlib.sha256(restored.read_bytes()).hexdigest(), entry["sha256"])
        expected_documents = 4 if (ROOT / "demo/dnn_v2.py").exists() else 1
        self.assertEqual(len(list((ROOT / "docs").glob("*.md"))), expected_documents)
        self.assertFalse((ROOT / "demo/gru_presentation.py").exists())
        self.assertTrue((ROOT / "demo/fe/gru_workflow.py").is_file())

    def test_restore_preview_is_non_mutating_and_all_dnn_files_are_available(self):
        plan = restore_module.restore()
        self.assertEqual(len(plan), 0 if (ROOT / "demo/dnn_v2.py").exists() else 39)
        self.assertTrue(all(not target.exists() for _, target in plan))

    def test_restore_copies_exact_bytes_and_refuses_conflicting_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "backupcuoiki/dnn/src/example.py"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"# preserved DNN source\n")
            target = root / "src/example.py"
            record = dict(group="dnn", source="src/example.py", archive=archive.relative_to(root).as_posix(), sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
            (root / "backupcuoiki/manifest.json").write_text(json.dumps(dict(files=[record])), encoding="utf-8")
            self.assertEqual(len(restore_module.restore(root)), 1)
            self.assertFalse(target.exists())
            restore_module.restore(root, apply=True)
            self.assertEqual(archive.read_bytes(), target.read_bytes())
            self.assertEqual(restore_module.restore(root, apply=True), [])
            target.write_bytes(b"# changed independently\n")
            with self.assertRaises(ValueError):
                restore_module.restore(root, apply=True)
            self.assertEqual(target.read_bytes(), b"# changed independently\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
