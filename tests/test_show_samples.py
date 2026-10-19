"""Samples routing/validation tests; no HAI data or predictions are fabricated here."""

from contextlib import redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch, sentinel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
import show_event as cli


def invoke(*arguments):
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = cli.main(list(arguments))
    return result, stdout.getvalue(), stderr.getvalue()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class TestSamplesValidation(unittest.TestCase):
    def test_samples_mode_forwards_run_detail_without_loading_logs_or_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with patch.object(cli, "load_run", return_value=sentinel.run) as load, \
                 patch.object(cli, "inspect_samples", return_value=sentinel.measured) as inspect, \
                 patch.object(cli, "show_samples") as show, \
                 patch.object(cli, "load_log", side_effect=AssertionError("samples must not load log")), \
                 patch.object(cli, "load_result", side_effect=AssertionError("samples must not load evaluation")), \
                 patch.object(cli, "inspect_event", side_effect=AssertionError("samples must not load attack labels")):
                result, output, error = invoke("samples", "--run", str(path), "--detail")
            self.assertEqual(result, 0, error)
            self.assertEqual(output, "")  # display was intercepted, no predicted values simulated
            load.assert_called_once_with(path)
            inspect.assert_called_once_with(sentinel.run)
            show.assert_called_once_with(sentinel.run, sentinel.measured, True)

    def test_missing_run_and_checkpoint_stop_without_any_prediction_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            result, output, error = invoke("samples", "--run", str(path / "missing"))
            self.assertEqual(result, 2)
            self.assertEqual(output, "")
            self.assertIn("settings.json", error)
            write_json(path / "settings.json", {})
            write_json(path / "training.json", {})
            result, output, error = invoke("samples", "--run", str(path))
            self.assertEqual(result, 2)
            self.assertEqual(output, "")
            self.assertIn("gru.pt", error)

    def test_samples_checks_checkpoint_hash_before_reading_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            # Copy an actual tracked checkpoint without modifying any of its bytes.
            source = ROOT / "models/primary/gru_w30_seed42.pt"
            before = cli.digest(source)
            shutil.copyfile(source, path / "gru.pt")
            write_json(path / "settings.json", {})
            write_json(path / "training.json", {"checkpoint_sha256": "mismatching metadata"})
            with patch.object(cli, "inspect_samples", side_effect=AssertionError("must stop before data")):
                result, output, error = invoke("samples", "--run", str(path))
            self.assertEqual(result, 2)
            self.assertEqual(output, "")
            self.assertIn("SHA-256", error)
            self.assertEqual(cli.digest(source), before)
            self.assertEqual(cli.digest(path / "gru.pt"), before)

    def test_missing_manifest_invalid_limits_and_missing_arrays_never_infer(self):
        cases = [
            (None, 0, "data_manifest.json"),
            ({}, 0, "train1"),
            ({"rows": 40, "used_rows": 40, "files": {}}, 0, "ít nhất 41 dòng"),
            ({"rows": 100, "used_rows": 50, "files": {}}, 0, "manifest/cấu hình"),
            ({"rows": 20000, "used_rows": 20000, "files": {}}, 10000, "manifest/cấu hình"),
            ({"rows": 100, "used_rows": 100, "files": {}}, 0, "thiếu đường dẫn features"),
            ({"rows": 100, "used_rows": 100, "files": {"features": "missing_train1_features.npy"}}, 0, "missing_train1_features.npy"),
        ]
        for item, limit, expected in cases:
            with self.subTest(error=expected), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)
                if item is not None:
                    write_json(path / "data_manifest.json", {"train1": item} if item else {})
                context = dict(path=path, trained=dict(window_size=30), settings=dict(row_limit=limit))
                with patch.object(cli, "load_run", return_value=context), \
                     patch("torch.load", side_effect=AssertionError("invalid input must not load model")):
                    result, output, error = invoke("samples", "--run", str(path))
                self.assertEqual(result, 2, error)
                self.assertEqual(output, "")
                self.assertIn(expected, error)

    def test_metadata_must_identify_hai_and_all_ordered_feature_names(self):
        metadata = cli.read_json(ROOT / "models/preprocessing/scaler_metadata.json")
        for key, invalid in [("dataset_version", "wrong"), ("feature_count", 78), ("output_dtype", "float64"),
                             ("ordered_feature_names", []), ("ordered_feature_names", ["duplicate"] * 79)]:
            with self.subTest(field=key):
                value = copy.deepcopy(metadata)
                value[key] = invalid
                with patch.object(cli, "read_json", return_value=value), \
                     patch("torch.load", side_effect=AssertionError("invalid schema must not load model")):
                    with self.assertRaises(cli.DemoError):
                        cli.inspect_samples({})

    def test_training_stride_is_read_from_the_repository_config(self):
        import yaml
        config = yaml.safe_load((ROOT / "configs/primary_gru.yaml").read_text(encoding="utf-8"))
        config["data"]["development_training"]["stride"] = 10
        with patch("yaml.safe_load", return_value=config):
            with self.assertRaisesRegex(cli.DemoError, "Bước nhảy huấn luyện hiện tại"):
                cli.inspect_samples({})


if __name__ == "__main__":
    unittest.main(verbosity=2)
