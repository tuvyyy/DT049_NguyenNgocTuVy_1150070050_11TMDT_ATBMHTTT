"""Real-job orchestration and UI regression tests. Synthetic data is test-only."""

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"
sys.path.insert(0, str(DEMO))
from be import gru_runtime as runtime
from fe.gru_workflow import STEPS


def fixture(source, count=96):
    rng = np.random.default_rng(49)
    for session in ["train1", "train2", "train3"] + runtime.SESSIONS:
        files = runtime._array_paths(source, session)
        files["features"].parent.mkdir(parents=True, exist_ok=True)
        features = rng.normal(0, .01, (count, 79)).astype(np.float32)
        times = np.datetime64("2020-01-01T00:00:00", "s") + np.arange(count).astype("timedelta64[s]")
        np.save(files["timestamps"], times)
        if session.startswith("test"):
            labels = np.zeros(count, dtype=np.int8)
            labels[35:40] = 1
            labels[65:70] = 1
            features[labels == 1] += 10
            for key in ("attack", "attack_P1", "attack_P2", "attack_P3"):
                np.save(files[key], labels)
        np.save(files["features"], features)


def settings(source):
    return dict(source_kind="processed", source=str(source), row_limit=10000, seed=42, epochs=1, learning_rate=.001, purpose="synthetic integration test")


def wait_job(path, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = runtime.read_state(path)
        if state["status"] != "running":
            if state["status"] != "done":
                log = path / f'{state["stage"]}.log'
                raise AssertionError(log.read_text(encoding="utf-8") if log.exists() else state)
            return state
        time.sleep(.1)
    runtime.stop(path)
    raise AssertionError("Job timed out")


class TestGRUExecutionUI(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(DEMO / "app.py"), default_timeout=30).run()

    def test_four_operation_steps_no_lecture_and_prerequisites(self):
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.radio(key="overview_mode").value, "Thực hiện GRU")
        self.assertEqual(self.app.radio(key="gru_step").options, STEPS)
        self.assertTrue(self.app.button(key="gru_back").disabled)
        self.assertTrue(self.app.button(key="gru_next").disabled)
        for step in STEPS[1:]:
            self.app.radio(key="gru_step").set_value(step).run()
            self.assertFalse(self.app.exception)
            self.assertTrue(self.app.info)
        self.assertEqual(len(self.app.get("popover")), 0)
        self.assertFalse(any("40/50" in element.value for element in self.app.markdown))

    def test_missing_source_is_reported_without_launch(self):
        self.app.text_input(key="gru_source").set_value("/this-folder-does-not-exist")
        with patch("be.gru_runtime.launch") as launch:
            next(button for button in self.app.button if button.label == "Chuẩn bị dữ liệu").click().run()
            self.assertFalse(self.app.exception)
            self.assertTrue(self.app.error)
            launch.assert_not_called()

    def test_other_gru_views_work_and_dnn_is_archived(self):
        self.app.radio(key="overview_mode").set_value("Tổng quan nghiên cứu").run()
        self.assertFalse(self.app.exception)
        if (DEMO / "dnn_v2.py").exists():
            self.assertIn("nav_dnn", [button.key for button in self.app.button])
        else:
            self.assertNotIn("nav_dnn", [button.key for button in self.app.button])
        for key in ("nav_models", "nav_attacks", "nav_errors", "nav_threshold", "nav_ablation", "nav_reproducibility"):
            self.app.button(key=key).click().run()
            self.assertFalse(self.app.exception, key)


class TestGRURuntime(unittest.TestCase):
    def test_paths_and_settings_reject_official_output_locations(self):
        for path in (ROOT, ROOT / "models/primary", ROOT / "results/metrics"):
            with self.assertRaises(ValueError):
                runtime.run_path(path)
        with tempfile.TemporaryDirectory() as temporary:
            value = settings(Path(temporary))
            for key, invalid in (("seed", 0), ("epochs", 0), ("row_limit", -1), ("learning_rate", 1.0)):
                with self.assertRaises(ValueError):
                    runtime.validate_settings({**value, key: invalid})

    def test_launch_is_allowlisted_isolated_and_does_not_use_shell(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            with patch.object(runtime, "RUNS", source / "runs"):
                path = runtime.create_run(settings(source))
                with self.assertRaises(ValueError):
                    runtime.launch(path, "evaluate", ["test1"])
                with self.assertRaises(ValueError):
                    runtime.launch(path, "arbitrary-command")
                with patch("be.gru_runtime.subprocess.Popen") as process:
                    process.return_value.pid = 123
                    runtime.launch(path, "data")
                    args, kwargs = process.call_args
                    self.assertEqual(args[0][0], sys.executable)
                    self.assertEqual(args[0][-2:], [str(path), "data"])
                    self.assertFalse(kwargs.get("shell", False))
                    self.assertEqual(runtime.read_json(path / "state.json")["status"], "running")
                    with self.assertRaises(ValueError):
                        runtime.launch(path, "train")

    def test_prepare_checks_actual_arrays_and_does_not_shorten_test(self):
        import logging
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            fixture(source, count=10040)
            with patch.object(runtime, "RUNS", Path(temporary) / "runs"):
                path = runtime.create_run(settings(source))
                runtime.prepare(path, settings(source), logging.getLogger("test"))
                data = runtime.read_json(path / "data_manifest.json")
                self.assertEqual(data["train1"]["used_rows"], 10000)
                self.assertEqual(data["test1"]["rows"], 10040)
                self.assertEqual(data["test1"]["used_rows"], 10040)
                invalid = np.load(source / "train1_features.npy")
                invalid[0, 0] = np.nan
                np.save(source / "train1_features.npy", invalid)
                with self.assertRaises(ValueError):
                    runtime.prepare(path, settings(source), logging.getLogger("test"))

    def test_worker_failure_is_visible_and_does_not_mark_data_complete(self):
        from contextlib import redirect_stderr
        import io
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            with patch.object(runtime, "RUNS", source / "runs"):
                path = runtime.create_run(settings(source))
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(runtime.work(path, "data"), 1)
                state = runtime.read_state(path)
                self.assertEqual(state["status"], "failed")
                self.assertEqual(state["completed"], [])
                self.assertIn("FileNotFoundError", state["error"])

    def test_stop_verifies_job_identity_before_terminating(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            with patch.object(runtime, "RUNS", source / "runs"):
                path = runtime.create_run(settings(source))
                runtime._write(path / "state.json", dict(status="running", pid=123, stage="train", completed=["data"]))
                with patch("psutil.pid_exists", return_value=True), patch("psutil.Process") as process:
                    process.return_value.cmdline.return_value = ["unrelated-process"]
                    with self.assertRaises(ValueError):
                        runtime.stop(path)
                    process.return_value.terminate.assert_not_called()
                    process.return_value.cmdline.return_value = [sys.executable, str(Path(runtime.__file__).resolve()), str(path), "train"]
                    runtime.stop(path)
                    process.return_value.terminate.assert_called_once()
                    self.assertEqual(runtime.read_state(path)["status"], "failed")


@unittest.skipIf(runtime.missing_dependencies(), "Requires ML environment (torch/matplotlib/sklearn/joblib)")
class TestRealGRUJob(unittest.TestCase):
    def test_csv_preparation_uses_frozen_scaler_without_modifying_it(self):
        import logging
        scaler = ROOT / "models/preprocessing/scaler.joblib"
        before = hashlib.sha256(scaler.read_bytes()).hexdigest()
        columns = runtime.read_json(ROOT / "models/preprocessing/scaler_metadata.json")["ordered_feature_names"]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "csv"
            source.mkdir()
            for session in ["train1", "train2", "train3"] + runtime.SESSIONS:
                frame = pd.DataFrame(np.zeros((64, 79)), columns=columns)
                frame["time"] = pd.date_range("2020-01-01", periods=64, freq="s")
                for key in ("attack", "attack_P1", "attack_P2", "attack_P3"):
                    frame[key] = 0
                frame.to_csv(source / f"{session}.csv.gz", index=False)
            with patch.object(runtime, "RUNS", Path(temporary) / "runs"):
                config = {**settings(source), "source_kind": "csv"}
                path = runtime.create_run(config)
                runtime.prepare(path, config, logging.getLogger("test"))
                manifest = runtime.read_json(path / "data_manifest.json")
                for item in manifest.values():
                    self.assertTrue(Path(item["files"]["features"]).is_relative_to(path))
                    self.assertEqual(item["rows"], 64)
        self.assertEqual(hashlib.sha256(scaler.read_bytes()).hexdigest(), before)

    def test_prepare_train_evaluate_and_ui_use_new_run_not_official_results(self):
        protected = [ROOT / "models/primary/gru_w30_seed42.pt", ROOT / "results/metrics/final_test_summary.csv"]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
        path = None
        try:
            with tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "source"
                fixture(source)
                app = AppTest.from_file(str(DEMO / "app.py"), default_timeout=30).run()
                app.text_input(key="gru_source").set_value(str(source))
                next(button for button in app.button if button.label == "Chuẩn bị dữ liệu").click().run()
                self.assertFalse(app.exception)
                path = runtime.run_path(app.session_state["gru_live_run"])
                self.assertIn("data", wait_job(path)["completed"])
                app.run()
                app.button(key="gru_next").click().run()
                app.number_input(key="gru_epochs").set_value(1)
                next(button for button in app.button if button.label == "Huấn luyện GRU").click().run()
                self.assertFalse(app.exception)
                self.assertIn("train", wait_job(path)["completed"])
                app.run()
                app.button(key="gru_next").click().run()
                app.button(key="gru_evaluate").click().run()
                self.assertFalse(app.exception)
                self.assertIn("evaluate", wait_job(path)["completed"])
                app.run()
                result = runtime.read_json(path / "evaluation.json")
                self.assertEqual(result["tp"] + result["fn"], 2)
                self.assertEqual(result["sessions"], ["test1"])
                self.assertEqual(result["checkpoint_sha256"], hashlib.sha256((path / "gru.pt").read_bytes()).hexdigest())
                history = pd.read_csv(path / "training_history.csv")
                self.assertEqual(len(history), 1)
                self.assertIn("Epoch", (path / "train.log").read_text(encoding="utf-8"))
                with np.load(path / "scores_test1.npz") as scores:
                    self.assertEqual(len(scores["scores"]), 66)
                for step in STEPS:
                    app.radio(key="gru_step").set_value(step).run()
                    self.assertFalse(app.exception)
                self.assertEqual(len(app.selectbox(key="gru_event").options), 2)
                self.assertFalse(any("40/50" in item.value for item in app.markdown))
                app.radio(key="gru_step").set_value(STEPS[2]).run()
                self.assertTrue(any(f'{result["tp"]}/2' in item.value for item in app.markdown))
                app.button(key="gru_new").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.radio(key="gru_step").value, STEPS[0])
                self.assertTrue(app.button(key="gru_next").disabled)
        finally:
            if path is not None:
                # Verify the exact recursive-delete target remains the test's run.
                assert path.resolve().parent == runtime.RUNS.resolve()
                assert runtime.read_json(path / "settings.json")["source"] == str(source.resolve())
                shutil.rmtree(path)
            self.assertEqual(before, {file: hashlib.sha256(file.read_bytes()).hexdigest() for file in protected})


if __name__ == "__main__":
    unittest.main(verbosity=2)
