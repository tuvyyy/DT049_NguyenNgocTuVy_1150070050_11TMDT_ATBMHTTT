"""Synthetic CLI tests only; these do not verify the user's absent Kali run."""

from contextlib import redirect_stdout, redirect_stderr
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
import show_event as cli


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def metadata_fixture(path):
    """Reference metadata for guard tests, not a trained checkpoint or real output."""
    (path / "gru.pt").write_bytes(b"metadata-only unit-test placeholder; not model weights")
    digest = hashlib.sha256((path / "gru.pt").read_bytes()).hexdigest()
    write_json(path / "settings.json", dict(seed=42, row_limit=10000, epochs=3))
    write_json(path / "training.json", dict(checkpoint_sha256=digest, profile="quick", total_epochs=3, threshold=1.605896))
    write_json(path / "evaluation.json", dict(checkpoint_sha256=digest, threshold=1.605896, seed=42, sessions=["test1"],
               tp=5, fn=0, event_f1=10/69, normal_fa_events_per_hour=4.9081, median_delay_sec=18))
    with (path / "attack_events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attack_event_id", "session", "orig_start", "orig_end", "detected", "delay_seconds"])
        writer.writeheader()
        writer.writerow(dict(attack_event_id=cli.EVENT, session="test1", orig_start=cli.START, orig_end=cli.END, detected=True, delay_seconds=18))
    (path / "train.log").write_text("\n".join(f"Epoch {epoch}/3 | Train Loss: {loss:.6f} | saved log" for epoch, loss in [(1,.3), (2,.2), (3,.1)]), encoding="utf-8")
    with (path / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss_mse"])
        writer.writeheader()
        writer.writerows([dict(epoch=i, train_loss_mse=v) for i,v in [(1,.3),(2,.2),(3,.1)]])


class TestTerminalDemo(unittest.TestCase):
    def test_missing_run_stops_without_any_fallback_or_output_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "missing"
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = cli.main(["all", "--run", str(target)])
            self.assertEqual(code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("settings.json", stderr.getvalue())
            self.assertFalse(target.exists())

    def test_threshold_and_result_mismatches_stop_before_printing_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            metadata_fixture(path)
            self.assertEqual(cli.load_run(path)["result"]["tp"], 5)
            for field, invalid in [("event_f1", .5), ("threshold", 1.058582), ("median_delay_sec", 84), ("checkpoint_sha256", "different")]:
                metadata_fixture(path)
                value = json.loads((path / "evaluation.json").read_text(encoding="utf-8"))
                value[field] = invalid
                write_json(path / "evaluation.json", value)
                before = {p.name: p.read_bytes() for p in path.iterdir()}
                stdout, stderr = io.StringIO(), io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    result = cli.main(["result", "--run", str(path)])
                self.assertEqual(result, 2, field)
                self.assertEqual(stdout.getvalue(), "", field)
                self.assertTrue(stderr.getvalue(), field)
                self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})

    def test_ascii_keeps_one_column_per_second_and_marks_first_alarm(self):
        times = np.datetime64(cli.START) - np.timedelta64(10, "s") + np.arange(118).astype("timedelta64[s]")
        scores = np.linspace(.1, 2., 118)
        data = dict(times=times, scores=scores, threshold=1.5, first=np.datetime64(cli.FIRST),
                    inside=(times >= np.datetime64(cli.START)) & (times <= np.datetime64(cli.END)))
        stream = io.StringIO()
        with redirect_stdout(stream):
            cli.show_timeline(data)
        output = stream.getvalue()
        labels = [line.split("|", 1)[1] for line in output.splitlines() if line.startswith("      NHAN|")]
        self.assertEqual(sum(map(len, labels)), 118)
        self.assertEqual("".join(labels)[10], "[")
        self.assertEqual("".join(labels)[107], "]")
        self.assertEqual(output.count("!"), 2)  # legend and exactly one data point
        self.assertEqual(len([line for line in output.splitlines() if line.startswith(" tau 1.500|")]), len(labels))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Requires existing torch environment")
    def test_read_only_inference_matches_existing_core_score_function(self):
        import torch
        sys.path.insert(0, str(ROOT / "src"))
        from gru_model import OneStepGRU
        from evaluate_final_test import score_gru
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "untrained-test-weights.pt"
            model = OneStepGRU().eval()  # initialization only, no training
            torch.save(dict(window_size=30, model_state_dict=model.state_dict()), path)
            features = np.random.default_rng(49).normal(size=(74,79)).astype(np.float32)
            original = features.copy()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            actual = cli.score_window(path, features)
            expected = score_gru(features, model, torch.device("cpu"))
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(features, original)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
