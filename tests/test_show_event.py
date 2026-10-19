"""Synthetic CLI tests only; no verification of the absent real HAI demo run."""

from contextlib import redirect_stdout, redirect_stderr
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
import show_event as cli

START = np.datetime64("2020-01-01T00:01:11")
END = START + np.timedelta64(97, "s")
FIRST = START + np.timedelta64(18, "s")
HAS_TORCH = importlib.util.find_spec("torch") is not None
BASH = str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe") if os.name == "nt" else shutil.which("bash")


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metadata_fixture(path, *, seed=42, epochs=3, threshold=1.5, row_limit=10000, weights=False, third=False):
    """Synthetic artifact consistency fixture, never presented as HAI results."""
    if weights:
        import torch
        sys.path.insert(0, str(ROOT / "src"))
        from gru_model import OneStepGRU
        model = OneStepGRU()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()  # controlled test weights; no optimization/training
        torch.save(dict(window_size=30, seed=seed, epoch=epochs, model_state_dict=model.state_dict()), path / "gru.pt")
    else:
        (path / "gru.pt").write_bytes(b"metadata-only test placeholder, not model weights")
    checksum = hashlib.sha256((path / "gru.pt").read_bytes()).hexdigest()
    history = [dict(epoch=i, train_loss_mse=round(.3/i,6), val_mean_rmse_s5=.2/i) for i in range(1, epochs+1)]
    write_json(path / "settings.json", dict(seed=seed, row_limit=row_limit, epochs=epochs))
    write_json(path / "training.json", dict(seed=seed, best_epoch=epochs, window_size=30, history=history,
               checkpoint_sha256=checksum, profile="quick" if row_limit else "full", total_epochs=epochs, threshold=threshold))
    detected = threshold < 2
    count = 3 if third else 2
    result = dict(checkpoint_sha256=checksum, threshold=threshold, seed=seed, sessions=["test1"],
                  tp=count if detected else 0, fn=0 if detected else count, fp=0, alarms=count if detected else 0,
                  event_precision=1.0 if detected else 0., event_recall=1.0 if detected else 0.,
                  event_f1=1.0 if detected else 0., normal_fa=0, normal_samples=101 if third else 67,
                  normal_fa_events_per_hour=0., median_delay_sec=(7. if third else 9.) if detected else None)
    write_json(path / "evaluation.json", result)
    write_csv(path / "per_session.csv", [dict(session="test1", **{key:result[key] for key in ("tp","fn","fp","alarms","normal_fa","normal_samples")})])
    intervals = [(1,START-np.timedelta64(36,"s"),START-np.timedelta64(32,"s"),0), (2,START,END,18)]
    if third:
        intervals.append((3,START+np.timedelta64(114,"s"),START+np.timedelta64(139,"s"),7))
    events = [dict(attack_event_id=f"test1_atk_{i:02d}", session="test1", orig_start=str(start).replace("T"," "),
                   orig_end=str(end).replace("T"," "), detected=detected, delay_seconds=delay if detected else "")
              for i,start,end,delay in intervals]
    write_csv(path / "attack_events.csv", events)
    (path / "train.log").write_text("\n".join(f"Epoch {row['epoch']}/{epochs} | Train Loss: {row['train_loss_mse']:.6f} | saved test log" for row in history), encoding="utf-8")
    write_csv(path / "training_history.csv", history)
    if weights:
        source = path / "input"
        source.mkdir()
        rows = 260 if third else 200
        times = np.datetime64("2020-01-01T00:00:00") + np.arange(rows).astype("timedelta64[s]")
        labels = np.zeros(rows, dtype=np.int8)
        labels[35:40] = 1
        labels[71:169] = 1
        features = np.zeros((rows,79), dtype=np.float32)
        features[35:40] = 2
        features[89:130] = 2
        if third:
            labels[185:211] = 1
            features[192:211] = 2
            features[200] = 4  # peak occurs later than the first threshold crossing
        files = {}
        for name, values in dict(features=features, timestamps=times, attack=labels).items():
            target = source / f"test1_{name}.npy"
            np.save(target, values, allow_pickle=False)
            files[name] = str(target)
        write_json(path / "data_manifest.json", dict(test1=dict(rows=rows, used_rows=rows, files=files)))
        np.savez(path / "scores_test1.npz", timestamps=times[30:], scores=np.sqrt(np.mean(features[30:].astype(np.float64)**2,axis=1)).astype(np.float32), attack=labels[30:])


def invoke(*arguments):
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = cli.main(list(arguments))
    return result, stdout.getvalue(), stderr.getvalue()


def snapshot(path):
    return {p.relative_to(path).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob("*") if p.is_file()}


class TestTerminalDemo(unittest.TestCase):
    def test_missing_run_stops_without_any_fallback_or_output_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "missing"
            result, stdout, stderr = invoke("all", "--run", str(target))
            self.assertEqual(result, 2)
            self.assertEqual(stdout, "")
            self.assertIn("settings.json", stderr)
            self.assertFalse(target.exists())
        with patch.dict(os.environ, {}, clear=True):
            result, _, error = invoke("log")
        self.assertEqual(result, 2)
        self.assertIn("Chưa chỉ định run", error)

    def test_threshold_and_result_mismatches_stop_before_printing_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            metadata_fixture(path)
            run = cli.load_run(path)
            cli.load_result(run)
            self.assertEqual(run["result"]["tp"], 2)
            for field, invalid in [("event_f1", .5), ("threshold", 1.058582), ("median_delay_sec", 84), ("checkpoint_sha256", "different"), ("tp", 5), ("seed", 44)]:
                metadata_fixture(path)
                value = cli.read_json(path / "evaluation.json")
                value[field] = invalid
                write_json(path / "evaluation.json", value)
                before = snapshot(path)
                result, stdout, stderr = invoke("result", "--run", str(path))
                self.assertEqual(result, 2, field)
                self.assertEqual(stdout, "", field)
                self.assertTrue(stderr, field)
                self.assertEqual(before, snapshot(path))

    def test_missing_checkpoint_corrupt_hash_and_invalid_log_are_clear(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            metadata_fixture(path)
            (path / "gru.pt").unlink()
            self.assertIn("gru.pt", invoke("log", "--run", str(path))[2])
            metadata_fixture(path)
            (path / "gru.pt").write_bytes(b"changed")
            for options in [(), ("--detail",)]:
                result, output, error = invoke("log", "--run", str(path), *options)
                self.assertEqual(result, 2)
                self.assertEqual(output, "")
                self.assertIn("SHA-256", error)
            metadata_fixture(path)
            (path / "train.log").write_text("not an epoch log", encoding="utf-8")
            self.assertIn("Số dòng epoch", invoke("log", "--run", str(path))[2])
            metadata_fixture(path)
            text = (path / "train.log").read_text().replace("0.300000", "0.999999")
            (path / "train.log").write_text(text)
            self.assertIn("Loss log/CSV", invoke("log", "--run", str(path))[2])

    def test_modes_are_separate_and_other_run_configurations_are_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            metadata_fixture(path, seed=43, epochs=5, row_limit=0, threshold=.75)
            with patch.object(cli, "inspect_event", side_effect=AssertionError("log/result must not infer")), patch.object(cli, "score_window", side_effect=AssertionError("no model load")):
                (path / "evaluation.json").unlink()
                result, output, error = invoke("log", "--run", str(path))
                self.assertEqual(result, 0, error)
                self.assertIn("seed 43", output)
                self.assertIn("5 epoch", output)
                self.assertIn("Mô hình đã được lưu: gru.pt (vòng học 5).", output)
                self.assertNotIn("SHA-256", output)
                result, detailed, error = invoke("log", "--detail", "--run", str(path))
                self.assertEqual(result, 0, error)
                self.assertIn("SHA-256 khớp.", detailed)
                metadata_fixture(path, seed=43, epochs=5, row_limit=0, threshold=.75)
                (path / "train.log").unlink()
                result, output, error = invoke("result", "--run", str(path))
                self.assertEqual(result, 0, error)
                self.assertIn("2/2", output)
                self.assertNotIn("Event F1", output)
                self.assertIn("Event F1", invoke("result", "--detail", "--run", str(path))[1])

    def test_ascii_keeps_one_column_per_second_and_marks_first_alarm(self):
        times = START - np.timedelta64(10,"s") + np.arange(118).astype("timedelta64[s]")
        data = dict(times=times, scores=np.linspace(.1,2.,118), threshold=1.5, first=FIRST, start=START, end=END,
                    inside=(times>=START)&(times<=END))
        stream = io.StringIO()
        with redirect_stdout(stream): cli.show_timeline(data)
        output = stream.getvalue()
        labels = [line.split("|",1)[1] for line in output.splitlines() if line.lstrip().startswith(("Nhãn|","NHAN|"))]
        self.assertEqual(sum(map(len,labels)),118)
        self.assertEqual("".join(labels)[10],"[")
        self.assertEqual("".join(labels)[107],"]")
        self.assertEqual(output.count("!"),2)
        self.assertEqual(len([line for line in output.splitlines() if line.startswith(" tau 1.500|")]),len(labels))

    @unittest.skipUnless(HAS_TORCH, "Requires existing torch environment")
    def test_read_only_inference_matches_existing_core_score_function(self):
        import torch
        sys.path.insert(0,str(ROOT/"src"))
        from gru_model import OneStepGRU
        from evaluate_final_test import score_gru
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"untrained-test-weights.pt"
            model=OneStepGRU().eval()  # no training
            torch.save(dict(window_size=30,model_state_dict=model.state_dict()),path)
            features=np.random.default_rng(49).normal(size=(74,79)).astype(np.float32)
            original=features.copy()
            checksum=cli.digest(path)
            np.testing.assert_array_equal(cli.score_window(path,features),score_gru(features,model,torch.device("cpu")))
            np.testing.assert_array_equal(features,original)
            self.assertEqual(cli.digest(path),checksum)

    @unittest.skipUnless(HAS_TORCH, "Requires existing torch environment")
    def test_event_recomputes_with_saved_weights_and_can_report_a_miss(self):
        for threshold, expected_text in [(1.5,"18 giây"),(3.,"không có"),(.75,"18 giây"),(1.234567,"18 giây"),(2.,"không có"),(1.9999996,"18 giây")]:
            with self.subTest(threshold=threshold), tempfile.TemporaryDirectory() as temporary:
                path=Path(temporary)
                metadata_fixture(path,weights=True,threshold=threshold,
                                 seed=43 if threshold==.75 else 42,
                                 epochs=5 if threshold==.75 else 3,
                                 row_limit=0 if threshold==.75 else 10000)
                before=snapshot(path)
                result,output,error=invoke("event","--run",str(path))
                self.assertEqual(result,0,error)
                self.assertIn(expected_text,output)
                self.assertIn("MINH HỌA 1 TRONG 2 SỰ KIỆN | test1_atk_02",output)
                self.assertIn("Sử dụng mô hình GRU đã lưu.",output)
                self.assertIn("Tấn công bắt đầu: 00:01:11",output)
                self.assertNotIn("2020-01-01",output)
                self.assertIn(f"Ngưỡng Tau: {threshold:.3f}",output)
                self.assertIn("Sai số cao nhất trong 98 giây: 2.000",output)
                if threshold < 2:
                    self.assertIn("GRU cảnh báo đầu tiên: 00:01:29",output)
                    self.assertIn("Kết luận: GRU phát hiện bất thường",output)
                else:
                    self.assertIn("GRU cảnh báo đầu tiên: không có",output)
                    self.assertIn("Thời gian phát hiện: không có",output)
                    self.assertNotIn("Thời gian phát hiện: 0",output)
                    self.assertIn("Kết luận: không có cảnh báo bất thường",output)
                    self.assertNotIn("Kết luận: GRU phát hiện bất thường",output)
                self.assertNotIn("Đỉnh RMSE",output)
                result,detailed,error=invoke("event","--detail","--run",str(path))
                self.assertEqual(result,0,error)
                self.assertIn("Tấn công bắt đầu: 2020-01-01 00:01:11",detailed)
                self.assertIn("Kết thúc theo nhãn HAI: 2020-01-01 00:02:48",detailed)
                self.assertIn("SHA-256 checkpoint khớp",detailed)
                self.assertIn("nhãn chỉ dùng đối chiếu sau suy luận",detailed)
                self.assertIn(f"tau: {threshold:.6f} | Đỉnh RMSE trong vùng nhãn: 2.000000",detailed)
                self.assertEqual(before,snapshot(path))

    def test_event_display_counts_only_its_session_and_handles_midnight(self):
        run = dict(event_id="test2_atk_03", event=dict(session="test2", orig_end="2020-01-02 00:00:02"),
                   events=[dict(session="test1") for _ in range(5)] + [dict(session="test2") for _ in range(3)])
        measured = dict(start=np.datetime64("2020-01-01T23:59:58"), end=np.datetime64("2020-01-02T00:00:02"),
                        first=np.datetime64("2020-01-02T00:00:00"), delay=2., threshold=1.234567, peak=2.345678)
        output = io.StringIO()
        with redirect_stdout(output):
            cli.show_event(run, measured)
        text = output.getvalue()
        self.assertIn("MINH HỌA 1 TRONG 3 SỰ KIỆN | test2_atk_03",text)
        self.assertIn("Tấn công bắt đầu: 23:59:58",text)
        self.assertIn("GRU cảnh báo đầu tiên: 00:00:00",text)
        self.assertIn("Thời gian phát hiện: 2 giây",text)
        self.assertIn("Ngưỡng Tau: 1.235",text)
        self.assertIn("Sai số cao nhất trong 5 giây: 2.346",text)

    @unittest.skipUnless(HAS_TORCH and BASH and Path(BASH).is_file(), "Requires existing torch environment and Bash")
    def test_four_event_commands_use_saved_checkpoint_through_launcher(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("demo", "src"):
                (root / directory).mkdir()
            for name in ("show", "demo/show_event.py", "src/gru_model.py", "src/gru_dataset.py"):
                shutil.copyfile(ROOT / name, root / name)
            interpreter = root / ".venv/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text("#!/usr/bin/env bash\nexec " + shlex.quote(sys.executable.replace("\\", "/")) + " \"$@\"\n", encoding="utf-8")
            interpreter.chmod(0o755)
            run = root / "synthetic run"
            run.mkdir()
            metadata_fixture(run, weights=True, third=True)
            before = snapshot(root)
            env = {**os.environ, "PYTHONUTF8": "1", "DT049_RUN": str(run)}
            cases = [
                (["event"], "02", "00:01:11", "00:01:29", 18, 98, 2.),
                (["event", "--detail"], "02", "2020-01-01 00:01:11", "2020-01-01 00:01:29", 18, 98, 2.),
                (["event", "--event", "test1_atk_01"], "01", "00:00:35", "00:00:35", 0, 5, 2.),
                (["event", "--event", "test1_atk_03"], "03", "00:03:05", "00:03:12", 7, 26, 4.),
            ]
            for arguments, ordinal, start, first, delay, duration, peak in cases:
                with self.subTest(command="./show " + " ".join(arguments)):
                    result = subprocess.run([BASH, str(root / "show"), *arguments], capture_output=True, text=True,
                                            encoding="utf-8", env=env, timeout=60)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertIn(f"MINH HỌA 1 TRONG 3 SỰ KIỆN | test1_atk_{ordinal}",result.stdout)
                    self.assertIn(f"Tấn công bắt đầu: {start}",result.stdout)
                    self.assertIn(f"GRU cảnh báo đầu tiên: {first}",result.stdout)
                    self.assertIn(f"Thời gian phát hiện: {delay} giây",result.stdout)
                    self.assertIn(f"Sai số cao nhất trong {duration} giây: {peak:.3f}",result.stdout)
            self.assertEqual(before,snapshot(root))

    @unittest.skipUnless(HAS_TORCH, "Requires existing torch environment")
    def test_event_missing_and_time_data_score_mismatches_stop(self):
        for failure in ("event","label-time","time-gap","shape","missing-data","saved-score","cached-time","nan-feature","delay","checkpoint-epoch"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                path=Path(temporary)
                metadata_fixture(path,weights=True)
                if failure=="event":
                    result,output,error=invoke("event","--event","test1_atk_99","--run",str(path))
                else:
                    if failure=="label-time":
                        rows=cli.read_csv(path/"attack_events.csv")
                        rows[1]["orig_start"]="2020-01-01 00:01:12"
                        write_csv(path/"attack_events.csv",rows)
                    elif failure=="time-gap":
                        times=np.load(path/"input/test1_timestamps.npy")
                        times[100]+=np.timedelta64(1,"s")
                        np.save(path/"input/test1_timestamps.npy",times)
                    elif failure=="shape": np.save(path/"input/test1_features.npy",np.zeros((200,78),np.float32))
                    elif failure=="missing-data": (path/"input/test1_features.npy").unlink()
                    elif failure=="saved-score":
                        with np.load(path/"scores_test1.npz") as saved: values={k:saved[k].copy() for k in saved.files}
                        values["scores"][80]+=5
                        np.savez(path/"scores_test1.npz",**values)
                    elif failure=="cached-time":
                        with np.load(path/"scores_test1.npz") as saved: values={k:saved[k].copy() for k in saved.files}
                        values["timestamps"][80]+=np.timedelta64(1,"s")
                        np.savez(path/"scores_test1.npz",**values)
                    elif failure=="nan-feature":
                        features=np.load(path/"input/test1_features.npy")
                        features[80,0]=np.nan
                        np.save(path/"input/test1_features.npy",features)
                    elif failure=="delay":
                        rows=cli.read_csv(path/"attack_events.csv")
                        rows[1]["delay_seconds"]="16"
                        write_csv(path/"attack_events.csv",rows)
                        value=cli.read_json(path/"evaluation.json")
                        value["median_delay_sec"]=8
                        write_json(path/"evaluation.json",value)
                    else:
                        value=cli.read_json(path/"training.json")
                        value["best_epoch"]=1
                        write_json(path/"training.json",value)
                    result,output,error=invoke("event","--run",str(path))
                self.assertEqual(result,2,error)
                self.assertEqual(output,"")
                self.assertTrue(error)

    @unittest.skipUnless(BASH and Path(BASH).is_file(), "Requires working Bash")
    def test_launcher_forwards_run_detail_and_does_not_choose_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"demo").mkdir()
            shutil.copyfile(ROOT/"demo/show_event.py",root/"demo/show_event.py")
            shutil.copyfile(ROOT/"show",root/"show")
            interpreter=root/".venv/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text("#!/usr/bin/env bash\nexec "+shlex.quote(sys.executable.replace("\\","/"))+" \"$@\"\n",encoding="utf-8")
            interpreter.chmod(0o755)
            run=root/"other run"
            run.mkdir()
            metadata_fixture(run,seed=43,epochs=5,threshold=.75,row_limit=0)
            args=[BASH,str(root/"show"),"log","--run",str(run),"--detail"]
            env={**os.environ,"PYTHONUTF8":"1","DT049_RUN":str(root/"missing")}
            result=subprocess.run(args,capture_output=True,text=True,encoding="utf-8",env=env)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("seed 43",result.stdout)
            self.assertIn("tau: 0.750000",result.stdout)
            missing=subprocess.run([BASH,str(root/"show"),"log"],capture_output=True,text=True,encoding="utf-8",env=env)
            self.assertEqual(missing.returncode,2)
            self.assertIn("missing",missing.stderr)
            env["DT049_RUN"]=str(run)
            selected=subprocess.run([BASH,str(root/"show"),"result"],capture_output=True,text=True,encoding="utf-8",env=env)
            self.assertEqual(selected.returncode,0,selected.stderr)
            self.assertIn("2/2",selected.stdout)


if __name__=="__main__":
    unittest.main(verbosity=2)
