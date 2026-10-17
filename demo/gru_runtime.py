"""Isolated, real GRU jobs for the local Streamlit UI.

The existing model, training loop and event-matching functions are reused.
Only runs/gru/<id> is writable. Official models/results are never outputs.
"""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
import traceback
import uuid

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs/gru"
STAGES = ("data", "train", "evaluate")
SESSIONS = [f"test{i}" for i in range(1, 6)]


def _write(path, value):
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def run_path(value):
    path = Path(value).resolve()
    if path.parent != RUNS.resolve() or not path.is_dir():
        raise ValueError("Thư mục lần chạy phải nằm trực tiếp trong runs/gru/.")
    return path


def missing_dependencies():
    return [name for name in ("torch", "numpy", "pandas", "yaml", "psutil", "matplotlib", "joblib", "sklearn")
            if importlib.util.find_spec(name) is None]


def validate_settings(settings):
    if settings["source_kind"] not in ("processed", "csv"):
        raise ValueError("Kiểu dữ liệu không hợp lệ.")
    if not Path(settings["source"]).is_dir():
        raise ValueError("Không tìm thấy thư mục dữ liệu trên máy đang chạy app.")
    if settings["seed"] not in (42, 43, 44) or not 1 <= settings["epochs"] <= 50:
        raise ValueError("Seed hoặc số epoch không hợp lệ.")
    if settings["row_limit"] not in (0, 10000) or not 0.00001 <= settings["learning_rate"] <= 0.01:
        raise ValueError("Thiết lập huấn luyện không hợp lệ.")


def create_run(settings):
    settings = dict(settings)
    settings["source"] = str(Path(settings["source"]).resolve())
    validate_settings(settings)
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    path.mkdir()
    _write(path / "settings.json", settings)
    _write(path / "state.json", {"stage": "data", "status": "ready", "completed": [], "pid": None})
    return path


def read_state(path):
    path = run_path(path)
    state = read_json(path / "state.json", {})
    if state.get("status") == "running" and state.get("pid"):
        import psutil
        if not psutil.pid_exists(state["pid"]):
            state.update(status="failed", error="Tiến trình đã kết thúc trước khi lưu trạng thái hoàn tất. Xem log để biết chi tiết.")
            state["pid"] = None
            _write(path / "state.json", state)
            (path / "active.lock").unlink(missing_ok=True)
    return state


def stop(path):
    import psutil
    path = run_path(path)
    state = read_state(path)
    if state.get("status") != "running" or not state.get("pid"):
        raise ValueError("Tiến trình chưa khởi động hoặc đã kết thúc; hãy cập nhật trạng thái.")
    process = psutil.Process(state["pid"])
    command = process.cmdline()
    if str(path) not in command or str(Path(__file__).resolve()) not in command:
        raise ValueError("Tiến trình không thuộc lần chạy GRU này.")
    process.terminate()
    try:
        process.wait(timeout=5)
    except psutil.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    state.update(status="failed", pid=None, error="Đã dừng theo yêu cầu. Có thể chạy lại bước này hoặc tạo lần chạy mới.")
    _write(path / "state.json", state)
    (path / "active.lock").unlink(missing_ok=True)


def launch(path, stage, sessions=None):
    """Launch an allowlisted Python worker; never execute user-supplied shell text."""
    path = run_path(path)
    if stage not in STAGES:
        raise ValueError("Thao tác không hợp lệ.")
    state = read_state(path)
    completed = state.get("completed", [])
    if stage in completed or state.get("status") == "running":
        raise ValueError("Bước đã hoàn tất hoặc đang có tác vụ chạy.")
    if stage != "data" and STAGES[STAGES.index(stage) - 1] not in completed:
        raise ValueError("Hoàn tất bước trước trước khi tiếp tục.")
    if stage == "evaluate":
        if not sessions or not set(sessions).issubset(SESSIONS):
            raise ValueError("Chọn ít nhất một phiên test1–test5.")
        _write(path / "evaluation_request.json", {"sessions": list(dict.fromkeys(sessions))})
    lock = path / "active.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.close(descriptor)
    state.update(stage=stage, status="running", error=None, pid=None)
    _write(path / "state.json", state)
    try:
        with (path / f"{stage}.log").open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), str(path), stage],
                cwd=str(ROOT), stdout=output, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        # The worker owns state changes, avoiding parent/child completion races.
        Thread(target=process.wait, daemon=True).start()
        return process.pid
    except Exception as error:
        state.update(status="failed", error=str(error))
        _write(path / "state.json", state)
        lock.unlink(missing_ok=True)
        raise


def _array_paths(source, session):
    base = source / "final_test" if session.startswith("test") else source
    suffixes = ["features", "timestamps"]
    if session.startswith("test"):
        suffixes += ["attack", "attack_P1", "attack_P2", "attack_P3"]
    return {suffix: base / f"{session}_{suffix}.npy" for suffix in suffixes}


def prepare(path, settings, logger):
    import numpy as np
    import pandas as pd
    source = Path(settings["source"])
    sessions = ["train1", "train2", "train3"] + SESSIONS
    manifest = {}
    for session in sessions:
        if settings["source_kind"] == "processed":
            files = _array_paths(source, session)
        else:
            import joblib
            # Read the trusted, frozen scaler; do not fit/modify it.
            meta = read_json(ROOT / "models/preprocessing/scaler_metadata.json")
            features = meta["ordered_feature_names"]
            scaler = joblib.load(ROOT / "models/preprocessing/scaler.joblib")
            csv = source / f"{session}.csv.gz"
            if not csv.exists():
                csv = source / f"{session}.csv"
            destination = path / "data"
            (destination / "final_test").mkdir(parents=True, exist_ok=True)
            files = _array_paths(destination, session)
            pieces = {key: [] for key in files}
            for chunk in pd.read_csv(csv, chunksize=20000):
                pieces["features"].append(scaler.transform(chunk[features].to_numpy(dtype=np.float64)).astype(np.float32))
                pieces["timestamps"].append(pd.to_datetime(chunk["time"]).to_numpy(dtype="datetime64[s]"))
                for key in files.keys() - {"features", "timestamps"}:
                    pieces[key].append(chunk[key].to_numpy(dtype=np.int8))
            for key, target in files.items():
                np.save(target, np.concatenate(pieces[key]), allow_pickle=False)
        arrays = {key: np.load(file, mmap_mode="r", allow_pickle=False) for key, file in files.items()}
        values = arrays["features"]
        if values.ndim != 2 or values.shape[1] != 79 or len(values) <= 30 or values.dtype != np.float32:
            raise ValueError(f"{session}: cần mảng float32 có dạng (N > 30, 79).")
        if any(len(array) != len(values) for array in arrays.values()):
            raise ValueError(f"{session}: dữ liệu và nhãn/thời gian không cùng số dòng.")
        times = arrays["timestamps"]
        if not np.issubdtype(times.dtype, np.datetime64) or np.isnat(times).any() or not np.all(np.diff(times) == np.timedelta64(1, "s")):
            raise ValueError(f"{session}: thời gian phải liên tục theo giây, không NaT.")
        for start in range(0, len(values), 20000):
            if not np.isfinite(values[start:start + 20000]).all():
                raise ValueError(f"{session}: có NaN hoặc Inf.")
        for key in files.keys() - {"features", "timestamps"}:
            if not np.isin(arrays[key], [0, 1]).all():
                raise ValueError(f"{session}: nhãn {key} phải là 0/1.")
        used = min(len(values), settings["row_limit"]) if settings["row_limit"] and session.startswith("train") else len(values)
        manifest[session] = {"rows": len(values), "used_rows": used, "files": {key: str(file.resolve()) for key, file in files.items()}}
        logger.info("DATA %s | rows=%s | used=%s | features=79 | validated", session, len(values), used)
    _write(path / "data_manifest.json", manifest)


def train(path, settings, logger):
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from gru_model import OneStepGRU
    from gru_dataset import SessionWindowDataset
    from train_gru import train_single_run, evaluate_model_on_loader
    config = yaml.safe_load((ROOT / "configs/primary_gru.yaml").read_text(encoding="utf-8"))
    config["training"]["maximum_epochs"] = settings["epochs"]
    config["training"]["learning_rate"] = settings["learning_rate"]
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    manifest = read_json(path / "data_manifest.json")
    def array(session):
        item = manifest[session]
        return np.load(item["files"]["features"], mmap_mode="r", allow_pickle=False)[:item["used_rows"]]
    training = [array("train1"), array("train2")]
    validation = array("train3")
    device = torch.device("cpu")
    checkpoint = path / "gru.pt"
    result = train_single_run(30, settings["seed"], training, [validation], config, checkpoint, logger, device)
    model = OneStepGRU().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model_state_dict"])
    loader = DataLoader(SessionWindowDataset([validation], window_size=30, stride=1), batch_size=512, shuffle=False)
    logger.info("CALIBRATE | normal train3 | stride=1 | Q99.9")
    _, scores = evaluate_model_on_loader(model, loader, device)
    result["threshold"] = float(np.quantile(scores, 0.999, method="linear"))
    result["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result["profile"] = "quick" if settings["row_limit"] else "full"
    _write(path / "training.json", result)
    import pandas as pd
    pd.DataFrame(result["history"]).to_csv(path / "training_history.csv", index=False)
    logger.info("TRAIN COMPLETE | best_epoch=%s | threshold=%.8f | checkpoint=%s", result["best_epoch"], result["threshold"], checkpoint)


def evaluate(path, settings, logger):
    import numpy as np
    import pandas as pd
    import torch
    from gru_model import OneStepGRU
    from evaluate_final_test import (score_gru, extract_ground_truth_events,
        extract_predicted_alarm_events, extract_normal_false_alarm_events, match_events_one_to_one)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    manifest = read_json(path / "data_manifest.json")
    training = read_json(path / "training.json")
    checkpoint = path / "gru.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != training["checkpoint_sha256"]:
        raise ValueError("Checkpoint của lần chạy đã thay đổi sau khi huấn luyện.")
    model = OneStepGRU()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"])
    threshold = training["threshold"]
    sessions = read_json(path / "evaluation_request.json")["sessions"]
    events, all_delays, per_session = [], [], []
    totals = dict(tp=0, fn=0, fp=0, alarms=0, normal_fa=0, normal_samples=0)
    for session in sessions:
        logger.info("EVALUATE %s | checkpoint=%s | threshold=%.8f", session, checkpoint.name, threshold)
        data = {key: np.load(file, mmap_mode="r", allow_pickle=False) for key, file in manifest[session]["files"].items()}
        scores = score_gru(data["features"], model, torch.device("cpu"))
        times = data["timestamps"][30:]
        attacks = data["attack"][30:]
        is_alarm = scores > threshold
        gt = extract_ground_truth_events(session, data["timestamps"], data["attack"], data["attack_P1"], data["attack_P2"], data["attack_P3"])
        eligible = [item for item in gt if item["evaluable"]]
        alarms = extract_predicted_alarm_events(session, "gru_demo", scores, threshold, times)
        normal_fa = extract_normal_false_alarm_events(session, is_alarm, attacks, times)
        matched = match_events_one_to_one(eligible, alarms, is_alarm, times)
        values = dict(tp=matched["tp_count"], fn=matched["fn_count"], fp=matched["fp_count"], alarms=len(alarms), normal_fa=len(normal_fa), normal_samples=int(np.sum(attacks == 0)))
        for key, value in values.items():
            totals[key] += value
        for event in eligible:
            delay = matched["delays"][event["event_id"]]
            detected = not np.isnan(delay)
            if detected:
                all_delays.append(float(delay))
            events.append({"attack_event_id": event["event_id"], "session": session,
                           "orig_start": str(event["orig_start"]), "orig_end": str(event["orig_end"]),
                           "detected": bool(detected), "delay_seconds": float(delay) if detected else None})
        np.savez_compressed(path / f"scores_{session}.npz", timestamps=times, scores=scores, attack=attacks)
        per_session.append({"session": session, **values})
        logger.info("RESULT %s | TP=%s FN=%s FP=%s", session, values["tp"], values["fn"], values["fp"])
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    normal_hours = totals["normal_samples"] / 3600
    result = {**totals, "sessions": sessions, "event_precision": precision, "event_recall": recall,
              "event_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
              "normal_fa_events_per_hour": totals["normal_fa"] / normal_hours if normal_hours else None,
              "median_delay_sec": float(np.median(all_delays)) if all_delays else None,
              "threshold": threshold, "seed": settings["seed"], "checkpoint_sha256": training["checkpoint_sha256"]}
    # Same pooling/formulas as evaluate_final_test; missed delays remain missing.
    pd.DataFrame(events, columns=["attack_event_id", "session", "orig_start", "orig_end", "detected", "delay_seconds"]).to_csv(path / "attack_events.csv", index=False)
    pd.DataFrame(per_session).to_csv(path / "per_session.csv", index=False)
    _write(path / "evaluation.json", result)
    logger.info("EVALUATION COMPLETE | TP=%s/%s | Event F1=%.6f", totals["tp"], totals["tp"] + totals["fn"], result["event_f1"])


def work(path, stage):
    path = run_path(path)
    if stage not in STAGES:
        raise ValueError("Thao tác không hợp lệ.")
    state = read_json(path / "state.json")
    state.update(status="running", stage=stage, pid=os.getpid())
    _write(path / "state.json", state)
    logger = logging.getLogger("gru_live")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    try:
        settings = read_json(path / "settings.json")
        validate_settings(settings)
        sys.path.insert(0, str(ROOT / "src"))
        {"data": prepare, "train": train, "evaluate": evaluate}[stage](path, settings, logger)
        state["completed"] = list(dict.fromkeys(state.get("completed", []) + [stage]))
        state.update(status="done", error=None)
        return_code = 0
    except Exception as error:
        traceback.print_exc()
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        return_code = 1
    finally:
        state["pid"] = None
        (path / "active.lock").unlink(missing_ok=True)
        _write(path / "state.json", state)
    return return_code


if __name__ == "__main__":
    sys.exit(work(Path(sys.argv[1]), sys.argv[2]))
