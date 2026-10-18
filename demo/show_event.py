#!/usr/bin/env python3
"""Read-only terminal demo for the saved quick run and test1_atk_02.

No training, threshold fitting, artifact writes, or benchmark substitution.
The constants below are the user's expected values, never displayed as measurements.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/gru/20261018T114814_2e1846de"
EVENT = "test1_atk_02"
START = "2020-07-07 17:28:11"
END = "2020-07-07 17:29:48"
FIRST = "2020-07-07 17:28:29"


class DemoError(ValueError):
    pass


def require(path):
    if not path.is_file():
        raise DemoError(f"Không tìm thấy file: {path}")
    return path


def read_json(path):
    return json.loads(require(path).read_text(encoding="utf-8"))


def read_csv(path):
    with require(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def expect(name, actual, expected, digits=None):
    if digits is not None:
        actual = f"{float(actual):.{digits}f}"
        expected = f"{float(expected):.{digits}f}"
    if actual != expected:
        raise DemoError(f"{name}: đọc được {actual!r}; số ní yêu cầu {expected!r}. Không chỉnh số cho khớp.")


def load_run(path):
    settings = read_json(path / "settings.json")
    trained = read_json(path / "training.json")
    result = read_json(path / "evaluation.json")
    checkpoint = require(path / "gru.pt")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expect("SHA256 checkpoint/training", digest, trained["checkpoint_sha256"])
    expect("SHA256 checkpoint/evaluation", digest, result["checkpoint_sha256"])
    expect("Chế độ", trained["profile"], "quick")
    expect("Giới hạn dòng học mỗi phiên", settings["row_limit"], 10000)
    expect("Số epoch cấu hình", settings["epochs"], 3)
    expect("Số epoch đã chạy", trained["total_epochs"], 3)
    expect("Ngưỡng training", trained["threshold"], 1.605896, 6)
    expect("Ngưỡng evaluation", result["threshold"], trained["threshold"])
    expect("Seed training/evaluation", result["seed"], settings["seed"])
    expect("Phiên đánh giá", result["sessions"], ["test1"])
    expect("Số sự kiện khớp nhãn", result["tp"], 5)
    expect("Số sự kiện bỏ sót", result["fn"], 0)
    expect("Event F1", result["event_f1"], .144928, 6)
    expect("Normal FA/h", result["normal_fa_events_per_hour"], 4.9081, 4)
    expect("Median Delay", result["median_delay_sec"], 18)
    events = [row for row in read_csv(path / "attack_events.csv") if row["attack_event_id"] == EVENT]
    if len(events) != 1:
        raise DemoError(f"Cần đúng một hàng {EVENT}; tìm thấy {len(events)}.")
    event = events[0]
    expect("Phiên của sự kiện", event["session"], "test1")
    expect("Bắt đầu", event["orig_start"], START)
    expect("Kết thúc", event["orig_end"], END)
    expect("Trạng thái khớp nhãn", event["detected"].lower(), "true")
    expect("Delay đã lưu", float(event["delay_seconds"]), 18)

    # Print compact excerpts of the actual saved log, checked against its CSV.
    history = read_csv(path / "training_history.csv")
    log_text = require(path / "train.log").read_text(encoding="utf-8")
    logs = re.findall(r"Epoch\s+(\d+)/(\d+)\s*\|\s*Train Loss:\s*([0-9.eE+-]+)", log_text)
    expect("Số dòng epoch trong log", len(logs), 3)
    expect("Số dòng history", len(history), 3)
    for number, ((epoch, maximum, loss), row) in enumerate(zip(logs, history), 1):
        expect("Epoch trong log", (int(epoch), int(maximum)), (number, 3))
        expect("Epoch history", int(row["epoch"]), number)
        expect("Loss log/history", float(loss), float(row["train_loss_mse"]), 6)
    if not float(logs[-1][2]) < float(logs[0][2]):
        raise DemoError("Loss lượt cuối không thấp hơn lượt đầu; không thể nói loss đã giảm.")
    return dict(path=path, settings=settings, trained=trained, result=result, event=event, logs=logs)


def score_window(checkpoint, features):
    """Same model, W=30 and float64 RMSE calculation as score_gru in src/."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(ROOT / "src"))
    from gru_dataset import SessionWindowDataset
    from gru_model import OneStepGRU

    torch.set_num_threads(min(4, __import__("os").cpu_count() or 1))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expect("Cửa sổ checkpoint", saved["window_size"], 30)
    model = OneStepGRU()
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    loader = DataLoader(SessionWindowDataset([features], window_size=30, stride=1),
                        batch_size=512, shuffle=False, num_workers=0)
    parts = []
    with torch.no_grad():
        for inputs, targets in loader:
            predictions = model(inputs).numpy().astype(np.float64)
            differences = targets.numpy().astype(np.float64) - predictions
            parts.append(np.sqrt(np.mean(differences ** 2, axis=1)).astype(np.float32))
    return np.concatenate(parts)


def inspect_event(run):
    import numpy as np
    path = run["path"]
    manifest = read_json(path / "data_manifest.json")
    files = manifest["test1"]["files"]
    arrays = {}
    for name in ("features", "timestamps", "attack"):
        location = Path(files[name])
        if not location.is_absolute():
            location = ROOT / location
        arrays[name] = np.load(require(location), mmap_mode="r", allow_pickle=False)
    features, times, labels = (arrays[name] for name in ("features", "timestamps", "attack"))
    if features.ndim != 2 or features.shape[1] != 79 or features.dtype != np.float32:
        raise DemoError("Test1 cần features float32 có 79 cột, giống đầu vào của lần chạy.")
    if len(features) != len(times) or len(labels) != len(times):
        raise DemoError("Số dòng features, timestamps và attack không khớp.")
    if not np.issubdtype(times.dtype, np.datetime64) or np.isnat(times).any():
        raise DemoError("Timestamp không hợp lệ.")
    if not np.all(np.diff(times) == np.timedelta64(1, "s")) or not np.isin(labels, [0, 1]).all():
        raise DemoError("Test1 phải liên tục mỗi giây và nhãn phải là 0/1.")
    edges = np.diff(np.pad(np.asarray(labels, dtype=np.int8), (1, 1)))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    if len(starts) < 2:
        raise DemoError("Không có sự kiện thứ hai trong nhãn test1.")
    start, end = times[starts[1]], times[ends[1]]
    expect("Bắt đầu trong nhãn test1", str(start.astype("datetime64[s]")).replace("T", " "), START)
    expect("Kết thúc trong nhãn test1", str(end.astype("datetime64[s]")).replace("T", " "), END)
    first_target = int(np.searchsorted(times, start - np.timedelta64(10, "s")))
    last_target = int(np.searchsorted(times, end + np.timedelta64(10, "s")))
    if first_target < 30 or last_target >= len(times):
        raise DemoError("Thiếu dữ liệu ngữ cảnh trước/sau sự kiện.")
    window = features[first_target - 30:last_target + 1]
    if not np.isfinite(window).all():
        raise DemoError("Features quanh sự kiện chứa NaN hoặc Inf.")
    scores = score_window(path / "gru.pt", window)
    targets = np.asarray(times[first_target:last_target + 1])
    inside = (targets >= start) & (targets <= end)
    threshold = run["trained"]["threshold"]
    alarms = np.flatnonzero(inside & (scores > threshold))
    if len(alarms) == 0:
        raise DemoError("Checkpoint không cho sai số vượt ngưỡng trong sự kiện; trái với kết quả đã lưu.")
    first = targets[alarms[0]]
    expect("Cảnh báo đầu tiên tính lại", str(first.astype("datetime64[s]")).replace("T", " "), FIRST)
    delay = float((first - start) / np.timedelta64(1, "s"))
    expect("Độ trễ tính lại", delay, float(run["event"]["delay_seconds"]))

    # Also cross-check saved scores if present; never use them as an inference substitute.
    stored = path / "scores_test1.npz"
    if stored.is_file():
        with np.load(stored, allow_pickle=False) as cached:
            indices = np.searchsorted(cached["timestamps"], targets)
            if np.any(indices >= len(cached["timestamps"])) or not np.array_equal(cached["timestamps"][indices], targets):
                raise DemoError("Mốc giờ scores_test1.npz không khớp dữ liệu test1.")
            if not np.allclose(scores, cached["scores"][indices], rtol=1e-5, atol=1e-6):
                raise DemoError("RMSE tính lại không khớp scores_test1.npz trong sai số số học cho phép.")
    return dict(times=targets, scores=scores, inside=inside, first=first,
                peak=float(np.max(scores[inside])), threshold=threshold, delay=delay)


def show_log(run):
    log_text = require(run["path"] / "train.log").read_text(encoding="utf-8")
    for line in log_text.splitlines():
        if "Epoch" in line:
            print(line.strip())
    print(f"Ngưỡng tau: {run['trained']['threshold']:.6f}")


def show_result(run):
    result = run["result"]
    print("Kết quả test1:")
    print(f"Khớp nhãn: {result['tp']}/{result['tp'] + result['fn']} | Event F1: {result['event_f1']:.6f}")
    print(f"Normal FA/h: {result['normal_fa_events_per_hour']:.4f} | Median Delay: {result['median_delay_sec']:g} giây")


def show_event(run, measured):
    print(f"Sự kiện {EVENT}:")
    print(f"Bắt đầu: {run['event']['orig_start']}")
    print(f"Cảnh báo: {str(measured['first'].astype('datetime64[s]')).replace('T', ' ')} | trễ {measured['delay']:g} giây")
    print(f"Kết thúc: {run['event']['orig_end']}")
    print(f"tau = {measured['threshold']:.6f} | Đỉnh RMSE = {measured['peak']:.6f}")


def show_timeline(measured):
    import numpy as np
    scores, threshold = measured["scores"], measured["threshold"]
    top = max(float(np.max(scores)), threshold * 1.1)
    levels = [top, (top + threshold) / 2, threshold, threshold / 2, 0]
    width = max(12, min(60, shutil.get_terminal_size((96, 30)).columns - 12))
    print("1 cột = 1 giây | * RMSE (phân mức); ! cảnh báo đầu; [=] nhãn tấn công")
    print(f"Vạch ngang tau = {threshold:.6f} (* tại hàng tau chỉ thể hiện mức gần ngưỡng)")
    for offset in range(0, len(scores), width):
        part = scores[offset:offset + width]
        times = measured["times"][offset:offset + width]
        inside = measured["inside"][offset:offset + width]
        rows = [["-" if row == 2 else " " for _ in part] for row in range(5)]
        for column, value in enumerate(part):
            # Keep every above-threshold sample strictly above the threshold row.
            if value > threshold:
                row = 0 if value > (top + threshold) / 2 else 1
            else:
                row = 2 + int(np.argmin(np.abs(np.asarray(levels[2:]) - value)))
            rows[row][column] = "!" if times[column] == measured["first"] else "*"
        print(f"\n{str(times[0])[11:19]} -> {str(times[-1])[11:19]}")
        for row, level in enumerate(levels):
            label = f"tau {level:.3f}" if row == 2 else f"{level:.3f}"
            print(f"{label:>10}|{''.join(rows[row])}")
        marks = ["=" if value else "." for value in inside]
        for column, time in enumerate(times):
            if time == np.datetime64(START): marks[column] = "["
            if time == np.datetime64(END): marks[column] = "]"
        print(f"{'Nhãn':>10}|{''.join(marks)}")


def auxiliary(mode):
    if mode == "seeds":
        source = ROOT / "results/metrics/final_test_summary.csv"
        rows = [r for r in read_csv(source) if r["model_id"] in ("gru_seed42", "gru_seed43", "gru_seed44")]
        if len(rows) != 3:
            raise DemoError("CSV tổng hợp không đủ ba seed GRU.")
        print("Benchmark (test1–test5):")
        for row in rows:
            print(f"{row['model_id']}: khớp {row['tp_detected']}/{row['evaluable_attacks']} | F1 {float(row['event_f1']):.6f} | FA/h {float(row['normal_fa_events_per_hour']):.4f}")
    elif mode == "benchmark-log":
        path = require(ROOT / "results/logs/train_gru.log")
        from collections import deque
        print(f"12 dòng cuối {path.relative_to(ROOT)}:")
        with path.open(encoding="utf-8", errors="replace") as handle:
            print("".join(deque(handle, maxlen=12)), end="")
    else:
        data = read_json(ROOT / "results/metrics/gru_thresholds.json")
        threshold = data["per_seed_thresholds"]["42"]["Q99.9"]["threshold"]
        expect("Ngưỡng benchmark", threshold, 1.058582, 6)
        print(f"Benchmark seed 42: tau = {threshold:.6f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["log", "result", "event", "timeline", "all", "seeds", "benchmark-log", "benchmark-threshold", "1", "2", "3", "4"])
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN, help="Thư mục run đã có; không tự chọn run khác.")
    args = parser.parse_args(argv)
    try:
        alias_map = {"1": "log", "2": "result", "3": "event", "4": "timeline"}
        mode = alias_map.get(args.mode, args.mode)
        if mode in ("seeds", "benchmark-log", "benchmark-threshold"):
            auxiliary(mode)
            return 0
        path = args.run if args.run.is_absolute() else ROOT / args.run
        run = load_run(path)
        measured = inspect_event(run) if mode in ("event", "timeline", "all") else None
        if mode in ("log", "all"): show_log(run)
        if mode in ("result", "all"): show_result(run)
        if mode in ("event", "all"): show_event(run, measured)
        if mode in ("timeline", "all"): show_timeline(measured)
        return 0
    except (DemoError, OSError, KeyError, ValueError, TypeError, ImportError, RuntimeError) as error:
        print(f"DỪNG: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
