#!/usr/bin/env python3
"""Terminal demo: read saved logs/results or recompute one event, without training."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EVENT = "test1_atk_02"


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
        actual, expected = (f"{float(value):.{digits}f}" for value in (actual, expected))
    if actual != expected:
        raise DemoError(f"{name}: giá trị đọc được {actual!r}, giá trị đối chiếu {expected!r}. Các nguồn không khớp.")


def number(value, name, integer=False):
    if isinstance(value, bool):
        raise DemoError(f"{name}: cần một số, không phải giá trị đúng/sai.")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (integer and not result.is_integer()):
        raise DemoError(f"{name}: cần số {'nguyên ' if integer else ''}hữu hạn không âm.")
    return int(result) if integer else result


def close(name, actual, expected):
    if expected is None:
        expect(name, actual, None)
    elif not math.isclose(number(actual, name), expected, rel_tol=1e-8, abs_tol=1e-9):
        raise DemoError(f"{name}: artifact ghi {actual}, tính từ các trường đã lưu là {expected}.")


def digest(path):
    return hashlib.sha256(require(path).read_bytes()).hexdigest()


def load_run(path):
    """Read metadata and verify checkpoint bytes; do not load a model or infer."""
    settings, trained = (read_json(path / name) for name in ("settings.json", "training.json"))
    checksum = digest(path / "gru.pt")
    expect("SHA-256 checkpoint/training.json", checksum, trained["checkpoint_sha256"])
    seed = number(settings["seed"], "Seed", integer=True)
    expect("Seed training/settings", trained["seed"], seed)
    limit = number(settings["row_limit"], "Giới hạn dòng", integer=True)
    if limit not in (0, 10000):
        raise DemoError("Giới hạn dòng không thuộc cấu hình hiện có: 0 hoặc 10000.")
    expect("Chế độ training/settings", trained["profile"], "quick" if limit else "full")
    maximum = number(settings["epochs"], "Số epoch cấu hình", integer=True)
    total = number(trained["total_epochs"], "Số epoch đã chạy", integer=True)
    best = number(trained["best_epoch"], "Epoch checkpoint", integer=True)
    if not 1 <= best <= total <= maximum:
        raise DemoError("Epoch checkpoint, số epoch đã chạy và cấu hình không nhất quán.")
    expect("Cửa sổ training", trained["window_size"], 30)
    number(trained["threshold"], "Ngưỡng của lần chạy")
    return dict(path=path, settings=settings, trained=trained, checksum=checksum)


def load_log(run):
    """Validate actual saved epoch lines against CSV and training metadata."""
    history = read_csv(run["path"] / "training_history.csv")
    text = require(run["path"] / "train.log").read_text(encoding="utf-8")
    logs = re.findall(r"Epoch\s+(\d+)/(\d+)\s*\|\s*Train Loss:\s*([0-9.eE+-]+)", text)
    total = run["trained"]["total_epochs"]
    expect("Số dòng epoch trong log", len(logs), total)
    expect("Số dòng history CSV", len(history), total)
    expect("Số dòng history JSON", len(run["trained"]["history"]), total)
    for index, ((epoch, maximum, loss), row, recorded) in enumerate(zip(logs, history, run["trained"]["history"]), 1):
        expect("Epoch trong log", (int(epoch), int(maximum)), (index, run["settings"]["epochs"]))
        expect("Epoch trong CSV", int(row["epoch"]), index)
        expect("Epoch trong JSON", recorded["epoch"], index)
        number(loss, "Train Loss")
        expect("Loss log/CSV", float(loss), float(row["train_loss_mse"]), 6)
        expect("Loss log/JSON", float(loss), float(recorded["train_loss_mse"]), 6)
    run["logs"] = logs


def load_result(run, event_id=EVENT):
    """Check saved evaluation, its checkpoint/threshold, counts and event rows."""
    result = read_json(run["path"] / "evaluation.json")
    expect("SHA-256 evaluation/checkpoint", result["checkpoint_sha256"], run["checksum"])
    expect("Ngưỡng evaluation/training", result["threshold"], run["trained"]["threshold"])
    expect("Seed evaluation/settings", result["seed"], run["settings"]["seed"])
    sessions = result["sessions"]
    if not sessions or len(set(sessions)) != len(sessions) or not set(sessions) <= {f"test{i}" for i in range(1, 6)}:
        raise DemoError("Danh sách phiên đánh giá không hợp lệ.")
    events = read_csv(run["path"] / "attack_events.csv")
    if len({row["attack_event_id"] for row in events}) != len(events):
        raise DemoError("Bảng sự kiện có mã trùng lặp.")
    for row in events:
        if row["session"] not in sessions or not row["attack_event_id"].startswith(row["session"] + "_atk_"):
            raise DemoError("Mã sự kiện hoặc phiên không khớp bảng đánh giá.")
        value = row["detected"].lower()
        if value not in ("true", "false"):
            raise DemoError("Trạng thái sự kiện phải là True hoặc False.")
        row["is_detected"] = value == "true"
        if row["is_detected"]:
            number(row["delay_seconds"], "Độ trễ đã lưu", integer=True)
        elif row["delay_seconds"].strip() not in ("", "nan"):
            raise DemoError("Sự kiện không khớp nhãn phải có độ trễ bỏ trống, không gán 0.")
    tp, fn, fp, alarms, normal_fa, normal_samples = (
        number(result[key], key, integer=True) for key in ("tp", "fn", "fp", "alarms", "normal_fa", "normal_samples"))
    expect("TP/bảng sự kiện", tp, sum(row["is_detected"] for row in events))
    expect("FN/bảng sự kiện", fn, len(events) - tp)
    expect("Số cảnh báo/TP+FP", alarms, tp + fp)
    per_session = read_csv(run["path"] / "per_session.csv")
    expect("Các phiên trong per_session.csv", sorted(row["session"] for row in per_session), sorted(sessions))
    for key in ("tp", "fn", "fp", "alarms", "normal_fa", "normal_samples"):
        expect(f"Tổng {key}/per_session.csv", result[key], sum(number(row[key], key, integer=True) for row in per_session))
    for row in per_session:
        subset = [event for event in events if event["session"] == row["session"]]
        expect("TP theo phiên", int(row["tp"]), sum(event["is_detected"] for event in subset))
        expect("FN theo phiên", int(row["fn"]), len(subset) - int(row["tp"]))
    precision, recall = (tp / (tp + fp) if tp + fp else 0), (tp / (tp + fn) if tp + fn else 0)
    close("Event Precision", result["event_precision"], precision)
    close("Event Recall", result["event_recall"], recall)
    close("Event F1", result["event_f1"], 2 * precision * recall / (precision + recall) if precision + recall else 0)
    close("Normal FA/h", result["normal_fa_events_per_hour"], normal_fa / (normal_samples / 3600) if normal_samples else None)
    delays = [float(row["delay_seconds"]) for row in events if row["is_detected"]]
    close("Median Delay", result["median_delay_sec"], statistics.median(delays) if delays else None)
    run.update(result=result, events=events, event_id=event_id)


def score_window(checkpoint, features, expected=None):
    """Same W=30 model and float64 RMSE calculation as score_gru in src/."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(ROOT / "src"))
    from gru_dataset import SessionWindowDataset
    from gru_model import OneStepGRU
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expect("Cửa sổ checkpoint", saved["window_size"], 30)
    if expected is not None:
        expect("Seed checkpoint/training", saved["seed"], expected["seed"])
        expect("Epoch checkpoint/training", saved["epoch"], expected["best_epoch"])
    model = OneStepGRU()
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    loader = DataLoader(SessionWindowDataset([features], window_size=30, stride=1), batch_size=512, shuffle=False, num_workers=0)
    parts = []
    with torch.no_grad():
        for inputs, targets in loader:
            predictions = model(inputs).numpy().astype(np.float64)
            differences = targets.numpy().astype(np.float64) - predictions
            parts.append(np.sqrt(np.mean(differences ** 2, axis=1)).astype(np.float32))
    return np.concatenate(parts)


def inspect_event(run):
    import numpy as np
    event_id, path = run["event_id"], run["path"]
    match = re.fullmatch(r"(test[1-5])_atk_(\d{2,})", event_id)
    if not match or int(match[2]) < 1:
        raise DemoError("Mã sự kiện không hợp lệ; ví dụ test1_atk_02.")
    session, ordinal = match[1], int(match[2])
    rows = [row for row in run["events"] if row["attack_event_id"] == event_id]
    if len(rows) != 1:
        raise DemoError(f"Không tìm thấy sự kiện {event_id} trong kết quả của run đã chọn.")
    event = rows[0]
    manifest = read_json(path / "data_manifest.json")
    files = manifest[session]["files"]
    arrays = {}
    for name in ("features", "timestamps", "attack"):
        location = Path(files[name])
        if not location.is_absolute():
            location = ROOT / location
        arrays[name] = np.load(require(location), mmap_mode="r", allow_pickle=False)
    features, times, labels = (arrays[name] for name in ("features", "timestamps", "attack"))
    if features.ndim != 2 or features.shape[1] != 79 or features.dtype != np.float32:
        raise DemoError(f"{session}: cần features float32 có 79 cột.")
    expect("Số dòng features/timestamps", len(features), len(times))
    expect("Số dòng nhãn/timestamps", len(labels), len(times))
    expect("Số dòng manifest/dữ liệu", manifest[session]["rows"], len(times))
    expect("Dữ liệu kiểm thử giữ đầy đủ", manifest[session]["used_rows"], len(times))
    if not np.issubdtype(times.dtype, np.datetime64) or np.isnat(times).any():
        raise DemoError("Timestamp không hợp lệ.")
    if not np.all(np.diff(times) == np.timedelta64(1, "s")) or not np.isin(labels, [0, 1]).all():
        raise DemoError("Phiên kiểm thử phải liên tục mỗi giây và nhãn phải là 0/1.")
    edges = np.diff(np.pad(np.asarray(labels, dtype=np.int8), (1, 1)))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    if ordinal > len(starts):
        raise DemoError(f"Không có {event_id} trong nhãn HAI của {session}.")
    start_index, end_index = int(starts[ordinal - 1]), int(ends[ordinal - 1])
    start, end = times[start_index], times[end_index]
    for field, measured in (("orig_start", start), ("orig_end", end)):
        expect(f"{field}: CSV/nhãn HAI", np.datetime64(event[field]), measured)
    if end_index < 30:
        raise DemoError("Sự kiện nằm trước mốc đánh giá chung W=30.")
    first_target, last_target = max(30, start_index - 10), min(len(times) - 1, end_index + 10)
    window = features[first_target - 30:last_target + 1]
    if not np.isfinite(window).all():
        raise DemoError("Features quanh sự kiện chứa NaN hoặc Inf.")
    scores = score_window(path / "gru.pt", window, expected=run["trained"])
    expect("SHA-256 checkpoint sau suy luận", digest(path / "gru.pt"), run["checksum"])
    targets = np.asarray(times[first_target:last_target + 1])
    if not np.isfinite(scores).all() or len(scores) != len(targets):
        raise DemoError("RMSE tính lại không hữu hạn hoặc lệch số dòng.")
    inside = (targets >= start) & (targets <= end)
    threshold = run["trained"]["threshold"]
    alarms = np.flatnonzero(inside & (scores > threshold))
    first = targets[alarms[0]] if len(alarms) else None
    if bool(len(alarms)) != event["is_detected"]:
        raise DemoError("Trạng thái vượt ngưỡng tính lại khác kết quả ghép sự kiện đã lưu. Cần kiểm tra ghép 1–1 trên toàn phiên; không kết luận thành công.")
    delay = float((first - times[max(30, start_index)]) / np.timedelta64(1, "s")) if first is not None else None
    if first is not None:
        close("Độ trễ tính lại/CSV", float(event["delay_seconds"]), delay)
    stored = path / f"scores_{session}.npz"
    if stored.is_file():
        with np.load(stored, allow_pickle=False) as cached:
            indices = np.searchsorted(cached["timestamps"], targets)
            if np.any(indices >= len(cached["timestamps"])) or not np.array_equal(cached["timestamps"][indices], targets):
                raise DemoError("Timestamp của scores đã lưu không khớp dữ liệu.")
            if not np.allclose(scores, cached["scores"][indices], rtol=1e-5, atol=1e-6):
                raise DemoError("RMSE tính lại không khớp scores đã lưu trong sai số số học cho phép.")
            if not np.array_equal(scores > threshold, cached["scores"][indices] > threshold):
                raise DemoError("Quyết định vượt ngưỡng khác scores đã lưu.")
            if "attack" in cached and not np.array_equal(cached["attack"][indices], labels[first_target:last_target + 1]):
                raise DemoError("Nhãn trong scores đã lưu không khớp nhãn dữ liệu.")
    run["event"] = event
    return dict(times=targets, scores=scores, inside=inside, first=first, start=start, end=end,
                peak=float(np.max(scores[inside])), threshold=threshold, delay=delay)


def show_log(run, detail=False):
    logs = run["logs"]
    print(f"ĐỌC LOG ĐÃ LƯU | seed {run['settings']['seed']} | {len(logs)} epoch")
    chosen = logs if detail or len(logs) <= 3 else [logs[i] for i in sorted({0, run['trained']['best_epoch'] - 1, len(logs) - 1})]
    for epoch, maximum, loss in chosen:
        print(f"Epoch {epoch}/{maximum} | Train Loss: {float(loss):.6f}")
    print(f"Mô hình đã được lưu: gru.pt (vòng học {run['trained']['best_epoch']}).")
    if len(chosen) != len(logs): print(f"Đang trích {len(chosen)}/{len(logs)} epoch; dùng --detail để xem đủ.")
    if detail:
        print("SHA-256 khớp.")
        print(f"Run: {run['path']} | tau: {run['trained']['threshold']:.6f}")


def show_result(run, detail=False):
    result = run["result"]
    print(f"ĐỌC KẾT QUẢ ĐÃ LƯU | {', '.join(result['sessions'])}")
    print(f"Cảnh báo khớp nhãn: {result['tp']}/{result['tp'] + result['fn']} sự kiện | Bỏ sót: {result['fn']}")
    if detail:
        delay, rate = result['median_delay_sec'], result['normal_fa_events_per_hour']
        print(f"Event F1: {result['event_f1']:.6f} | Normal FA/h: {rate:.4f}" if rate is not None else f"Event F1: {result['event_f1']:.6f} | Normal FA/h: N/A")
        print(f"Median Delay: {delay:g} giây" if delay is not None else "Median Delay: N/A")
        print(f"Ngưỡng của run: {result['threshold']:.6f}")


def show_event(run, measured, detail=False):
    print(f"TÍNH LẠI BẰNG CHECKPOINT | {run['event_id']}")
    print(f"Bắt đầu theo nhãn HAI: {run['event']['orig_start']}")
    if measured['first'] is None:
        print("Cảnh báo trong vùng nhãn: không có | Độ trễ: không áp dụng")
        print("Kết luận: không khớp sự kiện nhãn HAI.")
    else:
        print(f"Cảnh báo đầu trong vùng nhãn: {str(measured['first'].astype('datetime64[s]')).replace('T', ' ')}")
        print(f"Độ trễ: {measured['delay']:g} giây | Kết luận: sai số bất thường khớp nhãn HAI.")
    print(f"Ngưỡng cảnh báo (Q99.9): {measured['threshold']:.3f} | Sai số cao nhất: {measured['peak']:.3f}")
    if measured['peak'] > measured['threshold']:
        print("Sai số đã vượt ngưỡng cảnh báo.")
    if detail:
        print(f"Kết thúc theo nhãn: {run['event']['orig_end']}")
        print(f"tau: {measured['threshold']:.6f} | Đỉnh RMSE trong vùng nhãn: {measured['peak']:.6f}")
        print("Đầu vào GRU chỉ có 79 tín hiệu; nhãn chỉ dùng đối chiếu sau suy luận.")


def show_timeline(measured):
    import numpy as np
    scores, threshold = measured["scores"], measured["threshold"]
    top = max(float(np.max(scores)), threshold * 1.1, 1e-9)
    levels = [top, (top + threshold) / 2, threshold, threshold / 2, 0]
    width = max(12, min(60, shutil.get_terminal_size((96, 30)).columns - 12))
    print("1 cột = 1 giây | * RMSE (phân mức); ! cảnh báo đầu; [=] nhãn tấn công")
    print(f"Vạch ngang tau = {threshold:.6f} (* tại hàng tau chỉ thể hiện mức gần ngưỡng)")
    for offset in range(0, len(scores), width):
        part, times = scores[offset:offset + width], measured["times"][offset:offset + width]
        rows = [["-" if row == 2 else " " for _ in part] for row in range(5)]
        for column, value in enumerate(part):
            row = (0 if value > (top + threshold) / 2 else 1) if value > threshold else 2 + int(np.argmin(np.abs(np.asarray(levels[2:]) - value)))
            rows[row][column] = "!" if measured["first"] is not None and times[column] == measured["first"] else "*"
        print(f"\n{str(times[0])[11:19]} -> {str(times[-1])[11:19]}")
        for row, level in enumerate(levels):
            label = f"tau {level:.3f}" if row == 2 else f"{level:.3f}"
            print(f"{label:>10}|{''.join(rows[row])}")
        marks = ["=" if value else "." for value in measured["inside"][offset:offset + width]]
        for column, time in enumerate(times):
            if time == measured["start"]: marks[column] = "["
            if time == measured["end"]: marks[column] = "]"
        print(f"{'Nhãn':>10}|{''.join(marks)}")


def auxiliary(mode):
    if mode == "seeds":
        rows = [r for r in read_csv(ROOT / "results/metrics/final_test_summary.csv") if r["model_id"] in ("gru_seed42", "gru_seed43", "gru_seed44")]
        expect("Số seed trong benchmark", len(rows), 3)
        print("Benchmark (test1–test5):")
        for row in rows:
            print(f"{row['model_id']}: khớp {row['tp_detected']}/{row['evaluable_attacks']} | F1 {float(row['event_f1']):.6f} | FA/h {float(row['normal_fa_events_per_hour']):.4f}")
    elif mode == "benchmark-log":
        from collections import deque
        path = require(ROOT / "results/logs/train_gru.log")
        print(f"12 dòng cuối log đã lưu {path.relative_to(ROOT)}:")
        with path.open(encoding="utf-8", errors="replace") as handle:
            print("".join(deque(handle, maxlen=12)), end="")
    else:
        threshold = read_json(ROOT / "results/metrics/gru_thresholds.json")["per_seed_thresholds"]["42"]["Q99.9"]["threshold"]
        expect("Ngưỡng benchmark", threshold, 1.058582, 6)
        print(f"Benchmark seed 42: tau = {threshold:.6f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["log", "result", "event", "timeline", "all", "seeds", "benchmark-log", "benchmark-threshold", "1", "2", "3", "4"])
    parser.add_argument("--run", type=Path, default=os.environ.get("DT049_RUN"), help="Run đã lưu; mặc định lấy DT049_RUN do ./show thiết lập. Không tự chọn run mới nhất.")
    parser.add_argument("--event", default=EVENT, help="Mặc định test1_atk_02.")
    parser.add_argument("--detail", action="store_true", help="Thêm metric, ngưỡng, RMSE và thông tin đối chiếu.")
    args = parser.parse_args(argv)
    try:
        mode = {"1": "log", "2": "result", "3": "event", "4": "timeline"}.get(args.mode, args.mode)
        if mode in ("seeds", "benchmark-log", "benchmark-threshold"):
            auxiliary(mode)
            return 0
        if args.run is None:
            raise DemoError("Chưa chỉ định run. Dùng ./show hoặc truyền --run THƯ_MỤC_RUN; không tự chọn run thay thế.")
        path = args.run if args.run.is_absolute() else ROOT / args.run
        run = load_run(path)
        if mode in ("log", "all"): load_log(run)
        if mode in ("result", "event", "timeline", "all"): load_result(run, args.event)
        measured = inspect_event(run) if mode in ("event", "timeline", "all") else None
        if mode in ("log", "all"): show_log(run, args.detail)
        if mode in ("result", "all"): show_result(run, args.detail)
        if mode in ("event", "all"): show_event(run, measured, args.detail)
        if mode == "timeline": show_timeline(measured)
        return 0
    except (DemoError, OSError, KeyError, ValueError, TypeError, ImportError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        print(f"DỪNG: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
