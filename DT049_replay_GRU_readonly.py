#!/usr/bin/env python3
"""Independent read-only replay of DT049 GRU seed 42 on HAI 21.03.

From the repository on Kali:
    ./.venv/bin/python DT049_replay_GRU_readonly.py --repo . --session test1
    ./.venv/bin/python DT049_replay_GRU_readonly.py --repo .
    ./.venv/bin/python DT049_replay_GRU_readonly.py --smoke

No training, model modifications, threshold selection, or writing result files.
--smoke uses synthetic inputs; it is not an HAI evaluation.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

WINDOW = 30
FEATURES = 79
THRESHOLD = 1.0585819482803345
EXPECTED_CKPT_SHA = "48013cdb36a7490330ba615874399ff21f0a2c3f9b5f7f6edc756168233e2c3d"
EXPECTED_SCALER_SHA = "cbc43757fb1155a36285b711d9b134d387e20e9775a7ab19c8776749a9000d26"
EXPECTED = {"test1": (43201, 5), "test2": (118801, 20), "test3": (108001, 8), "test4": (39601, 5), "test5": (92401, 12)}
RAW_SHA = {
    "test1": "532de41bed372206941609d05b763958f59abecfd63c8c123649f823452e445e",
    "test2": "c5572e0ed19906107ceb56c4b4f40e165153fc7ba87b3506230d53d2478c29bb",
    "test3": "bff835894baff0c73871bc01ece4f5cf759d455a049e808b012128767420caaf",
    "test4": "51380fd28fd985b94a10cae3d8b80926becb3af9f233f2d3a38ab8be9482a482",
    "test5": "f0d59308b167ed95d0e81fb3e27c906682c9b21d3dba0ab4c4d696565df35179",
}


class OneStepGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=FEATURES, hidden_size=64, num_layers=1, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(64, FEATURES)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def intervals(mask):
    arr = np.asarray(mask, dtype=np.int8)
    edges = np.diff(np.pad(arr, (1, 1)))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    (np.flatnonzero(edges == -1) - 1).tolist()))


def load_session(repo, session):
    directory = repo / "data/processed/hai_21_03/final_test"
    feature_path = directory / f"{session}_features.npy"
    attack_path = directory / f"{session}_attack.npy"
    if feature_path.is_file() and attack_path.is_file():
        manifest_path = repo / "results/audit/final_test_preprocessing_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing checksum manifest {manifest_path}")
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            manifest = list(csv.DictReader(handle))
        for role, path in [("features", feature_path), ("attack", attack_path)]:
            matching = [row for row in manifest if row["session_name"] == session and row["array_role"] == role]
            if len(matching) != 1 or sha256(path) != matching[0]["sha256"]:
                raise ValueError(f"Processed {session} {role}: SHA-256 mismatch")
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        attack = np.load(attack_path, mmap_mode="r", allow_pickle=False)
        source = "preprocessed NPY (SHA-256 verified)"
    else:
        import joblib
        import pandas as pd
        raw = repo / "data/source/hai_repo/hai-21.03" / f"{session}.csv.gz"
        scaler_path = repo / "models/preprocessing/scaler.joblib"
        meta_path = repo / "models/preprocessing/scaler_metadata.json"
        if not raw.is_file():
            raise FileNotFoundError(f"Missing processed NPY and source CSV: {raw}")
        if sha256(raw) != RAW_SHA[session]:
            raise ValueError(f"Raw {session} CSV SHA-256 mismatch")
        if sha256(scaler_path) != EXPECTED_SCALER_SHA:
            raise ValueError("Frozen scaler SHA-256 mismatch")
        columns = json.loads(meta_path.read_text(encoding="utf-8"))["ordered_feature_names"]
        df = pd.read_csv(raw, compression="gzip", usecols=columns + ["attack"])
        features = joblib.load(scaler_path).transform(df[columns].to_numpy(dtype=np.float64)).astype(np.float32)
        attack = df["attack"].to_numpy(dtype=np.int8)
        source = "official compressed CSV + frozen StandardScaler"
    if features.ndim != 2 or features.shape != (EXPECTED[session][0], FEATURES):
        raise ValueError(f"Unexpected feature shape {features.shape} for {session}")
    if len(attack) != EXPECTED[session][0] or np.any((attack != 0) & (attack != 1)):
        raise ValueError(f"Unexpected attack labels for {session}")
    if not np.all(np.isfinite(features)):
        raise ValueError(f"Non-finite features for {session}")
    if len(intervals(attack == 1)) != EXPECTED[session][1]:
        raise ValueError(f"Unexpected number of attack events for {session}")
    return features, attack, source


@torch.inference_mode()
def score_session(model, features, batch_size):
    count = len(features) - WINDOW
    scores = np.empty(count, dtype=np.float32)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        batch = np.stack([features[i:i + WINDOW] for i in range(start, end)]).astype(np.float32)
        predicted = model(torch.from_numpy(batch)).cpu().numpy().astype(np.float64)
        target = np.asarray(features[start + WINDOW:end + WINDOW], dtype=np.float64)
        scores[start:end] = np.sqrt(np.mean((target - predicted) ** 2, axis=1)).astype(np.float32)
    return scores


def evaluate_session(scores, labels):
    attack = np.asarray(labels[WINDOW:] == 1, dtype=bool)
    alarms = np.asarray(scores > THRESHOLD, dtype=bool)
    ground_truth = [(max(s, WINDOW) - WINDOW, e - WINDOW)
                    for s, e in intervals(labels == 1) if e >= WINDOW]
    predicted = intervals(alarms)
    matched = set()
    delays = []
    for start, end in ground_truth:
        for index, (a_start, a_end) in enumerate(predicted):
            if index not in matched and max(start, a_start) <= min(end, a_end):
                matched.add(index)
                alarm_positions = np.flatnonzero(alarms[start:end + 1])
                if len(alarm_positions):
                    delays.append(float(alarm_positions[0]))
                break
    normal = ~attack
    return {
        "tp": len(matched),
        "fn": len(ground_truth) - len(matched),
        "fp_events": len(predicted) - len(matched),
        "normal_fa": len(intervals(alarms & normal)),
        "normal_seconds": int(normal.sum()),
        "delays": delays,
    }


def read_saved(repo, session=None):
    path = (repo / "results/metrics/final_per_file.csv" if session
            else repo / "results/metrics/final_test_summary.csv")
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        matches = [row for row in csv.DictReader(handle)
                   if row["model_id"] == "gru_seed42" and (session is None or row["session"] == session)]
    if len(matches) != 1:
        raise ValueError(f"Expected one gru_seed42 row in {path}; found {len(matches)}")
    return matches[0]


def summarize(result):
    tp, fn, fp = result["tp"], result["fn"], result["fp_events"]
    delays = result["delays"]
    return {
        "tp": tp, "fn": fn, "fp_events": fp,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "fa_h": result["normal_fa"] / (result["normal_seconds"] / 3600),
        "mean_delay": float(np.mean(delays)) if delays else float("nan"),
    }


def compare(result, saved):
    if saved is None:
        print("Saved metrics CSV not found: no comparison possible")
        return
    fields = [
        ("TP", "tp", "tp_events" if "tp_events" in saved else "tp_detected", 0),
        ("FN", "fn", "fn_events" if "fn_events" in saved else "fn_missed", 0),
        ("Unmatched alarm events", "fp_events", "event_level_fp", 0),
        ("Event Recall", "recall", "event_recall", 0.0001),
        ("Event F1", "f1", "event_f1", 0.0001),
        ("Normal false alarms/hour", "fa_h", "normal_fa_events_per_hour", 0.0002),
        ("Mean detection delay (sec)", "mean_delay", "mean_delay_sec", 0.02),
    ]
    for label, observed_key, saved_key, tolerance in fields:
        if saved_key not in saved or not saved[saved_key]:
            print(f"  {label}: no saved value")
            continue
        observed = float(result[observed_key])
        expected = float(saved[saved_key])
        delta = observed - expected
        print(f"  {label}: replay={observed:.6f}, saved={expected:.6f}, "
              f"delta={delta:+.6f} => {'MATCH' if abs(delta) <= tolerance else 'DIFF'}")


def smoke():
    torch.manual_seed(42)
    model = OneStepGRU()
    data = torch.randn(2, WINDOW, FEATURES)
    target = torch.randn(2, FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model.fc.weight.detach().clone()
    optimizer.zero_grad()
    prediction = model(data)
    torch.nn.functional.mse_loss(prediction, target).backward()
    optimizer.step()
    assert prediction.shape == (2, FEATURES)
    assert sum(p.numel() for p in model.parameters()) == 32975
    assert not torch.equal(before, model.fc.weight)
    print("SMOKE TEST PASS (SYNTHETIC DATA ONLY; NOT HAI): forward, backward, Adam update")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--session", choices=["all", *EXPECTED.keys()], default="all")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    torch.set_num_threads(4)
    if args.smoke:
        smoke()
        return
    repo = args.repo.resolve()
    checkpoint_path = repo / "models/primary/gru_w30_seed42.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if sha256(checkpoint_path) != EXPECTED_CKPT_SHA:
        raise ValueError("GRU checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = OneStepGRU()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    sessions = list(EXPECTED) if args.session == "all" else [args.session]
    totals = {"tp": 0, "fn": 0, "fp_events": 0, "normal_fa": 0, "normal_seconds": 0, "delays": []}
    print(f"HAI 21.03 GRU SEED 42 REPLAY: threshold Q99.9 = {THRESHOLD:.9f}")
    for session in sessions:
        features, labels, source = load_session(repo, session)
        metrics = evaluate_session(score_session(model, features, args.batch_size), labels)
        print(f"{session}: {source}; TP={metrics['tp']}, FN={metrics['fn']}, "
              f"unmatched alarm events={metrics['fp_events']}, normal FA={metrics['normal_fa']}")
        compare(summarize(metrics), read_saved(repo, session))
        for key in ["tp", "fn", "fp_events", "normal_fa", "normal_seconds"]:
            totals[key] += metrics[key]
        totals["delays"].extend(metrics["delays"])
    print("TOTAL REPLAY:", summarize(totals))
    if args.session == "all":
        print("COMPARISON TO SAVED BENCHMARK (replay minus saved):")
        compare(summarize(totals), read_saved(repo))
    print("The paper classifies four states on different robot data; there is no valid direct accuracy delta.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        sys.exit(f"REPLAY FAILED (no results claimed): {exc}")
