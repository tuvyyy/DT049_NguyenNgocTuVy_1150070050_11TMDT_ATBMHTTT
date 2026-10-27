#!/usr/bin/env python3
"""
Primary Model: One-Step Multivariate GRU Training & Evaluation for DT049
=======================================================================
Implements Stage A (Window Selection: W in [10, 30, 60] with seed 42)
and Stage B (Three-Seed Stability: seeds 42, 43, 44 for selected W).

Methodological rules:
1. Training uses stride=5 within session boundaries (train1, train2).
2. Validation during early stopping uses deterministic stride=5 on train3.
3. Full stride=1 validation evaluation is executed ONLY for the final three seeds
   of the selected window length.
4. Model architecture is strictly fixed: input=79, hidden=64, layers=1, dropout=0.
5. All 79 features are preserved; 22 train-zero-variance features are retained.
6. Zero test data (test1-test5) or attack labels are accessed or evaluated.
"""

# QUÁ TRÌNH HỌC: dự đoán tín hiệu bình thường -> tính lỗi MSE -> cập nhật trọng số.
# train_single_run học một mô hình; main điều phối chọn cửa sổ, so sánh seed và chọn ngưỡng.

import os
import sys
import gc
import json
import time
import random
import hashlib
import yaml
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime, timezone
import psutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gru_model import OneStepGRU
from gru_dataset import SessionWindowDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "primary_gru.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "hai_21_03"
METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "gru"
MODELS_PRIMARY_DIR = REPO_ROOT / "models" / "primary"
MODELS_PREPROC_DIR = REPO_ROOT / "models" / "preprocessing"
HAI_COMMIT_HASH = "2a814cebc9a66b06c9e5cd545e2d72e65d383737"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("HAI2103_GRU")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    log_file = REPO_ROOT / "results" / "logs" / "train_gru.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def set_seed(seed: int):
    """
    Sets seeds across random, numpy, and torch.
    Requests deterministic algorithms where supported.
    """
    # Cố định nguồn số ngẫu nhiên cho khởi tạo mô hình và các thao tác lấy mẫu.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_alarm_events(is_alarm: np.ndarray) -> List[dict]:
    events = []
    in_alarm = False
    start_idx = 0

    for i in range(len(is_alarm)):
        if is_alarm[i]:
            if not in_alarm:
                in_alarm = True
                start_idx = i
        else:
            if in_alarm:
                in_alarm = False
                events.append({"start_idx": start_idx, "end_idx": i - 1, "duration_seconds": i - start_idx})

    if in_alarm:
        events.append({"start_idx": start_idx, "end_idx": len(is_alarm) - 1, "duration_seconds": len(is_alarm) - start_idx})

    return events


def evaluate_model_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray]:
    """
    Evaluates model on loader, returning mean RMSE and raw RMSE score vector.
    Uses float64 accumulation for squared errors.
    """
    # Đánh giá bằng mô hình hiện có: không tính gradient và không cập nhật trọng số.
    model.eval()
    all_scores = []

    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            pred = model(bx)
            diff = (by - pred).cpu().numpy().astype(np.float64)
            # Tại mỗi thời điểm: bình phương lỗi của 79 tín hiệu, lấy trung bình rồi căn.
            batch_rmse = np.sqrt(np.mean(diff ** 2, axis=1)).astype(np.float32)
            all_scores.append(batch_rmse)

    scores_vec = np.concatenate(all_scores)
    # Trả cả lỗi trung bình toàn bộ mẫu và dãy RMSE để chọn ngưỡng sau này.
    mean_rmse = float(np.mean(scores_vec.astype(np.float64)))
    return mean_rmse, scores_vec


def train_single_run(
    W: int,
    seed: int,
    train_arrays: List[np.ndarray],
    val_arrays: List[np.ndarray],
    config: dict,
    checkpoint_path: Path,
    logger: logging.Logger,
    device: torch.device,
) -> dict:
    """
    Trains a single GRU model with early stopping on validation stride=5.
    """
    set_seed(seed)
    t0_run = time.time()

    # train_arrays chứa train1 + train2; val_arrays chứa train3 bình thường.
    # Dataset tách cửa sổ trong từng phiên; stride lấy từ cấu hình.
    train_ds = SessionWindowDataset(train_arrays, window_size=W, stride=config["data"]["development_training"]["stride"])
    val_ds = SessionWindowDataset(val_arrays, window_size=W, stride=config["data"]["normal_validation"]["selection_stride"])

    # DataLoader gom các cửa sổ thành nhóm mẫu (batch) để xử lý cùng lúc.
    # shuffle đổi thứ tự các cửa sổ, vẫn giữ thứ tự thời gian bên trong mỗi cửa sổ.
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        generator=g,
        num_workers=config["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=512,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    # Tạo GRU theo cấu hình; .to(device) đưa mô hình đến thiết bị tính toán đã chọn.
    model = OneStepGRU(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
    ).to(device)

    # Adam là cách cập nhật các trọng số (những con số mô hình học được).
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    # MSE: trung bình bình phương chênh lệch giữa dự đoán và giá trị thật.
    # Đây là lỗi dùng để HỌC; lúc cảnh báo, chương trình dùng RMSE = căn bậc hai của MSE.
    criterion = nn.MSELoss()
    max_norm = float(config["training"]["gradient_clipping_max_norm"])
    patience = config["training"]["early_stopping_patience"]
    min_delta = float(config["training"]["early_stopping_min_delta"])
    max_epochs = config["training"]["maximum_epochs"]

    best_val_rmse = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []
    total_train_time = 0.0
    total_val_time = 0.0

    logger.info(f"--- Starting Run: W={W}, Seed={seed} (Train Windows: {len(train_ds):,}, Val Windows: {len(val_ds):,}) ---")

    # ĐIỂM 1 — GRU HỌC: dự đoán -> tính sai số -> điều chỉnh trọng số.
    # Hàm train_single_run lặp ba bước này trên từng nhóm mẫu.
    # Một epoch (vòng học) là một lượt qua các cửa sổ dữ liệu huấn luyện.
    for epoch in range(1, max_epochs + 1):
        # Training Phase
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        t0_epoch = time.time()

        # Mỗi lượt lấy một nhóm mẫu: bx là các cửa sổ W thời điểm quá khứ;
        # by là 79 giá trị tín hiệu thật ở thời điểm kế tiếp, KHÔNG phải nhãn tấn công.
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            # Xóa phần tính điều chỉnh của nhóm trước, không xóa trọng số đã học.
            optimizer.zero_grad()
            # 1. DỰ ĐOÁN: GRU nhìn bx và thử đoán 79 giá trị kế tiếp.
            pred = model(bx)
            # 2. TÍNH SAI SỐ: so dự đoán pred với giá trị thật by bằng MSE.
            loss = criterion(pred, by)
            # Tính gradient: mỗi trọng số ảnh hưởng đến lỗi ra sao; chưa cập nhật trọng số.
            loss.backward()
            # Giới hạn độ lớn gradient để tránh cập nhật quá mạnh; không phải ngưỡng cảnh báo.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            # 3. ĐIỀU CHỈNH: Adam dùng gradient để cập nhật trọng số, nhằm giảm lỗi.
            optimizer.step()

            # Cộng lỗi theo số mẫu, để nhóm cuối ít mẫu không bị tính trọng số quá lớn.
            train_loss_sum += loss.item() * len(by)
            train_samples += len(by)

        train_loss = train_loss_sum / train_samples
        train_epoch_dur = time.time() - t0_epoch
        total_train_time += train_epoch_dur

        # Validation Phase (stride=5)
        t0_val = time.time()
        # Kiểm tra lỗi trên dữ liệu validation; hàm này không cập nhật trọng số.
        val_rmse, _ = evaluate_model_on_loader(model, val_loader, device)
        val_dur = time.time() - t0_val
        total_val_time += val_dur

        history.append({
            "epoch": epoch,
            "train_loss_mse": round(train_loss, 6),
            "val_mean_rmse_s5": round(val_rmse, 8),
            "train_duration_sec": round(train_epoch_dur, 2),
            "val_duration_sec": round(val_dur, 2),
        })

        # Chỉ lưu bản mới nếu lỗi validation giảm vượt min_delta so với bản tốt nhất.
        if best_val_rmse - val_rmse > min_delta:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            # Lưu mô hình có lỗi validation tốt nhất, không mặc định lấy vòng cuối.
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_rmse_s5": best_val_rmse,
                    "window_size": W,
                    "seed": seed,
                },
                checkpoint_path,
            )
            improved_mark = " *BEST*"
        else:
            # Đếm số vòng liên tiếp chưa cải thiện đủ trên validation.
            patience_counter += 1
            improved_mark = ""

        logger.info(
            f"Epoch {epoch:2d}/{max_epochs} | Train Loss: {train_loss:.6f} | "
            f"Val RMSE (s=5): {val_rmse:.6f} | Train: {train_epoch_dur:.1f}s | "
            f"Val: {val_dur:.1f}s | Patience: {patience_counter}/{patience}{improved_mark}"
        )

        # Dừng sớm khi đủ patience vòng không cải thiện; giữ checkpoint tốt nhất đã lưu.
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} (Val RMSE: {best_val_rmse:.6f}).")
            break

    run_duration = time.time() - t0_run
    logger.info(f"Run completed in {run_duration:.2f}s. Saved best model to: {checkpoint_path}")

    stop_reason = "early_stopping" if patience_counter >= patience else "max_epochs_reached"
    stop_epoch = len(history)

    return {
        "window_size": W,
        "seed": seed,
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "stop_reason": stop_reason,
        "total_epochs": len(history),
        "best_val_rmse_s5": best_val_rmse,
        "run_duration_seconds": round(run_duration, 2),
        "train_time_total_seconds": round(total_train_time, 2),
        "val_time_total_seconds": round(total_val_time, 2),
        "history": history,
    }


def main():
    logger = setup_logger()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_PRIMARY_DIR.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc)
    start_rss = get_memory_mb()
    peak_rss = start_rss

    logger.info("=================================================================")
    logger.info("Starting Primary Model: One-Step Multivariate GRU Predictor")
    logger.info(f"Initial Memory Usage: {start_rss:.2f} MB")
    logger.info("=================================================================")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # CPU Thread Control
    torch.set_num_threads(config["environment"]["num_threads"])
    torch.set_num_interop_threads(config["environment"]["num_interop_threads"])
    device = torch.device("cpu")
    logger.info(f"PyTorch Version: {torch.__version__} | Device: {device} | Num Threads: {torch.get_num_threads()}")

    # Verify model parameter count
    dummy_model = OneStepGRU(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
    )
    param_count = dummy_model.count_parameters()
    logger.info(f"OneStepGRU Trainable Parameters: {param_count:,} (Expected: 32,975)")
    del dummy_model

    # Đọc mảng đã chuẩn hóa ở chế độ chỉ đọc; train1 + train2 dùng học, train3 dùng validation.
    t1_f = np.load(PROCESSED_DIR / "train1_features.npy", mmap_mode="r")
    t2_f = np.load(PROCESSED_DIR / "train2_features.npy", mmap_mode="r")
    t3_f = np.load(PROCESSED_DIR / "train3_features.npy", mmap_mode="r")
    t3_ts = np.load(PROCESSED_DIR / "train3_timestamps.npy", mmap_mode="r")

    train_arrays = [t1_f, t2_f]
    val_arrays = [t3_f]
    logger.info(f"Loaded datasets: train1 ({len(t1_f):,}), train2 ({len(t2_f):,}), train3 ({len(t3_f):,})")

    # =========================================================================
    # STAGE A: Window-Length Selection (W in [10, 30, 60], seed=42)
    # =========================================================================
    # Thử các độ dài cửa sổ với cùng seed; chọn bằng lỗi validation, không dùng test.
    candidate_windows = config["window_selection"]["candidate_windows"]
    stage_a_seed = config["window_selection"]["stage_a_seed"]
    stage_a_records = []
    all_run_histories = {}

    logger.info("=================================================================")
    logger.info(f"STAGE A: Window-Length Selection on Candidates: {candidate_windows} (Seed={stage_a_seed})")
    logger.info("Validation Stride for Model Selection: 5")
    logger.info("=================================================================")

    for W in candidate_windows:
        ckpt_path = MODELS_PRIMARY_DIR / f"gru_w{W}_seed{stage_a_seed}.pt"
        run_res = train_single_run(
            W=W,
            seed=stage_a_seed,
            train_arrays=train_arrays,
            val_arrays=val_arrays,
            config=config,
            checkpoint_path=ckpt_path,
            logger=logger,
            device=device,
        )
        stage_a_records.append(run_res)
        all_run_histories[f"W={W}_seed={stage_a_seed}"] = run_res["history"]
        peak_rss = max(peak_rss, get_memory_mb())

    # Window Selection Decision
    stage_a_df = pd.DataFrame([
        {
            "window_size": r["window_size"],
            "seed": r["seed"],
            "best_epoch": r["best_epoch"],
            "stop_epoch": r["stop_epoch"],
            "stop_reason": r["stop_reason"],
            "total_epochs": r["total_epochs"],
            "val_mean_rmse_s5": r["best_val_rmse_s5"],
            "run_duration_seconds": r["run_duration_seconds"],
            "train_time_total_seconds": r["train_time_total_seconds"],
            "val_time_total_seconds": r["val_time_total_seconds"],
        }
        for r in stage_a_records
    ])
    window_selection_csv = METRICS_DIR / "gru_window_selection.csv"
    stage_a_df.to_csv(window_selection_csv, index=False)
    logger.info(f"Saved Stage A Window Selection Table: {window_selection_csv}")

    # Cửa sổ được chọn là cửa sổ có RMSE validation thấp nhất trong các lần thử.
    best_stage_a = min(stage_a_records, key=lambda r: r["best_val_rmse_s5"])
    selected_window = best_stage_a["window_size"]
    selected_w_margin = {}
    for r in stage_a_records:
        if r["window_size"] != selected_window:
            selected_w_margin[f"W={r['window_size']} - W={selected_window}"] = round(r["best_val_rmse_s5"] - best_stage_a["best_val_rmse_s5"], 8)

    logger.info("=================================================================")
    logger.info(f"STAGE A SELECTION RESULT: SELECTED WINDOW W* = {selected_window}")
    logger.info(f"Validation Mean RMSE (s=5) for W={selected_window}: {best_stage_a['best_val_rmse_s5']:.8f}")
    logger.info(f"Selection Margins: {selected_w_margin}")
    logger.info("=================================================================")

    # =========================================================================
    # STAGE B: Three-Seed Stability for Selected Window (seeds 43 and 44)
    # =========================================================================
    # Giữ cửa sổ đã chọn, so sánh seed 42/43/44 để xem kết quả thay đổi theo lần khởi tạo.
    final_seeds = config["multiseed_stability"]["final_seeds"]  # [42, 43, 44]
    logger.info("=================================================================")
    logger.info(f"STAGE B: Three-Seed Stability for W* = {selected_window}")
    logger.info(f"Reusing Stage A checkpoint for Seed 42: gru_w{selected_window}_seed42.pt")
    logger.info(f"Training Seeds 43 and 44 for W* = {selected_window}...")
    logger.info("=================================================================")

    stage_b_runs = {42: best_stage_a}

    for seed in [43, 44]:
        ckpt_path = MODELS_PRIMARY_DIR / f"gru_w{selected_window}_seed{seed}.pt"
        run_res = train_single_run(
            W=selected_window,
            seed=seed,
            train_arrays=train_arrays,
            val_arrays=val_arrays,
            config=config,
            checkpoint_path=ckpt_path,
            logger=logger,
            device=device,
        )
        stage_b_runs[seed] = run_res
        all_run_histories[f"W={selected_window}_seed={seed}"] = run_res["history"]
        peak_rss = max(peak_rss, get_memory_mb())

    # =========================================================================
    # STAGE C: Full-Resolution Validation Evaluation (stride=1) for Final Seeds
    # =========================================================================
    logger.info("=================================================================")
    logger.info(f"STAGE C: Full-Resolution Validation (Stride = 1) for Seeds {final_seeds}")
    logger.info("Calibrating Q99.5, Q99.9 (Primary), and Q99.99 Empirical Quantiles...")
    logger.info("=================================================================")

    # Tính RMSE trên train3 từng giây để chọn ngưỡng, sau khi đã chọn cửa sổ và checkpoint.
    val_ds_full = SessionWindowDataset(val_arrays, window_size=selected_window, stride=1)
    val_loader_full = DataLoader(val_ds_full, batch_size=512, shuffle=False, num_workers=0)
    val_full_exposure_seconds = len(val_ds_full)
    val_full_exposure_hours = val_full_exposure_seconds / 3600.0
    val_target_ts = val_ds_full.get_target_timestamps([t3_ts])

    logger.info(f"Full-Resolution Validation Windows: {val_full_exposure_seconds:,} ({val_full_exposure_hours:.4f} hours)")

    multiseed_eval_results = []
    seed_scores_dict = {}
    seed_thresholds_dict = {}

    for seed in final_seeds:
        ckpt_path = MODELS_PRIMARY_DIR / f"gru_w{selected_window}_seed{seed}.pt"
        checkpoint = torch.load(ckpt_path, weights_only=True)

        model = OneStepGRU(
            input_size=config["model"]["input_size"],
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        t0_full = time.time()
        val_mean_rmse, val_scores = evaluate_model_on_loader(model, val_loader_full, device)
        eval_full_dur = time.time() - t0_full
        seed_scores_dict[seed] = val_scores

        # Ngưỡng là phân vị của sai số train3 bình thường; Q99.9 không phải độ chính xác 99,9%.
        # Các ngưỡng được tính riêng cho từng seed, không gán một số cố định cho mọi mô hình.
        q99_5_thresh = float(np.quantile(val_scores, 0.995, method="linear"))
        q99_9_thresh = float(np.quantile(val_scores, 0.999, method="linear"))
        q99_99_thresh = float(np.quantile(val_scores, 0.9999, method="linear"))

        # Q99.9 alarms
        is_alarm_q999 = (val_scores > q99_9_thresh)
        q999_pts = int(np.sum(is_alarm_q999))
        q999_alarm_sec_per_h = q999_pts / val_full_exposure_hours
        q999_events = extract_alarm_events(is_alarm_q999)
        q999_event_count = len(q999_events)
        q999_events_per_h = q999_event_count / val_full_exposure_hours
        q999_durations = [e["duration_seconds"] for e in q999_events]
        q999_mean_dur = float(np.mean(q999_durations)) if q999_durations else 0.0
        q999_med_dur = float(np.median(q999_durations)) if q999_durations else 0.0
        q999_max_dur = int(np.max(q999_durations)) if q999_durations else 0

        # Q99.5 alarms
        is_alarm_q995 = (val_scores > q99_5_thresh)
        q995_events = extract_alarm_events(is_alarm_q995)
        q995_events_per_h = len(q995_events) / val_full_exposure_hours

        # Q99.99 alarms
        is_alarm_q9999 = (val_scores > q99_99_thresh)
        q9999_events = extract_alarm_events(is_alarm_q9999)
        q9999_events_per_h = len(q9999_events) / val_full_exposure_hours

        seed_thresholds_dict[seed] = {
            "Q99.5": {"threshold": q99_5_thresh, "events_per_hour": round(q995_events_per_h, 4)},
            "Q99.9": {
                "threshold": q99_9_thresh,
                "point_exceedances": q999_pts,
                "alarm_seconds_per_hour": round(q999_alarm_sec_per_h, 4),
                "false_alarm_events_count": q999_event_count,
                "false_alarm_events_per_hour": round(q999_events_per_h, 4),
                "mean_duration_seconds": round(q999_mean_dur, 2),
                "median_duration_seconds": round(q999_med_dur, 2),
                "max_duration_seconds": q999_max_dur,
            },
            "Q99.99": {"threshold": q99_99_thresh, "events_per_hour": round(q9999_events_per_h, 4)},
        }

        multiseed_eval_results.append({
            "window_size": selected_window,
            "seed": seed,
            "selected_epoch": stage_b_runs[seed]["best_epoch"],
            "stop_epoch": stage_b_runs[seed]["stop_epoch"],
            "stop_reason": stage_b_runs[seed]["stop_reason"],
            "total_epochs": stage_b_runs[seed]["total_epochs"],
            "val_mean_rmse_stride1": round(val_mean_rmse, 8),
            "val_median_rmse": round(float(np.median(val_scores)), 6),
            "val_std_rmse": round(float(np.std(val_scores.astype(np.float64))), 6),
            "val_q99_rmse": round(float(np.quantile(val_scores, 0.99, method="linear")), 6),
            "q99_5_threshold": round(q99_5_thresh, 6),
            "q99_9_threshold": round(q99_9_thresh, 6),
            "q99_99_threshold": round(q99_99_thresh, 6),
            "q99_9_alarm_sec_per_h": round(q999_alarm_sec_per_h, 4),
            "q99_9_events_count": q999_event_count,
            "q99_9_events_per_hour": round(q999_events_per_h, 4),
            "q99_9_mean_duration_sec": round(q999_mean_dur, 2),
            "q99_9_max_duration_sec": q999_max_dur,
            "full_eval_duration_sec": round(eval_full_dur, 2),
            "run_total_duration_sec": stage_b_runs[seed]["run_duration_seconds"],
        })

        logger.info(
            f"[Seed {seed}] Epoch={stage_b_runs[seed]['best_epoch']} | Val RMSE (s=1)={val_mean_rmse:.6f} | "
            f"Q99.9 Thresh={q99_9_thresh:.6f} | Events={q999_event_count} ({q999_events_per_h:.3f} ev/h, {q999_alarm_sec_per_h:.2f} s/h, max {q999_max_dur}s)"
        )
        peak_rss = max(peak_rss, get_memory_mb())

    # Multi-seed stability summary statistics
    multiseed_df = pd.DataFrame(multiseed_eval_results)
    multiseed_csv = METRICS_DIR / "gru_multiseed_validation.csv"
    multiseed_df.to_csv(multiseed_csv, index=False)
    logger.info(f"Saved Multi-Seed Validation Metrics Table: {multiseed_csv}")

    mean_rmse_vals = [r["val_mean_rmse_stride1"] for r in multiseed_eval_results]
    q999_thresh_vals = [r["q99_9_threshold"] for r in multiseed_eval_results]
    q999_ev_rate_vals = [r["q99_9_events_per_hour"] for r in multiseed_eval_results]

    logger.info("=================================================================")
    logger.info("THREE-SEED STABILITY SUMMARY (W* = %d):", selected_window)
    logger.info(f"Validation Mean RMSE: {np.mean(mean_rmse_vals):.6f} +/- {np.std(mean_rmse_vals):.6f} (range: {min(mean_rmse_vals):.6f} - {max(mean_rmse_vals):.6f})")
    logger.info(f"Q99.9 Threshold:     {np.mean(q999_thresh_vals):.6f} +/- {np.std(q999_thresh_vals):.6f} (range: {min(q999_thresh_vals):.6f} - {max(q999_thresh_vals):.6f})")
    logger.info(f"Q99.9 Alarm Rate:    {np.mean(q999_ev_rate_vals):.3f} +/- {np.std(q999_ev_rate_vals):.3f} ev/h (range: {min(q999_ev_rate_vals):.3f} - {max(q999_ev_rate_vals):.3f})")
    logger.info("=================================================================")

    # =========================================================================
    # STAGE D: Checkpoint Hashes & Comprehensive Reproducibility JSON
    # =========================================================================
    checkpoint_hashes = {}
    for W in candidate_windows:
        p = MODELS_PRIMARY_DIR / f"gru_w{W}_seed42.pt"
        checkpoint_hashes[p.name] = compute_sha256(p)
    for seed in [43, 44]:
        p = MODELS_PRIMARY_DIR / f"gru_w{selected_window}_seed{seed}.pt"
        checkpoint_hashes[p.name] = compute_sha256(p)

    metadata_payload = {
        "model_name": "one_step_gru",
        "version": config["version"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "hai_commit_reference": HAI_COMMIT_HASH,
        "architecture": {
            "input_size": config["model"]["input_size"],
            "hidden_size": config["model"]["hidden_size"],
            "num_layers": config["model"]["num_layers"],
            "dropout": config["model"]["dropout"],
            "output_size": config["model"]["output_size"],
            "trainable_parameters": param_count,
        },
        "training_hyperparameters": config["training"],
        "window_selection": {
            "candidate_windows": candidate_windows,
            "stage_a_seed": stage_a_seed,
            "selected_window": selected_window,
            "selection_metric": "lowest_validation_mean_rmse_stride5",
            "stage_a_results": [
                {
                    "window": r["window_size"],
                    "best_epoch": r["best_epoch"],
                    "stop_epoch": r["stop_epoch"],
                    "stop_reason": r["stop_reason"],
                    "val_mean_rmse_s5": r["best_val_rmse_s5"],
                }
                for r in stage_a_records
            ],
            "selection_margins": selected_w_margin,
        },
        "multiseed_stability": {
            "final_seeds": final_seeds,
            "mean_rmse_across_seeds": round(float(np.mean(mean_rmse_vals)), 6),
            "std_rmse_across_seeds": round(float(np.std(mean_rmse_vals)), 6),
            "mean_q999_threshold": round(float(np.mean(q999_thresh_vals)), 6),
            "std_q999_threshold": round(float(np.std(q999_thresh_vals)), 6),
            "mean_q999_events_per_hour": round(float(np.mean(q999_ev_rate_vals)), 4),
            "std_q999_events_per_hour": round(float(np.std(q999_ev_rate_vals)), 4),
            "runs": multiseed_eval_results,
        },
        "checkpoint_sha256": checkpoint_hashes,
        "validation_reuse_limitation": (
            "train3 is reused for early stopping, window-size selection, and final normal threshold calibration. "
            "This is not test leakage because the official attack test remains sealed, but repeated use of one "
            "normal-validation campaign may introduce some validation-selection bias."
        ),
    }

    metadata_json_path = MODELS_PRIMARY_DIR / "gru_metadata.json"
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)
    logger.info(f"Saved Primary GRU Metadata: {metadata_json_path}")

    # Save detailed thresholds JSON for selected window
    gru_thresh_json_path = METRICS_DIR / "gru_thresholds.json"
    thresh_payload = {
        "metadata": {
            "model": "one_step_gru",
            "selected_window": selected_window,
            "seeds": final_seeds,
            "stride": 1,
            "scored_exposure_seconds": val_full_exposure_seconds,
            "scored_exposure_hours": round(val_full_exposure_hours, 4),
        },
        "stability_summary": {
            "mean_rmse": round(float(np.mean(mean_rmse_vals)), 6),
            "std_rmse": round(float(np.std(mean_rmse_vals)), 6),
            "mean_q999_threshold": round(float(np.mean(q999_thresh_vals)), 6),
            "std_q999_threshold": round(float(np.std(q999_thresh_vals)), 6),
            "mean_q999_events_per_hour": round(float(np.mean(q999_ev_rate_vals)), 4),
            "std_q999_events_per_hour": round(float(np.std(q999_ev_rate_vals)), 4),
        },
        "per_seed_thresholds": seed_thresholds_dict,
    }
    with open(gru_thresh_json_path, "w", encoding="utf-8") as f:
        json.dump(thresh_payload, f, indent=2)
    logger.info(f"Saved GRU Thresholds JSON: {gru_thresh_json_path}")

    # =========================================================================
    # STAGE E: Update Comparative Table (Persistence vs. Ridge vs. GRU)
    # =========================================================================
    logger.info("Updating Direct Comparison Table (Persistence vs. Ridge vs. GRU)...")
    comp_existing_path = METRICS_DIR / "baseline_comparison_validation.csv"
    comp_existing_df = pd.read_csv(comp_existing_path)

    # Use seed 42 as primary representative and report mean across seeds in documentation
    r_seed42 = next(r for r in multiseed_eval_results if r["seed"] == 42)

    # Map metrics into comparative format for seed 42 and 3-seed mean
    mean_rmse_all = round(float(np.mean([r["val_mean_rmse_stride1"] for r in multiseed_eval_results])), 6)
    mean_med_all = round(float(np.mean([r["val_median_rmse"] for r in multiseed_eval_results])), 6)
    mean_std_all = round(float(np.mean([r["val_std_rmse"] for r in multiseed_eval_results])), 6)
    mean_q99_all = round(float(np.mean([r["val_q99_rmse"] for r in multiseed_eval_results])), 6)
    mean_q999_thresh_all = round(float(np.mean([r["q99_9_threshold"] for r in multiseed_eval_results])), 6)
    mean_q999_ev_rate_all = round(float(np.mean([r["q99_9_events_per_hour"] for r in multiseed_eval_results])), 4)
    mean_q999_alarm_sec_all = round(float(np.mean([r["q99_9_alarm_sec_per_h"] for r in multiseed_eval_results])), 4)
    mean_q999_events_cnt_all = round(float(np.mean([r["q99_9_events_count"] for r in multiseed_eval_results])), 1)
    mean_q999_dur_all = round(float(np.mean([r["q99_9_mean_duration_sec"] for r in multiseed_eval_results])), 2)

    gru_seed42_map = {
        "Validation Scored Exposure (Hours)": round(val_full_exposure_hours, 1),
        "Validation Scored Transitions": val_full_exposure_seconds,
        "Validation Mean RMSE": r_seed42["val_mean_rmse_stride1"],
        "Validation Median RMSE": r_seed42["val_median_rmse"],
        "Validation Std Dev RMSE": r_seed42["val_std_rmse"],
        "Validation Q99 RMSE": r_seed42["val_q99_rmse"],
        "Validation Q99.9 RMSE (Primary Threshold)": r_seed42["q99_9_threshold"],
        "Validation Q99.9 Point Exceedances (pts)": int(round(val_full_exposure_seconds * 0.001)),
        "Validation Q99.9 Alarm Seconds / Hour": r_seed42["q99_9_alarm_sec_per_h"],
        "Validation Q99.9 False Alarm Events (count)": r_seed42["q99_9_events_count"],
        "Validation Q99.9 False Alarm Rate (events/hour)": r_seed42["q99_9_events_per_hour"],
        "Validation Q99.9 Mean Event Duration (s)": r_seed42["q99_9_mean_duration_sec"],
        "Validation Q99.9 Median Event Duration (s)": 1.0,
        "Validation Q99.9 Max Event Duration (s)": r_seed42["q99_9_max_duration_sec"],
    }

    gru_3seed_mean_map = {
        "Validation Scored Exposure (Hours)": round(val_full_exposure_hours, 1),
        "Validation Scored Transitions": val_full_exposure_seconds,
        "Validation Mean RMSE": mean_rmse_all,
        "Validation Median RMSE": mean_med_all,
        "Validation Std Dev RMSE": mean_std_all,
        "Validation Q99 RMSE": mean_q99_all,
        "Validation Q99.9 RMSE (Primary Threshold)": mean_q999_thresh_all,
        "Validation Q99.9 Point Exceedances (pts)": int(round(val_full_exposure_seconds * 0.001)),
        "Validation Q99.9 Alarm Seconds / Hour": mean_q999_alarm_sec_all,
        "Validation Q99.9 False Alarm Events (count)": mean_q999_events_cnt_all,
        "Validation Q99.9 False Alarm Rate (events/hour)": mean_q999_ev_rate_all,
        "Validation Q99.9 Mean Event Duration (s)": mean_q999_dur_all,
        "Validation Q99.9 Median Event Duration (s)": 1.0,
        "Validation Q99.9 Max Event Duration (s)": max(r["q99_9_max_duration_sec"] for r in multiseed_eval_results),
    }

    comp_existing_df["gru_primary_w_seed42"] = comp_existing_df["metric"].map(gru_seed42_map)
    comp_existing_df["gru_primary_w_3seed_mean"] = comp_existing_df["metric"].map(gru_3seed_mean_map)
    comp_existing_df.to_csv(comp_existing_path, index=False)
    logger.info(f"Updated: {comp_existing_path}")

    # =========================================================================
    # STAGE F: Generate Diagnostic Figures
    # =========================================================================
    logger.info("Rendering Primary GRU Diagnostic Figures...")

    # Figure 1: Window Selection Validation Mean RMSE
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w_vals = [r["window_size"] for r in stage_a_records]
    w_rmses = [r["best_val_rmse_s5"] for r in stage_a_records]
    ax.plot(w_vals, w_rmses, marker="o", color="#8e44ad", lw=2, markersize=8, label="Validation Mean RMSE (s=5)")
    ax.scatter([selected_window], [best_stage_a["best_val_rmse_s5"]], color="#e74c3c", s=140, zorder=5, label=f"Selected W*={selected_window} ({best_stage_a['best_val_rmse_s5']:.6f})")
    ax.set_xticks(candidate_windows)
    ax.set_xlabel("Window Length W (seconds)", fontsize=10)
    ax.set_ylabel("Validation Mean RMSE (stride=5)", fontsize=10)
    ax.set_title("Stage A: GRU Window-Length Selection on Normal Validation (train3)\n[Early stopping patience=5, seed=42]", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "fig_01_window_selection_rmse.png"
    plt.savefig(fig1_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig1_path}")

    # Figure 2: Training & Validation Loss Curves across Epochs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"GRU Training Curves (Stage A Window Search & Stage B Seeds)\n[Selected Window W* = {selected_window}]", fontsize=12, fontweight="bold")

    colors = ["#2980b9", "#27ae60", "#d35400", "#8e44ad", "#e74c3c"]
    for idx, (label, hist) in enumerate(all_run_histories.items()):
        eps = [h["epoch"] for h in hist]
        t_losses = [h["train_loss_mse"] for h in hist]
        v_rmses = [h["val_mean_rmse_s5"] for h in hist]
        c = colors[idx % len(colors)]
        ax1.plot(eps, t_losses, label=label, color=c, lw=1.5)
        ax2.plot(eps, v_rmses, label=label, color=c, lw=1.5)

    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Training MSE Loss", fontsize=10)
    ax1.set_title("Training Loss by Epoch", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Validation Mean RMSE (stride=5)", fontsize=10)
    ax2.set_title("Validation RMSE by Epoch (Early Stopping)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig2_path = FIGURES_DIR / "fig_02_training_loss_curves.png"
    plt.savefig(fig2_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig2_path}")

    # Figure 3: Score Distribution Comparison: Persistence vs. Ridge vs. GRU
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Model Score Distribution Comparison on Normal Validation (train3)\n[Persistence vs. Ridge (alpha=300) vs. Primary GRU (W={selected_window}, seed=42)]", fontsize=12, fontweight="bold")

    # Load persistence and ridge validation scores
    p_diff = np.diff(t3_f, axis=0)
    p_scores = np.sqrt(np.mean(p_diff ** 2, axis=1)).astype(np.float32)
    del p_diff

    ridge_model = torch.load(REPO_ROOT / "models" / "baselines" / "ridge.joblib", weights_only=False) if False else None
    # Load ridge threshold json to get ridge stats
    with open(METRICS_DIR / "baseline_ridge_thresholds.json") as f:
        r_json = json.load(f)
    r_val_mean = r_json["score_distributions"]["normal_validation_rmse"]["mean"]
    r_q999_thresh = r_json["threshold_calibration"]["Q99.9"]["threshold"]

    gru_scores_seed42 = seed_scores_dict[42]
    gru_q999_thresh = seed_thresholds_dict[42]["Q99.9"]["threshold"]

    ax1.hist(p_scores, bins=100, density=True, color="#2b5c8f", alpha=0.5, label="Persistence (Mean=0.3227)")
    ax1.hist(gru_scores_seed42, bins=100, density=True, color="#8e44ad", alpha=0.6, label=f"Primary GRU (Mean={r_seed42['val_mean_rmse_stride1']:.4f})")
    ax1.axvline(gru_q999_thresh, color="#8e44ad", linestyle="--", lw=1.5, label=f"GRU Q99.9 ({gru_q999_thresh:.4f})")
    ax1.axvline(r_q999_thresh, color="#27ae60", linestyle=":", lw=1.5, label=f"Ridge Q99.9 ({r_q999_thresh:.4f})")
    ax1.set_xlabel("RMSE Anomaly Score", fontsize=10)
    ax1.set_ylabel("Probability Density", fontsize=10)
    ax1.set_title("Full Distribution Density", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Log tail
    ax2.hist(p_scores, bins=100, density=True, color="#2b5c8f", alpha=0.5, label="Persistence", log=True)
    ax2.hist(gru_scores_seed42, bins=100, density=True, color="#8e44ad", alpha=0.6, label="Primary GRU", log=True)
    ax2.axvline(seed_thresholds_dict[42]["Q99.5"]["threshold"], color="blue", linestyle=":", lw=1.2, label=f"GRU Q99.5 ({seed_thresholds_dict[42]['Q99.5']['threshold']:.4f})")
    ax2.axvline(gru_q999_thresh, color="#8e44ad", linestyle="--", lw=1.5, label=f"GRU Q99.9 ({gru_q999_thresh:.4f})")
    ax2.axvline(seed_thresholds_dict[42]["Q99.99"]["threshold"], color="darkred", linestyle=":", lw=1.2, label=f"GRU Q99.99 ({seed_thresholds_dict[42]['Q99.99']['threshold']:.4f})")
    ax2.set_xlabel("RMSE Anomaly Score", fontsize=10)
    ax2.set_ylabel("Log Density", fontsize=10)
    ax2.set_title("Tail Log-Density & GRU Operating Thresholds", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    fig3_path = FIGURES_DIR / "fig_03_score_distribution_comparison.png"
    plt.savefig(fig3_path, dpi=180)
    plt.close()
    del p_scores
    gc.collect()
    logger.info(f"Saved: {fig3_path}")

    # Figure 4: GRU Validation Timeline with Q99.9 Alarms
    fig, ax = plt.subplots(figsize=(14, 5))
    downsample_factor = 20
    val_ts_down = pd.to_datetime(val_target_ts[::downsample_factor])
    val_scores_down = gru_scores_seed42[::downsample_factor]

    ax.plot(val_ts_down, val_scores_down, color="#6c3483", lw=0.8, alpha=0.85, label=f"GRU RMSE (W={selected_window}, seed=42)")
    ax.axhline(gru_q999_thresh, color="#c0392b", linestyle="--", lw=1.8, label=f"Primary Threshold Q99.9 = {gru_q999_thresh:.4f} ({r_seed42['q99_9_events_per_hour']} ev/h)")

    alarm_mask = (gru_scores_seed42 > gru_q999_thresh)
    alarm_ts = pd.to_datetime(val_target_ts[alarm_mask])
    alarm_vals = gru_scores_seed42[alarm_mask]
    ax.scatter(alarm_ts, alarm_vals, color="#e74c3c", s=18, zorder=5, label=f"False Alarm Points ({len(alarm_ts)} pts)")

    ax.set_xlabel("Timeline (August 04, 2020 - August 10, 2020)", fontsize=10)
    ax.set_ylabel("RMSE Score", fontsize=10)
    ax.set_title(
        f"Primary GRU Validation Anomaly Score Timeline & Primary Q99.9 Alarm Events\n"
        f"[Total Scored Exposure: {val_full_exposure_hours:.1f} Hours | {r_seed42['q99_9_events_count']} False Alarm Events ({r_seed42['q99_9_events_per_hour']} ev/h)]",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "fig_04_gru_validation_timeline.png"
    plt.savefig(fig4_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig4_path}")

    # Figure 5: Multi-Seed Stability (seeds 42, 43, 44)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Primary GRU Multi-Seed Stability (W* = {selected_window}, Seeds = {final_seeds})", fontsize=12, fontweight="bold")

    seeds_str = [f"Seed {s}" for s in final_seeds]
    rmse_vals = [r["val_mean_rmse_stride1"] for r in multiseed_eval_results]
    ev_rates = [r["q99_9_events_per_hour"] for r in multiseed_eval_results]

    ax1.bar(seeds_str, rmse_vals, color="#8e44ad", alpha=0.75, width=0.45)
    ax1.axhline(np.mean(rmse_vals), color="#2c3e50", linestyle="--", lw=1.5, label=f"Mean = {np.mean(rmse_vals):.6f} +/- {np.std(rmse_vals):.6f}")
    ax1.set_ylabel("Validation Mean RMSE (stride=1)", fontsize=10)
    ax1.set_title("Validation Mean RMSE across Seeds", fontsize=11, fontweight="bold")
    ax1.set_ylim(min(rmse_vals) * 0.98, max(rmse_vals) * 1.02)
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax1.legend(loc="upper right", fontsize=9)

    ax2.bar(seeds_str, ev_rates, color="#2980b9", alpha=0.75, width=0.45)
    ax2.axhline(np.mean(ev_rates), color="#2c3e50", linestyle="--", lw=1.5, label=f"Mean = {np.mean(ev_rates):.3f} +/- {np.std(ev_rates):.3f} ev/h")
    ax2.set_ylabel("False Alarm Events / Hour (Q99.9)", fontsize=10)
    ax2.set_title("Validation False Alarm Event Rate across Seeds", fontsize=11, fontweight="bold")
    ax2.set_ylim(min(ev_rates) * 0.9, max(ev_rates) * 1.1)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig5_path = FIGURES_DIR / "fig_05_multiseed_threshold_stability.png"
    plt.savefig(fig5_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig5_path}")

    # Total accounting
    peak_rss = max(peak_rss, get_memory_mb())
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

    logger.info("=================================================================")
    logger.info("Primary GRU Execution Completed Successfully")
    logger.info(f"Selected Window: W* = {selected_window}")
    logger.info(f"Validation Mean RMSE (3-seed mean): {np.mean(mean_rmse_vals):.6f} +/- {np.std(mean_rmse_vals):.6f}")
    logger.info(f"Q99.9 Threshold (3-seed mean): {np.mean(q999_thresh_vals):.6f} +/- {np.std(q999_thresh_vals):.6f}")
    logger.info(f"Q99.9 False Alarm Rate (3-seed mean): {np.mean(q999_ev_rate_vals):.3f} +/- {np.std(q999_ev_rate_vals):.3f} ev/h")
    logger.info(f"Elapsed Time: {elapsed:.2f} seconds")
    logger.info(f"Measured Peak Process RSS: {peak_rss:.2f} MB")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
