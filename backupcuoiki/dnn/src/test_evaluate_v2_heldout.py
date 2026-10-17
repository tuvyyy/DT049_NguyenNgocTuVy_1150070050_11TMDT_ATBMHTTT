#!/usr/bin/env python3
"""
Unit Test Suite for Held-Out V2 Evaluation (`src/evaluate_v2_heldout.py`)
=========================================================================
Verifies:
1. Timestamp alignment (`t >= 30`), warmup censoring, and real `test4 + test5`
   ground-truth attack counts (5 + 12 = 17 events, 131,942 scored rows).
2. Deterministic 1-to-1 chronological event matching (`match_events_one_to_one_session`)
   and exact first-detection delay calculation.
3. Normal false alarm interval extraction (`alarm_t AND attack_t == 0`).
4. Pooled metric formulas (`compute_session_and_pooled_metrics`): proves that
   pooled Event Precision/Recall/F1, pooled FA/h, and pooled median delay are
   computed from pooled counts and pooled delay vectors rather than averaging
   per-file metrics.
5. Split isolation (`test4, test5` strictly disjoint from `train1, train2, test1, test2, test3`)
   and pre-evaluation freeze SHA-256 verification.
6. Zero-discrepancy cross-verification of V1 per-file metrics against frozen
   `results/metrics/final_per_file.csv`.
"""

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_v2_heldout import (
    EVAL_SESSIONS,
    START_IDX,
    compute_session_and_pooled_metrics,
    extract_ground_truth_events_session,
    extract_normal_false_alarm_events_session,
    extract_predicted_alarm_events_session,
    match_events_one_to_one_session,
    sha256_file,
)


class TestEvaluateV2Heldout(unittest.TestCase):
    def test_1_timestamp_and_t30_alignment(self):
        """Tests warmup censoring (`t < 30`), left-censoring across `t = 30`, and real test4+test5 counts."""
        n = 100
        ts = pd.date_range("2021-03-01 00:00:00", periods=n, freq="1s").to_numpy()
        attack = np.zeros(n, dtype=np.int8)
        p1 = np.zeros(n, dtype=np.int8)
        p2 = np.zeros(n, dtype=np.int8)
        p3 = np.zeros(n, dtype=np.int8)

        # Attack 1: strictly inside warmup [5..15] -> not evaluable
        attack[5:16] = 1
        p1[5:16] = 1
        # Attack 2: spans warmup boundary [25..40] -> evaluable, left_censored=True, effective_start_idx=30
        attack[25:41] = 1
        p2[25:41] = 1
        # Attack 3: strictly inside common region [60..75] -> evaluable, left_censored=False, effective_start_idx=60
        attack[60:76] = 1
        p1[60:76] = 1
        p3[60:76] = 1

        events = extract_ground_truth_events_session(
            "synth", ts, attack, p1, p2, p3, start_idx_alignment=START_IDX
        )
        self.assertEqual(len(events), 3)

        self.assertFalse(events[0]["evaluable"])
        self.assertTrue(events[0]["left_censored"])
        self.assertIsNone(events[0]["effective_start_idx"])

        self.assertTrue(events[1]["evaluable"])
        self.assertTrue(events[1]["left_censored"])
        self.assertEqual(events[1]["effective_start_idx"], 30)
        self.assertEqual(events[1]["effective_start"], pd.Timestamp(ts[30]))

        self.assertTrue(events[2]["evaluable"])
        self.assertFalse(events[2]["left_censored"])
        self.assertEqual(events[2]["effective_start_idx"], 60)
        self.assertTrue(events[2]["is_multiprocess"])
        self.assertEqual(events[2]["process_set"], "P1;P3")

        # Verify real test4 + test5 session counts directly from processed arrays
        test_dir = PROJECT_ROOT / "data" / "processed" / "hai_21_03" / "final_test"
        expected_scored = {"test4": 39571, "test5": 92371}
        expected_atks = {"test4": 5, "test5": 12}
        total_scored = 0
        total_normal_hours = 0.0

        for sname in EVAL_SESSIONS:
            ts_arr = np.load(test_dir / f"{sname}_timestamps.npy", allow_pickle=True)
            atk_arr = np.load(test_dir / f"{sname}_attack.npy", mmap_mode="r")
            p1_arr = np.load(test_dir / f"{sname}_attack_P1.npy", mmap_mode="r")
            p2_arr = np.load(test_dir / f"{sname}_attack_P2.npy", mmap_mode="r")
            p3_arr = np.load(test_dir / f"{sname}_attack_P3.npy", mmap_mode="r")

            evs = extract_ground_truth_events_session(
                sname, ts_arr, atk_arr, p1_arr, p2_arr, p3_arr, start_idx_alignment=START_IDX
            )
            eval_evs = [e for e in evs if e["evaluable"]]
            self.assertEqual(len(eval_evs), expected_atks[sname])

            scored_rows = len(atk_arr) - START_IDX
            self.assertEqual(scored_rows, expected_scored[sname])
            total_scored += scored_rows

            norm_samples = int(np.sum(atk_arr[START_IDX:] == 0))
            total_normal_hours += norm_samples / 3600.0

        self.assertEqual(total_scored, 131942)
        self.assertAlmostEqual(total_normal_hours, 35.72444444, places=5)

    def test_2_one_to_one_event_matching_and_delays(self):
        """Tests deterministic 1-to-1 chronological event matching and detection delay."""
        n = 120
        ts = pd.date_range("2021-03-01 00:00:00", periods=n, freq="1s").to_numpy()
        attack = np.zeros(n, dtype=np.int8)
        p1 = np.ones(n, dtype=np.int8)
        p2 = np.zeros(n, dtype=np.int8)
        p3 = np.zeros(n, dtype=np.int8)

        # Attack 1: [40..50], Attack 2: [60..70], Attack 3: [90..100]
        attack[40:51] = 1
        attack[60:71] = 1
        attack[90:101] = 1

        ev_all = extract_ground_truth_events_session(
            "synth", ts, attack, p1, p2, p3, start_idx_alignment=START_IDX
        )
        evaluable_atks = [e for e in ev_all if e["evaluable"]]
        self.assertEqual(len(evaluable_atks), 3)

        target_ts = ts[START_IDX:]
        scores_common = np.zeros(len(target_ts), dtype=np.float64)

        # Case A: One continuous alarm spanning [42..65] overlaps BOTH Attack 1 and Attack 2.
        # Under 1-to-1 matching, it must match Attack 1 (delay = 2s) and leave Attack 2 unmatched!
        scores_common[42 - START_IDX : 65 - START_IDX + 1] = 0.9
        # Case B: Two separate chatter alarms inside Attack 3: [93..94] and [97..99].
        # First alarm matches Attack 3 (delay = 3s); second alarm is an unmatched FP alarm!
        scores_common[93 - START_IDX : 94 - START_IDX + 1] = 0.9
        scores_common[97 - START_IDX : 99 - START_IDX + 1] = 0.9

        pred_alarms = extract_predicted_alarm_events_session(
            "synth", "m1", scores_common, 0.5, target_ts
        )
        self.assertEqual(len(pred_alarms), 3)

        m_res = match_events_one_to_one_session(
            evaluable_atks,
            pred_alarms,
            scores_common > 0.5,
            target_ts,
            start_idx_alignment=START_IDX,
        )
        self.assertEqual(m_res["tp_count"], 2)  # Attack 1 and Attack 3
        self.assertEqual(m_res["fn_count"], 1)  # Attack 2 unmatched
        self.assertEqual(m_res["fp_count"], 1)  # Second chatter alarm inside Attack 3
        self.assertAlmostEqual(m_res["delays"]["synth_atk_01"], 2.0, places=6)
        self.assertTrue(np.isnan(m_res["delays"]["synth_atk_02"]))
        self.assertAlmostEqual(m_res["delays"]["synth_atk_03"], 3.0, places=6)

    def test_3_normal_false_alarm_splitting(self):
        """Tests that normal false alarm intervals (`alarm_t AND attack_t == 0`) split across attacks."""
        target_ts = pd.date_range("2021-03-01 00:00:30", periods=50, freq="1s").to_numpy()
        target_atk = np.zeros(50, dtype=np.int8)
        target_atk[20:30] = 1  # Attack during indices [20..29]

        is_alarm = np.zeros(50, dtype=bool)
        is_alarm[15:35] = True  # Continuous alarm from 15 to 34 bridging the attack

        norm_fa = extract_normal_false_alarm_events_session(
            "synth", is_alarm, target_atk, target_ts
        )
        # Must split into [15..19] and [30..34]
        self.assertEqual(len(norm_fa), 2)
        self.assertEqual(norm_fa[0]["start_idx"], 15)
        self.assertEqual(norm_fa[0]["end_idx"], 19)
        self.assertEqual(norm_fa[0]["duration_seconds"], 5)
        self.assertEqual(norm_fa[1]["start_idx"], 30)
        self.assertEqual(norm_fa[1]["end_idx"], 34)
        self.assertEqual(norm_fa[1]["duration_seconds"], 5)

    def test_4_pooled_metric_formulas_never_average_per_file(self):
        """Proves pooled F1, pooled FA/h, and pooled median delay use pooled counts/vectors."""
        n1, n2 = 200, 400
        ts1 = pd.date_range("2021-03-01 00:00:00", periods=n1, freq="1s").to_numpy()
        ts2 = pd.date_range("2021-03-02 00:00:00", periods=n2, freq="1s").to_numpy()

        atk1 = np.zeros(n1, dtype=np.int8)
        atk1[40:50] = 1  # 1 attack in s1
        atk2 = np.zeros(n2, dtype=np.int8)
        atk2[50:60] = 1  # 3 attacks in s2
        atk2[100:110] = 1
        atk2[150:160] = 1

        p_ones1 = atk1.copy()
        p_zeros1 = np.zeros_like(atk1)
        p_ones2 = atk2.copy()
        p_zeros2 = np.zeros_like(atk2)

        ev1 = [
            e
            for e in extract_ground_truth_events_session(
                "s1", ts1, atk1, p_ones1, p_zeros1, p_zeros1
            )
            if e["evaluable"]
        ]
        ev2 = [
            e
            for e in extract_ground_truth_events_session(
                "s2", ts2, atk2, p_ones2, p_zeros2, p_zeros2
            )
            if e["evaluable"]
        ]

        t_atk1 = atk1[START_IDX:]
        t_atk2 = atk2[START_IDX:]
        norm_s1 = int(np.sum(t_atk1 == 0))
        norm_s2 = int(np.sum(t_atk2 == 0))

        session_meta = {
            "s1": {
                "session_name": "s1",
                "total_rows": n1,
                "common_rows": len(t_atk1),
                "target_ts": ts1[START_IDX:],
                "target_atk": t_atk1,
                "normal_samples": norm_s1,
                "normal_hours": norm_s1 / 3600.0,
                "evaluable_attacks": ev1,
            },
            "s2": {
                "session_name": "s2",
                "total_rows": n2,
                "common_rows": len(t_atk2),
                "target_ts": ts2[START_IDX:],
                "target_atk": t_atk2,
                "normal_samples": norm_s2,
                "normal_hours": norm_s2 / 3600.0,
                "evaluable_attacks": ev2,
            },
        }

        scores1 = np.zeros(len(t_atk1), dtype=np.float64)
        # s1: detect attack 1 with delay 1s, plus 3 false alarm events in normal region
        scores1[41 - START_IDX : 45 - START_IDX] = 0.9
        scores1[80 - START_IDX] = 0.9
        scores1[90 - START_IDX] = 0.9
        scores1[100 - START_IDX] = 0.9

        scores2 = np.zeros(len(t_atk2), dtype=np.float64)
        # s2: detect all 3 attacks with delays [2s, 5s, 8s], 0 false alarms
        scores2[52 - START_IDX : 55 - START_IDX] = 0.9  # delay 2s
        scores2[105 - START_IDX : 108 - START_IDX] = 0.9  # delay 5s
        scores2[158 - START_IDX : 160 - START_IDX] = 0.9  # delay 8s

        _, pooled, _ = compute_session_and_pooled_metrics(
            model_id="synth_model",
            model_name="Synth",
            paradigm="Test",
            operating_mode="test",
            threshold_by_session={"s1": 0.5, "s2": 0.5},
            scores_by_session={"s1": scores1, "s2": scores2},
            session_meta=session_meta,
        )

        # In s1: TP=1, FP=3, FN=0 -> Prec=0.25, Rec=1.0, F1=0.40, median_delay=1.0
        # In s2: TP=3, FP=0, FN=0 -> Prec=1.00, Rec=1.0, F1=1.00, median_delay=5.0
        # Naive average of per-file F1 = (0.40 + 1.00)/2 = 0.70
        # True pooled: TP=4, FP=3, FN=0 -> Prec=4/7, Rec=1.0, F1=8/11 = 0.727273
        self.assertAlmostEqual(pooled["event_f1"], 8.0 / 11.0, places=5)
        self.assertNotAlmostEqual(pooled["event_f1"], 0.70, places=2)

        # True pooled delays = [1.0, 2.0, 5.0, 8.0] -> median = (2.0 + 5.0)/2 = 3.5s
        # Naive average of per-file medians = (1.0 + 5.0)/2 = 3.0s
        self.assertAlmostEqual(pooled["median_delay_sec"], 3.5, places=5)
        self.assertNotAlmostEqual(pooled["median_delay_sec"], 3.0, places=2)

        # True pooled FA/h = (3 + 0) / ((norm_s1 + norm_s2)/3600)
        expected_fa_h = 3.0 / ((norm_s1 + norm_s2) / 3600.0)
        self.assertAlmostEqual(pooled["normal_fa_events_per_hour"], expected_fa_h, places=4)

    def test_5_split_isolation_and_pre_eval_freeze_verification(self):
        """Verifies split isolation and pre-evaluation freeze manifest integrity."""
        self.assertEqual(EVAL_SESSIONS, ("test4", "test5"))
        forbidden = {"train1", "train2", "test1", "test2", "test3"}
        self.assertEqual(len(set(EVAL_SESSIONS).intersection(forbidden)), 0)

        manifest_path = PROJECT_ROOT / "results" / "v2_dnn" / "v2_pre_evaluation_freeze_manifest.json"
        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["pre_evaluation_verification_status"], "PASS")
        for seed_key, expected_tau in [
            ("seed_42", 0.953203),
            ("seed_43", 0.958436),
            ("seed_44", 0.958462),
        ]:
            info = manifest["frozen_checkpoints"][seed_key]
            self.assertAlmostEqual(
                float(info["frozen_calibrated_threshold_test3"]),
                expected_tau,
                places=6,
            )
            ckpt_path = PROJECT_ROOT / info["checkpoint_path"]
            self.assertEqual(sha256_file(ckpt_path), info["sha256"])

    def test_6_cross_check_v1_frozen_per_file_and_v2_outputs(self):
        """Verifies that V1 per-file metrics on test4 and test5 match frozen results/metrics/final_per_file.csv."""
        v2_per_file_path = PROJECT_ROOT / "results/v2_dnn/v2_heldout_per_file.csv"
        v1_per_file_path = PROJECT_ROOT / "results/metrics/final_per_file.csv"
        self.assertTrue(v2_per_file_path.exists())
        self.assertTrue(v1_per_file_path.exists())

        df_v2 = pd.read_csv(v2_per_file_path)
        df_v1 = pd.read_csv(v1_per_file_path)
        df_v1_t45 = df_v1[df_v1["session"].isin(["test4", "test5"])]

        for _, row1 in df_v1_t45.iterrows():
            sub = df_v2[
                (df_v2["model_id"] == row1["model_id"])
                & (df_v2["session"] == row1["session"])
            ]
            self.assertEqual(len(sub), 1)
            row2 = sub.iloc[0]
            self.assertEqual(int(row1["tp_events"]), int(row2["tp_events"]))
            self.assertEqual(int(row1["event_level_fp"]), int(row2["event_level_fp"]))
            self.assertEqual(int(row1["fn_events"]), int(row2["fn_events"]))
            self.assertAlmostEqual(
                float(row1["event_f1"]), float(row2["event_f1"]), places=6
            )
            self.assertAlmostEqual(
                float(row1["normal_fa_events_per_hour"]),
                float(row2["normal_fa_events_per_hour"]),
                places=4,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
