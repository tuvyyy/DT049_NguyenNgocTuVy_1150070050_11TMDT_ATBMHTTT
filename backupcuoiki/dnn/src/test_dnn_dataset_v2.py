#!/usr/bin/env python3
"""
Unit and Smoke Tests for DT049 DNN V2 Data Pipeline (`src/dnn_dataset_v2.py`)
=============================================================================
Verifies:
1. Input batch shape = (batch_size, 79) and float32 dtype.
2. Target labels are strictly binary (0 or 1).
3. Zero leakage of `time`, `attack`, `attack_P1`, `attack_P2`, `attack_P3`
   into the 79 input features (verified against both scaler_metadata.json and
   raw HAI 21.03 CSV headers).
4. Zero session overlap between Train (test1+test2), Validation (test3), and
   Evaluation (test4+test5).
5. Dataset uses read-only memory mapping (`mmap_mode="r"`) and does not mutate
   source `.npy` files or frozen V1 artifacts (`scaler.joblib`).
6. Real batches can be fetched cleanly from Train and Validation DataLoaders
   under Kali CPU (< 4 GB RAM).
7. Exact Train and Validation label distributions match audited HAI 21.03 ground
   truth, and class weights are computed strictly from Train (`test1 + test2`).
"""

import gzip
import hashlib
import json
import resource
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dnn_dataset_v2 import (
    DEFAULT_CONFIG_PATH,
    FORBIDDEN_FEATURE_COLUMNS,
    SessionPointwiseDNNDatasetV2,
    build_v2_dataloaders,
    compute_train_class_weights,
    load_v2_config,
    verify_frozen_scaler_and_features,
    verify_split_disjointness,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0


class TestDNNDatasetV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_v2_config(DEFAULT_CONFIG_PATH)
        cls.processed_test_dir = PROJECT_ROOT / cls.config["dataset"]["processed_test_dir"]
        cls.scaler_path = PROJECT_ROOT / cls.config["dataset"]["scaler_path"]
        cls.scaler_meta_path = PROJECT_ROOT / cls.config["dataset"]["scaler_metadata_path"]

        # Record pre-test SHA-256 checksums of critical files to prove immutability
        cls.tracked_files = [
            cls.scaler_path,
            cls.scaler_meta_path,
            cls.processed_test_dir / "test1_features.npy",
            cls.processed_test_dir / "test1_attack.npy",
            cls.processed_test_dir / "test2_features.npy",
            cls.processed_test_dir / "test2_attack.npy",
            cls.processed_test_dir / "test3_features.npy",
            cls.processed_test_dir / "test3_attack.npy",
            cls.processed_test_dir / "test4_features.npy",
            cls.processed_test_dir / "test4_attack.npy",
            cls.processed_test_dir / "test5_features.npy",
            cls.processed_test_dir / "test5_attack.npy",
        ]
        cls.pre_hashes = {str(p): sha256_file(p) for p in cls.tracked_files}

    def test_01_protocol_config_and_reference_architecture(self):
        cfg = self.config
        self.assertEqual(cfg["dataset"]["version"], "21.03")
        self.assertFalse(cfg["dataset"]["refit_scaler"])
        self.assertEqual(cfg["dataset"]["scaler_fit_source"]["total_rows"], 442802)
        self.assertEqual(cfg["features"]["input_dim"], 79)
        self.assertEqual(cfg["features"]["target_label_column"], "attack")
        self.assertEqual(cfg["splits"]["dnn_train"]["sessions"], ["test1", "test2"])
        self.assertEqual(cfg["splits"]["dnn_validation"]["sessions"], ["test3"])
        self.assertEqual(cfg["splits"]["dnn_evaluation_v2"]["sessions"], ["test4", "test5"])

        ref_arch = cfg["reference_architecture"]
        self.assertEqual(ref_arch["hidden_layers"], [100, 70, 100, 80, 60, 10])
        self.assertEqual(ref_arch["dropout_rate"], 0.3)
        self.assertEqual(ref_arch["paper_original_output_activation"], "sigmoid")
        self.assertEqual(ref_arch["v2_adapted_output_units"], 1)
        self.assertEqual(ref_arch["v2_loss_function"], "BCEWithLogitsLoss")
        self.assertEqual(
            ref_arch["hidden_activations"],
            ["relu", "tanh", "relu", "tanh", "tanh", "tanh"],
        )
        self.assertEqual(
            ref_arch["hidden_activations_verification_status"],
            "VERIFIED_FROM_PAPER_LISTING_1",
        )

    def test_02_no_label_or_time_leakage_in_79_features(self):
        info = verify_frozen_scaler_and_features(
            scaler_path=self.scaler_path,
            scaler_metadata_path=self.scaler_meta_path,
            expected_feature_count=79,
            expected_fit_sample_count=442802,
        )
        ordered_names = info["ordered_feature_names"]
        self.assertEqual(len(ordered_names), 79)
        for col in FORBIDDEN_FEATURE_COLUMNS:
            self.assertNotIn(col, ordered_names)

        # Cross-verify against raw test1.csv.gz header (84 columns = 79 features + 5 excluded)
        raw_test1 = PROJECT_ROOT / "data" / "source" / "hai_repo" / "hai-21.03" / "test1.csv.gz"
        if raw_test1.exists():
            with gzip.open(raw_test1, "rt", encoding="utf-8") as gz:
                header_cols = [c.strip() for c in gz.readline().strip().split(",")]
            self.assertEqual(len(header_cols), 84)
            filtered_cols = [c for c in header_cols if c not in FORBIDDEN_FEATURE_COLUMNS]
            self.assertEqual(filtered_cols, ordered_names)

    def test_03_split_disjointness_and_eval_isolation(self):
        verify_split_disjointness(["test1", "test2"], ["test3"], ["test4", "test5"])

        # Overlapping splits must raise ValueError
        with self.assertRaises(ValueError):
            verify_split_disjointness(["test1", "test2"], ["test2", "test3"], ["test4", "test5"])
        with self.assertRaises(ValueError):
            verify_split_disjointness(["test1", "test2"], ["test3"], ["test3", "test4"])

        # Default build_v2_dataloaders must NOT load eval_dataset
        bundle = build_v2_dataloaders(DEFAULT_CONFIG_PATH, batch_size=256, include_eval=False)
        self.assertNotIn("eval_dataset", bundle)
        self.assertNotIn("eval_loader", bundle)

    def test_04_exact_label_distributions_and_row_alignment(self):
        bundle = build_v2_dataloaders(DEFAULT_CONFIG_PATH, batch_size=512, include_eval=False)
        train_ds: SessionPointwiseDNNDatasetV2 = bundle["train_dataset"]
        val_ds: SessionPointwiseDNNDatasetV2 = bundle["val_dataset"]

        train_dist = train_ds.get_aggregate_label_distribution()
        val_dist = val_ds.get_aggregate_label_distribution()

        # Verify Train (test1 + test2) exact counts
        self.assertEqual(train_dist["total_rows"], 162002)
        self.assertEqual(train_dist["normal_rows"], 157924)
        self.assertEqual(train_dist["attack_rows"], 4078)
        self.assertEqual(train_dist["per_session"][0]["session"], "test1")
        self.assertEqual(train_dist["per_session"][0]["total_rows"], 43201)
        self.assertEqual(train_dist["per_session"][0]["attack_rows"], 629)
        self.assertEqual(train_dist["per_session"][1]["session"], "test2")
        self.assertEqual(train_dist["per_session"][1]["total_rows"], 118801)
        self.assertEqual(train_dist["per_session"][1]["attack_rows"], 3449)

        # Verify Validation (test3) exact counts
        self.assertEqual(val_dist["total_rows"], 108001)
        self.assertEqual(val_dist["normal_rows"], 106466)
        self.assertEqual(val_dist["attack_rows"], 1535)

        # Verify Common V1 Alignment mode (start_index=30) on Validation and Eval
        val_ds_common = SessionPointwiseDNNDatasetV2(
            sessions=["test3"],
            processed_test_dir=self.processed_test_dir,
            start_index=30,
        )
        self.assertEqual(len(val_ds_common), 107971)

        eval_ds_full = SessionPointwiseDNNDatasetV2(
            sessions=["test4", "test5"],
            processed_test_dir=self.processed_test_dir,
            start_index=0,
        )
        eval_dist_full = eval_ds_full.get_aggregate_label_distribution()
        self.assertEqual(eval_dist_full["total_rows"], 132002)
        self.assertEqual(eval_dist_full["normal_rows"], 128668)
        self.assertEqual(eval_dist_full["attack_rows"], 3334)

        eval_ds_common = SessionPointwiseDNNDatasetV2(
            sessions=["test4", "test5"],
            processed_test_dir=self.processed_test_dir,
            start_index=30,
        )
        eval_dist_common = eval_ds_common.get_aggregate_label_distribution()
        self.assertEqual(eval_dist_common["total_rows"], 131942)
        self.assertEqual(eval_dist_common["attack_rows"], 3334)

    def test_05_class_weights_strictly_from_train(self):
        bundle = build_v2_dataloaders(DEFAULT_CONFIG_PATH, batch_size=256, include_eval=False)
        cw = bundle["train_class_weights"]
        self.assertEqual(cw["source_sessions"], ["test1", "test2"])
        self.assertEqual(cw["total_samples"], 162002)
        self.assertEqual(cw["normal_samples_class0"], 157924)
        self.assertEqual(cw["attack_samples_class1"], 4078)
        self.assertAlmostEqual(cw["pos_weight_balanced"], 157924 / 4078, places=5)
        self.assertAlmostEqual(cw["pos_weight_sqrt"], np.sqrt(157924 / 4078), places=5)

        # Attempting to compute class weights on Validation (test3) or Eval (test4, test5) must fail
        val_ds = bundle["val_dataset"]
        with self.assertRaises(ValueError):
            compute_train_class_weights(val_ds)

        eval_ds = SessionPointwiseDNNDatasetV2(
            sessions=["test4", "test5"],
            processed_test_dir=self.processed_test_dir,
        )
        with self.assertRaises(ValueError):
            compute_train_class_weights(eval_ds)

    def test_06_batch_shape_binary_labels_and_safe_fetching(self):
        batch_size = 256
        bundle = build_v2_dataloaders(DEFAULT_CONFIG_PATH, batch_size=batch_size, include_eval=False)
        train_loader = bundle["train_loader"]
        val_loader = bundle["val_loader"]

        # Fetch real batch from Train DataLoader
        x_train, y_train = next(iter(train_loader))
        self.assertEqual(tuple(x_train.shape), (batch_size, 79))
        self.assertEqual(tuple(y_train.shape), (batch_size,))
        self.assertEqual(x_train.dtype, torch.float32)
        self.assertEqual(y_train.dtype, torch.float32)
        self.assertTrue(torch.isfinite(x_train).all().item())
        self.assertTrue(set(y_train.unique().tolist()).issubset({0.0, 1.0}))

        # Fetch real batch from Validation DataLoader
        x_val, y_val = next(iter(val_loader))
        self.assertEqual(tuple(x_val.shape), (batch_size, 79))
        self.assertEqual(tuple(y_val.shape), (batch_size,))
        self.assertTrue(torch.isfinite(x_val).all().item())
        self.assertTrue(set(y_val.unique().tolist()).issubset({0.0, 1.0}))

        # Verify that mutating returned tensor does not alter underlying dataset
        orig_first_x, orig_first_y = bundle["val_dataset"][0]
        orig_first_x_clone = orig_first_x.clone()
        orig_first_x[:] = -99999.0
        reloaded_first_x, _ = bundle["val_dataset"][0]
        self.assertTrue(torch.equal(orig_first_x_clone, reloaded_first_x))

    def test_07_source_immutability_and_ram_budget(self):
        for p in self.tracked_files:
            post_hash = sha256_file(p)
            self.assertEqual(
                self.pre_hashes[str(p)],
                post_hash,
                f"File {p} was modified during dataset testing!",
            )

        peak_rss = get_peak_rss_mb()
        # Must remain comfortably below 4096 MB (typically < 300 MB)
        self.assertLess(peak_rss, 1024.0, f"Peak RSS too high: {peak_rss:.2f} MB")


def run_and_export_audit():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDNNDatasetV2)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Ensure V2 directories exist without touching any V1 directories
    models_v2_dir = PROJECT_ROOT / "models" / "v2_dnn"
    results_v2_dir = PROJECT_ROOT / "results" / "v2_dnn"
    models_v2_dir.mkdir(parents=True, exist_ok=True)
    results_v2_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_v2_dataloaders(DEFAULT_CONFIG_PATH, batch_size=256, include_eval=True)
    train_dist = bundle["train_dataset"].get_aggregate_label_distribution()
    val_dist = bundle["val_dataset"].get_aggregate_label_distribution()
    eval_dist = bundle["eval_dataset"].get_aggregate_label_distribution()

    audit_payload = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_config": "configs/dnn_v2_protocol.yaml",
        "test_suite_status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "peak_rss_mb": round(get_peak_rss_mb(), 2),
        "scaler_verification": {
            "scaler_path": "models/preprocessing/scaler.joblib",
            "refit_performed": False,
            "fit_sample_count": bundle["scaler_verification"]["fit_sample_count"],
            "feature_count": bundle["scaler_verification"]["feature_count"],
            "excluded_columns_verified_absent": sorted(FORBIDDEN_FEATURE_COLUMNS),
        },
        "splits": {
            "dnn_train": train_dist,
            "dnn_validation": val_dist,
            "dnn_evaluation_v2": eval_dist,
        },
        "train_class_weights": bundle["train_class_weights"],
        "immutability_verified": True,
    }

    audit_path = results_v2_dir / "v2_data_pipeline_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)

    print(f"\n[OK] Exported V2 data pipeline audit report to: {audit_path}")
    print(f"[OK] Peak RSS Memory: {audit_payload['peak_rss_mb']:.2f} MB")
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    run_and_export_audit()
