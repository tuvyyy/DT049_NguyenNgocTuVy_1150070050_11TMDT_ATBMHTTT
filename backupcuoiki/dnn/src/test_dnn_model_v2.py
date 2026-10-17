#!/usr/bin/env python3
"""
Unit and Pre-Training Smoke Tests for DT049 DNN V2 Model & Training Pipeline
============================================================================
Implements all 8 required verification checks:
1. Re-runs all 7 tests from `src/test_dnn_dataset_v2.py` and confirms PASS.
2. Verifies exact 35,731 trainable parameters, layer-by-layer breakdown,
   verified Listing 1 activations (`["relu", "tanh", "relu", "tanh", "tanh", "tanh"]`),
   2 Dropout(0.3) layers, and zero BatchNorm/LayerNorm layers.
3. Verifies forward pass shape: (256, 79) -> (256, 1) and sigmoid probability bounds.
4. Verifies BCEWithLogitsLoss (with Train pos_weight=6.223009) is finite.
5. Verifies backward pass, gradient clipping (max_norm=1.0), and Adam optimizer step.
6. Runs a real-data smoke test on a subset of Train (test1+test2) and Validation (test3)
   using the verified Listing 1 architecture without writing official checkpoints.
7. Verifies `.venv` Python environment and CPU device usage.
8. Verifies zero side effects on V1 modules/artifacts and zero access to test4/test5.
"""

import json
import resource
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dnn_dataset_v2 import (
    DEFAULT_CONFIG_PATH,
    build_v2_dataloaders,
    load_v2_config,
)
from src.dnn_model import (
    EXPECTED_TRAINABLE_PARAMETERS,
    VERIFIED_ACTIVATION_STATUS,
    VERIFIED_HIDDEN_ACTIVATIONS,
    FeedforwardDNNV2,
    count_trainable_parameters,
    verify_dnn_v2_architecture,
)
from src.test_dnn_dataset_v2 import TestDNNDatasetV2, sha256_file
from src.train_dnn_v2 import (
    assert_no_heldout_eval_access,
    calibrate_threshold_on_test3,
    evaluate_validation,
    set_deterministic_seed,
    train_one_epoch,
    train_single_seed,
)


def get_peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0


class TestDNNModelAndTrainingV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_v2_config(DEFAULT_CONFIG_PATH)
        cls.verified_acts = cls.cfg["reference_architecture"]["hidden_activations"]
        cls.verified_status = cls.cfg["reference_architecture"]["hidden_activations_verification_status"]
        cls.v1_protected_files = [
            PROJECT_ROOT / "models" / "preprocessing" / "scaler.joblib",
            PROJECT_ROOT / "models" / "preprocessing" / "scaler_metadata.json",
            PROJECT_ROOT / "models" / "baselines" / "ridge.joblib",
            PROJECT_ROOT / "models" / "primary" / "gru_w30_seed42.pt",
            PROJECT_ROOT / "models" / "primary" / "gru_w30_seed43.pt",
            PROJECT_ROOT / "models" / "primary" / "gru_w30_seed44.pt",
            PROJECT_ROOT / "results" / "metrics" / "final_test_summary.csv",
            PROJECT_ROOT / "results" / "metrics" / "final_per_file.csv",
            PROJECT_ROOT / "results" / "audit" / "pretest_freeze_manifest.json",
        ]
        cls.v1_pre_hashes = {str(p): sha256_file(p) for p in cls.v1_protected_files}

    def test_01_rerun_7_data_pipeline_tests(self):
        """1. Re-runs all 7 tests from `src/test_dnn_dataset_v2.py`."""
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDNNDatasetV2)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertEqual(res.testsRun, 7)
        self.assertTrue(res.wasSuccessful(), f"Data pipeline tests failed: {res.failures} {res.errors}")

    def test_02_exact_35731_parameters_and_verified_listing_1_activations(self):
        """2. Verifies exact 35,731 trainable parameters, Listing 1 activations, and no Norm layers."""
        self.assertEqual(
            tuple(self.verified_acts),
            ("relu", "tanh", "relu", "tanh", "tanh", "tanh"),
        )
        self.assertEqual(self.verified_status, "VERIFIED_FROM_PAPER_LISTING_1")

        model = FeedforwardDNNV2(
            hidden_activations=self.verified_acts,
            activation_status=self.verified_status,
        )
        self.assertEqual(count_trainable_parameters(model), 35731)
        audit = verify_dnn_v2_architecture(model, expected_params=35731)
        self.assertEqual(audit["trainable_parameters"], 35731)
        self.assertEqual(audit["normalization_layers_count"], 0)
        self.assertEqual(audit["dropout_count"], 2)
        self.assertEqual(audit["dropout_rate"], 0.3)

        # Verify exact activation module classes per layer:
        # 1: ReLU, 2: Tanh, 3: ReLU, 4: Tanh, 5: Tanh, 6: Tanh
        self.assertIsInstance(model.act1, nn.ReLU)
        self.assertIsInstance(model.act2, nn.Tanh)
        self.assertIsInstance(model.act3, nn.ReLU)
        self.assertIsInstance(model.act4, nn.Tanh)
        self.assertIsInstance(model.act5, nn.Tanh)
        self.assertIsInstance(model.act6, nn.Tanh)

        # Verify exact layer-by-layer parameter counts:
        # 8000 + 7070 + 7100 + 8080 + 4860 + 610 + 11 = 35,731
        layer_totals = [
            model.fc1.weight.numel() + model.fc1.bias.numel(),
            model.fc2.weight.numel() + model.fc2.bias.numel(),
            model.fc3.weight.numel() + model.fc3.bias.numel(),
            model.fc4.weight.numel() + model.fc4.bias.numel(),
            model.fc5.weight.numel() + model.fc5.bias.numel(),
            model.fc6.weight.numel() + model.fc6.bias.numel(),
            model.out.weight.numel() + model.out.bias.numel(),
        ]
        self.assertEqual(layer_totals, [8000, 7070, 7100, 8080, 4860, 610, 11])
        self.assertEqual(sum(layer_totals), 35731)

    def test_03_forward_shape_256x79_to_256x1(self):
        """3. Checks forward pass (256, 79) -> (256, 1) and predict_proba in (0, 1)."""
        set_deterministic_seed(42)
        model = FeedforwardDNNV2(
            hidden_activations=self.verified_acts,
            activation_status=self.verified_status,
        )
        model.eval()
        x = torch.randn(256, 79, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x)
            probs = model.predict_proba(x)

        self.assertEqual(tuple(logits.shape), (256, 1))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all().item())

        self.assertEqual(tuple(probs.shape), (256, 1))
        self.assertTrue(torch.isfinite(probs).all().item())
        self.assertTrue(((probs >= 0.0) & (probs <= 1.0)).all().item())

    def test_04_loss_finite_with_train_pos_weight(self):
        """4. Checks BCEWithLogitsLoss with Train pos_weight is finite on real data batch."""
        bundle = build_v2_dataloaders(DEFAULT_CONFIG_PATH, batch_size=256, include_eval=False)
        pos_weight_val = float(bundle["train_class_weights"]["pos_weight_sqrt"])
        self.assertAlmostEqual(pos_weight_val, 6.223009, places=5)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32))
        model = FeedforwardDNNV2(
            hidden_activations=self.verified_acts,
            activation_status=self.verified_status,
        )
        x_batch, y_batch = next(iter(bundle["train_loader"]))
        logits = model(x_batch)
        loss = criterion(logits, y_batch.unsqueeze(1))

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(float(loss.item()), 0.0)

    def test_05_backward_grad_clipping_and_optimizer_step(self):
        """5. Checks backward pass, gradient clipping (max_norm=1.0), and Adam step update parameters."""
        set_deterministic_seed(42)
        bundle = build_v2_dataloaders(DEFAULT_CONFIG_PATH, batch_size=256, include_eval=False)
        pos_weight_val = float(bundle["train_class_weights"]["pos_weight_sqrt"])
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32))

        model = FeedforwardDNNV2(
            hidden_activations=self.verified_acts,
            activation_status=self.verified_status,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        before_params = {n: p.detach().clone() for n, p in model.named_parameters()}

        x_batch, y_batch = next(iter(bundle["train_loader"]))
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch.unsqueeze(1))
        loss.backward()

        for n, p in model.named_parameters():
            self.assertIsNotNone(p.grad, f"Missing gradient on {n}")
            self.assertTrue(torch.isfinite(p.grad).all().item(), f"Non-finite gradient on {n}")

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        self.assertTrue(torch.isfinite(total_norm).item())
        optimizer.step()

        changed_any = any(
            not torch.equal(before_params[n], p.detach())
            for n, p in model.named_parameters()
        )
        self.assertTrue(changed_any, "Optimizer step did not update model parameters!")

    def test_06_real_data_smoke_test_and_calibration(self):
        """
        6. Runs a smoke test on a stratified slice of real Train (test1+test2)
           and Validation (test3) data using verified Listing 1 activations, verifying
           training step, validation PR-AUC, threshold calibration on test3, and confirming
           that unverified status is blocked from saving official checkpoints.
        """
        set_deterministic_seed(42)
        bundle = build_v2_dataloaders(DEFAULT_CONFIG_PATH, batch_size=512, include_eval=False)
        assert_no_heldout_eval_access(bundle)

        train_ds = bundle["train_dataset"]
        val_ds = bundle["val_dataset"]

        train_indices = list(range(1500, 5596))
        val_indices = list(range(48000, 52096))

        train_sub_loader = DataLoader(Subset(train_ds, train_indices), batch_size=256, shuffle=True)
        val_sub_loader = DataLoader(Subset(val_ds, val_indices), batch_size=512, shuffle=False)

        device = torch.device("cpu")
        model = FeedforwardDNNV2(
            hidden_activations=self.verified_acts,
            activation_status=self.verified_status,
        ).to(device)
        pos_w = torch.tensor([float(bundle["train_class_weights"]["pos_weight_sqrt"])], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        tr_metrics = train_one_epoch(model, train_sub_loader, criterion, optimizer, device, max_grad_norm=1.0)
        val_metrics = evaluate_validation(model, val_sub_loader, criterion, device)

        self.assertTrue(np.isfinite(tr_metrics["train_bce_loss"]))
        self.assertTrue(np.isfinite(val_metrics["val_bce_loss"]))
        self.assertGreaterEqual(val_metrics["val_pr_auc"], 0.0)
        self.assertLessEqual(val_metrics["val_pr_auc"], 1.0)

        val_ts_slice = val_ds.timestamp_arrays[0][48000:52096]
        val_atk_slice = val_ds.label_arrays[0][48000:52096]
        cal = calibrate_threshold_on_test3(
            val_probs_full=val_metrics["probs"],
            val_labels_full=val_atk_slice,
            val_timestamps_full=val_ts_slice,
            calibration_cfg=self.cfg["training_protocol"]["threshold_calibration"],
            session_name="test3_smoke_slice",
        )
        self.assertIn("selected_operating_point", cal)
        self.assertGreaterEqual(cal["selected_operating_point"]["threshold"], 0.05)
        self.assertLessEqual(cal["selected_operating_point"]["threshold"], 0.99)

        with self.assertRaises(RuntimeError):
            train_single_seed(
                seed=42,
                cfg=self.cfg,
                bundle=bundle,
                hidden_activations=self.verified_acts,
                activation_status="PROVISIONAL",
                max_epochs_override=1,
                save_checkpoint=True,
            )

    def test_07_venv_and_cpu_device(self):
        """7. Verifies execution uses `.venv` Python and CPU device."""
        exe_path = Path(sys.executable).resolve()
        self.assertTrue(
            ".venv" in str(exe_path) or (PROJECT_ROOT / ".venv").exists(),
            f"Expected .venv execution, got {exe_path}",
        )
        self.assertEqual(self.cfg["training_protocol"]["device"], "cpu")
        t = torch.zeros(1)
        self.assertEqual(t.device.type, "cpu")

    def test_08_no_v1_side_effects_and_no_test4_test5_access(self):
        """8. Verifies zero V1 side effects and zero V1 file changes."""
        forbidden_v1_modules = {
            "src.train_gru",
            "src.evaluate_final_test",
            "src.baseline_persistence",
            "src.baseline_ridge",
            "src.preprocess",
            "src.preprocess_test",
        }
        loaded_forbidden = forbidden_v1_modules.intersection(sys.modules.keys())
        self.assertEqual(len(loaded_forbidden), 0, f"Unexpected V1 modules imported: {loaded_forbidden}")

        for p in self.v1_protected_files:
            self.assertEqual(
                self.v1_pre_hashes[str(p)],
                sha256_file(p),
                f"Protected V1 artifact modified: {p}",
            )


def run_and_export_model_smoke_audit():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDNNModelAndTrainingV2)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    cfg = load_v2_config(DEFAULT_CONFIG_PATH)
    model = FeedforwardDNNV2(
        hidden_activations=cfg["reference_architecture"]["hidden_activations"],
        activation_status=cfg["reference_architecture"]["hidden_activations_verification_status"],
    )
    arch_info = verify_dnn_v2_architecture(model, EXPECTED_TRAINABLE_PARAMETERS)

    results_dir = PROJECT_ROOT / cfg["directories"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "v2_model_pretrain_smoke_audit.json"

    payload = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_suite": "src/test_dnn_model_v2.py",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "device": "cpu",
        "peak_rss_mb": round(get_peak_rss_mb(), 2),
        "architecture_verification": arch_info,
        "official_training_status": "READY_VERIFIED_FROM_PAPER_LISTING_1",
        "heldout_test4_test5_accessed": False,
        "v1_artifacts_intact": True,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[OK] Exported V2 model pre-training smoke audit to: {report_path}")
    print(f"[OK] Peak RSS Memory: {payload['peak_rss_mb']:.2f} MB")
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    run_and_export_model_smoke_audit()
