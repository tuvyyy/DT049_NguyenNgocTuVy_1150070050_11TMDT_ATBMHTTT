"""Exercise research controls against saved CSVs, including filter regressions."""

from pathlib import Path
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


class TestDashboardInteractions(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / "demo/app.py"), default_timeout=30).run()

    def test_each_threshold_displays_its_own_frozen_csv_rows(self):
        self.app.button(key="nav_threshold").click().run()
        original = pd.read_csv(ROOT / "results/metrics/final_test_sensitivity_complete.csv")
        for label, point in zip(self.app.radio(key="threshold_point").options, ["Q99.5", "Q99.9", "Q99.99"]):
            self.app.radio(key="threshold_point").set_value(label).run()
            self.assertFalse(self.app.exception)
            shown = self.app.dataframe[0].value
            expected = original.loc[original.operating_point.eq(point)]
            self.assertEqual(shown["Threshold"].tolist(), expected.threshold_value.tolist())
            self.assertEqual(shown["TP"].tolist(), expected.tp_detected.tolist())
            self.assertEqual(shown["Event F1"].tolist(), expected.event_f1.tolist())

    def test_attack_filters_and_literal_search(self):
        self.app.button(key="nav_attacks").click().run()
        source = pd.read_csv(ROOT / "results/metrics/final_attack_events.csv")
        for session in sorted(source.session.unique()):
            self.app.selectbox(key="attack_session").set_value(session).run()
            self.assertFalse(self.app.exception)
            expected = source.loc[source.session.eq(session), "attack_event_id"].tolist()
            self.assertEqual(self.app.dataframe[0].value["Attack ID"].tolist(), expected)
        self.app.selectbox(key="attack_session").set_value("All").run()
        for process in sorted(source.process_set.unique()):
            self.app.selectbox(key="attack_process").set_value(process).run()
            self.assertFalse(self.app.exception)
            expected = source.loc[source.process_set.eq(process), "attack_event_id"].tolist()
            self.assertEqual(self.app.dataframe[0].value["Attack ID"].tolist(), expected)
        self.app.selectbox(key="attack_process").set_value("All").run()
        masks = {
            "All": pd.Series(True, index=source.index),
            "Detected by GRU s42": source.gru_seed42_detected,
            "Missed by GRU s42": ~source.gru_seed42_detected,
            "Detected by All Models": source.persistence_detected & source.ridge_detected & source.gru_seed42_detected,
            "Missed by All Models": ~source.persistence_detected & ~source.ridge_detected & ~source.gru_seed42_detected,
            "Ridge detected, GRU missed": source.ridge_detected & ~source.gru_seed42_detected,
        }
        for status, mask in masks.items():
            self.app.selectbox(key="attack_status").set_value(status).run()
            self.assertFalse(self.app.exception)
            self.assertEqual(self.app.dataframe[0].value["Attack ID"].tolist(), source.loc[mask, "attack_event_id"].tolist())
        self.app.selectbox(key="attack_status").set_value("All").run()
        for query in ("atk_02", "[", ".*", "not-an-event"):
            self.app.text_input(key="attack_search").set_value(query).run()
            self.assertFalse(self.app.exception)
            expected = source.loc[source.attack_event_id.str.contains(query, regex=False, case=False), "attack_event_id"].tolist()
            self.assertEqual(self.app.dataframe[0].value["Attack ID"].tolist(), expected)

    def test_benchmark_metrics_overview_choices_and_exports(self):
        self.app.button(key="overview_mode_1").click().run()
        for metric in self.app.selectbox(key="overview_metric").options:
            self.app.selectbox(key="overview_metric").set_value(metric).run()
            self.assertFalse(self.app.exception)
        self.app.button(key="nav_models").click().run()
        source = pd.read_csv(ROOT / "results/metrics/final_test_summary.csv")
        for metric in self.app.selectbox(key="comparison_metric").options:
            self.app.selectbox(key="comparison_metric").set_value(metric).run()
            self.assertFalse(self.app.exception)
            self.assertEqual(self.app.dataframe[0].value["Detected (TP)"].tolist(), source.tp_detected.tolist())
        self.app.button(key="nav_reproducibility").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.get("download_button")), 6)
        self.assertTrue(self.app.success)


if __name__ == "__main__":
    unittest.main(verbosity=2)
