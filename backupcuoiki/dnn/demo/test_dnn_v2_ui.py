"""UI integration tests; no training, inference or artifact writes.

Run: python -m unittest discover -s demo -p test_dnn_v2_ui.py -v
"""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df
from streamlit.testing.v1 import AppTest

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
sys.path.insert(0, str(DEMO))
from dnn_v2 import BASE_MODELS, EVENT_MODELS, METRICS, PRIMARY_MODELS, TABS


class TestDNNV2UI(unittest.TestCase):
    def setUp(self):
        self.summary = pd.read_csv(ROOT / "results/v2_dnn/v2_heldout_test4_test5_summary.csv").set_index("model_id")
        self.events = pd.read_csv(ROOT / "results/v2_dnn/v2_heldout_17_attack_events.csv").set_index("attack_event_id")
        self.app = AppTest.from_file(str(DEMO / "app.py"), default_timeout=30).run()
        self.app.button(key="nav_dnn").click().run()
        self.assertFalse(self.app.exception)

    def tab(self, index):
        self.app.radio(key="dnn_tab").set_value(TABS[index]).run()
        self.assertFalse(self.app.exception)

    def html(self):
        return "\n".join(element.value for element in self.app.markdown)

    def test_five_tabs_render_only_selected_content_and_source_kpis(self):
        self.assertEqual(self.app.radio(key="dnn_tab").options, TABS)
        self.assertEqual(len(self.app.dataframe), 0)
        for model in ("dnn_v2_seed42_tau_star", "gru_seed42"):
            row = self.summary.loc[model]
            for value in (f'{int(row.tp_events)}/17', f'{row.event_f1:.6f}', f'{row.normal_fa_events_per_hour:.4f}', f'{row.median_delay_sec:.0f} giây'):
                self.assertIn(value, self.html())
        for index in (1, 2, 3, 4, 0):
            self.tab(index)
        self.assertFalse(self.app.selectbox)
        self.assertFalse(self.app.get("vega_lite_chart"))

    def test_four_metrics_chart_data_all_seeds_and_hidden_tables(self):
        self.tab(1)
        for all_seeds, expected_ids in ((False, BASE_MODELS), (True, PRIMARY_MODELS)):
            self.app.checkbox(key="dnn_all_seeds").set_value(all_seeds).run()
            for label, column in METRICS.items():
                self.app.selectbox(key="dnn_metric").select(label).run()
                chart = self.app.get("vega_lite_chart")[0].proto
                plotted = convert_arrow_bytes_to_pandas_df(chart.datasets[0].data.data).set_index("model_id")
                self.assertEqual(list(plotted.index), expected_ids)
                self.assertTrue(plotted.evaluable_attacks.eq(17).all())
                pd.testing.assert_series_equal(plotted[column], self.summary.loc[expected_ids, column])
                self.assertIn(column, json.loads(chart.spec)["layer"][0]["encoding"]["x"]["field"])
        self.assertFalse(self.app.expander[0].proto.expanded)
        for option in self.app.selectbox(key="dnn_table_view").options:
            self.app.selectbox(key="dnn_table_view").select(option).run()
            self.assertFalse(self.app.exception)
            self.assertEqual(len(self.app.dataframe), 1)

    def test_event_pagination_default_detail_and_filter_reset(self):
        self.tab(2)
        self.assertEqual(self.app.selectbox(key="dnn_event_selected").value, "test4_atk_01")
        self.assertIn("2020-07-28 12:43:03", self.html())
        self.assertIn("22.0 giây", self.html())
        self.assertTrue(self.app.button(key="dnn_prev").disabled)
        self.app.selectbox(key="dnn_session").select("Tất cả").run()
        seen = []
        for page in range(3):
            preview = self.app.dataframe[0].value
            self.assertLessEqual(len(preview), 6)
            seen.extend(preview["Sự kiện"].tolist())
            if page < 2:
                self.app.button(key="dnn_next").click().run()
                self.assertFalse(self.app.exception)
        self.assertEqual(seen, list(self.events.index))
        self.assertTrue(self.app.button(key="dnn_next").disabled)
        self.app.button(key="dnn_prev").click().run()
        self.assertEqual(self.app.session_state["dnn_event_page"], 1)
        self.app.selectbox(key="dnn_session").select("test4").run()
        self.assertEqual(self.app.session_state["dnn_event_page"], 0)
        self.assertEqual(self.app.selectbox(key="dnn_event_selected").value, "test4_atk_01")

    def test_event_models_source_values_status_process_and_empty_result(self):
        self.tab(2)
        for label, prefix in EVENT_MODELS.items():
            self.app.selectbox(key="dnn_event_model").select(label).run()
            preview = self.app.dataframe[0].value
            for _, row in preview.iterrows():
                original = self.events.loc[row["Sự kiện"]]
                self.assertEqual(row["Kết quả"] == "Phát hiện", bool(original[prefix + "_detected"]))
                if pd.notna(row["Delay (s)"]):
                    self.assertEqual(row["Delay (s)"], original[prefix + "_delay_seconds"])
        self.app.selectbox(key="dnn_event_model").select("DNN seed 42 · τ*").run()
        self.app.selectbox(key="dnn_status").select("Bỏ sót").run()
        self.assertFalse(self.app.dataframe)
        self.assertTrue(self.app.info)
        self.app.selectbox(key="dnn_status").select("Tất cả").run()
        for process in self.app.selectbox(key="dnn_process").options:
            self.app.selectbox(key="dnn_process").select(process).run()
            self.assertFalse(self.app.exception)
            if self.app.dataframe:
                for event_id in self.app.dataframe[0].value["Sự kiện"]:
                    row = self.events.loc[event_id]
                    if process in ("P1", "P2", "P3"):
                        self.assertTrue(row["attack_" + process])
                    elif process in ("Đơn phân hệ", "Đa phân hệ"):
                        self.assertEqual(bool(row.is_multiprocess), process == "Đa phân hệ")

    def test_stability_figures_and_reference_access(self):
        self.tab(3)
        for seed in (42, 43, 44):
            row = self.summary.loc[f"dnn_v2_seed{seed}_tau_star"]
            for value in (f'{row.event_recall:.2%}', f'{row.event_f1:.6f}', f'{row.normal_fa_events_per_hour:.4f}', f'{row.threshold:.6f}'):
                self.assertIn(value, self.html())
        for option in self.app.selectbox(key="dnn_stability_view").options[1:]:
            self.app.selectbox(key="dnn_stability_view").select(option).run()
            self.assertFalse(self.app.exception)
            images = self.app.get("image") or self.app.get("imgs")
            self.assertEqual(len(images), 1)
            self.assertFalse(self.app.dataframe)
        self.tab(4)
        self.assertFalse(self.app.expander[0].proto.expanded)
        self.assertIn("norm_layer", self.html())
        self.assertIn("max_point_fpr_cap", self.html())
        for option in self.app.selectbox(key="dnn_document").options[1:]:
            self.app.selectbox(key="dnn_document").select(option).run()
            self.assertFalse(self.app.exception)
            self.assertEqual(len(self.app.json), 1)

    def test_other_seven_pages_and_scoped_styles(self):
        for key in ("overview", "models", "attacks", "errors", "threshold", "ablation", "reproducibility"):
            self.app.button(key="nav_" + key).click().run()
            self.assertFalse(self.app.exception)
            self.assertNotIn("dnn_tab", [radio.key for radio in self.app.radio])
            self.assertNotIn("body:has(.st-key-dnn_v2)", self.html())

    def test_rejects_full_v1_benchmark_in_v2(self):
        incorrect = pd.DataFrame({"evaluable_attacks": [50], "eval_split": ["test1-test5"]})
        with patch("dnn_v2._read", return_value=incorrect):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.error), 1)
        self.assertNotIn('class="v2-kpi"', self.html())


if __name__ == "__main__":
    unittest.main(verbosity=2)
