"""Overview presentation using the frozen benchmark artifacts."""

from html import escape
from pathlib import Path

import streamlit as st

from ui_utils import display_figure, load_csv, render_download_button


def _html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def _navigate(destination):
    st.session_state["navigation"] = destination


def _kpi(label, value, unit, note, context, accent=False):
    _html(
        f'<article class="ov-kpi {"ov-kpi-accent" if accent else ""}">'
        f'<div class="ov-label">{escape(label)}</div>'
        f'<div class="ov-kpi-value">{escape(value)}<span>{escape(unit)}</span></div>'
        f'<div class="ov-kpi-note">{escape(note)}</div>'
        f'<div class="ov-kpi-context">{escape(context)}</div></article>'
    )


def render_overview():
    css = Path(__file__).with_name("overview.css").read_text(encoding="utf-8")
    _html(f"<style>{css}</style>")
    _html(
        '<div class="ov-topline"><div><span class="ov-brand">DT049</span>'
        '<span class="ov-top-divider">/</span>Research overview</div>'
        '<span class="ov-status"><i></i> Frozen final test</span></div>'
    )

    summary = load_csv("results/metrics/final_test_summary.csv")
    required = {
        "model_id", "event_recall", "event_f1", "evaluable_attacks", "tp_detected",
        "fn_missed", "normal_fa_events_per_hour", "delay_median_sec", "total_normal_hours",
    }
    model_ids = {"persistence", "ridge", "gru_seed42", "gru_seed43", "gru_seed44"}
    if summary is None:
        st.info("Add the official benchmark summary to view the overview.")
        return
    if not required.issubset(summary.columns) or not model_ids.issubset(set(summary["model_id"])):
        st.warning("The benchmark summary is missing required models or metrics.")
        return

    rows = summary.set_index("model_id")
    gru, ridge = rows.loc["gru_seed42"], rows.loc["ridge"]
    seeds = rows.loc[["gru_seed42", "gru_seed43", "gru_seed44"]]
    attacks, detected, missed = (int(gru[key]) for key in ("evaluable_attacks", "tp_detected", "fn_missed"))
    mean_fa = seeds["normal_fa_events_per_hour"].mean()
    fa_reduction = (1 - mean_fa / ridge["normal_fa_events_per_hour"]) * 100
    recall_gain = (gru["event_recall"] - ridge["event_recall"]) * 100
    tiles = "".join(
        f'<span class="ov-attack-tile {"detected" if i < detected else "missed"}"></span>'
        for i in range(attacks)
    )

    _html(
        '<section class="ov-hero"><div class="ov-hero-copy">'
        '<div class="ov-eyebrow">INDUSTRIAL SECURITY · HAI 21.03</div>'
        '<h1>See the signals.<br><span>Understand the threats.</span></h1>'
        '<p>A clear view of multivariate anomaly detection. Explore how a compact GRU '
        'identifies physical attacks across five unseen industrial test sessions.</p>'
        '<div class="ov-chips"><span>79 telemetry channels</span><span>5 test sessions</span>'
        '<span>Q<sub>99.9</sub> threshold</span></div></div>'
        '<div class="ov-hero-result"><div class="ov-hero-result-label">ATTACK DETECTION · GRU SEED 42</div>'
        f'<div class="ov-hero-number">{detected}<span>/ {attacks}</span></div>'
        '<div class="ov-hero-result-subtitle">physical attacks detected</div>'
        f'<div class="ov-attack-grid" aria-hidden="true">{tiles}</div>'
        f'<div class="ov-legend"><span><i class="ov-dot teal"></i>{detected} detected</span>'
        f'<span><i class="ov-dot muted"></i>{missed} missed</span></div>'
        '<div class="ov-aggregate-note">Aggregate count · each tile represents one attack</div>'
        '</div></section>'
    )

    _html('<div class="ov-section-heading"><h2>Performance at a glance</h2><span>Official benchmark · Q<sub>99.9</sub></span></div>')
    columns = st.columns(4, gap="small")
    with columns[0]:
        _kpi("Attack recall", f'{gru["event_recall"] * 100:.1f}', "%",
             f"+{recall_gain:.0f} percentage points vs Ridge", "GRU · canonical seed 42", True)
    with columns[1]:
        _kpi("Normal false alarms", f"{mean_fa:.2f}", " / h",
             f"{fa_reduction:.1f}% fewer events vs Ridge", "GRU · mean of seeds 42, 43, 44")
    with columns[2]:
        _kpi("Median detection delay", f'{gru["delay_median_sec"]:.1f}', " s",
             "Time to first matched alarm", "GRU · canonical seed 42")
    with columns[3]:
        _kpi("Event F1", f'{gru["event_f1"]:.4f}', "",
             "Strict one-to-one event matching", "GRU · canonical seed 42")

    left, right = st.columns([1.65, 1], gap="medium")
    with left:
        with st.container(border=False, key="overview_benchmark"):
            _html('<div class="ov-panel-eyebrow">THE BENCHMARK</div><h2 class="ov-panel-title">Three models. One protocol.</h2>')
            metrics = {
                "Attack recall": ("event_recall", True, "Higher is better", "%"),
                "Event F1": ("event_f1", False, "Higher is better", ""),
                "Normal false alarms": ("normal_fa_events_per_hour", False, "Lower is better", " ev/h"),
                "Median detection delay": ("delay_median_sec", False, "Lower is better", " s"),
            }
            selected = st.selectbox("Compare a benchmark metric", list(metrics),
                                    label_visibility="collapsed", key="overview_metric")
            metric, percent, direction, unit = metrics[selected]
            _html(f'<div class="ov-chart-meta"><span>{escape(selected)}</span><span>{direction}</span></div>')
            models = [("persistence", "Persistence", "Last-value baseline", "baseline"),
                      ("ridge", "Ridge", "Linear baseline · α = 300", "ridge"),
                      ("gru_seed42", "GRU", "Primary model · seed 42", "primary")]
            maximum = 1.0 if percent or metric == "event_f1" else max(float(rows.loc[m[0], metric]) for m in models)
            bars = []
            for model_id, label, detail, color in models:
                value = float(rows.loc[model_id, metric])
                formatted = f"{value * 100:.1f}%" if percent else (f"{value:.4f}" if metric == "event_f1" else f"{value:.2f}{unit}")
                width = min(100, max(0, value / maximum * 100)) if maximum > 0 else 0
                bars.append(
                    f'<div class="ov-bar-row"><div class="ov-bar-heading"><div><strong>{label}</strong>'
                    f'<span>{detail}</span></div><b>{formatted}</b></div>'
                    f'<div class="ov-bar-track"><div class="ov-bar-fill {color}" style="width:{width:.3f}%"></div></div></div>'
                )
            _html('<div class="ov-bars">' + "".join(bars) + '</div>')
            st.caption("GRU bars use canonical seed 42. Recall and F1 use a 0–1 scale; rate and delay scale to the largest value.")
            st.button("Explore model comparison →", key="overview_compare", on_click=_navigate,
                      args=("Model Comparison",), width="stretch")

    with right:
        with st.container(border=False, key="overview_scope"):
            _html(
                '<div class="ov-panel-eyebrow">EVALUATION SCOPE</div><h2 class="ov-panel-title">Built on real process signals.</h2>'
                '<div class="ov-scope-grid">'
                f'<div><strong>{attacks}</strong><span>attack events</span></div>'
                '<div><strong>5</strong><span>unseen sessions</span></div>'
                '<div><strong>79</strong><span>sensor channels</span></div>'
                f'<div><strong>{gru["total_normal_hours"]:.2f}<small> h</small></strong><span>normal exposure</span></div></div>'
                '<div class="ov-scope-foot">HAI 21.03 · test1–test5<br>Common evaluation timeline: t ≥ 30</div>'
            )
            st.button("Inspect attack events →", key="overview_attacks", on_click=_navigate,
                      args=("Attack Explorer",), width="stretch")

    _html('<div class="ov-section-heading"><h2>What the results tell us</h2><span>Beyond a single score</span></div>')
    insights = st.columns(3, gap="small")
    with insights[0]:
        _html(
            '<article class="ov-insight"><div class="ov-insight-index">01 / DETECTION</div>'
            '<h3>Temporal context helps.</h3>'
            f'<p>GRU detects {detected} of {attacks} attacks, compared with {int(ridge["tp_detected"])} for Ridge '
            f'and {int(rows.loc["persistence", "tp_detected"])} for Persistence.</p>'
            '<span class="ov-insight-foot">30-step window · 1-layer GRU</span></article>'
        )
    with insights[1]:
        _html(
            '<article class="ov-insight"><div class="ov-insight-index">02 / REPEATABILITY</div>'
            '<h3>Check across three seeds.</h3>'
            f'<p>Mean event F1 is <strong>{seeds["event_f1"].mean():.4f} ± {seeds["event_f1"].std(ddof=0):.4f}</strong>. '
            f'Mean median detection delay is {seeds["delay_median_sec"].mean():.2f} seconds.</p>'
            '<span class="ov-insight-foot">Seeds 42, 43, 44 · population SD</span></article>'
        )
    with insights[2]:
        _html(
            '<article class="ov-insight"><div class="ov-insight-index">03 / INTERPRETATION</div>'
            '<h3>Every extra alarm counts.</h3>'
            '<p>Strict event matching counts redundant alarm intervals as false positives. '
            'Read event F1 alongside recall and normal false alarms.</p>'
            '<span class="ov-insight-foot">Research benchmark · further validation needed</span></article>'
        )

    with st.expander("Benchmark figure & evaluation details"):
        st.markdown("**Evaluation:** Five unseen test sessions, normal-only training, and validation quantile Q₉₉.₉. "
                    "Event matching is one-to-one. The false-alarm KPI averages three GRU seeds; the comparison uses seed 42.")
        display_figure("results/figures/final_test/fig_01_final_model_comparison.png",
                       caption="Official event metrics, normal false-alarm rates, and detection latency.")
        render_download_button("results/metrics/final_test_summary.csv", "Download benchmark CSV")

    _html('<footer class="ov-footer"><span><b>DT049</b> · HAI 21.03 anomaly detection</span><span>Persistence / Ridge / GRU</span></footer>')
