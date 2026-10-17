"""Read-only V2 presentation. All scientific metrics come from frozen artifacts."""

from html import escape
from math import ceil
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from ui_utils import REPO_ROOT, load_csv, load_json

RESULTS = "results/v2_dnn/"
TABS = ["Tổng quan", "So sánh mô hình", "Sự kiện tấn công", "Độ ổn định 3 Seeds", "Phương pháp & Giới hạn"]
BASE_MODELS = ["persistence", "ridge", "gru_seed42", "dnn_v2_seed42_tau_star"]
PRIMARY_MODELS = ["persistence", "ridge"] + [f"gru_seed{s}" for s in (42, 43, 44)] + [f"dnn_v2_seed{s}_tau_star" for s in (42, 43, 44)]
MODEL_NAMES = {"persistence": "Persistence", "ridge": "Ridge"}
MODEL_NAMES.update({f"gru_seed{s}": f"GRU · seed {s}" for s in (42, 43, 44)})
MODEL_NAMES.update({f"dnn_v2_seed{s}_tau_star": f"DNN · seed {s} (τ*)" for s in (42, 43, 44)})
EVENT_MODELS = {f"DNN seed {s} · τ*": f"dnn_seed{s}_tau_star" for s in (42, 43, 44)}
EVENT_MODELS.update({f"GRU seed {s}": f"gru_seed{s}" for s in (42, 43, 44)})
EVENT_MODELS.update({"Ridge": "ridge", "Persistence": "persistence"})
EVENT_MODELS.update({f"DNN seed {s} · τ=0.50 (phụ)": f"dnn_seed{s}_tau_050" for s in (42, 43, 44)})
METRICS = {"Event Recall": "event_recall", "Event F1": "event_f1", "Normal FA/h": "normal_fa_events_per_hour", "Median Delay": "median_delay_sec"}
PAGE_SIZE = 6


def _html(text):
    st.markdown(text, unsafe_allow_html=True)


def _read(name):
    return load_csv(RESULTS + name)


def _table(frame, height=250):
    st.dataframe(frame, hide_index=True, width="stretch", height=height)


def _icon(name):
    paths = {
        "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><path d="m16 8 5-5M17 3h4v4"/>',
        "score": '<path d="M4 19V5m0 14h16M8 15l4-5 4 2 4-7"/>',
        "clock": '<circle cx="12" cy="12" r="8"/><path d="M12 7v5l3 2"/>',
        "alarm": '<path d="M6 9a6 6 0 0 1 12 0v5l2 3H4l2-3V9m4 11h4"/>',
    }
    return f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths[name]}</svg>'


def _section(title, subtitle):
    _html(f'<div class="v2-section"><div class="v2-section-title" role="heading" aria-level="2">{escape(title)}</div><span>{escape(subtitle)}</span></div>')


def _note(text, warning=False):
    _html(f'<div class="v2-note {"v2-caution" if warning else ""}"><span class="v2-note-mark" aria-hidden="true">{"!" if warning else "i"}</span><p>{escape(text)}</p></div>')


def _download(name, label):
    path = REPO_ROOT / RESULTS / name
    if path.is_file():
        st.download_button(label, path.read_bytes(), file_name=name, mime="text/csv", key="dnn_download_" + name)


def _overview(summary):
    rows = summary.set_index("model_id")
    dnn, gru = rows.loc["dnn_v2_seed42_tau_star"], rows.loc["gru_seed42"]
    specs = [
        ("Sự kiện phát hiện", "target", "Trên 17 sự kiện", lambda r: f'{int(r["tp_events"])}/{int(r["evaluable_attacks"])}'),
        ("Event F1", "score", "Event matching 1–1", lambda r: f'{r["event_f1"]:.6f}'),
        ("Median Delay", "clock", "Thời gian phát hiện", lambda r: f'{r["median_delay_sec"]:.0f} giây'),
        ("Normal FA/h", "alarm", "Báo động giả / giờ", lambda r: f'{r["normal_fa_events_per_hour"]:.4f}'),
    ]
    cards = []
    for label, icon, context, fmt in specs:
        cards.append(f'<article class="v2-kpi"><div class="v2-kpi-top"><div class="v2-label">{label}</div>'
                     f'<span class="v2-metric-icon">{_icon(icon)}</span></div>'
                     f'<div class="v2-model"><i></i>DNN seed 42</div><strong>{fmt(dnn)}</strong>'
                     f'<div class="v2-kpi-divider"></div><div class="v2-baseline"><span>GRU seed 42</span>'
                     f'<b>{fmt(gru)}</b></div><div class="v2-kpi-context">{context}</div></article>')
    # A single HTML grid controls spacing independently of Streamlit block gaps.
    _html('<section class="v2-overview"><div class="v2-section">'
          '<div class="v2-section-title" role="heading" aria-level="2">Kết quả đại diện</div>'
          '<span>Seed 42 · ngưỡng đã đóng băng</span></div>'
          '<div class="v2-kpi-grid">' + "".join(cards) + '</div>'
          '<div class="v2-readout"><div><span class="v2-eyebrow">ĐÁNH ĐỔI QUAN SÁT ĐƯỢC</span>'
          '<p>DNN phát hiện thêm 1 sự kiện và có Normal FA/h thấp hơn; GRU có Median Delay ngắn hơn.</p>'
          '<small>Hai mô hình dùng chế độ giám sát khác nhau.</small></div>'
          '<div class="v2-protocol"><span class="v2-eyebrow">NGƯỠNG ĐÁNH GIÁ</span>'
          f'<div><span>DNN · test3</span><b>τ* {dnn["threshold"]:.6f}</b></div>'
          f'<div><span>GRU · train3</span><b>Q₉₉.₉ {gru["threshold"]:.6f}</b></div></div></div></section>')


def _comparison(summary):
    _section("Đối chiếu trên cùng tập kiểm định", "Baselines · GRU · DNN")
    metric_col, seeds_col = st.columns([1.3, 1], vertical_alignment="bottom")
    with metric_col:
        selected = st.selectbox("Chỉ số so sánh", list(METRICS), key="dnn_metric")
    with seeds_col:
        all_seeds = st.checkbox("Hiện đủ seeds 42 / 43 / 44", key="dnn_all_seeds")
    ids = PRIMARY_MODELS if all_seeds else BASE_MODELS
    frame = summary.set_index("model_id").loc[ids].reset_index()
    frame["Mô hình"] = frame["model_id"].map(MODEL_NAMES)
    frame["Nhóm"] = frame["model_id"].map(lambda x: "DNN" if x.startswith("dnn") else "GRU" if x.startswith("gru") else "Baseline")
    metric = METRICS[selected]
    fmt = ".1%" if metric == "event_recall" else ".6f" if metric == "event_f1" else ".4f" if metric == "normal_fa_events_per_hour" else ".1f"
    scale = alt.Scale(domain=[0, 1]) if metric in ("event_recall", "event_f1") else alt.Scale(zero=True)
    base = alt.Chart(frame).encode(
        y=alt.Y("Mô hình:N", sort=frame["Mô hình"].tolist(), title=None, axis=alt.Axis(labelLimit=180)),
        x=alt.X(f"{metric}:Q", title=selected + (" (giây)" if selected == "Median Delay" else ""), scale=scale, axis=alt.Axis(format=fmt)),
        tooltip=[alt.Tooltip("Mô hình:N"), alt.Tooltip(f"{metric}:Q", title=selected, format=fmt)],
    )
    bars = base.mark_bar(size=16, cornerRadiusEnd=3).encode(color=alt.Color("Nhóm:N", scale=alt.Scale(domain=["Baseline", "GRU", "DNN"], range=["#627383", "#a4b7c8", "#9ac9b3"]), legend=None))
    labels = base.mark_text(align="left", dx=9, color="#e4eaf0", fontSize=12).encode(text=alt.Text(f"{metric}:Q", format=fmt))
    chart = (bars + labels).properties(height=230 if all_seeds else 188, padding={"right": 70, "top": 10, "bottom": 3, "left": 5}).configure_view(stroke=None).configure(background="#16232e").configure_axis(labelColor="#b0c1cd", titleColor="#899eac", gridColor="#293a47", gridDash=[2, 4], domain=False, labelFontSize=12, labelPadding=10, titleFontWeight="normal")
    st.altair_chart(chart, width="stretch", theme=None)
    st.caption(("Cao hơn tốt hơn." if selected in ("Event Recall", "Event F1") else "Thấp hơn tốt hơn.") + " DNN có giám sát; GRU và baselines học từ normal. Cùng tập V2 17 sự kiện, t ≥ 30.")
    with st.expander("Bảng so sánh đầy đủ & dữ liệu gốc", expanded=False):
        view = st.selectbox("Bảng chi tiết", ["Tổng hợp 17 sự kiện", "Theo phiên test4 / test5", "Confusion matrices", "Theo phân hệ"], key="dnn_table_view")
        if view == "Tổng hợp 17 sự kiện":
            _table(summary)
            _download("v2_heldout_test4_test5_summary.csv", "Tải bảng tổng hợp CSV")
        else:
            names = {"Theo phiên test4 / test5": "v2_heldout_per_file.csv", "Confusion matrices": "v2_heldout_confusion_matrices.csv", "Theo phân hệ": "v2_heldout_subsystem_breakdown.csv"}
            data = _read(names[view])
            if data is not None:
                _table(data)
                _download(names[view], "Tải bảng CSV")


def _reset_events():
    st.session_state["dnn_event_page"] = 0
    st.session_state.pop("dnn_event_selected", None)


def _move_page(step):
    st.session_state["dnn_event_page"] += step
    st.session_state.pop("dnn_event_selected", None)


def _events():
    events = _read("v2_heldout_17_attack_events.csv")
    if events is None:
        return
    filters = st.columns([.85, 1.5, 1.1, 1])
    with filters[0]:
        session = st.selectbox("Phiên", ["test4", "test5", "Tất cả"], key="dnn_session", on_change=_reset_events)
    with filters[1]:
        model = st.selectbox("Mô hình", list(EVENT_MODELS), key="dnn_event_model", on_change=_reset_events)
    with filters[2]:
        status = st.selectbox("Phát hiện", ["Tất cả", "Đã phát hiện", "Bỏ sót", "Chỉ DNN s42 phát hiện"], key="dnn_status", on_change=_reset_events)
    with filters[3]:
        process = st.selectbox("Phân hệ", ["Tất cả", "P1", "P2", "P3", "Đơn phân hệ", "Đa phân hệ"], key="dnn_process", on_change=_reset_events)
    prefix = EVENT_MODELS[model]
    detected, delay = prefix + "_detected", prefix + "_delay_seconds"
    subset = events.copy()
    if session != "Tất cả":
        subset = subset[subset["session"] == session]
    if process in ("P1", "P2", "P3"):
        subset = subset[subset["attack_" + process].eq(True)]
    elif process != "Tất cả":
        subset = subset[subset["is_multiprocess"].eq(process == "Đa phân hệ")]
    if status in ("Đã phát hiện", "Bỏ sót"):
        subset = subset[subset[detected].eq(status == "Đã phát hiện")]
    elif status == "Chỉ DNN s42 phát hiện":
        subset = subset[subset["dnn_seed42_tau_star_detected"].eq(True) & subset["gru_seed42_detected"].eq(False)]
    if subset.empty:
        st.info("Không có sự kiện khớp bộ lọc. Hãy đổi phiên, mô hình hoặc tình trạng phát hiện.")
        return
    pages = ceil(len(subset) / PAGE_SIZE)
    page = max(0, min(st.session_state.get("dnn_event_page", 0), pages - 1))
    st.session_state["dnn_event_page"] = page
    visible = subset.iloc[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
    listing, detail = st.columns([1.1, 1], gap="medium")
    with listing:
        st.caption(f"{len(subset)} / 17 sự kiện · Trang {page + 1}/{pages}")
        preview = pd.DataFrame({"Sự kiện": visible["attack_event_id"], "Kết quả": visible[detected].map({True: "Phát hiện", False: "Bỏ sót"}), "Delay (s)": visible[delay]})
        _table(preview, 250)
        prev, nxt = st.columns(2)
        prev.button("← Trước", key="dnn_prev", disabled=page == 0, on_click=_move_page, args=(-1,), width="stretch")
        nxt.button("Tiếp →", key="dnn_next", disabled=page == pages - 1, on_click=_move_page, args=(1,), width="stretch")
    with detail:
        ids = visible["attack_event_id"].tolist()
        if st.session_state.get("dnn_event_selected") not in ids:
            st.session_state["dnn_event_selected"] = ids[0]
        event_id = st.selectbox("Chi tiết sự kiện", ids, key="dnn_event_selected")
        row = visible.set_index("attack_event_id").loc[event_id]
        found = bool(row[detected])
        delay_text = f'{row[delay]:.1f} giây' if pd.notna(row[delay]) else "— (bỏ sót)"
        _html('<article class="v2-event-detail">'
              '<div class="v2-eyebrow">HỒ SƠ SỰ KIỆN</div>'
              f'<div class="v2-event-heading"><h3>{escape(event_id)}</h3>'
              f'<span class="v2-event-status {"" if found else "v2-missed"}">{"Đã phát hiện" if found else "Bỏ sót"}</span></div>'
              f'<dl><dt>Bắt đầu</dt><dd>{escape(str(row["effective_start"]))}</dd>'
              f'<dt>Kết thúc</dt><dd>{escape(str(row["orig_end"]))}</dd>'
              f'<dt>Phân hệ · thời lượng</dt><dd>{escape(str(row["process_set"]))} · {int(row["duration_seconds"])} giây</dd>'
              f'<dt>Detection Delay</dt><dd class="v2-detail-delay">{delay_text}</dd></dl></article>')
    with st.expander("Điểm số, ngưỡng & đối chiếu sự kiện", expanded=False):
        cols = ["attack_event_id", "orig_start", "effective_start", "orig_end", "left_censored", "process_set", "is_multiprocess"]
        cols += [f"{p}_{field}" for p in EVENT_MODELS.values() for field in ("detected", "delay_seconds", "peak_score", "threshold", "peak_to_threshold_ratio")]
        _table(visible.loc[visible["attack_event_id"].eq(event_id), cols], 120)
        st.caption("τ=0.50 chỉ là phân tích phụ. ‘Chỉ DNN s42 phát hiện’ luôn đối chiếu DNN s42 τ* với GRU s42.")


def _stability(summary):
    figures = {
        "Chỉ số 3 Seeds": None,
        "Độ nhạy ngưỡng · phân phối xác suất": "v2_probability_saturation_analysis.png",
        "Phân tích lỗi · Confusion matrices": "v2_confusion_matrices_seed42.png",
        "Phát hiện & độ trễ 17 sự kiện": "v2_attack_event_heatmap_17.png",
        "Đối chiếu Event metrics": "v2_event_metrics_comparison.png",
    }
    selected = st.selectbox("Nội dung hiển thị", list(figures), key="dnn_stability_view")
    _note("Seed 44: phát hiện 8/17, Event Recall 47.06%. F1 cao không đồng nghĩa phát hiện tốt hơn; seed 42 vẫn là canonical.", warning=True)
    if figures[selected]:
        path = REPO_ROOT / RESULTS / "figures" / figures[selected]
        if path.is_file():
            st.image(str(path), width="stretch")
            st.caption("Hình gốc · test4 + test5. τ* là ngưỡng chính; τ=0.50 chỉ dùng phân tích phụ. Phân phối xác suất cho thấy độ nhạy, không phải đường quét ngưỡng mới.")
        else:
            st.warning(f"Thiếu hình: {path.name}")
    else:
        rows = summary.set_index("model_id")
        for column, seed in zip(st.columns(3, gap="small"), (42, 43, 44)):
            row = rows.loc[f"dnn_v2_seed{seed}_tau_star"]
            with column:
                _html(f'<article class="v2-seed {"v2-seed-missed" if seed == 44 else ""}">'
                      f'<div class="v2-seed-heading"><h3>Seed {seed}</h3><span>{"Đại diện" if seed == 42 else "Lặp lại"}</span></div>'
                      f'<strong>{int(row["tp_events"])}/17</strong><span> sự kiện phát hiện</span>'
                      f'<div class="v2-recall-track" aria-hidden="true"><div style="width:{float(row["event_recall"]) * 100:.4f}%"></div></div>'
                      f'<div class="v2-seed-metrics"><div><span>Event Recall</span><b>{row["event_recall"]:.2%}</b></div>'
                      f'<div><span>Event F1</span><b>{row["event_f1"]:.6f}</b></div>'
                      f'<div><span>Normal FA/h</span><b>{row["normal_fa_events_per_hour"]:.4f}</b></div></div>'
                      f'<div class="v2-label">τ* = {row["threshold"]:.6f} · test3</div></article>')
        with st.expander("Thống kê Mean ± Std, validation & phân tích chi tiết", expanded=False):
            view = st.selectbox("Dữ liệu", ["Thống kê 3 Seeds", "Validation test3", "Phân tích seed 44"], key="dnn_seed_details")
            if view == "Phân tích seed 44":
                _note("Seed 44 có Event Precision 42.11% (8/19), 11 Event FP và chỉ 3 normal FA; bỏ sót 9/17. 8/9 đỉnh xác suất bỏ sót nằm sát dưới τ*. Cơ chế Tanh(10), early stopping, pos_weight và số sự kiện validation ít chỉ là giả thuyết, chưa có causal ablation.")
            else:
                frame = _read("v2_heldout_multiseed_stats.csv" if view == "Thống kê 3 Seeds" else "v2_dnn_validation_summary.csv")
                if frame is not None:
                    _table(frame, 180)


def _methods(summary):
    manifest = load_json(RESULTS + "v2_pre_evaluation_freeze_manifest.json")
    if manifest is None:
        return
    splits = manifest["splits"]
    _html('<div class="v2-flow">'
          f'<div><span>01 / Train</span><strong>{escape(" + ".join(splits["dnn_train"]))}</strong><small>DNN · có nhãn attack</small></div>'
          '<b aria-hidden="true">→</b>'
          f'<div><span>02 / Validation</span><strong>{escape(" + ".join(splits["dnn_validation"]))}</strong><small>Chọn epoch & ngưỡng τ*</small></div>'
          '<b aria-hidden="true">→</b>'
          f'<div><span>03 / Held-out Test</span><strong>{escape(" + ".join(splits["dnn_heldout_evaluation_v2"]))}</strong><small>17 sự kiện · t ≥ 30</small></div></div>')
    a, b = st.columns(2)
    with a:
        _html('<article class="v2-method"><h3>DNN · Supervised</h3><p>Phân loại Normal/Attack từ 79 đặc trưng. Tái lập kiến trúc có điều chỉnh từ Kuchař & Fujdiak (2025), không tái lập 100% bài báo hoặc accuracy 4 lớp 99.72%.</p></article>')
    with b:
        _html('<article class="v2-method"><h3>GRU · Unsupervised</h3><p>Học dự báo từ normal train1 + train2, hiệu chuẩn Q₉₉.₉ trên train3. So sánh hai chế độ giám sát, không cô lập riêng tác động kiến trúc.</p></article>')
    _note("Giới hạn: DNN nhạy ngưỡng giữa seeds; còn báo động giả; test4/test5 đã được phân tích ở V1 nên không mù hoàn toàn ở cấp dự án. Hiệu chỉnh ngưỡng trên test3 là hậu nghiệm. Chưa chứng minh sẵn sàng vận hành ICS.", warning=True)
    with st.expander("Tài liệu phương pháp đầy đủ, giới hạn & kiểm toán", expanded=False):
        reference = REPO_ROOT / "docs/dnn_v2_dashboard_reference.md"
        doc = st.selectbox("Tài liệu", ["Diễn giải học thuật gốc", "Kiến trúc & freeze manifest", "Data pipeline audit", "Held-out audit"], key="dnn_document")
        if doc == "Diễn giải học thuật gốc":
            st.markdown(reference.read_text(encoding="utf-8"))
            st.download_button("Tải tài liệu Markdown", reference.read_bytes(), file_name=reference.name, mime="text/markdown")
        else:
            name = {"Kiến trúc & freeze manifest": "v2_pre_evaluation_freeze_manifest.json", "Data pipeline audit": "v2_data_pipeline_audit.json", "Held-out audit": "v2_heldout_evaluation_audit.json"}[doc]
            data = load_json(RESULTS + name)
            if data is not None:
                st.json(data, expanded=False)


def render_dnn_v2():
    css = Path(__file__).with_suffix(".css").read_text(encoding="utf-8")
    with st.container(key="dnn_v2"):
        _html(f"<style>{css}</style>")
        _html('<header class="v2-heading"><div><div class="v2-eyebrow">HAI 21.03 / NGHIÊN CỨU THỰC NGHIỆM</div>'
              '<div class="v2-page-title" role="heading" aria-level="1">DNN Replication <span>V2</span></div></div>'
              '<div class="v2-header-scope"><span>HELD-OUT TEST</span><div>test4 + test5<b>17 sự kiện</b></div></div></header>')
        tab = st.radio("Nội dung DNN V2", TABS, horizontal=True, label_visibility="collapsed", width="stretch", key="dnn_tab")
        summary = _read("v2_heldout_test4_test5_summary.csv")
        if summary is None:
            return
        # Presentation guard: never substitute the full V1 50-event benchmark.
        if not summary["evaluable_attacks"].eq(17).all() or not summary["eval_split"].eq("test4+test5").all():
            st.error("Dữ liệu không thuộc held-out V2 17 sự kiện. Dừng hiển thị để tránh trộn V1/V2.")
            return
        if tab == TABS[0]:
            _overview(summary)
        elif tab == TABS[1]:
            _comparison(summary)
        elif tab == TABS[2]:
            _events()
        elif tab == TABS[3]:
            _stability(summary)
        else:
            _methods(summary)
