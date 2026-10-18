"""Interactive execution UI: prepare data, train GRU, evaluate, inspect."""

from collections import deque
from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from be import gru_runtime as runtime
from fe.navigation import navigation

STEPS = ["1 · Dữ liệu", "2 · Huấn luyện GRU", "3 · Đánh giá", "4 · Kiểm tra sự kiện"]


def _html(value):
    st.markdown(value, unsafe_allow_html=True)


def _go(index):
    st.session_state["gru_step"] = STEPS[max(0, min(index, len(STEPS) - 1))]


def _reset():
    st.session_state.pop("gru_live_run", None)
    st.session_state["gru_step"] = STEPS[0]
    for key in ("gru_seed", "gru_epochs", "gru_rate", "gru_eval_sessions", "gru_event"):
        st.session_state.pop(key, None)


def _section(number, title, description=""):
    desc_html = f'<p>{escape(description)}</p>' if description else ''
    _html(f'<div class="gf-section"><span class="gf-eyebrow">BƯỚC {number:02d} / 04</span><h2>{escape(title)}</h2>{desc_html}</div>')


def _cards(items):
    html = "".join(f'<article class="gf-stat"><span>{escape(label)}</span><strong>{escape(str(value))}</strong><small>{escape(note)}</small></article>'
                   for label, value, note in items)
    _html(f'<div class="gf-stats" style="--columns:{len(items)}">{html}</div>')


def _plot(frame, x, series, *, temporal=False, attack=None, height=190):
    """Presentation styling only; plot the original series without resampling."""
    long = frame.melt(id_vars=[x], value_vars=series, var_name="Chỉ số", value_name="Giá trị")
    axis = alt.X(f'{x}:{"T" if temporal else "Q"}', title=None,
                 axis=alt.Axis(format="%H:%M:%S" if temporal else "d", tickCount=5))
    colors = alt.Scale(domain=series, range=["#28594c", "#aa693b"])
    curve = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=axis, y=alt.Y("Giá trị:Q", title=None),
        color=alt.Color("Chỉ số:N", scale=colors, legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("Chỉ số:N", scale=alt.Scale(domain=series, range=[[1, 0], [5, 4]]), legend=None),
        tooltip=[alt.Tooltip(f'{x}:{"T" if temporal else "Q"}', title="Thời điểm" if temporal else "Epoch"), "Chỉ số:N", alt.Tooltip("Giá trị:Q", format=".6f")])
    if attack is not None:
        band = alt.Chart(pd.DataFrame([{"start": pd.Timestamp(attack[0]), "end": pd.Timestamp(attack[1])}])).mark_rect(color="#aa693b", opacity=.09).encode(x="start:T", x2="end:T")
        curve = band + curve
    chart = curve.properties(height=height).configure(background="#ffffff").configure_view(stroke=None).configure_axis(
        labelColor="#64716f", titleColor="#394740", gridColor="#e9ede6", domain=False, tickColor="#dce2de", labelFontSize=14
    ).configure_legend(labelColor="#435148", labelFontSize=15)
    st.altair_chart(chart, theme=None, width="stretch")


def _log(path, stage):
    file = path / f"{stage}.log"
    if file.exists():
        with file.open(encoding="utf-8", errors="replace") as handle:
            lines = list(deque(handle, maxlen=12))
        st.code("".join(lines) or "Tiến trình đã khởi động, đang chờ log…", language="text", height=190)


@st.fragment(run_every=1.5)
def _live_job(path, stage):
    state = runtime.read_state(path)
    if state.get("status") != "running":
        st.rerun()
    st.info({"data": "Đang đọc và kiểm tra dữ liệu…", "train": "GRU đang huấn luyện…", "evaluate": "Đang dự đoán và tính metric…"}[stage])
    _log(path, stage)


def _start(path, stage, sessions=None):
    try:
        runtime.launch(path, stage, sessions)
    except (OSError, ValueError) as error:
        st.error(str(error))
    else:
        st.rerun()


def _current_run():
    value = st.session_state.get("gru_live_run")
    try:
        return runtime.run_path(value) if value else None
    except ValueError:
        st.session_state.pop("gru_live_run", None)
        return None


def _data(path, state):
    _section(1, "Chọn dữ liệu")
    if path is not None:
        settings = runtime.read_json(path / "settings.json")
        if "data" in state.get("completed", []):
            manifest = runtime.read_json(path / "data_manifest.json")
            st.success("Dữ liệu sẵn sàng. Tiếp tục sang huấn luyện.")
            _cards([("Huấn luyện", f'{sum(manifest[s]["used_rows"] for s in ("train1", "train2")):,}', "dòng · train1 + train2"),
                    ("Chọn ngưỡng", f'{manifest["train3"]["used_rows"]:,}', "dòng · train3"),
                    ("Kiểm thử", "5 phiên", "test1–test5 · giữ đầy đủ")])
            with st.expander("Chi tiết dữ liệu"):
                st.caption(settings["source"])
                table = pd.DataFrame([{"Phiên": session, "Số dòng": item["rows"], "Dòng sử dụng": item["used_rows"]} for session, item in manifest.items()])
                st.dataframe(table, hide_index=True, height=235)
        elif state.get("status") != "running":
            if st.button("Thử chuẩn bị lại", key="gru_retry_data", type="primary"):
                _start(path, "data")
        return
    with st.form("gru_data_form", border=False):
        left, right = st.columns(2, gap="medium")
        with left:
            source = st.text_input("Thư mục dữ liệu", "data/processed/hai_21_03", key="gru_source",
                                   help=".npy: thư mục chứa train1/2/3 và final_test/. CSV: thư mục chứa train1–3 và test1–5.")
        with right:
            kind = st.selectbox("Định dạng", ["Mảng đã xử lý (.npy)", "CSV HAI gốc (.csv / .csv.gz)"], key="gru_source_kind",
                                help="CSV dùng scaler đã lưu, không học lại scaler. Dữ liệu mới được ghi riêng trong runs/gru/.")
        profile = st.radio("Chế độ chạy", ["Chạy nhanh · 10.000 dòng / phiên", "Toàn bộ dữ liệu"], horizontal=True, key="gru_profile",
                           help="Chạy nhanh giới hạn mỗi phiên train/validation ở 10.000 dòng. Dữ liệu kiểm thử vẫn dùng đầy đủ. Kết quả mới không thay thế benchmark đã lưu.")
        submitted = st.form_submit_button("Chuẩn bị dữ liệu", type="primary", key="gru_prepare", icon=":material/arrow_forward:")
    if submitted:
        try:
            source_path = Path(source).expanduser()
            if not source_path.is_absolute():
                source_path = runtime.ROOT / source_path
            path = runtime.create_run({"source_kind": "processed" if kind.startswith("Mảng") else "csv",
                "source": str(source_path), "row_limit": 10000 if profile.startswith("Chạy nhanh") else 0,
                "seed": 42, "epochs": 3, "learning_rate": 0.001})
            st.session_state["gru_live_run"] = str(path)
            _start(path, "data")
        except (OSError, ValueError) as error:
            st.error(str(error))


def _train(path, state):
    _section(2, "Chạy huấn luyện GRU")
    if path is None or "data" not in state.get("completed", []):
        st.info("Chuẩn bị dữ liệu ở bước 1 trước.")
        return
    settings = runtime.read_json(path / "settings.json")
    completed = "train" in state.get("completed", [])
    busy = state.get("status") == "running"
    if completed:
        trained = runtime.read_json(path / "training.json")
        st.caption(f'Seed {settings["seed"]} · Epoch tối đa {settings["epochs"]} · Learning rate {settings["learning_rate"]}')
        _cards([("Checkpoint", f'Epoch {trained["best_epoch"]}', "gru.pt của lần chạy này"),
                ("Đã chạy", f'{trained["total_epochs"]} epoch', trained["stop_reason"]),
                ("Ngưỡng mới", f'{trained["threshold"]:.6f}', "Q99.9 trên train3 của lần chạy")])
        view = st.selectbox("Xem quá trình chạy", ["Log huấn luyện", "Đường học"], key="gru_training_view")
        if view == "Log huấn luyện":
            _log(path, "train")
        else:
            history = pd.read_csv(path / "training_history.csv")
            _plot(history, "epoch", ["train_loss_mse", "val_mean_rmse_s5"])
            st.caption("Train Loss là MSE; validation dùng RMSE. Hai đường có thang đo khác nhau.")
        return
    missing = runtime.missing_dependencies()
    if missing:
        st.warning("Môi trường Python đang chạy app thiếu: " + ", ".join(missing) + ". Chạy app bằng môi trường đã cài thư viện GRU trên Kali.")
    with st.form("gru_train_form", border=False):
        a, b = st.columns(2, gap="medium")
        with a:
            seed = st.selectbox("Seed", [42, 43, 44], index=[42, 43, 44].index(settings["seed"]), disabled=completed or busy, key="gru_seed",
                                help="Mã khởi tạo ngẫu nhiên. Giữ 42 để demo cùng một thiết lập.")
        with b:
            epochs = st.number_input("Số lượt học (epoch)", 1, 50, settings["epochs"], disabled=completed or busy, key="gru_epochs",
                                     help="Số lượt tối đa mô hình học qua dữ liệu. Mặc định 3 để chạy thử.")
        with st.expander("Tùy chỉnh nâng cao"):
            rate = st.number_input("Learning rate", 0.00001, 0.01, settings["learning_rate"], step=0.0001, format="%.5f", disabled=completed or busy, key="gru_rate")
            st.caption("Cửa sổ 30 · 79 tín hiệu · hidden size 64 · 1 lớp GRU · CPU")
        submitted = st.form_submit_button("Huấn luyện GRU", type="primary", disabled=completed or busy or bool(missing), icon=":material/play_arrow:")
    if submitted:
        settings.update(seed=int(seed), epochs=int(epochs), learning_rate=float(rate))
        runtime.validate_settings(settings)
        runtime._write(path / "settings.json", settings)
        _start(path, "train")


def _evaluate(path, state):
    _section(3, "Kiểm tra kết quả mô hình")
    if path is None or "train" not in state.get("completed", []):
        st.info("Huấn luyện xong ở bước 2 trước khi đánh giá.")
        return
    completed = "evaluate" in state.get("completed", [])
    request = runtime.read_json(path / "evaluation_request.json", {"sessions": ["test1"]})
    if not completed:
        sessions = st.multiselect("Phiên kiểm thử", runtime.SESSIONS, default=request["sessions"], disabled=state.get("status") == "running", key="gru_eval_sessions",
                                  help="Chọn test1 để chạy thử nhanh. Chọn đủ test1–test5 để đánh giá cả 50 sự kiện.")
        if st.button("Chạy đánh giá", key="gru_evaluate", type="primary", disabled=not sessions or state.get("status") == "running", icon=":material/play_arrow:"):
            _start(path, "evaluate", sessions)
    if completed:
        result = runtime.read_json(path / "evaluation.json")
        total = result["tp"] + result["fn"]
        rate = result["normal_fa_events_per_hour"]
        delay = result["median_delay_sec"]
        _cards([("Sự kiện phát hiện", f'{result["tp"]}/{total}', " + ".join(result["sessions"])),
                ("Event F1", f'{result["event_f1"]:.6f}', "F1 Score"),
                ("Normal FA/h", f"{rate:.4f}" if rate is not None else "N/A", "Báo động giả / giờ"),
                ("Median Delay", f"{delay:g} giây" if delay is not None else "N/A", "Độ trễ trung vị")])
        st.success("Đánh giá xong. Mở sự kiện để kiểm tra từng cảnh báo.")
        with st.expander("Log đánh giá & tải kết quả", expanded=False):
            _log(path, "evaluate")
            st.download_button("Tải kết quả lần chạy", (path / "evaluation.json").read_bytes(), file_name=f"{path.name}_evaluation.json", mime="application/json")


def _events(path, state):
    _section(4, "Xem một sự kiện cụ thể")
    if path is None or "evaluate" not in state.get("completed", []):
        st.info("Chạy đánh giá ở bước 3 trước.")
        return
    events = pd.read_csv(path / "attack_events.csv")
    if events.empty:
        st.info("Không có sự kiện tấn công có thể đánh giá trong các phiên đã chọn.")
        return
    a, b, c = st.columns(3, gap="medium")
    with a:
        session = st.selectbox("Phiên", events.session.unique(), key="gru_session")
    with b:
        status = st.selectbox("Kết quả", ["Tất cả", "Phát hiện được", "Bỏ sót"], key="gru_status")
    filtered = events.loc[events.session.eq(session)]
    if status != "Tất cả":
        filtered = filtered.loc[filtered.detected.eq(status == "Phát hiện được")]
    if filtered.empty:
        st.info("Không có sự kiện phù hợp bộ lọc.")
        return
    with c:
        options = filtered.attack_event_id.tolist()
        if st.session_state.get("gru_event") not in options:
            st.session_state["gru_event"] = "test1_atk_02" if "test1_atk_02" in options else options[0]
        selected = st.selectbox("Sự kiện", options, key="gru_event")
    row = filtered.set_index("attack_event_id").loc[selected]
    detected = bool(row.detected)
    delay = f"{row.delay_seconds:g} giây" if detected else "Không áp dụng"
    _html(f'<div class="gf-event"><div><span>{escape(selected)}</span><p>Bắt đầu <strong>{escape(row.orig_start)}</strong></p><p>Kết thúc <strong>{escape(row.orig_end)}</strong></p></div>'
          f'<div class="gf-event-outcome"><h3 class="{"gf-detected" if detected else "gf-missed"}">{"Phát hiện được" if detected else "Bỏ sót"}</h3><p>Độ trễ <strong>{delay}</strong></p></div></div>')
    with np.load(path / f"scores_{session}.npz", allow_pickle=False) as scores:
        timestamps = pd.to_datetime(scores["timestamps"])
        mask = (timestamps >= pd.Timestamp(row.orig_start) - pd.Timedelta(seconds=30)) & (timestamps <= pd.Timestamp(row.orig_end) + pd.Timedelta(seconds=30))
        threshold = runtime.read_json(path / "training.json")["threshold"]
        frame = pd.DataFrame({"Thời gian": timestamps[mask], "Sai số dự báo": scores["scores"][mask], "Ngưỡng cảnh báo": threshold})
    _plot(frame, "Thời gian", ["Sai số dự báo", "Ngưỡng cảnh báo"], temporal=True, attack=(row.orig_start, row.orig_end))
    st.caption("Vùng tô nhạt: thời gian tấn công · Đường đứt: ngưỡng cảnh báo")


def render_gru_workflow():
    css = Path(__file__).parent / "styles/gru_workflow.css"
    _html(f'<style>{css.read_text(encoding="utf-8")}</style>')
    with st.container(key="gru_flow"):
        path = _current_run()
        state = runtime.read_state(path) if path else {}
        busy = state.get("status") == "running"
        status = "Đang chạy" if busy else "Cần kiểm tra" if state.get("status") == "failed" else "Hoàn tất" if "evaluate" in state.get("completed", []) else "Sẵn sàng" if path else "Chưa bắt đầu"
        _html('<header class="gf-header"><div><span class="gf-eyebrow">DT049 / HAI 21.03</span>'
              f'<h1>Thực hiện GRU<span class="gf-header-dot">.</span></h1></div><span class="gf-status {"gf-status-live" if busy else ""}"><i></i>{status}</span></header>')
        step = navigation(STEPS, key="gru_step")
        index = STEPS.index(step)
        with st.container(key="gru_workspace"):
            if state.get("status") == "failed":
                st.error(state.get("error", "Tác vụ lỗi. Xem log."))
                with st.expander("Chi tiết lỗi", expanded=True):
                    _log(path, state["stage"])
            [_data, _train, _evaluate, _events][index](path, state)
            if busy:
                _live_job(path, state["stage"])
                if st.button("Dừng tác vụ", key="gru_stop"):
                    try:
                        runtime.stop(path)
                    except (OSError, ValueError) as error:
                        st.error(str(error))
                    else:
                        st.rerun()
        with st.container(key="gru_footer"):
            back, middle, forward = st.columns([1, 2, 1.4], vertical_alignment="center")
            with back:
                st.button("Quay lại", key="gru_back", disabled=index == 0, on_click=_go, args=(index - 1,), width="stretch", icon=":material/arrow_back:")
            with middle:
                prerequisite = runtime.STAGES[index] if index < 3 else "evaluate"
                st.caption("Đã xong bước này" if prerequisite in state.get("completed", []) else "Hoàn thành thao tác để tiếp tục")
            with forward:
                prerequisite = runtime.STAGES[index] if index < 3 else "evaluate"
                next_label = ["Tiếp: Huấn luyện", "Tiếp: Đánh giá", "Tiếp: Xem sự kiện", "Đã hoàn tất"][index]
                st.button(next_label, key="gru_next", type="primary", disabled=index == 3 or prerequisite not in state.get("completed", []), on_click=_go, args=(index + 1,), width="stretch", icon=":material/arrow_forward:")
        previous = sorted(runtime.RUNS.glob("*/settings.json"), reverse=True)[:20] if runtime.RUNS.exists() else []
        with st.expander("Quản lý lần chạy", expanded=False):
            if path:
                settings = runtime.read_json(path / "settings.json")
                st.caption(f'{path.name} · Seed {settings["seed"]} · {"Chạy nhanh" if settings["row_limit"] else "Toàn bộ dữ liệu"} · Kết quả mới, lưu riêng với benchmark')
            st.button("Tạo lần chạy mới", key="gru_new", disabled=busy or path is None, on_click=_reset)
            if previous:
                selected = st.selectbox("Lần chạy", [item.parent.name for item in previous], key="gru_previous")
                if st.button("Mở lần chạy", disabled=busy, key="gru_open_previous"):
                    st.session_state["gru_live_run"] = str(runtime.run_path(runtime.RUNS / selected))
                    st.rerun()
