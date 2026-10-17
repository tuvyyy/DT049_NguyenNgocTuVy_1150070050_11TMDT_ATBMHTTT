"""Interactive execution UI: prepare data, train GRU, evaluate, inspect."""

from collections import deque
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from be import gru_runtime as runtime

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
    _html(f'<div class="gf-section"><h2>{escape(title)}</h2>{desc_html}</div>')


def _cards(items):
    html = "".join(f'<article class="gf-stat"><span>{escape(label)}</span><strong>{escape(str(value))}</strong><small>{escape(note)}</small></article>'
                   for label, value, note in items)
    _html(f'<div class="gf-stats" style="--columns:{len(items)}">{html}</div>')


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
    _section(1, "1. Chuẩn bị dữ liệu")
    if path is not None:
        settings = runtime.read_json(path / "settings.json")
        st.caption(f'Nguồn: {settings["source"]} · {"Nhanh: tối đa 10.000 dòng mỗi phiên train/validation" if settings["row_limit"] else "Toàn bộ dữ liệu"}')
        if "data" in state.get("completed", []):
            manifest = runtime.read_json(path / "data_manifest.json")
            st.success("Dữ liệu đã được đọc và kiểm tra. Bấm Tiếp theo để thiết lập GRU.")
            table = pd.DataFrame([{"Phiên": session, "Số dòng": item["rows"], "Dòng sử dụng": item["used_rows"]} for session, item in manifest.items()])
            st.dataframe(table, hide_index=True, height=235)
        elif state.get("status") != "running":
            if st.button("Thử chuẩn bị lại", key="gru_retry_data", type="primary"):
                _start(path, "data")
        return
    with st.form("gru_data_form"):
        left, right = st.columns([1, 1.4])
        with left:
            kind = st.selectbox("Định dạng dữ liệu", ["Mảng đã xử lý (.npy)", "CSV HAI gốc (.csv / .csv.gz)"], key="gru_source_kind")
        with right:
            source = st.text_input("Thư mục dữ liệu", "data/processed/hai_21_03", key="gru_source",
                                   help=".npy: thư mục chứa train1/2/3 và final_test/. CSV: thư mục chứa train1–3 và test1–5.")
        profile = st.radio("Khối lượng huấn luyện", ["Chạy nhanh · 10.000 dòng / phiên", "Toàn bộ dữ liệu"], horizontal=True, key="gru_profile")
        submitted = st.form_submit_button("Chuẩn bị dữ liệu", type="primary")
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
    st.caption("CSV được chuẩn hóa bằng scaler đã có, chỉ ghi mảng mới vào thư mục lần chạy. Dữ liệu kiểm thử luôn được giữ đầy đủ.")


def _train(path, state):
    _section(2, "2. Huấn luyện mô hình GRU")
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
            st.line_chart(history.set_index("epoch")[["train_loss_mse", "val_mean_rmse_s5"]], height=190)
            st.caption("Train Loss là MSE; validation dùng RMSE. Hai đường có thang đo khác nhau.")
        return
    missing = runtime.missing_dependencies()
    if missing:
        st.warning("Môi trường Python đang chạy app thiếu: " + ", ".join(missing) + ". Chạy app bằng môi trường đã cài thư viện GRU trên Kali.")
    with st.form("gru_train_form"):
        a, b, c = st.columns(3)
        with a:
            seed = st.selectbox("Seed", [42, 43, 44], index=[42, 43, 44].index(settings["seed"]), disabled=completed or busy, key="gru_seed")
        with b:
            epochs = st.number_input("Số epoch tối đa", 1, 50, settings["epochs"], disabled=completed or busy, key="gru_epochs")
        with c:
            rate = st.number_input("Learning rate", 0.00001, 0.01, settings["learning_rate"], step=0.0001, format="%.5f", disabled=completed or busy, key="gru_rate")
        st.caption("Cửa sổ 30 · 79 tín hiệu · hidden size 64 · 1 lớp GRU · CPU")
        submitted = st.form_submit_button("Huấn luyện GRU", type="primary", disabled=completed or busy or bool(missing))
    if submitted:
        settings.update(seed=int(seed), epochs=int(epochs), learning_rate=float(rate))
        runtime.validate_settings(settings)
        runtime._write(path / "settings.json", settings)
        _start(path, "train")


def _evaluate(path, state):
    _section(3, "3. Đánh giá kiểm thử")
    if path is None or "train" not in state.get("completed", []):
        st.info("Huấn luyện xong ở bước 2 trước khi đánh giá.")
        return
    completed = "evaluate" in state.get("completed", [])
    request = runtime.read_json(path / "evaluation_request.json", {"sessions": ["test1"]})
    sessions = st.multiselect("Phiên kiểm thử", runtime.SESSIONS, default=request["sessions"], disabled=completed or state.get("status") == "running", key="gru_eval_sessions")
    if st.button("Chạy đánh giá", key="gru_evaluate", type="primary", disabled=completed or not sessions or state.get("status") == "running"):
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
        st.success("Đánh giá hoàn tất. Bấm Tiếp theo để xem chi tiết từng sự kiện.")
        with st.expander("Log đánh giá & tải kết quả", expanded=False):
            _log(path, "evaluate")
            st.download_button("Tải kết quả lần chạy", (path / "evaluation.json").read_bytes(), file_name=f"{path.name}_evaluation.json", mime="application/json")


def _events(path, state):
    _section(4, "4. Chi tiết sự kiện & Cảnh báo")
    if path is None or "evaluate" not in state.get("completed", []):
        st.info("Chạy đánh giá ở bước 3 trước.")
        return
    events = pd.read_csv(path / "attack_events.csv")
    if events.empty:
        st.info("Không có sự kiện tấn công có thể đánh giá trong các phiên đã chọn.")
        return
    a, b, c = st.columns([1, 1, 1.4])
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
        frame = pd.DataFrame({"Sai số dự báo": scores["scores"][mask], "Ngưỡng cảnh báo": threshold}, index=timestamps[mask])
    st.line_chart(frame, height=210)
    st.caption("Sai số và ngưỡng lấy từ lần chạy này. Độ trễ ở trên là của riêng sự kiện đang chọn.")


def render_gru_workflow():
    css = Path(__file__).parent / "styles/gru_workflow.css"
    _html(f'<style>{css.read_text(encoding="utf-8")}</style>')
    with st.container(key="gru_flow"):
        _html('<header class="gf-header"><h1>Thực hiện mô hình GRU</h1></header>')
        path = _current_run()
        state = runtime.read_state(path) if path else {}
        busy = state.get("status") == "running"
        if path:
            settings = runtime.read_json(path / "settings.json")
            st.caption(f'Lần chạy: {path.name} · Seed {settings["seed"]} · {"Chạy nhanh" if settings["row_limit"] else "Toàn bộ dữ liệu"}')
        if st.session_state.get("gru_step") not in STEPS:
            st.session_state["gru_step"] = STEPS[0]
        step = st.radio("Các bước thực hiện", STEPS, horizontal=True, label_visibility="collapsed", key="gru_step", width="stretch")
        index = STEPS.index(step)
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
                st.button("Trước", key="gru_back", disabled=index == 0, on_click=_go, args=(index - 1,), width="stretch")
            with middle:
                st.caption(f"Bước {index + 1} / {len(STEPS)}")
            with forward:
                prerequisite = runtime.STAGES[index] if index < 3 else "evaluate"
                st.button("Tiếp theo", key="gru_next", type="primary", disabled=index == 3 or prerequisite not in state.get("completed", []), on_click=_go, args=(index + 1,), width="stretch")
        st.button("Tạo lần chạy mới", key="gru_new", disabled=busy or path is None, on_click=_reset)
        previous = sorted(runtime.RUNS.glob("*/settings.json"), reverse=True)[:20] if runtime.RUNS.exists() else []
        if previous:
            with st.expander("Mở lần chạy trước", expanded=False):
                selected = st.selectbox("Lần chạy", [item.parent.name for item in previous], key="gru_previous")
                if st.button("Mở lần chạy", disabled=busy, key="gru_open_previous"):
                    st.session_state["gru_live_run"] = str(runtime.run_path(runtime.RUNS / selected))
                    st.rerun()
