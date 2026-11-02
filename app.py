"""Streamlit dashboard for the academic smart traffic signal prototype."""

import os
import tempfile
import time
from typing import Dict

import altair as alt
import pandas as pd

from config import DIRECTIONS, MAX_TICK_SECONDS, TICK_SECONDS
from intersection_simulation import IntersectionSimulation
from signal_controller import SignalController
from traffic_density import summarize_traffic


# ---------------------------------------------------------------------------
# Custom CSS for professional look – parameterised by theme
# ---------------------------------------------------------------------------

_LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #f8f9fb;
    --bg-secondary: #ffffff;
    --bg-card: #ffffff;
    --text-primary: #1a1d23;
    --text-secondary: #5a6270;
    --border-color: #e2e6ec;
    --accent: #4f6df5;
    --accent-hover: #3b57d9;
    --green: #22c55e;
    --yellow: #eab308;
    --red: #ef4444;
    --card-shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    --signal-bg: #f1f3f8;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    font-family: 'Inter', sans-serif !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: var(--card-shadow);
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

/* Sidebar polish */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdown"] {
    color: var(--text-primary);
}

/* Section headers */
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Rounded images */
[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid var(--border-color);
}

/* Progress bars */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent), #6f8afa) !important;
    border-radius: 4px;
}

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(79,109,245,0.25);
}

/* Divider */
hr { border-color: var(--border-color) !important; }
</style>
"""

_DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #0e1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2129;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --border-color: #30363d;
    --accent: #6f8afa;
    --accent-hover: #8fa3fb;
    --green: #3fb950;
    --yellow: #d29922;
    --red: #f85149;
    --card-shadow: 0 1px 3px rgba(0,0,0,.3);
    --signal-bg: #21262d;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stAppViewContainer"] > section {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: var(--card-shadow);
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color) !important;
}

h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.02em;
    color: var(--text-primary) !important;
}

[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid var(--border-color);
}

.stProgress > div > div {
    background: linear-gradient(90deg, #6f8afa, #a78bfa) !important;
    border-radius: 4px;
}

.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(111,138,250,0.35);
}

hr { border-color: var(--border-color) !important; }
</style>
"""


def _close_runtime(runtime: Dict[str, object]) -> None:
    """Release video captures and temporary uploaded-video files."""
    monitors = runtime.get("monitors", {})
    for monitor in monitors.values():
        monitor.close()
    for path in runtime.get("temporary_files", []):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    runtime["monitors"] = {}
    runtime["temporary_files"] = []


def _prepare_video_runtime(
    mode: str,
    uploads: Dict[str, object],
    stream_urls: Dict[str, str],
    model_path: str,
    confidence: float,
) -> Dict[str, object]:
    """Validate all four video inputs and initialize their YOLO monitors."""
    from vehicle_detection import VideoVehicleMonitor, YOLOVehicleDetector

    sources = {}
    temporary_files = []
    monitors = {}
    try:
        for direction in DIRECTIONS:
            if mode == "Uploaded videos":
                upload = uploads.get(direction)
                if upload is None:
                    raise ValueError("Choose a video for the {} approach.".format(direction))
                suffix = os.path.splitext(upload.name)[1] or ".mp4"
                handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temporary_files.append(handle.name)
                try:
                    handle.write(upload.getbuffer())
                    sources[direction] = handle.name
                finally:
                    handle.close()
            else:
                source = stream_urls.get(direction, "").strip()
                if not source:
                    raise ValueError("Enter a CCTV stream URL for {}.".format(direction))
                sources[direction] = source

        detector = YOLOVehicleDetector(model_path, confidence)
        for direction, source in sources.items():
            monitors[direction] = VideoVehicleMonitor(source, detector)
    except (OSError, RuntimeError, ValueError):
        for monitor in monitors.values():
            monitor.close()
        for path in temporary_files:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        raise
    return {"monitors": monitors, "temporary_files": temporary_files}


# ---------------------------------------------------------------------------
# Helper: build Altair charts with axis titles
# ---------------------------------------------------------------------------

def _traffic_line_chart(history: list, dark: bool) -> alt.Chart:
    """Vehicle counts over time with labelled axes."""
    rows = []
    for i, row in enumerate(history):
        for d in ("North", "South", "East", "West"):
            rows.append({"Tick": i, "Time": row["time"], "Direction": d, "Vehicles": row[d]})
    df = pd.DataFrame(rows)
    text_color = "#e6edf3" if dark else "#1a1d23"
    grid_color = "#30363d" if dark else "#e2e6ec"
    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.5, opacity=0.9)
        .encode(
            x=alt.X("Tick:Q", title="Time (ticks)", axis=alt.Axis(labelColor=text_color, titleColor=text_color, gridColor=grid_color)),
            y=alt.Y("Vehicles:Q", title="Vehicle Count", axis=alt.Axis(labelColor=text_color, titleColor=text_color, gridColor=grid_color)),
            color=alt.Color(
                "Direction:N",
                title="Approach",
                scale=alt.Scale(
                    domain=["North", "South", "East", "West"],
                    range=["#4f6df5", "#22c55e", "#f59e0b", "#ef4444"],
                ),
                legend=alt.Legend(labelColor=text_color, titleColor=text_color),
            ),
            tooltip=["Time:N", "Direction:N", "Vehicles:Q"],
        )
        .properties(height=240)
        .configure_view(strokeWidth=0)
        .configure(background="transparent")
    )
    return chart


def _green_area_chart(history: list, dark: bool) -> alt.Chart:
    """Green allocation over time with labelled axes."""
    df = pd.DataFrame({
        "Tick": list(range(len(history))),
        "Green Duration": [row["green_seconds"] for row in history],
    })
    text_color = "#e6edf3" if dark else "#1a1d23"
    grid_color = "#30363d" if dark else "#e2e6ec"
    chart = (
        alt.Chart(df)
        .mark_area(
            opacity=0.45,
            line={"color": "#6f8afa", "strokeWidth": 2},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#6f8afa", offset=0),
                    alt.GradientStop(color="transparent", offset=1),
                ],
                x1=0, x2=0, y1=0, y2=1,
            ),
        )
        .encode(
            x=alt.X("Tick:Q", title="Time (ticks)", axis=alt.Axis(labelColor=text_color, titleColor=text_color, gridColor=grid_color)),
            y=alt.Y("Green Duration:Q", title="Green Duration (seconds)", axis=alt.Axis(labelColor=text_color, titleColor=text_color, gridColor=grid_color)),
            tooltip=["Tick:Q", "Green Duration:Q"],
        )
        .properties(height=180)
        .configure_view(strokeWidth=0)
        .configure(background="transparent")
    )
    return chart


def _vehicle_bar_chart(traffic_counts: dict, dark: bool) -> alt.Chart:
    """Current vehicle counts per direction – with axis titles."""
    df = pd.DataFrame({
        "Direction": list(traffic_counts.keys()),
        "Vehicles": list(traffic_counts.values()),
    })
    text_color = "#e6edf3" if dark else "#1a1d23"
    grid_color = "#30363d" if dark else "#e2e6ec"
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, opacity=0.9)
        .encode(
            x=alt.X("Direction:N", title="Approach Direction", axis=alt.Axis(labelColor=text_color, titleColor=text_color, labelAngle=0)),
            y=alt.Y("Vehicles:Q", title="Number of Vehicles", axis=alt.Axis(labelColor=text_color, titleColor=text_color, gridColor=grid_color)),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(
                    domain=["North", "South", "East", "West"],
                    range=["#4f6df5", "#22c55e", "#f59e0b", "#ef4444"],
                ),
                legend=None,
            ),
            tooltip=["Direction:N", "Vehicles:Q"],
        )
        .properties(height=220)
        .configure_view(strokeWidth=0)
        .configure(background="transparent")
    )
    return chart


def _signal_lights_html(colors: dict, dark: bool) -> str:
    """Render the four signal lights as styled HTML."""
    bg = "#21262d" if dark else "#f1f3f8"
    text = "#e6edf3" if dark else "#1a1d23"
    color_map = {
        "green": "#22c55e",
        "yellow": "#eab308",
        "red": "#ef4444",
    }
    glow_map = {
        "green": "0 0 18px #22c55e80",
        "yellow": "0 0 18px #eab30880",
        "red": "0 0 18px #ef444480",
    }
    parts = [
        '<div style="display:flex;gap:20px;justify-content:space-around;'
        'padding:22px;background:{};border-radius:14px;border:1px solid {}">'
        .format(bg, "#30363d" if dark else "#e2e6ec")
    ]
    for direction in DIRECTIONS:
        c = colors[direction]
        hex_c = color_map[c]
        glow = glow_map[c]
        parts.append(
            '<div style="text-align:center;color:{text}">'
            '<div style="margin:auto;width:58px;height:58px;border-radius:50%;'
            'background:{color};border:3px solid {border};box-shadow:{glow};'
            'transition:all 0.3s ease"></div>'
            '<strong style="display:block;margin-top:8px;font-size:0.95rem">{dir}</strong>'
            '<span style="font-size:0.8rem;opacity:0.75;text-transform:uppercase;'
            'letter-spacing:0.05em">{state}</span></div>'
            .format(text=text, color=hex_c, border="#111" if dark else "#bbb",
                    glow=glow, dir=direction, state=c.upper())
        )
    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    import streamlit as st

    tick_started = time.monotonic()
    st.set_page_config(
        page_title="Smart Traffic Signal Management",
        page_icon="🚦",
        layout="wide",
    )

    # ── Session state initialisation ──────────────────────────────────────
    if "controller" not in st.session_state:
        st.session_state.controller = SignalController()
        st.session_state.simulation = IntersectionSimulation()
        st.session_state.running = False
        st.session_state.runtime = {"monitors": {}, "temporary_files": []}
        st.session_state.traffic_counts = {direction: 0 for direction in DIRECTIONS}
        st.session_state.waiting = {direction: 0.0 for direction in DIRECTIONS}
        st.session_state.frames = {}
        st.session_state.history = []
        st.session_state.last_tick = time.monotonic()
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True  # default to dark

    dark = st.session_state.dark_mode

    # ── Inject theme CSS ──────────────────────────────────────────────────
    st.markdown(_DARK_CSS if dark else _LIGHT_CSS, unsafe_allow_html=True)

    # ── Header row ────────────────────────────────────────────────────────
    header_left, header_right = st.columns([5, 1])
    with header_left:
        st.markdown("# 🚦 AI Smart Traffic Signal Management")
        st.caption("Real-time adaptive signal control — simulated signals only")
    with header_right:
        st.markdown("")  # spacer
        theme_label = "☀️ Light Mode" if dark else "🌙 Dark Mode"
        if st.button(theme_label, use_container_width=True, key="theme_toggle"):
            st.session_state.dark_mode = not dark
            st.rerun()

    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        mode = st.selectbox(
            "Traffic input",
            ("Simulation", "Uploaded videos", "CCTV streams"),
            disabled=st.session_state.running,
        )
        uploads = {}
        stream_urls = {}
        model_path = st.text_input("YOLOv8 model", value="yolov8n.pt")
        confidence = st.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)
        if mode == "Uploaded videos":
            for direction in DIRECTIONS:
                uploads[direction] = st.file_uploader(
                    "{} approach video".format(direction),
                    type=("mp4", "avi", "mov", "mkv"),
                    key="upload_{}".format(direction),
                    disabled=st.session_state.running,
                )
        elif mode == "CCTV streams":
            for direction in DIRECTIONS:
                stream_urls[direction] = st.text_input(
                    "{} stream URL".format(direction),
                    key="stream_{}".format(direction),
                    placeholder="rtsp://... or http(s)://...",
                    disabled=st.session_state.running,
                )

        st.markdown("---")
        start_col, pause_col, reset_col = st.columns(3)
        start_clicked = start_col.button("▶ Start", disabled=st.session_state.running, use_container_width=True)
        pause_clicked = pause_col.button("⏸ Pause", disabled=not st.session_state.running, use_container_width=True)
        reset_clicked = reset_col.button("⟲ Reset", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔀 Manual Override")
        override_phase = st.selectbox("Next phase", ("NS", "EW"))
        override_clicked = st.button("Request Override", use_container_width=True)
        clear_override_clicked = st.button("Clear Override", use_container_width=True)
        st.caption("Overrides take effect at the next safe phase boundary.")

        # Sidebar footer info
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center;opacity:0.6;font-size:0.75rem">'
            "AI Traffic Controller v2.0<br>Academic Prototype"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Button logic ──────────────────────────────────────────────────────
    if start_clicked:
        _close_runtime(st.session_state.runtime)
        if mode == "Simulation":
            st.session_state.runtime = {"monitors": {}, "temporary_files": []}
            st.session_state.running = True
            st.session_state.last_tick = time.monotonic()
        else:
            try:
                st.session_state.runtime = _prepare_video_runtime(
                    mode, uploads, stream_urls, model_path, confidence
                )
                st.session_state.running = True
                st.session_state.last_tick = time.monotonic()
            except (OSError, RuntimeError, ValueError) as exc:
                st.error("Unable to start video monitoring: {}".format(exc))
    if pause_clicked:
        st.session_state.running = False
    if reset_clicked:
        st.session_state.running = False
        _close_runtime(st.session_state.runtime)
        st.session_state.controller.reset()
        st.session_state.simulation.reset()
        st.session_state.traffic_counts = {direction: 0 for direction in DIRECTIONS}
        st.session_state.waiting = {direction: 0.0 for direction in DIRECTIONS}
        st.session_state.frames = {}
        st.session_state.history = []
        st.session_state.last_tick = time.monotonic()
    if override_clicked:
        st.session_state.controller.request_manual_phase(override_phase)
    if clear_override_clicked:
        st.session_state.controller.clear_manual_override()

    # ── Tick logic ────────────────────────────────────────────────────────
    signal = st.session_state.controller.snapshot()
    latest_class_counts = {}
    if st.session_state.running:
        now = time.monotonic()
        delta_seconds = min(MAX_TICK_SECONDS, max(0.0, now - st.session_state.last_tick))
        st.session_state.last_tick = now
        if mode == "Simulation":
            frames, counts, waits, latest_class_counts = st.session_state.simulation.step(
                delta_seconds, signal["lights"]
            )
            st.session_state.frames = frames
            st.session_state.traffic_counts = counts
            st.session_state.waiting = waits
        else:
            for direction, monitor in st.session_state.runtime["monitors"].items():
                try:
                    result = monitor.read(delta_seconds)
                except (RuntimeError, OSError) as exc:
                    st.error("{} feed error: {}".format(direction, exc))
                    continue
                if result is None:
                    st.warning("{} feed ended or returned no frame.".format(direction))
                    continue
                st.session_state.frames[direction] = result["frame"]
                st.session_state.traffic_counts[direction] = result["total"]
                st.session_state.waiting[direction] = result["waiting"]
                latest_class_counts[direction] = result["counts"]
        signal = st.session_state.controller.update(
            delta_seconds,
            st.session_state.traffic_counts,
            st.session_state.waiting,
        )
        st.session_state.history.append({
            "time": time.strftime("%H:%M:%S"),
            "North": st.session_state.traffic_counts["North"],
            "South": st.session_state.traffic_counts["South"],
            "East": st.session_state.traffic_counts["East"],
            "West": st.session_state.traffic_counts["West"],
            "green_seconds": signal["green_duration"],
        })
        st.session_state.history = st.session_state.history[-120:]
    elif mode == "Simulation" and not st.session_state.frames:
        frames, _, _, _ = st.session_state.simulation.step(0.0, signal["lights"])
        st.session_state.frames = frames

    density = summarize_traffic(st.session_state.traffic_counts)
    total_vehicles = sum(st.session_state.traffic_counts.values())

    # ── KPI summary row ──────────────────────────────────────────────────
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("🚗 Total Vehicles", total_vehicles)
    kpi_cols[1].metric("🟢 Current Phase", signal["phase"])
    kpi_cols[2].metric("⏱️ Countdown", "{:.1f}s".format(signal["remaining"]))
    kpi_cols[3].metric("📊 Green Alloc.", "{:.1f}s".format(signal["green_duration"]))
    busiest_dir = max(st.session_state.traffic_counts, key=st.session_state.traffic_counts.get)
    kpi_cols[4].metric("🔥 Busiest", "{} ({})".format(busiest_dir, st.session_state.traffic_counts[busiest_dir]))

    st.divider()

    # ── Live feeds ────────────────────────────────────────────────────────
    st.markdown("### 📹 Live Approach Feeds")
    feed_columns = st.columns(4)
    for column, direction in zip(feed_columns, DIRECTIONS):
        with column:
            light_color = signal["lights"][direction]
            badge_colors = {"green": "#22c55e", "yellow": "#eab308", "red": "#ef4444"}
            st.markdown(
                '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                '<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
                'background:{};box-shadow:0 0 8px {}80"></span>'
                '<strong style="font-size:1rem">{}</strong></div>'.format(
                    badge_colors[light_color], badge_colors[light_color], direction
                ),
                unsafe_allow_html=True,
            )
            frame = st.session_state.frames.get(direction)
            if frame is not None:
                st.image(frame, channels="RGB", use_container_width=True)
            else:
                st.info("No frame available.")
            st.metric("Vehicles", st.session_state.traffic_counts[direction])
            st.progress(float(density[direction]["ratio"]))
            st.caption(
                "**{}** density · waiting **{:.0f}s**".format(
                    density[direction]["density"],
                    st.session_state.waiting[direction],
                )
            )
            classes = latest_class_counts.get(direction)
            if classes is not None:
                st.caption(
                    "🚗{car}  🏍️{motorcycle}  🚌{bus}  🚛{truck}".format(**classes)
                )

    st.divider()

    # ── Signal status + bar chart row ─────────────────────────────────────
    st.markdown("### 🚦 Intersection Signals")
    sig_left, sig_right = st.columns([3, 2])

    with sig_left:
        st.markdown(_signal_lights_html(signal["lights"], dark), unsafe_allow_html=True)
        st.markdown("")
        # Signal details
        detail_cols = st.columns(4)
        detail_cols[0].metric("Phase", signal["phase"])
        detail_cols[1].metric("Stage", signal["stage"].replace("_", " ").title())
        detail_cols[2].metric("Remaining", "{:.1f}s".format(signal["remaining"]))
        detail_cols[3].metric("Green Duration", "{:.1f}s".format(signal["green_duration"]))
        if signal["manual_override"]:
            st.info("🔀 Manual override queued: **{}**".format(signal["manual_override"]))

    with sig_right:
        st.markdown("#### 📊 Current Vehicle Distribution")
        st.altair_chart(
            _vehicle_bar_chart(st.session_state.traffic_counts, dark),
            use_container_width=True,
        )

    st.divider()

    # ── Charts section ────────────────────────────────────────────────────
    st.markdown("### 📈 Traffic & Timing History")
    if st.session_state.history:
        chart_col_left, chart_col_right = st.columns(2)
        with chart_col_left:
            st.markdown("#### Vehicle Counts Over Time")
            st.altair_chart(
                _traffic_line_chart(st.session_state.history, dark),
                use_container_width=True,
            )
        with chart_col_right:
            st.markdown("#### Green Allocation Over Time")
            st.altair_chart(
                _green_area_chart(st.session_state.history, dark),
                use_container_width=True,
            )
    else:
        st.info("▶ Start the simulation or connect all four video feeds to build chart history.")

    st.divider()

    # ── Additional information panels ─────────────────────────────────────
    st.markdown("### 📋 System Information")
    info_left, info_mid, info_right = st.columns(3)

    with info_left:
        st.markdown("#### ⏳ Wait Times")
        for d in DIRECTIONS:
            wait_val = st.session_state.waiting[d]
            emoji = "🟢" if wait_val < 15 else ("🟡" if wait_val < 30 else "🔴")
            st.markdown("{} **{}**: {:.1f}s".format(emoji, d, wait_val))
        # Average wait
        avg_wait = sum(st.session_state.waiting.values()) / max(1, len(st.session_state.waiting))
        st.metric("Avg. Wait Time", "{:.1f}s".format(avg_wait))

    with info_mid:
        st.markdown("#### 🔬 Phase Demand")
        demand = signal.get("demand", {})
        if demand:
            for phase_name, phase_data in demand.items():
                vehicles = phase_data.get("vehicles", 0)
                wait_s = phase_data.get("waiting_seconds", 0)
                st.markdown(
                    "**{}** — {:.0f} vehicles, max wait {:.1f}s".format(
                        phase_name, vehicles, wait_s
                    )
                )
        else:
            st.caption("Demand data will appear once the simulation is running.")

        st.markdown("#### 🏷️ Density Summary")
        for d in DIRECTIONS:
            label = density[d]["density"]
            color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(label, "⚪")
            st.markdown("{} **{}**: {} ({} vehicles)".format(
                color, d, label, st.session_state.traffic_counts[d]
            ))

    with info_right:
        st.markdown("#### 🧠 Algorithm Details")
        st.markdown(
            "This controller uses a **demand-responsive adaptive** algorithm:\n\n"
            "1. **Volume weighting** ({:.1f}×) prioritises congested approaches\n"
            "2. **Wait bonus** ({:.2f}×) prevents starvation\n"
            "3. **Starvation cap** at {:.0f}s forces phase changes\n"
            "4. **Saturation headway** of {:.1f}s per vehicle\n"
            "5. Green time range: **{:.0f}–{:.0f}s**".format(
                1.4, 0.12, 60.0, 2.0, 10.0, 45.0,
            )
        )
        if st.session_state.history:
            recent = st.session_state.history[-20:]
            avg_green = sum(r["green_seconds"] for r in recent) / max(1, len(recent))
            avg_total = sum(r["North"] + r["South"] + r["East"] + r["West"] for r in recent) / max(1, len(recent))
            st.metric("Avg. Green (last 20)", "{:.1f}s".format(avg_green))
            st.metric("Avg. Vehicles (last 20)", "{:.0f}".format(avg_total))

    # ── Footer ────────────────────────────────────────────────────────────
    st.divider()
    footer_left, footer_right = st.columns(2)
    with footer_left:
        if st.session_state.running:
            st.caption("🟢 Dashboard is **live** — refreshing every ~1 second.")
        else:
            st.caption("⏸ Dashboard is **paused**.")
    with footer_right:
        st.caption(
            "⚠️ Simulation and scheduling are for demonstration only. "
            "The software does not control physical signals or guarantee real-world traffic safety."
        )

    # ── Auto-refresh ──────────────────────────────────────────────────────
    if st.session_state.running:
        # Sleep only for what is left of the tick so detection time is not
        # added on top of the 1 s refresh period.
        time.sleep(max(0.05, TICK_SECONDS - (time.monotonic() - tick_started)))
        st.rerun()


if __name__ == "__main__":
    main()
