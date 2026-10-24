"""Streamlit dashboard for the academic smart traffic signal prototype."""

import os
import tempfile
import time
from typing import Dict

from config import DIRECTIONS, MAX_TICK_SECONDS, TICK_SECONDS
from intersection_simulation import IntersectionSimulation
from signal_controller import SignalController
from traffic_density import summarize_traffic


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


def main() -> None:
    import streamlit as st

    tick_started = time.monotonic()
    st.set_page_config(page_title="Smart Traffic Signal Management", layout="wide")
    st.title("AI Smart Traffic Signal Management")
    st.caption(
        " simulated signals only "
        
    )

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

    with st.sidebar:
        st.header("Input and controls")
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

        start_col, pause_col, reset_col = st.columns(3)
        start_clicked = start_col.button("Start", disabled=st.session_state.running, use_container_width=True)
        pause_clicked = pause_col.button("Pause", disabled=not st.session_state.running, use_container_width=True)
        reset_clicked = reset_col.button("Reset", use_container_width=True)

        override_phase = st.selectbox("Manual next phase", ("NS", "EW"))
        override_clicked = st.button("Request override", use_container_width=True)
        clear_override_clicked = st.button("Clear override", use_container_width=True)
        st.caption("Overrides take effect at the next safe phase boundary.")

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

    signal = st.session_state.controller.snapshot()
    feed_columns = st.columns(4)
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
    st.subheader("Live approach feeds")
    for column, direction in zip(feed_columns, DIRECTIONS):
        with column:
            st.markdown("#### {}".format(direction))
            frame = st.session_state.frames.get(direction)
            if frame is not None:
                st.image(frame, channels="RGB", use_container_width=True)
            else:
                st.info("No frame available.")
            st.metric("Vehicles in view", st.session_state.traffic_counts[direction])
            st.progress(float(density[direction]["ratio"]))
            st.caption(
                "{} density · waiting {:.0f}s".format(
                    density[direction]["density"],
                    st.session_state.waiting[direction],
                )
            )
            classes = latest_class_counts.get(direction)
            if classes is not None:
                st.caption(
                    "Cars {car} · Motorcycles {motorcycle} · Buses {bus} · Trucks {truck}".format(
                        **classes
                    )
                )

    st.subheader("Intersection signals")
    phase_col, light_columns = st.columns([1, 4])
    phase_col.metric("Current phase", signal["phase"])
    phase_col.metric("Stage", signal["stage"].replace("_", " ").title())
    phase_col.metric("Countdown", "{:.1f} s".format(signal["remaining"]))
    phase_col.metric("Allocated green", "{:.1f} s".format(signal["green_duration"]))
    if signal["manual_override"]:
        phase_col.info("Next phase override: {}".format(signal["manual_override"]))
    colors = signal["lights"]
    html = ['<div style="display:flex;gap:16px;justify-content:space-around;'
            'padding:18px;background:#252a30;border-radius:12px">']
    for direction in DIRECTIONS:
        color = {"green": "#29c45a", "yellow": "#f0c62e", "red": "#e94343"}[colors[direction]]
        html.append(
            '<div style="text-align:center;color:white"><div style="margin:auto;width:54px;'
            'height:54px;border-radius:50%;background:{};border:3px solid #111"></div>'
            '<strong>{}</strong><br>{}</div>'.format(color, direction, colors[direction].upper())
        )
    html.append("</div>")
    light_columns.markdown("".join(html), unsafe_allow_html=True)

    st.subheader("Traffic and timing history")
    if st.session_state.history:
        chart_data = {
            key: [row[key] for row in st.session_state.history]
            for key in ("North", "South", "East", "West")
        }
        st.line_chart(chart_data, height=220)
        st.area_chart(
            {"Green allocation (s)": [row["green_seconds"] for row in st.session_state.history]},
            height=160,
        )
    else:
        st.info("Start the simulation or connect all four video feeds to build chart history.")

    if st.session_state.running:
        st.caption("Dashboard ticks about once per second while running.")

    st.caption(
        "Simulation and scheduling are for demonstration only. The software does not "
        "control physical signals or guarantee real-world traffic safety."
    )

    if st.session_state.running:
        # Sleep only for what is left of the tick so detection time is not
        # added on top of the 1 s refresh period.
        time.sleep(max(0.05, TICK_SECONDS - (time.monotonic() - tick_started)))
        st.rerun()


if __name__ == "__main__":
    main()
