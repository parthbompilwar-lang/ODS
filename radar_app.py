"""
AI Football Tactical Radar & Spatial Heatmap Platform.
Streamlit Localhost Interactive Web Application.
Features:
1. Side-by-Side Live Broadcast & 2D Tactical Pitch Radar.
2. Team Compactness Convex Hulls & Team Centroids (Team Center of Mass).
3. Post-Match Spatial Heatmap Studio (Team Territory & Individual Player Trails).
4. Physical Tactical Metrics (Pitch Control %, Defensive Line Height, Compactness m^2).
"""

import os
import cv2
import tempfile
import numpy as np
import imageio
import streamlit as st
import torch
from ultralytics import YOLO
from typing import Optional, Dict, Any, Tuple, List

from src.tracking.player_tracker import PlayerTracker
from src.tracking.ball_tracker import BallTracker
from src.team.team_classifier import TeamClassifier
from src.geometry.pitch_geometry import PitchGeometry
from src.visualization.tactical_radar_heatmap import TacticalRadarEngine


STYLE_CSS = """<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0f172a;
    text-align: center;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
}
.sub-header {
    font-size: 0.95rem;
    font-weight: 600;
    color: #475569;
    text-align: center;
    margin-bottom: 22px;
    letter-spacing: 0.8px;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.metric-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.55rem;
    font-weight: 900;
    color: #0f172a;
}
.badge-team0 {
    background-color: #2563eb;
    color: white;
    padding: 4px 10px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.9rem;
}
.badge-team1 {
    background-color: #dc2626;
    color: white;
    padding: 4px 10px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.9rem;
}
</style>"""


@st.cache_resource
def load_yolo_model():
    """Loads trained 4-class YOLO detector."""
    model_path = "models/yolo11_v2_4class_best.pt"
    if not os.path.exists(model_path):
        model_path = "yolo11n.pt"
    return YOLO(model_path)


def get_pitch_geometry(width: int, height: int) -> PitchGeometry:
    """Calibrated sideline broadcast homography mapping."""
    cam_pts = np.array([
        [width * 0.23, height * 0.22],   # 18-yard line at far touchline (X=16.5m, Y=0.0m)
        [width * 0.78, height * 0.22],   # Halfway line at far touchline (X=52.5m, Y=0.0m)
        [width * 0.86, height * 0.91],   # Halfway line at near touchline (X=52.5m, Y=68.0m)
        [width * 0.24, height * 0.91]    # 18-yard line at near touchline (X=16.5m, Y=68.0m)
    ], dtype=np.float32)
    pitch_pts = np.array([
        [16.5, 0.0],
        [52.5, 0.0],
        [52.5, 68.0],
        [16.5, 68.0]
    ], dtype=np.float32)
    geom = PitchGeometry(pitch_length=105.0, pitch_width=68.0)
    geom.set_homography(cam_pts, pitch_pts)
    return geom


def process_tactical_stream(
    input_video_path: str,
    output_video_path: str,
    show_hulls: bool = True,
    show_centroids: bool = True,
    progress_bar = None,
    status_text = None
) -> Tuple[TacticalRadarEngine, Dict[str, Any]]:
    """
    Processes video stream, executes tracking + homography projection,
    and produces synchronized side-by-side broadcast + 2D tactical radar video.
    """
    model = load_yolo_model()
    device = "0" if torch.cuda.is_available() else "cpu"

    player_tracker = PlayerTracker(high_conf_thresh=0.40, low_conf_thresh=0.15, cost_thresh=0.65)
    ball_tracker = BallTracker(max_missing_frames=5, initial_gating_dist=85.0)
    team_classifier = TeamClassifier(n_teams=2, vote_window=10, min_votes_required=2)
    radar_engine = TacticalRadarEngine(pitch_length=105.0, pitch_width=68.0)

    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    geom = get_pitch_geometry(in_w, in_h)

    # Output Dimensions: Side-by-Side (640x360 broadcast + 640x360 radar = 1280x360)
    pane_w, pane_h = 640, 380
    out_w = pane_w * 2
    out_h = pane_h

    writer = imageio.get_writer(
        output_video_path,
        fps=fps,
        codec='libx264',
        format='FFMPEG',
        pixelformat='yuv420p'
    )

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. YOLO Inference
        res = model(frame, imgsz=1280, device=device, conf=0.12, verbose=False)[0]
        p_dets = []
        b_dets = []
        for box in res.boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [float(v) for v in box.xyxy[0].cpu().numpy().tolist()]
            if cid in [0, 1, 2]:
                p_dets.append({"bbox": xyxy, "class_id": cid, "conf": conf})
            elif cid == 3:
                b_dets.append({"bbox": xyxy, "conf": conf})

        # 2. Tracking & Team Classification
        p_tracks = player_tracker.update(p_dets, frame_index=frame_idx)
        b_state = ball_tracker.update(b_dets, frame_index=frame_idx)
        p_tracks = team_classifier.classify_tracks(frame, p_tracks, frame_index=frame_idx)

        ball_pos = b_state.get("position")

        # 3. Update Tactical Radar Engine
        radar_engine.update_frame(p_tracks, ball_pos=ball_pos, pitch_geom=geom)

        # 4. Render Broadcast Left Pane
        broadcast_pane = frame.copy()
        for p in p_tracks:
            bx1, by1, bx2, by2 = map(int, p["bbox"])
            t_id = p.get("team_id", -1)
            role = p.get("role", "player")

            if role == "referee":
                col = (180, 50, 180)
                lbl = "REF"
            elif role == "goalkeeper":
                col = (0, 220, 255)
                lbl = "GK"
            elif t_id == 0:
                col = (240, 160, 40)
                lbl = f"T1 #{p.get('track_id', '')}"
            elif t_id == 1:
                col = (40, 70, 235)
                lbl = f"T2 #{p.get('track_id', '')}"
            else:
                col = (150, 150, 150)
                lbl = f"#{p.get('track_id', '')}"

            cv2.rectangle(broadcast_pane, (bx1, by1), (bx2, by2), col, 2, cv2.LINE_AA)
            cv2.putText(broadcast_pane, lbl, (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        if ball_pos is not None:
            cx, cy = map(int, ball_pos)
            cv2.circle(broadcast_pane, (cx, cy), 8, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(broadcast_pane, (cx, cy), 10, (255, 255, 255), 2, cv2.LINE_AA)

        # Header on broadcast pane
        cv2.rectangle(broadcast_pane, (0, 0), (in_w, 40), (15, 15, 15), -1)
        cv2.putText(broadcast_pane, f"MATCH BROADCAST FEED | FRAME #{frame_idx}", (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # 5. Render 2D Tactical Radar Right Pane
        radar_pane = radar_engine.render_tactical_radar(
            players=p_tracks,
            ball_pos=ball_pos,
            pitch_geom=geom,
            width=pane_w,
            height=pane_h,
            margin=20,
            show_convex_hulls=show_hulls,
            show_centroids=show_centroids
        )

        # Resize broadcast pane to fit side-by-side
        resized_broadcast = cv2.resize(broadcast_pane, (pane_w, pane_h))

        # Combine into side-by-side canvas
        combined = np.hstack([resized_broadcast, radar_pane])

        # Convert to RGB for video writer
        rgb_frame = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)

        frame_idx += 1
        if progress_bar and total_frames > 0:
            progress_bar.progress(min(1.0, frame_idx / total_frames))
        if status_text:
            status_text.text(f"Analyzing frame {frame_idx}/{total_frames}...")

    cap.release()
    writer.close()

    metrics = radar_engine.compute_tactical_metrics()
    return radar_engine, metrics


def main():
    try:
        st.set_page_config(
            page_title="AI Football Tactical Radar & Heatmap Studio",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass

    st.html(STYLE_CSS)
    st.html('<div class="main-header">⚽ AI FOOTBALL TACTICAL RADAR & HEATMAP STUDIO</div>')
    st.html('<div class="sub-header">TOP-DOWN 2D METRIC PROJECTION • TEAM COMPACTNESS HULLS • GAUSSIAN KDE SPATIAL HEATMAPS</div>')

    with st.sidebar:
        st.header("⚙️ Radar Configuration")
        show_hulls = st.checkbox("Show Team Compactness Polygons (Convex Hull)", value=True)
        show_centroids = st.checkbox("Show Team Center of Mass (Centroids)", value=True)
        kde_smooth = st.slider("Heatmap Kernel Smoothing:", min_value=25, max_value=95, value=51, step=2)

        st.markdown("---")
        st.subheader("Team Legend")
        st.markdown('<span class="badge-team0">TEAM 1 (White / Home)</span>', unsafe_allow_html=True)
        st.write("")
        st.markdown('<span class="badge-team1">TEAM 2 (Navy / Away)</span>', unsafe_allow_html=True)
        st.write("")
        st.caption("🟡 Yellow: Goalkeepers | 🟣 Purple: Referees | 🔴 Red: Match Ball")

    tab_video, tab_heatmaps, tab_analytics = st.tabs([
        "📹 Match Radar & Broadcast Stream",
        "🔥 Tactical Heatmap Studio",
        "📊 Sports Science Analytics"
    ])

    # -------------------------------------------------------------
    # TAB 1: Video Ingestion & Synchronized Radar Stream
    # -------------------------------------------------------------
    with tab_video:
        st.subheader("Match Footage Ingestion & Real-Time Tracking")
        video_choice = st.radio(
            "Select Video Source:",
            [
                "Tottenham Hotspur vs Leeds (offside_spurs_match.mp4)",
                "Continuous Match Footage (continuous_match_sample.mp4)",
                "Standard Benchmark Clip (sample_match.mp4)",
                "Upload Custom Football Match (.mp4)"
            ],
            index=0,
            horizontal=True
        )

        input_path = None
        if video_choice == "Tottenham Hotspur vs Leeds (offside_spurs_match.mp4)":
            if os.path.exists("offside_spurs_match.mp4"):
                input_path = "offside_spurs_match.mp4"
            else:
                st.error("offside_spurs_match.mp4 not found.")
        elif video_choice == "Continuous Match Footage (continuous_match_sample.mp4)":
            if os.path.exists("continuous_match_sample.mp4"):
                input_path = "continuous_match_sample.mp4"
            else:
                st.error("continuous_match_sample.mp4 not found.")
        elif video_choice == "Standard Benchmark Clip (sample_match.mp4)":
            if os.path.exists("sample_match.mp4"):
                input_path = "sample_match.mp4"
            else:
                st.error("sample_match.mp4 not found.")
        else:
            uploaded = st.file_uploader("Upload Football Match MP4:", type=["mp4", "mov", "avi"], key="radar_up")
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded.read())
                input_path = tfile.name

        if input_path and os.path.exists(input_path):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📥 Source Match Feed")
                st.video(input_path)

            output_radar_video = "output_tactical_radar.mp4"

            with col2:
                st.markdown("### 🚀 Real-Time Tactical Radar Analysis")
                if st.button("Generate Tactical Radar Stream", type="primary", use_container_width=True):
                    prog = st.progress(0.0)
                    stat = st.empty()
                    engine, metrics = process_tactical_stream(
                        input_video_path=input_path,
                        output_video_path=output_radar_video,
                        show_hulls=show_hulls,
                        show_centroids=show_centroids,
                        progress_bar=prog,
                        status_text=stat
                    )
                    st.session_state["radar_engine"] = engine
                    st.session_state["radar_metrics"] = metrics
                    st.session_state["radar_video"] = output_radar_video
                    st.success("Tactical Radar & Spatial Heatmaps successfully generated!")

            if "radar_video" in st.session_state and os.path.exists(st.session_state["radar_video"]):
                st.markdown("### 🖥️ Synchronized Broadcast & 2D Tactical Radar")
                st.video(st.session_state["radar_video"])

                m = st.session_state["radar_metrics"]
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.html(f'<div class="metric-card"><div class="metric-title">Team 1 Pitch Control</div><div class="metric-value">{m["team0_control_pct"]}%</div></div>')
                with m2:
                    st.html(f'<div class="metric-card"><div class="metric-title">Team 2 Pitch Control</div><div class="metric-value">{m["team1_control_pct"]}%</div></div>')
                with m3:
                    st.html(f'<div class="metric-card"><div class="metric-title">Tracked Players</div><div class="metric-value">{m["tracked_players_count"]}</div></div>')
                with m4:
                    st.html(f'<div class="metric-card"><div class="metric-title">Total Observations</div><div class="metric-value">{m["team0_samples"] + m["team1_samples"]}</div></div>')

    # -------------------------------------------------------------
    # TAB 2: Tactical Heatmap Studio
    # -------------------------------------------------------------
    with tab_heatmaps:
        st.subheader("Tactical Heatmap Studio (Kernel Density Estimation)")
        st.caption("Thermal visualization of accumulated team spatial territory and individual player movement patterns.")

        if "radar_engine" not in st.session_state:
            st.info("💡 Run the video analysis in Tab 1 first to accumulate spatial trajectories for heatmap rendering.")
        else:
            engine = st.session_state["radar_engine"]
            hm_col1, hm_col2 = st.columns(2)

            with hm_col1:
                st.markdown("#### 🔵 Team 1 Territory Heatmap")
                heat_t0 = engine.generate_heatmap(
                    engine.team0_positions,
                    width=720, height=480,
                    colormap=cv2.COLORMAP_JET,
                    kernel_size=kde_smooth,
                    title="TEAM 1 (HOME) SPATIAL DOMINANCE"
                )
                st.image(cv2.cvtColor(heat_t0, cv2.COLOR_BGR2RGB), use_container_width=True)

            with hm_col2:
                st.markdown("#### 🔴 Team 2 Territory Heatmap")
                heat_t1 = engine.generate_heatmap(
                    engine.team1_positions,
                    width=720, height=480,
                    colormap=cv2.COLORMAP_HOT,
                    kernel_size=kde_smooth,
                    title="TEAM 2 (AWAY) SPATIAL DOMINANCE"
                )
                st.image(cv2.cvtColor(heat_t1, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🏃 Individual Player Movement Heatmap")
            available_tracks = sorted(list(engine.player_trajectories.keys()))
            if available_tracks:
                pick_track = st.selectbox("Select Player Track ID:", available_tracks, index=0)
                player_pts = engine.player_trajectories.get(pick_track, [])
                heat_p = engine.generate_heatmap(
                    player_pts,
                    width=800, height=520,
                    colormap=cv2.COLORMAP_TURBO,
                    kernel_size=kde_smooth,
                    title=f"INDIVIDUAL TRAIL: PLAYER TRACK #{pick_track}"
                )
                st.image(cv2.cvtColor(heat_p, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"Total positions tracked for Player #{pick_track}: {len(player_pts)} frames")

    # -------------------------------------------------------------
    # TAB 3: Sports Science Analytics
    # -------------------------------------------------------------
    with tab_analytics:
        st.subheader("Professional Sports Science & Tactical Report")
        st.caption("Quantitative metrics computed from physical 2D pitch coordinates.")

        if "radar_metrics" not in st.session_state:
            st.info("💡 Run the video analysis in Tab 1 to compute tactical analytics.")
        else:
            m = st.session_state["radar_metrics"]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.html(f"""
                <div class="metric-card">
                    <div class="metric-title">Defensive Line Height</div>
                    <div class="metric-value">T1: {m['team0_def_line_m']}m | T2: {m['team1_def_line_m']}m</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Distance from respective defending goal lines</div>
                </div>""")
            with c2:
                st.html(f"""
                <div class="metric-card">
                    <div class="metric-title">Team Tactical Compactness</div>
                    <div class="metric-value">T1: {m['team0_compactness_m2']} m² | T2: {m['team1_compactness_m2']} m²</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Average Convex Hull Area (lower = more compact)</div>
                </div>""")
            with c3:
                st.html(f"""
                <div class="metric-card">
                    <div class="metric-title">Final Third Penetration</div>
                    <div class="metric-value">T1: {m['team0_attacking_third_pct']}% | T2: {m['team1_attacking_third_pct']}%</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Percentage of presence in opponent final 35m</div>
                </div>""")

            st.write("")
            st.markdown("### 📋 Executive Summary Table")
            summary_table = [
                {"Tactical Metric": "Pitch Territory Control (%)", "Team 1 (Home)": f"{m['team0_control_pct']}%", "Team 2 (Away)": f"{m['team1_control_pct']}%", "Tactical Significance": "Possession & field occupancy dominance"},
                {"Tactical Metric": "Team Center of Mass (X, Y)", "Team 1 (Home)": f"({m['team0_centroid'][0]}m, {m['team0_centroid'][1]}m)", "Team 2 (Away)": f"({m['team1_centroid'][0]}m, {m['team1_centroid'][1]}m)", "Tactical Significance": "High press vs mid-block positioning"},
                {"Tactical Metric": "Defensive Line Height", "Team 1 (Home)": f"{m['team0_def_line_m']} m", "Team 2 (Away)": f"{m['team1_def_line_m']} m", "Tactical Significance": "Offside trap depth and backline discipline"},
                {"Tactical Metric": "Tactical Compactness Area", "Team 1 (Home)": f"{m['team0_compactness_m2']} m²", "Team 2 (Away)": f"{m['team1_compactness_m2']} m²", "Tactical Significance": "Spatial organization and defensive structure"},
                {"Tactical Metric": "Final-Third Penetration Rate", "Team 1 (Home)": f"{m['team0_attacking_third_pct']}%", "Team 2 (Away)": f"{m['team1_attacking_third_pct']}%", "Tactical Significance": "Offensive threat and danger creation"}
            ]
            st.table(summary_table)


if __name__ == "__main__":
    main()
