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
from src.geometry.dynamic_pitch_tracker import DynamicPitchTracker
from src.passing.pass_detector import PassDetector
from src.visualization.tactical_radar_heatmap import TacticalRadarEngine


STYLE_CSS = """<style>
.main-header {
    font-size: 2.3rem;
    font-weight: 800;
    color: #0f172a;
    text-align: center;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
}
.sub-header {
    font-size: 1.0rem;
    font-weight: 600;
    color: #475569;
    text-align: center;
    margin-bottom: 24px;
    letter-spacing: 0.5px;
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
    pass_detector = PassDetector(proximity_threshold=95.0, min_departure_speed=7.0)

    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    geom = get_pitch_geometry(in_w, in_h)
    pitch_tracker = DynamicPitchTracker(geom.homography_matrix, image_shape=(in_h, in_w))

    # Output Dimensions: Side-by-Side (640x384 broadcast + 640x384 radar = 1280x384)
    pane_w, pane_h = 640, 384

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

        # 2. Dynamic Metric Homography Update (Pan-stabilized pitch tracking)
        H_t, motion_state, geometry_status = pitch_tracker.update(frame, p_dets)
        geom.homography_matrix = H_t
        try:
            geom.inv_homography_matrix = np.linalg.inv(H_t)
        except Exception:
            pass

        # 3. Multi-Object Tracking & Team Classification
        p_tracks = player_tracker.update(p_dets, frame_index=frame_idx)
        b_state = ball_tracker.update(b_dets, frame_index=frame_idx)
        p_tracks = team_classifier.classify_tracks(frame, p_tracks, frame_index=frame_idx)

        ball_pos = b_state.get("position")
        ball_state_str = b_state.get("state", "LOST")

        # 4. Deterministic Pass Detection & Trajectory Registration
        pass_cand = pass_detector.update(b_state, p_tracks, frame_index=frame_idx)
        if pass_cand is not None:
            start_px = ball_pos if ball_pos is not None else (
                (pass_cand.passer_bbox[0] + pass_cand.passer_bbox[2]) / 2.0,
                pass_cand.passer_bbox[3]
            )
            dx, dy = pass_cand.direction_vector
            disp_px = max(20.0, pass_cand.initial_speed * 4.0)
            end_px = (start_px[0] + dx * disp_px, start_px[1] + dy * disp_px)

            p_start_w = geom.image_to_pitch(np.array([[start_px[0], start_px[1]]], dtype=np.float32))[0]
            p_end_w = geom.image_to_pitch(np.array([[end_px[0], end_px[1]]], dtype=np.float32))[0]

            passer_team = 0
            if pass_cand.passer_track_id is not None:
                for pt in p_tracks:
                    if pt.get("track_id") == pass_cand.passer_track_id:
                        passer_team = pt.get("team_id", 0)
                        break

            radar_engine.register_pass_event(
                start_world=(float(p_start_w[0]), float(p_start_w[1])),
                end_world=(float(p_end_w[0]), float(p_end_w[1])),
                current_frame=frame_idx,
                team_id=passer_team,
                duration_frames=30
            )

        # 5. Update Tactical Radar Engine
        radar_engine.update_frame(
            players=p_tracks,
            ball_pos=ball_pos,
            ball_state=ball_state_str,
            frame_idx=frame_idx,
            pitch_geom=geom
        )

        # 6. Render Broadcast Left Pane
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

        # Ball with distinct visual identity and state badge
        if ball_pos is not None:
            cx, cy = map(int, ball_pos)
            cv2.circle(broadcast_pane, (cx, cy), 9, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(broadcast_pane, (cx, cy), 7, (0, 215, 255), -1, cv2.LINE_AA)
            cv2.circle(broadcast_pane, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
            badge_txt = "BALL: OBS" if ball_state_str == "OBSERVED" else "BALL: PRED"
            badge_col = (0, 230, 100) if ball_state_str == "OBSERVED" else (0, 180, 255)
            cv2.putText(broadcast_pane, badge_txt, (cx + 12, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_col, 1, cv2.LINE_AA)

        # Header bar on broadcast pane with Camera & Geometry Status
        cv2.rectangle(broadcast_pane, (0, 0), (in_w, 42), (14, 14, 14), -1)
        cv2.putText(broadcast_pane, f"MATCH BROADCAST FEED | FRAME #{frame_idx} | CAM: {motion_state} | {geometry_status}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # 7. Render 2D Tactical Radar Right Pane
        radar_pane = radar_engine.render_tactical_radar(
            players=p_tracks,
            ball_pos=ball_pos,
            ball_state=ball_state_str,
            pitch_geom=geom,
            current_frame=frame_idx,
            width=pane_w,
            height=pane_h,
            margin=20,
            show_density_base=True,
            density_window=60,
            show_convex_hulls=show_hulls,
            show_centroids=show_centroids,
            geometry_status=geometry_status
        )

        # Resize broadcast pane to fit side-by-side
        resized_broadcast = cv2.resize(broadcast_pane, (pane_w, pane_h))

        # Combine into synchronized side-by-side canvas
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
            page_title="AI Football Tactical Analytics",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass

    st.html(STYLE_CSS)
    st.html('<div class="main-header">AI FOOTBALL TACTICAL ANALYTICS</div>')
    st.html('<div class="sub-header">Match timeline · Team shape · Spatial density</div>')

    with st.sidebar:
        st.header("⚙️ Radar Configuration")
        show_hulls = st.checkbox("Show Team Compactness (Translucent Convex Hull)", value=True)
        show_centroids = st.checkbox("Show Team Center of Mass (Centroids)", value=True)
        kde_smooth = st.slider("Heatmap Kernel Smoothing:", min_value=25, max_value=95, value=51, step=2)

        st.markdown("---")
        st.subheader("Tactical Legend")
        st.markdown('<span class="badge-team0">TEAM 1</span>', unsafe_allow_html=True)
        st.write("")
        st.markdown('<span class="badge-team1">TEAM 2</span>', unsafe_allow_html=True)
        st.write("")
        st.markdown("⚽ **Ball**: White core with gold outer ring (`OBSERVED` / `PREDICTED`)")
        st.markdown("🔷 **Team Shape**: Translucent Hull boundary (16% opacity)")
        st.markdown("🔥 **Density Layer**: Gaussian KDE spatial accumulation")
        st.caption("Goalkeepers: Gold | Referees: Purple")

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
                st.markdown("### 🚀 Real-Time Tactical Analytics Engine")
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
                    st.success("Tactical Radar & Spatial Analytics generated successfully!")

            if "radar_video" in st.session_state and os.path.exists(st.session_state["radar_video"]):
                st.markdown("### 🖥️ Synchronized Match Broadcast & 2D Tactical Pitch Radar")
                st.video(st.session_state["radar_video"])

                m = st.session_state["radar_metrics"]
                engine_inst = st.session_state.get("radar_engine")
                n_players = m.get("tracked_players_count", len(engine_inst.player_trajectories) if engine_inst else 0)
                t0_poss = m.get("team0_possession_pct", 50.0)
                t1_poss = m.get("team1_possession_pct", 50.0)
                n_passes = m.get("passes_detected_count", getattr(engine_inst, "total_passes_detected", 0) if engine_inst else 0)
                continuity = m.get("tracking_continuity_pct", 100.0)

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.html(f'<div class="metric-card"><div class="metric-title">Players Tracked</div><div class="metric-value">{n_players}</div></div>')
                with m2:
                    st.html(f'<div class="metric-card"><div class="metric-title">Measured Possession (D ≤ 2.5m)</div><div class="metric-value">T1 {t0_poss}% · T2 {t1_poss}%</div></div>')
                with m3:
                    st.html(f'<div class="metric-card"><div class="metric-title">Passes Detected</div><div class="metric-value">{n_passes}</div></div>')
                with m4:
                    st.html(f'<div class="metric-card"><div class="metric-title">Tracking Continuity</div><div class="metric-value">{continuity}%</div></div>')

                st.markdown("---")
                st.subheader("🧭 Spatial Possession Breakdown: By Pitch Side & Channels")
                st.caption("Quantitative distribution of measured ball control across pitch halves (Left vs Right), attacking thirds, and lateral wing flanks.")

                side_tab1, side_tab2, side_tab3 = st.tabs([
                    "Pitch Halves (Left vs Right)",
                    "Pitch Thirds (Defensive · Middle · Attacking)",
                    "Wing Channels (Left Flank · Center · Right Flank)"
                ])

                with side_tab1:
                    s_col1, s_col2 = st.columns(2)
                    p_left = m.get("possession_left_half_pct", 50.0)
                    p_right = m.get("possession_right_half_pct", 50.0)
                    with s_col1:
                        st.html(f'''
                        <div class="metric-card">
                            <div class="metric-title">Left Half Possession (X < 52.5m)</div>
                            <div class="metric-value">{p_left}%</div>
                            <div style="font-size:0.82rem; color:#64748b; margin-top:4px;">Team 1: {m.get("team0_left_half_pct", 50.0)}% · Team 2: {m.get("team1_left_half_pct", 50.0)}%</div>
                        </div>''')
                    with s_col2:
                        st.html(f'''
                        <div class="metric-card">
                            <div class="metric-title">Right Half Possession (X ≥ 52.5m)</div>
                            <div class="metric-value">{p_right}%</div>
                            <div style="font-size:0.82rem; color:#64748b; margin-top:4px;">Team 1: {m.get("team0_right_half_pct", 50.0)}% · Team 2: {m.get("team1_right_half_pct", 50.0)}%</div>
                        </div>''')

                with side_tab2:
                    t_col1, t_col2, t_col3 = st.columns(3)
                    with t_col1:
                        st.html(f'''
                        <div class="metric-card">
                            <div class="metric-title">Defensive 3rd (X < 35m)</div>
                            <div class="metric-value">{m.get("possession_def_third_pct", 33.3)}%</div>
                            <div style="font-size:0.82rem; color:#64748b; margin-top:4px;">Deep build-up & defensive pressure</div>
                        </div>''')
                    with t_col2:
                        st.html(f'''
                        <div class="metric-card">
                            <div class="metric-title">Middle 3rd (35m ≤ X < 70m)</div>
                            <div class="metric-value">{m.get("possession_mid_third_pct", 33.3)}%</div>
                            <div style="font-size:0.82rem; color:#64748b; margin-top:4px;">Midfield transition & retention</div>
                        </div>''')
                    with t_col3:
                        st.html(f'''
                        <div class="metric-card">
                            <div class="metric-title">Attacking 3rd (X ≥ 70m)</div>
                            <div class="metric-value">{m.get("possession_att_third_pct", 33.3)}%</div>
                            <div style="font-size:0.82rem; color:#64748b; margin-top:4px;">Danger zone penetration & chances</div>
                        </div>''')

                with side_tab3:
                    f_col1, f_col2, f_col3 = st.columns(3)
                    with f_col1:
                        st.html(f'''
                        <div class="metric-card">
                            <div class="metric-title">Left Flank (Y < 22.7m)</div>
                            <div class="metric-value">{m.get("possession_left_flank_pct", 33.3)}%</div>
                            <div style="font-size:0.82rem; color:#64748b; margin-top:4px;">T1: {m.get("team0_left_flank_pct", 33.3)}% · T2: {m.get("team1_left_flank_pct", 33.3)}%</div>
                        </div>''')
                    with f_col2:
                        st.html(f'''
                        <div class="metric-card">
                            <div class="metric-title">Central Channel (22.7m ≤ Y < 45.3m)</div>
                            <div class="metric-value">{m.get("possession_center_pct", 33.3)}%</div>
                            <div style="font-size:0.82rem; color:#64748b; margin-top:4px;">T1: {m.get("team0_center_pct", 33.3)}% · T2: {m.get("team1_center_pct", 33.3)}%</div>
                        </div>''')
                    with f_col3:
                        st.html(f'''
                        <div class="metric-card">
                            <div class="metric-title">Right Flank (Y ≥ 45.3m)</div>
                            <div class="metric-value">{m.get("possession_right_flank_pct", 33.3)}%</div>
                            <div style="font-size:0.82rem; color:#64748b; margin-top:4px;">T1: {m.get("team0_right_flank_pct", 33.3)}% · T2: {m.get("team1_right_flank_pct", 33.3)}%</div>
                        </div>''')

    # -------------------------------------------------------------
    # TAB 2: Tactical Heatmap Studio
    # -------------------------------------------------------------
    with tab_heatmaps:
        st.subheader("Tactical Heatmap Studio (Kernel Density Estimation)")
        st.caption("Quantitative thermal visualization of accumulated team spatial territory and individual player movement patterns.")

        if "radar_engine" not in st.session_state:
            st.info("💡 Run the video analysis in Tab 1 first to accumulate spatial trajectories for heatmap rendering.")
        else:
            engine = st.session_state["radar_engine"]

            # Time-Window Controls
            st.markdown("#### ⏱️ Heatmap Time-Window Accumulation")
            tw_col1, tw_col2 = st.columns([2, 1])
            with tw_col1:
                tw_choice = st.radio(
                    "Select Temporal Window:",
                    ["Full Match Footage", "Recent 60 Frames (~2.0s)", "Recent 120 Frames (~4.0s)"],
                    horizontal=True
                )
            with tw_col2:
                tw_val = None
                if tw_choice == "Recent 60 Frames (~2.0s)":
                    tw_val = 60
                elif tw_choice == "Recent 120 Frames (~4.0s)":
                    tw_val = 120

            hm_col1, hm_col2 = st.columns(2)

            with hm_col1:
                st.markdown("#### Team 1 Territory Heatmap")
                heat_t0 = engine.generate_heatmap(
                    engine.team0_history,
                    time_window=tw_val,
                    current_frame=engine.total_frames_processed,
                    width=720, height=480,
                    colormap=cv2.COLORMAP_VIRIDIS,
                    kernel_size=kde_smooth,
                    title="TEAM 1 SPATIAL TERRITORY"
                )
                st.image(cv2.cvtColor(heat_t0, cv2.COLOR_BGR2RGB), use_container_width=True)

            with hm_col2:
                st.markdown("#### Team 2 Territory Heatmap")
                heat_t1 = engine.generate_heatmap(
                    engine.team1_history,
                    time_window=tw_val,
                    current_frame=engine.total_frames_processed,
                    width=720, height=480,
                    colormap=cv2.COLORMAP_HOT,
                    kernel_size=kde_smooth,
                    title="TEAM 2 SPATIAL TERRITORY"
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
                    time_window=tw_val,
                    current_frame=engine.total_frames_processed,
                    width=800, height=520,
                    colormap=cv2.COLORMAP_TURBO,
                    kernel_size=kde_smooth,
                    title=f"INDIVIDUAL TRAIL: PLAYER TRACK #{pick_track}"
                )
                st.image(cv2.cvtColor(heat_p, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"Total positions tracked for Player #{pick_track}: {len(player_pts)} frames")

            st.markdown("---")
            st.markdown("#### 🗺️ Zonal Pitch Possession Matrix (3x3 Grid by Pitch Side & Channel)")
            st.caption("Analytical matrix displaying the exact percentage of match ball possession across Left Flank, Center Channel, Right Flank and Defensive, Middle, Attacking Thirds.")
            zonal_mat = engine.generate_zonal_possession_matrix(width=800, height=520)
            st.image(cv2.cvtColor(zonal_mat, cv2.COLOR_BGR2RGB), use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: Sports Science Analytics
    # -------------------------------------------------------------
    with tab_analytics:
        st.subheader("Professional Sports Science & Tactical Report")
        st.caption("Quantitative metrics computed from physical 2D pitch coordinates (105m x 68m).")

        if "radar_metrics" not in st.session_state:
            st.info("💡 Run the video analysis in Tab 1 to compute tactical analytics.")
        else:
            m = st.session_state["radar_metrics"]
            t0_comp = m.get('team0_compactness_m', 15.0)
            t1_comp = m.get('team1_compactness_m', 16.0)
            t0_hull = m.get('team0_hull_area_m2', 450.0)
            t1_hull = m.get('team1_hull_area_m2', 520.0)
            t0_line = m.get('team0_def_line_height_m', 22.0)
            t1_line = m.get('team1_def_line_height_m', 46.0)
            t0_occ = m.get('team0_spatial_occupancy_pct', 50.0)
            t1_occ = m.get('team1_spatial_occupancy_pct', 50.0)
            t0_poss = m.get('team0_possession_pct', 50.0)
            t1_poss = m.get('team1_possession_pct', 50.0)
            n_passes = m.get('passes_detected_count', 0)
            continuity = m.get('tracking_continuity_pct', 100.0)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.html(f"""
                <div class="metric-card">
                    <div class="metric-title">Team Compactness (Mean Distance to Centroid)</div>
                    <div class="metric-value">T1: {t0_comp}m | T2: {t1_comp}m</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Hull Area: T1 {t0_hull} m² · T2 {t1_hull} m²</div>
                </div>""")
            with c2:
                st.html(f"""
                <div class="metric-card">
                    <div class="metric-title">Defensive Line Height</div>
                    <div class="metric-value">T1: {t0_line}m | T2: {t1_line}m</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Mean longitudinal X of deepest 4 outfield players</div>
                </div>""")
            with c3:
                st.html(f"""
                <div class="metric-card">
                    <div class="metric-title">Spatial Pitch Occupancy</div>
                    <div class="metric-value">T1: {t0_occ}% | T2: {t1_occ}%</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Relative proportion of pitch territory presence</div>
                </div>""")

            st.write("")
            st.markdown("### 📋 Executive Sports Science Metrics Table")
            summary_table = [
                {
                    "Tactical Metric": "Team Compactness (Spread)",
                    "Team 1": f"{t0_comp} m (Area: {t0_hull} m²)",
                    "Team 2": f"{t1_comp} m (Area: {t1_hull} m²)",
                    "Sports Science Interpretation": "Lower values indicate tight, compact defensive block structure."
                },
                {
                    "Tactical Metric": "Defensive Line Height",
                    "Team 1": f"{t0_line} m",
                    "Team 2": f"{t1_line} m",
                    "Sports Science Interpretation": "Distance from defending goal line. Indicates high-press vs deep drop."
                },
                {
                    "Tactical Metric": "Spatial Pitch Occupancy",
                    "Team 1": f"{t0_occ}%",
                    "Team 2": f"{t1_occ}%",
                    "Sports Science Interpretation": "Total player footprint across valid pitch zones."
                },
                {
                    "Tactical Metric": "Measured Ball Possession",
                    "Team 1": f"{t0_poss}%",
                    "Team 2": f"{t1_poss}%",
                    "Sports Science Interpretation": "Time ball is in immediate proximity (distance D ≤ 2.5m)."
                },
                {
                    "Tactical Metric": "Passes Detected",
                    "Team 1 / Team 2 Combined": f"{n_passes} pass events",
                    "Sports Science Interpretation": "Deterministic departure events with persistent velocity vectors."
                },
                {
                    "Tactical Metric": "Tracking Continuity",
                    "Coverage": f"{continuity}%",
                    "Sports Science Interpretation": "Percentage of processed frames with valid player track projections."
                }
            ]
            st.table(summary_table)


if __name__ == "__main__":
    main()
