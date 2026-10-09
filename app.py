"""
AI Video Offside Detection Platform (ODS V2) — Streamlit Localhost GUI.
Full End-to-End Pipeline with Immutable Evidence Synchronization:
- YOLO11 V2 (4-Class: Player, Goalkeeper, Referee, Ball @ 1280px)
- Unsupervised Jersey Color Clustering (TeamClassifier in CIE-Lab space with temporal majority voting)
- Multi-Object Tracking (PlayerTracker + BallTracker with explicit OBSERVED state)
- Event-Driven Pass Candidate Detection (PassDetector)
- Contact Moment Estimation (ContactEstimator strictly on direct OBSERVED ball evidence)
- Law 11 State Machine (PlayStateMachine with immutable EvidenceSnapshot)
- Projected Perspective Offside Pitch Lines & Pitch-Clipped Shaded Zone via Calibrated Geometry
- Distinct Semantics for OFFSIDE POSITION vs OFFSIDE OFFENCE
- Visible UNDETERMINED states for Invalid Calibration or Missing Ball Evidence
"""

import os
import cv2
import json
import tempfile
import numpy as np
import imageio
import streamlit as st
from ultralytics import YOLO
import torch
from typing import Optional, Dict, Any, Tuple, List

from src.tracking.player_tracker import PlayerTracker
from src.tracking.ball_tracker import BallTracker
from src.passing.pass_detector import PassDetector
from src.passing.contact_estimator import ContactEstimator
from src.engine.play_state import PlayStateMachine, PlayStateEnum
from src.engine.offside_engine import OffsideEngine, OffsideResult
from src.engine.evidence_snapshot import EvidenceSnapshot
from src.geometry.pitch_geometry import PitchGeometry
from src.team.team_classifier import TeamClassifier

STYLE_CSS = """<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1a1a24;
    text-align: center;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
}
.sub-header {
    font-size: 0.95rem;
    font-weight: 600;
    color: #555566;
    text-align: center;
    margin-bottom: 24px;
    letter-spacing: 1px;
}
.status-box-offside {
    background: linear-gradient(135deg, #e90052, #8b002f);
    color: white;
    padding: 16px;
    border-radius: 8px;
    text-align: center;
    font-weight: 900;
    font-size: 1.5rem;
    letter-spacing: 1px;
    box-shadow: 0 4px 15px rgba(233,0,82,0.3);
}
.status-box-position {
    background: linear-gradient(135deg, #f59e0b, #b45309);
    color: white;
    padding: 16px;
    border-radius: 8px;
    text-align: center;
    font-weight: 900;
    font-size: 1.5rem;
    letter-spacing: 1px;
    box-shadow: 0 4px 15px rgba(245,158,11,0.3);
}
.status-box-onside {
    background: linear-gradient(135deg, #00ff85, #008f4c);
    color: #1a1a24;
    padding: 16px;
    border-radius: 8px;
    text-align: center;
    font-weight: 900;
    font-size: 1.5rem;
    letter-spacing: 1px;
    box-shadow: 0 4px 15px rgba(0,255,133,0.3);
}
.status-box-undetermined {
    background: linear-gradient(135deg, #374151, #1f2937);
    color: #fbbf24;
    padding: 16px;
    border: 2px solid #fbbf24;
    border-radius: 8px;
    text-align: center;
    font-weight: 800;
    font-size: 1.35rem;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(251,191,36,0.2);
}
.metric-card {
    background: #f8f9fc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}
.metric-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #718096;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.35rem;
    font-weight: 800;
    color: #2d3748;
}
</style>"""


@st.cache_resource
def load_yolo_v2_model():
    """Loads the trained 4-class YOLO11n V2 detector."""
    model_path = "models/yolo11_v2_4class_best.pt"
    if not os.path.exists(model_path):
        model_path = "yolo11n.pt"
    return YOLO(model_path)


def get_default_pitch_geometry(width: int, height: int) -> PitchGeometry:
    """
    Creates a calibrated pitch geometry for standard sideline broadcast camera view.
    Length (X: 0m to 105m) runs horizontally across the screen (Leeds goal left, Spurs goal right).
    Width (Y: 0m to 68m) runs vertically from far touchline (top) to near touchline (bottom).
    """
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


def process_video_stream(
    input_path: str,
    output_path: str,
    attack_direction: str = "right",
    attack_team_id: int = 0,
    overlay_mode: str = "VAR Mode",
    progress_bar = None,
    status_text = None
):
    """
    Executes the complete event-driven Law 11 pipeline with synchronized evidence.
    """
    model = load_yolo_v2_model()
    device = "0" if torch.cuda.is_available() else "cpu"

    player_tracker = PlayerTracker(high_conf_thresh=0.40, low_conf_thresh=0.15, cost_thresh=0.65)
    ball_tracker = BallTracker(max_missing_frames=5, initial_gating_dist=85.0)
    pass_detector = PassDetector(proximity_threshold=85.0, min_departure_speed=6.5, min_departure_displacement=20.0)
    contact_estimator = ContactEstimator(search_half_window=6)
    play_machine = PlayStateMachine(freeze_duration_frames=45)  # 3s freeze on offside
    team_classifier = TeamClassifier(n_teams=2, vote_window=10, min_votes_required=2)

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 15.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    geom = get_default_pitch_geometry(width, height)
    offside_engine = OffsideEngine(pitch_geometry=geom)

    # Standard web render resolution: 1280x720
    out_w = 1280
    out_h = int(height * (out_w / width))
    if out_h % 2 != 0:
        out_h += 1

    # ImageIO libx264 writer for direct browser playback
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',
        format='FFMPEG',
        pixelformat='yuv420p'
    )

    frame_index = 0
    sliding_tracking_history = []
    primary_snapshot: Optional[EvidenceSnapshot] = None
    all_snapshots = []
    os.makedirs("offside_incidents", exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Run 4-Class YOLO11 V2 Inference
        results = model(frame, imgsz=1280, device=device, conf=0.10, verbose=False)[0]

        player_dets = []
        ball_dets = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [float(v) for v in box.xyxy[0].cpu().numpy().tolist()]

            if cls_id in [0, 1, 2]:  # Player (0), Goalkeeper (1), Referee (2)
                player_dets.append({
                    "bbox": xyxy,
                    "class_id": cls_id,
                    "conf": conf
                })
            elif cls_id == 3:  # Ball (3)
                ball_dets.append({"bbox": xyxy, "conf": conf})

        # 2. Update Multi-Object Trackers
        p_tracks = player_tracker.update(player_dets, frame_index=frame_index)
        b_state = ball_tracker.update(ball_dets, frame_index=frame_index)

        # 3. Classify Teams using Online CIE-Lab Clustering & Temporal Majority Voting
        p_tracks = team_classifier.classify_tracks(frame, p_tracks, frame_index=frame_index)

        # 4. Maintain sliding history for contact search window (save frame image for exact freeze)
        hist_rec = {
            "frame": frame_index,
            "ball_state": b_state,
            "ball_pos": b_state.get("position"),
            "player_tracks": p_tracks,
            "frame_image": frame.copy()
        }
        sliding_tracking_history.append(hist_rec)
        if len(sliding_tracking_history) > 20:
            sliding_tracking_history.pop(0)

        # 5. Pass Detection & Contact Moment Estimation
        pass_cand = pass_detector.update(b_state, p_tracks, frame_index=frame_index)
        contact_est = None
        offside_res = None
        contact_frame_img = frame

        if pass_cand is not None:
            contact_est = contact_estimator.estimate_contact(pass_cand, sliding_tracking_history)
            if contact_est is not None:
                # Retrieve exact frozen frame at contact frame t_hat*
                for r in sliding_tracking_history:
                    if r["frame"] == contact_est.frame_hat and "frame_image" in r:
                        contact_frame_img = r["frame_image"]
                        break

                # Perform Law 11 Evaluation strictly at contact moment t_hat*
                offside_res = offside_engine.evaluate_contact_moment(
                    players=p_tracks,
                    attack_team_id=attack_team_id,
                    attack_direction=attack_direction,
                    ball_pos=contact_est.ball_position,
                    image_shape=(height, width)
                )

        # 6. Update Law 11 State Machine with exact contact image
        curr_state, should_freeze, active_snap = play_machine.update(
            frame_index=frame_index,
            frame_image=contact_frame_img,
            pass_candidate=pass_cand,
            contact_estimate=contact_est,
            offside_result=offside_res
        )

        # 7. Render Output Frame
        if should_freeze and active_snap is not None:
            assert active_snap.frame_index >= 0, "FAILED: Invalid snapshot frame index"

            # Render Broadcast-Grade VAR Evidence Frame with Pitch-Clipped Shading
            def render_broadcast_evidence(base_img):
                img = base_img.copy()
                ih, iw = img.shape[:2]
                dir_tag = "+X (right)" if attack_direction.lower() == "right" else ("-X (left)" if attack_direction.lower() == "left" else attack_direction.upper())

                # 1. Pitch Shaded Zone (strictly clipped to pitch surface)
                if active_snap.offside_line_endpoints is not None and geom is not None:
                    try:
                        img = geom.create_offside_shaded_overlay(
                            img,
                            active_snap.offside_line_endpoints,
                            goal_direction=attack_direction,
                            boundary_world_x=active_snap.offside_boundary_world_x,
                            alpha=0.42
                        )
                    except TypeError:
                        img = geom.create_offside_shaded_overlay(
                            img,
                            active_snap.offside_line_endpoints,
                            goal_direction=attack_direction,
                            color=(10, 30, 10),
                            alpha=0.42
                        )

                # 2. Render Calibrated Offside Boundary Line (Red, Clipped to Playable Pitch)
                if active_snap.offside_line_endpoints is not None:
                    p1, p2 = active_snap.offside_line_endpoints
                    if geom is not None and hasattr(geom, 'clip_line_to_pitch'):
                        p1, p2 = geom.clip_line_to_pitch(p1, p2, (ih, iw))
                    cv2.line(img, p1, p2, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.line(img, p1, p2, (0, 0, 235), 2, cv2.LINE_AA)
                    cv2.putText(img, "OFFSIDE BOUNDARY", (20, max(20, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 235), 1, cv2.LINE_AA)

                # 3. Render Attacker Line if present (Cyan/White, Clipped to Playable Pitch)
                if active_snap.attacker_line_endpoints is not None:
                    ap1, ap2 = active_snap.attacker_line_endpoints
                    if geom is not None and hasattr(geom, 'clip_line_to_pitch'):
                        ap1, ap2 = geom.clip_line_to_pitch(ap1, ap2, (ih, iw))
                    cv2.line(img, ap1, ap2, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.line(img, ap1, ap2, (240, 240, 240), 2, cv2.LINE_AA)
                    cv2.putText(img, f"ATTACKER (+{active_snap.margin_val:.2f}{active_snap.margin_unit})", (20, min(ih - 10, ap1[1] + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

                # 4. Ball evidence at contact
                if active_snap.ball_evidence_state == "OBSERVED" and hasattr(active_snap, 'ball_pos') and active_snap.ball_pos is not None:
                    bx, by = map(int, active_snap.ball_pos)
                    cv2.circle(img, (bx, by), 8, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(img, (bx, by), 10, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, "BALL [CONTACT]", (bx + 12, by + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

                # Check boundary uncertainty status
                b_status = getattr(active_snap, 'boundary_status', 'DECISIVE')
                err_budget = getattr(active_snap, 'empirical_error_budget_m', 0.59)

                # 5. Premier League Style Top-Left TV Decision Graphic
                box_w = 530 if b_status == "MARGINAL" else 440
                box_h = 46
                bx = 30
                by = 24
                cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), (50, 0, 45), -1)
                cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), (255, 255, 255), 1, cv2.LINE_AA)

                if active_snap.decision == "UNDETERMINED":
                    dec_title = "DECISION: UNDETERMINED"
                elif active_snap.decision == "OFFSIDE":
                    state_lbl = getattr(active_snap, 'law11_state', 'OFFSIDE_POSITION')
                    if b_status == "MARGINAL":
                        dec_title = "VAR: OFFSIDE POSITION (MARGINAL)"
                    elif state_lbl == "OFFSIDE_OFFENCE":
                        dec_title = "DECISION NO GOAL - OFFSIDE"
                    else:
                        dec_title = "VAR: OFFSIDE POSITION DETECTED"
                else:
                    if b_status == "MARGINAL":
                        dec_title = "DECISION - ONSIDE (MARGINAL)"
                    else:
                        dec_title = "DECISION - GOAL / ONSIDE"

                cv2.putText(img, dec_title, (bx + 16, by + 31), cv2.FONT_HERSHEY_DUPLEX, 0.60 if b_status == "MARGINAL" else 0.68, (255, 255, 255), 2, cv2.LINE_AA)

                # 6. Technical Telemetry Sub-pill at bottom left
                sub_w = 660 if b_status == "MARGINAL" else 540
                sub_h = 32
                sby = ih - 48
                cv2.rectangle(img, (bx, sby), (bx + sub_w, sby + sub_h), (20, 20, 20), -1)
                cv2.rectangle(img, (bx, sby), (bx + sub_w, sby + sub_h), (80, 80, 80), 1, cv2.LINE_AA)
                m_sign = "+" if active_snap.margin_val > 0 else ""
                if b_status == "MARGINAL":
                    sub_txt = f"CONTACT #{active_snap.frame_index} | MARGIN: {m_sign}{active_snap.margin_val:.2f}{active_snap.margin_unit} [MARGINAL ±{err_budget:.2f}m] | REVIEW RECOMMENDED"
                else:
                    sub_txt = f"CONTACT #{active_snap.frame_index} | MARGIN: {m_sign}{active_snap.margin_val:.2f}{active_snap.margin_unit} [DECISIVE] | ATTACK: TEAM {attack_team_id} ({dir_tag})"
                cv2.putText(img, sub_txt, (bx + 10, sby + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40 if b_status == "MARGINAL" else 0.44, (230, 230, 230), 1, cv2.LINE_AA)

                return img

            if primary_snapshot is None:
                primary_snapshot = active_snap
                all_snapshots.append(active_snap)

                snap_path = os.path.join("offside_incidents", f"offside_contact_frame_{active_snap.frame_index}.jpg")
                annotated_snap = render_broadcast_evidence(active_snap.image)
                cv2.imwrite(snap_path, annotated_snap)

            annotated = render_broadcast_evidence(active_snap.image)
        else:
            # Normal Live Play Rendering
            annotated = frame.copy()
            dir_tag = "+X (right)" if attack_direction.lower() == "right" else ("-X (left)" if attack_direction.lower() == "left" else attack_direction.upper())

            # Draw Players
            for p in p_tracks:
                bx1, by1, bx2, by2 = map(int, p["bbox"])
                role = p.get("role", "player")
                t_id = p.get("team_id", -1)

                if role == "referee":
                    col = (180, 50, 180)  # Purple for Referee
                    lbl = f"REF #{p.get('track_id', '')}"
                elif role == "goalkeeper":
                    col = (0, 220, 255)   # Yellow for Goalkeeper
                    lbl = f"GK #{p.get('track_id', '')}"
                elif t_id == 0:
                    col = (235, 120, 30)  # Blue/Cyan for Team 0
                    lbl = f"T0 #{p.get('track_id', '')}"
                elif t_id == 1:
                    col = (40, 60, 220)   # Orange/Red for Team 1
                    lbl = f"T1 #{p.get('track_id', '')}"
                else:
                    col = (160, 160, 160) # Neutral Gray for UNKNOWN
                    lbl = f"? #{p.get('track_id', '')}"

                thickness = 2 if overlay_mode.startswith("Debug") else 1
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), col, thickness, cv2.LINE_AA)
                cv2.putText(annotated, lbl, (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

            # Ball Marker Rendering: strictly distinguish OBSERVED, PREDICTED, and LOST
            if b_state["state"] == "OBSERVED" and b_state["position"] is not None:
                cx, cy = map(int, b_state["position"])
                cv2.circle(annotated, (cx, cy), 8, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, (cx, cy), 10, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated, "BALL [OBS]", (cx + 12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            elif b_state["state"] == "PREDICTED" and b_state["position"] is not None:
                cx, cy = map(int, b_state["position"])
                cv2.circle(annotated, (cx, cy), 9, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.circle(annotated, (cx, cy), 2, (0, 220, 255), -1, cv2.LINE_AA)
                age = b_state.get("consecutive_missing", 1)
                cv2.putText(annotated, f"BALL [PRED +{age}]", (cx + 12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
            # When LOST: DO NOT DRAW ANY MARKER AT ALL. Zero stale ghost positions.

            # Persistent Top HUD Bar
            cv2.rectangle(annotated, (0, 0), (width, 45), (20, 20, 20), -1)
            hud_txt = f"MATCH PLAY | FRAME #{frame_index} | ATTACKING: TEAM {attack_team_id} ({dir_tag}) | BALL: {b_state['state']}"
            cv2.putText(annotated, hud_txt, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

        resized = cv2.resize(annotated, (out_w, out_h))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)

        frame_index += 1

        if progress_bar and total_frames > 0:
            progress_bar.progress(min(1.0, frame_index / total_frames))
        if status_text:
            status_text.text(f"Processing frame {frame_index}/{total_frames} (State: {curr_state.value})...")

    cap.release()
    writer.close()

    return {
        "primary_snapshot": primary_snapshot,
        "all_snapshots": all_snapshots,
        "total_frames": total_frames,
        "final_verdict": primary_snapshot.decision if primary_snapshot is not None else "ONSIDE"
    }


def analyze_single_incident_frame(
    image: np.ndarray,
    attack_direction: str = "right",
    attack_team_id: int = 0,
    overlay_mode: str = "VAR Mode"
) -> Tuple[np.ndarray, OffsideResult, Dict[str, Any]]:
    """
    Evaluates a single incident frame with 4-class YOLO, CIE-Lab team clustering,
    Law 11 engine, and pitch-clipped broadcast overlay.
    """
    model = load_yolo_v2_model()
    device = "0" if torch.cuda.is_available() else "cpu"
    h, w = image.shape[:2]

    # 1. 4-Class YOLO Detection
    res = model(image, imgsz=1280, device=device, conf=0.15, verbose=False)[0]
    player_dets = []
    ball_pos = None
    ball_state = "UNOBSERVED"

    for box in res.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = [float(v) for v in box.xyxy[0].cpu().numpy().tolist()]

        if cls_id in [0, 1, 2]:
            player_dets.append({
                "bbox": xyxy,
                "class_id": cls_id,
                "conf": conf,
                "track_id": len(player_dets)
            })
        elif cls_id == 3:
            ball_pos = ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)
            ball_state = "OBSERVED"

    # 2. Team Classification
    tc = TeamClassifier(n_teams=2)
    tc.fit_from_image(image, player_dets)
    classified_players = tc.classify_tracks(image, player_dets)

    # 3. Geometry Calibration
    geom = get_default_pitch_geometry(w, h)

    # 4. Offside Evaluation
    engine = OffsideEngine(pitch_geometry=geom)
    offside_result = engine.evaluate_contact_moment(
        players=classified_players,
        attack_team_id=attack_team_id,
        attack_direction=attack_direction,
        ball_pos=ball_pos,
        image_shape=(h, w)
    )

    # 5. Broadcast Rendering with Pitch-Clipped Shading
    annotated = image.copy()

    # Pitch Shaded Zone
    if offside_result.offside_line is not None:
        try:
            annotated = geom.create_offside_shaded_overlay(
                annotated,
                offside_result.offside_line,
                goal_direction=attack_direction,
                boundary_world_x=offside_result.offside_boundary_world_x,
                alpha=0.42
            )
        except TypeError:
            annotated = geom.create_offside_shaded_overlay(
                annotated,
                offside_result.offside_line,
                goal_direction=attack_direction,
                color=(10, 30, 10),
                alpha=0.42
            )
        # Red Boundary Line (Clipped to Playable Pitch)
        p1, p2 = offside_result.offside_line
        if geom is not None and hasattr(geom, 'clip_line_to_pitch'):
            p1, p2 = geom.clip_line_to_pitch(p1, p2, (h, w))
        cv2.line(annotated, p1, p2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(annotated, p1, p2, (0, 0, 235), 2, cv2.LINE_AA)
        cv2.putText(annotated, "OFFSIDE BOUNDARY", (20, max(20, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 235), 1, cv2.LINE_AA)

    if offside_result.attacker_line is not None:
        ap1, ap2 = offside_result.attacker_line
        if geom is not None and hasattr(geom, 'clip_line_to_pitch'):
            ap1, ap2 = geom.clip_line_to_pitch(ap1, ap2, (h, w))
        cv2.line(annotated, ap1, ap2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(annotated, ap1, ap2, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"ATTACKER LINE (+{offside_result.margin_val:.2f}{offside_result.margin_unit})", (20, min(h - 10, ap1[1] + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

    # In VAR Mode: prominently highlight the key offending attacker and 2nd-last defender
    if overlay_mode.startswith("VAR"):
        if offside_result.offside_players:
            for atk in offside_result.offside_players:
                ax1, ay1, ax2, ay2 = map(int, atk["bbox"])
                cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), (0, 0, 0), 4, cv2.LINE_AA)
                cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), (240, 240, 240), 2, cv2.LINE_AA)
                cv2.putText(annotated, f"OFFENDING ATTACKER (+{offside_result.margin_val:.2f}{offside_result.margin_unit})", (ax1, max(15, ay1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

        if offside_result.second_last_defender:
            df = offside_result.second_last_defender
            dx1, dy1, dx2, dy2 = map(int, df["bbox"])
            cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), (0, 0, 0), 4, cv2.LINE_AA)
            cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), (0, 0, 235), 2, cv2.LINE_AA)
            cv2.putText(annotated, "2ND-LAST DEFENDER", (dx1, max(15, dy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 235), 1, cv2.LINE_AA)

    # Draw Other Player Boxes
    for p in classified_players:
        is_key = False
        if overlay_mode.startswith("VAR"):
            if offside_result.offside_players and any(np.array_equal(p["bbox"], atk["bbox"]) for atk in offside_result.offside_players):
                is_key = True
            if offside_result.second_last_defender and np.array_equal(p["bbox"], offside_result.second_last_defender["bbox"]):
                is_key = True
        if is_key:
            continue

        bx1, by1, bx2, by2 = map(int, p["bbox"])
        role = p.get("role", "player")
        t_id = p.get("team_id", -1)

        if role == "referee":
            col = (180, 50, 180)
            lbl = "REF"
        elif role == "goalkeeper":
            col = (0, 220, 255)
            lbl = "GK"
        elif t_id == 0:
            col = (235, 120, 30)
            lbl = f"T0 #{p['track_id']}"
        elif t_id == 1:
            col = (40, 60, 220)
            lbl = f"T1 #{p['track_id']}"
        else:
            col = (160, 160, 160)
            lbl = "UNKNOWN"

        thickness = 2 if overlay_mode.startswith("Debug") else 1
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), col, thickness, cv2.LINE_AA)
        if overlay_mode.startswith("Debug") or role in ["referee", "goalkeeper"]:
            cv2.putText(annotated, lbl, (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    # Ball marker
    if ball_pos is not None:
        cx, cy = int(ball_pos[0]), int(ball_pos[1])
        cv2.circle(annotated, (cx, cy), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy), 10, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, "BALL [OBS]", (cx + 12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA)

    # Check boundary uncertainty status
    b_status = getattr(offside_result, 'boundary_status', 'DECISIVE')
    err_budget = getattr(offside_result, 'empirical_error_budget_m', 0.59)

    # Premier League Banner
    bx, by = 30, 24
    box_w = 530 if b_status == "MARGINAL" else 440
    box_h = 46
    cv2.rectangle(annotated, (bx, by), (bx + box_w, by + box_h), (50, 0, 45), -1)
    cv2.rectangle(annotated, (bx, by), (bx + box_w, by + box_h), (255, 255, 255), 1, cv2.LINE_AA)

    if offside_result.decision == "UNDETERMINED":
        title_txt = "DECISION: UNDETERMINED"
    elif offside_result.decision == "OFFSIDE":
        if b_status == "MARGINAL":
            title_txt = "VAR: OFFSIDE POSITION (MARGINAL)"
        else:
            title_txt = "VAR: OFFSIDE POSITION DETECTED"
    else:
        if b_status == "MARGINAL":
            title_txt = "DECISION - ONSIDE (MARGINAL)"
        else:
            title_txt = "DECISION - GOAL / ONSIDE"

    cv2.putText(annotated, title_txt, (bx + 16, by + 31), cv2.FONT_HERSHEY_DUPLEX, 0.60 if b_status == "MARGINAL" else 0.68, (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom technical telemetry sub-pill
    sub_w = 660 if b_status == "MARGINAL" else 540
    sub_h = 32
    sby = h - 48
    cv2.rectangle(annotated, (bx, sby), (bx + sub_w, sby + sub_h), (20, 20, 20), -1)
    cv2.rectangle(annotated, (bx, sby), (bx + sub_w, sby + sub_h), (80, 80, 80), 1, cv2.LINE_AA)
    m_sign = "+" if offside_result.margin_val > 0 else ""
    if b_status == "MARGINAL":
        sub_txt = f"INCIDENT REVIEW | MARGIN: {m_sign}{offside_result.margin_val:.2f}{offside_result.margin_unit} [MARGINAL ±{err_budget:.2f}m] | REVIEW RECOMMENDED"
    else:
        sub_txt = f"INCIDENT REVIEW | MARGIN: {m_sign}{offside_result.margin_val:.2f}{offside_result.margin_unit} [DECISIVE] | ATTACK: TEAM {attack_team_id}"
    cv2.putText(annotated, sub_txt, (bx + 10, sby + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40 if b_status == "MARGINAL" else 0.44, (230, 230, 230), 1, cv2.LINE_AA)

    stats = {
        "team_0_count": sum(1 for p in classified_players if p.get("team_id") == 0 and p.get("role") == "player"),
        "team_1_count": sum(1 for p in classified_players if p.get("team_id") == 1 and p.get("role") == "player"),
        "gk_count": sum(1 for p in classified_players if p.get("role") == "goalkeeper"),
        "ref_count": sum(1 for p in classified_players if p.get("role") == "referee"),
        "ball_state": ball_state,
        "ball_position": ball_pos
    }

    return annotated, offside_result, stats


def main():
    try:
        st.set_page_config(
            page_title="AI VAR Offside Detection Platform",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass

    st.html(STYLE_CSS)
    st.html('<div class="main-header">⚽ AI VAR OFFSIDE DETECTOR (ODS V2)</div>')
    st.html('<div class="sub-header">CIE-LAB TEAM CLUSTERING • IMMUTABLE EVIDENCE SNAPSHOT • PITCH-CLIPPED BROADCAST SHADING</div>')

    with st.sidebar:
        st.header("🚩 Tactical & Match Configuration")
        match_preset = st.selectbox(
            "Match Configuration Preset:",
            [
                "Tottenham Hotspur vs Leeds (Spurs attacking LEFT)",
                "Standard Benchmark (Attacking RIGHT)",
                "Custom / Manual Configuration"
            ],
            index=0
        )
        if match_preset == "Tottenham Hotspur vs Leeds (Spurs attacking LEFT)":
            default_dir_idx = 1   # "left"
            default_team_idx = 1  # Team 2 (Spurs)
            st.caption("ℹ️ Preset: Tottenham Hotspur (dark kit, Team 2) attacking LEFT towards Leeds goal.")
        elif match_preset == "Standard Benchmark (Attacking RIGHT)":
            default_dir_idx = 0   # "right"
            default_team_idx = 0  # Team 1
            st.caption("ℹ️ Preset: Attacking Team 1 attacking RIGHT towards 105m.")
        else:
            default_dir_idx = 0
            default_team_idx = 0

        attack_direction = st.selectbox(
            "Attacking Direction (Towards Defending Goal):",
            ["right", "left", "top", "bottom"],
            index=default_dir_idx,
            format_func=lambda x: f"{x.upper()} ({'+X toward 105m' if x=='right' else ('-X toward 0m' if x=='left' else x)})"
        )
        attack_team_id = st.selectbox(
            "Attacking Team:",
            [0, 1],
            index=default_team_idx,
            format_func=lambda x: f"Team {x+1} (Internal: TEAM_{x})"
        )

        st.markdown("---")
        st.header("🖥️ Telemetry & Overlay")
        overlay_mode = st.radio(
            "Overlay Density:",
            ["VAR Mode (Broadcast Clean)", "Debug Mode (Full Telemetry)"],
            index=0,
            help="VAR Mode focuses on the incident: highlighted offending attacker, 2nd-last defender, ball, offside line, and TV decision graphic. Debug Mode shows all 22 player boxes, track IDs, roles, and technical telemetry."
        )

        st.markdown("---")
        st.caption("FIFA Law 11 Rules Invariant:")
        st.caption("• Only players of attacking team evaluated for offside position.")
        st.caption("• Defending team provides second-last opponent baseline.")
        st.caption("• Ball detection required for legal contact verification.")

    tab_video, tab_incident = st.tabs(["📹 Full Match Video Stream", "🖼️ Single Incident Frame VAR Review"])

    # -------------------------------------------------------------
    # TAB 1: Match Video Stream
    # -------------------------------------------------------------
    with tab_video:
        st.subheader("Match Video Ingestion")
        video_choice = st.radio(
            "Select Video Source:",
            [
                "Tottenham Hotspur Match Footage (offside_spurs_match.mp4)",
                "Continuous Match Footage (continuous_match_sample.mp4)",
                "Standard Benchmark Clip (sample_match.mp4)",
                "Upload Custom Match Video (.mp4)"
            ],
            index=0,
            horizontal=True
        )

        input_video_path = None
        if video_choice == "Tottenham Hotspur Match Footage (offside_spurs_match.mp4)":
            if os.path.exists("offside_spurs_match.mp4"):
                input_video_path = "offside_spurs_match.mp4"
            else:
                st.error("offside_spurs_match.mp4 not found.")
        elif video_choice == "Continuous Match Footage (continuous_match_sample.mp4)":
            if os.path.exists("continuous_match_sample.mp4"):
                input_video_path = "continuous_match_sample.mp4"
            else:
                st.error("continuous_match_sample.mp4 not found.")
        elif video_choice == "Standard Benchmark Clip (sample_match.mp4)":
            if os.path.exists("sample_match.mp4"):
                input_video_path = "sample_match.mp4"
            else:
                st.error("sample_match.mp4 not found.")
        else:
            uploaded = st.file_uploader("Upload Match Video:", type=["mp4", "mov", "avi", "mkv"], key="vid_uploader")
            custom_path = st.text_input("Or enter local file path:", value="", placeholder="e.g. C:/Users/ACER/Downloads/offside Spurs .mp4")
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded.read())
                input_video_path = tfile.name
            elif custom_path and os.path.exists(custom_path.strip('\"')):
                input_video_path = custom_path.strip('\"')

        if input_video_path and os.path.exists(input_video_path):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📥 Input Match Video")
                st.video(input_video_path)

            output_video_path = "output_var_match.mp4"

            with col2:
                st.markdown("### 📤 VAR Stream Analysis")
                if st.button("🚀 Run Full Video Offside Detection", type="primary", use_container_width=True, key="btn_run_video"):
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    results = process_video_stream(
                        input_path=input_video_path,
                        output_path=output_video_path,
                        attack_direction=attack_direction,
                        attack_team_id=attack_team_id,
                        overlay_mode=overlay_mode,
                        progress_bar=progress_bar,
                        status_text=status_text
                    )

                    progress_bar.empty()
                    status_text.empty()
                    st.session_state["results"] = results
                    st.session_state["processed_video"] = output_video_path

                if "results" in st.session_state and os.path.exists(st.session_state.get("processed_video", "")):
                    res = st.session_state["results"]
                    snap = res.get("primary_snapshot")

                    if snap is not None:
                        if snap.decision == "UNDETERMINED":
                            st.html(
                                '<div class="status-box-undetermined">UNDETERMINED — CALIBRATION OR EVIDENCE INVALID<br>'
                                f'<span style="font-size: 0.95rem; font-weight: 500;">{snap.explanation}</span></div>'
                            )
                        elif snap.decision == "OFFSIDE":
                            state_lbl = getattr(snap, 'law11_state', 'OFFSIDE_POSITION')
                            box_cls = "status-box-offside" if state_lbl == "OFFSIDE_OFFENCE" else "status-box-position"
                            title_txt = "OFFSIDE COMMITTED — PLAY STOPPED" if state_lbl == "OFFSIDE_OFFENCE" else "OFFSIDE POSITION DETECTED"
                            st.html(
                                f'<div class="{box_cls}">{title_txt} (+{snap.margin_val:.2f}{snap.margin_unit})<br>'
                                f'<span style="font-size: 0.95rem; font-weight: 500;">Evaluated at Contact Moment t* = Frame #{snap.frame_index} (Confidence: {snap.confidence*100:.1f}%)</span></div>'
                            )
                        else:
                            st.html(
                                '<div class="status-box-onside">ONSIDE — PLAY CONTINUES<br>'
                                '<span style="font-size: 0.95rem; font-weight: 500;">All attacking players level or behind defensive baseline</span></div>'
                            )
                    else:
                        st.html(
                            '<div class="status-box-onside">ONSIDE — NO PASS/OFFSIDE OFFENCE DETECTED<br>'
                            '<span style="font-size: 0.95rem; font-weight: 500;">No illegal attacking passes detected during sequence</span></div>'
                        )

                    st.video(st.session_state["processed_video"])

                    # Metrics Summary
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        dec_val = snap.decision if snap else "ONSIDE"
                        st.html(f'<div class="metric-card"><div class="metric-title">VAR Decision</div><div class="metric-value">{dec_val}</div></div>')
                    with m2:
                        frm_val = f"#{snap.frame_index}" if snap else "N/A"
                        st.html(f'<div class="metric-card"><div class="metric-title">Contact Frame</div><div class="metric-value">{frm_val}</div></div>')
                    with m3:
                        mrg_val = f"+{snap.margin_val:.2f} {snap.margin_unit}" if snap else "0.00 m"
                        st.html(f'<div class="metric-card"><div class="metric-title">Offside Margin</div><div class="metric-value">{mrg_val}</div></div>')
                    with m4:
                        ev_val = snap.ball_evidence_state if snap else "N/A"
                        st.html(f'<div class="metric-card"><div class="metric-title">Ball Evidence</div><div class="metric-value">{ev_val}</div></div>')

    # -------------------------------------------------------------
    # TAB 2: Single Incident Frame Review
    # -------------------------------------------------------------
    with tab_incident:
        st.subheader("Single Incident Frame VAR Review")
        st.caption("Perform immediate, deterministic offside evaluation on individual match incident frames.")

        inc_col1, inc_col2 = st.columns([1, 1])

        with inc_col1:
            sample_scenes = ["spurs_incident_frame_106.jpg", "0.jpg", "1.jpg", "10.jpg", "104.jpg", "114.jpg", "126.jpg", "136.jpg", "146.jpg", "156.jpg", "166.jpg", "176.jpg", "189.jpg", "214.jpg", "225.jpg", "237.jpg"]
            inc_source = st.radio("Incident Image Source:", ["Benchmark Match Scene", "Upload Custom Image"], horizontal=True)

            selected_image_path = None
            if inc_source == "Benchmark Match Scene":
                scene_pick = st.selectbox("Select Benchmark Scene:", sample_scenes, index=0)
                selected_image_path = os.path.join("Offside_Images", scene_pick)
            else:
                up_file = st.file_uploader("Upload Incident Photo:", type=["jpg", "jpeg", "png"], key="inc_uploader")
                if up_file:
                    t_img = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    t_img.write(up_file.read())
                    selected_image_path = t_img.name

            if selected_image_path and os.path.exists(selected_image_path):
                st.image(selected_image_path, caption="Input Incident Scene", use_container_width=True)

        with inc_col2:
            if selected_image_path and os.path.exists(selected_image_path):
                if st.button("🔍 Analyze Incident Frame", type="primary", use_container_width=True, key="btn_run_incident"):
                    raw_img = cv2.imread(selected_image_path)
                    annotated_img, result, stats = analyze_single_incident_frame(
                        raw_img,
                        attack_direction=attack_direction,
                        attack_team_id=attack_team_id,
                        overlay_mode=overlay_mode
                    )

                    st.session_state["incident_annotated"] = annotated_img
                    st.session_state["incident_result"] = result
                    st.session_state["incident_stats"] = stats

                if "incident_result" in st.session_state:
                    res = st.session_state["incident_result"]
                    st_data = st.session_state["incident_stats"]

                    if res.decision == "UNDETERMINED":
                        st.html(
                            f'<div class="status-box-undetermined">UNDETERMINED — CALIBRATION INVALID<br>'
                            f'<span style="font-size: 0.95rem; font-weight: 500;">{res.explanation}</span></div>'
                        )
                    elif res.decision == "OFFSIDE":
                        st.html(
                            f'<div class="status-box-offside">OFFSIDE POSITION DETECTED (+{res.margin_val:.2f}{res.margin_unit})<br>'
                            f'<span style="font-size: 0.95rem; font-weight: 500;">Attacker closer to goal line than second-last opponent</span></div>'
                        )
                    else:
                        st.html(
                            f'<div class="status-box-onside">ONSIDE — NO OFFSIDE OFFENCE<br>'
                            f'<span style="font-size: 0.95rem; font-weight: 500;">All attacking players behind defensive line or ball</span></div>'
                        )

                    # Display Annotated Output
                    ann_rgb = cv2.cvtColor(st.session_state["incident_annotated"], cv2.COLOR_BGR2RGB)
                    st.image(ann_rgb, caption="Premier League Broadcast VAR Overlay", use_container_width=True)

                    # Diagnostic Breakdown Cards
                    d1, d2, d3, d4 = st.columns(4)
                    with d1:
                        st.html(f'<div class="metric-card"><div class="metric-title">Team 0 Players</div><div class="metric-value">{st_data["team_0_count"]}</div></div>')
                    with d2:
                        st.html(f'<div class="metric-card"><div class="metric-title">Team 1 Players</div><div class="metric-value">{st_data["team_1_count"]}</div></div>')
                    with d3:
                        st.html(f'<div class="metric-card"><div class="metric-title">Goalkeepers / Refs</div><div class="metric-value">{st_data["gk_count"]} / {st_data["ref_count"]}</div></div>')
                    with d4:
                        st.html(f'<div class="metric-card"><div class="metric-title">Ball Evidence</div><div class="metric-value">{st_data["ball_state"]}</div></div>')


if __name__ == "__main__":
    main()
