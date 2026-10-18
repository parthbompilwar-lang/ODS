"""
Command-Line Interface (CLI) for Football Tactical Radar & Heatmap Engine.
Usage:
    python football_tactical_radar.py --video offside_spurs_match.mp4 --output tactical_radar_out.mp4
"""

import os
import cv2
import argparse
import numpy as np
import imageio
import torch
from ultralytics import YOLO

from src.tracking.player_tracker import PlayerTracker
from src.tracking.ball_tracker import BallTracker
from src.team.team_classifier import TeamClassifier
from src.geometry.pitch_geometry import PitchGeometry
from src.geometry.dynamic_pitch_tracker import DynamicPitchTracker
from src.passing.pass_detector import PassDetector
from src.visualization.tactical_radar_heatmap import TacticalRadarEngine


def get_pitch_geometry(width: int, height: int) -> PitchGeometry:
    cam_pts = np.array([
        [width * 0.23, height * 0.22],
        [width * 0.78, height * 0.22],
        [width * 0.86, height * 0.91],
        [width * 0.24, height * 0.91]
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


def run_pipeline(video_path: str, output_path: str):
    print("=" * 70)
    print("AI FOOTBALL TACTICAL RADAR & SPATIAL HEATMAP PIPELINE")
    print("=" * 70)
    print(f"Input video:  {video_path}")
    print(f"Output video: {output_path}")

    model_path = "models/yolo11_v2_4class_best.pt"
    if not os.path.exists(model_path):
        model_path = "yolo11n.pt"
    model = YOLO(model_path)
    device = "0" if torch.cuda.is_available() else "cpu"

    player_tracker = PlayerTracker(high_conf_thresh=0.40, low_conf_thresh=0.15, cost_thresh=0.65)
    ball_tracker = BallTracker(max_missing_frames=5, initial_gating_dist=85.0)
    team_classifier = TeamClassifier(n_teams=2, vote_window=10, min_votes_required=2)
    radar_engine = TacticalRadarEngine(pitch_length=105.0, pitch_width=68.0)
    pass_detector = PassDetector(proximity_threshold=95.0, min_departure_speed=7.0)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    geom = get_pitch_geometry(in_w, in_h)
    pitch_tracker = DynamicPitchTracker(geom.homography_matrix, image_shape=(in_h, in_w))

    pane_w, pane_h = 640, 384
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', format='FFMPEG', pixelformat='yuv420p')

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

        # 2. Dynamic Metric Homography Update
        H_t, motion_state = pitch_tracker.update(frame, p_dets)
        geom.homography_matrix = H_t
        try:
            geom.inv_homography_matrix = np.linalg.inv(H_t)
        except Exception:
            pass

        # 3. Tracking & Team Classification
        p_tracks = player_tracker.update(p_dets, frame_index=frame_idx)
        b_state = ball_tracker.update(b_dets, frame_index=frame_idx)
        p_tracks = team_classifier.classify_tracks(frame, p_tracks, frame_index=frame_idx)

        ball_pos = b_state.get("position")
        ball_state_str = b_state.get("state", "LOST")

        # 4. Pass Detection
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

        if ball_pos is not None:
            cx, cy = map(int, ball_pos)
            cv2.circle(broadcast_pane, (cx, cy), 9, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(broadcast_pane, (cx, cy), 7, (0, 215, 255), -1, cv2.LINE_AA)
            cv2.circle(broadcast_pane, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
            badge_txt = "BALL: OBS" if ball_state_str == "OBSERVED" else "BALL: PRED"
            badge_col = (0, 230, 100) if ball_state_str == "OBSERVED" else (0, 180, 255)
            cv2.putText(broadcast_pane, badge_txt, (cx + 12, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_col, 1, cv2.LINE_AA)

        cv2.rectangle(broadcast_pane, (0, 0), (in_w, 42), (14, 14, 14), -1)
        cv2.putText(broadcast_pane, f"MATCH BROADCAST FEED | FRAME #{frame_idx} | CAM: {motion_state}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

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
            show_convex_hulls=True,
            show_centroids=True
        )

        resized_broadcast = cv2.resize(broadcast_pane, (pane_w, pane_h))
        combined = np.hstack([resized_broadcast, radar_pane])

        rgb_frame = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)

        frame_idx += 1
        if frame_idx % 20 == 0 or frame_idx == total_frames:
            pct = (frame_idx / total_frames) * 100.0 if total_frames > 0 else 0.0
            print(f"Processed frame {frame_idx}/{total_frames} ({pct:.1f}%)")

    cap.release()
    writer.close()
    print("Video rendering complete!")

    # Save Heatmaps
    print("Exporting Tactical Heatmaps...")
    heat_t0 = radar_engine.generate_heatmap(radar_engine.team0_history, title="TEAM 1 TERRITORY HEATMAP", colormap=cv2.COLORMAP_VIRIDIS)
    heat_t1 = radar_engine.generate_heatmap(radar_engine.team1_history, title="TEAM 2 TERRITORY HEATMAP", colormap=cv2.COLORMAP_HOT)
    cv2.imwrite("tactical_heatmap_team0.jpg", heat_t0)
    cv2.imwrite("tactical_heatmap_team1.jpg", heat_t1)
    print("Saved tactical_heatmap_team0.jpg and tactical_heatmap_team1.jpg")

    metrics = radar_engine.compute_tactical_metrics()
    print("\n--- MATCH TACTICAL METRICS ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Tactical Radar & Heatmap Engine")
    parser.add_argument("--video", type=str, default="sample_match.mp4", help="Path to input match video")
    parser.add_argument("--output", type=str, default="tactical_radar_output.mp4", help="Path for rendered output video")
    args = parser.parse_args()

    run_pipeline(args.video, args.output)
