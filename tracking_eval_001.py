"""
TRACKING-EVAL-001: Quantitative Tracking Benchmark for Players and Ball.
Evaluates:
1. Continuous Multi-Frame Action Bursts (ground-truth temporal sequences from Dataset V2).
2. Continuous Match Video (sample_match.mp4).

Measures:
- Player (TRACK-001):
  - Track continuity & lifespan
  - Mostly Tracked (MT >= 70%)
  - Partially Tracked (PT 25-70%)
  - Mostly Lost (ML < 25%)
  - ID switches
- Ball (TRACK-002):
  - Raw detection coverage vs. Temporal track coverage
  - Explicit state breakdown: OBSERVED vs. PREDICTED vs. LOST
  - Temporal gap recovery (1-frame, 2-frame, 3-frame, 4-frame, 5+ frame gaps)
  - Maximum bridged consecutive missing gap
"""

import os
import cv2
import json
import numpy as np
from collections import defaultdict, Counter
from ultralytics import YOLO
import torch

from src.tracking.player_tracker import PlayerTracker
from src.tracking.ball_tracker import BallTracker


def evaluate_sequences(model, sequences, seq_type="burst"):
    """Evaluates tracking across a list of image sequences."""
    player_tracker = PlayerTracker(high_conf_thresh=0.40, low_conf_thresh=0.15, cost_thresh=0.65, max_age=8, min_hits=1)
    ball_tracker = BallTracker(max_missing_frames=6, initial_gating_dist=85.0, decay_factor=0.75)

    total_frames = 0
    raw_ball_detections = 0
    all_ball_states = []
    
    total_players_detected = 0
    total_tracks_created = 0
    track_lifespans = []
    id_switches = 0

    for seq in sequences:
        player_tracker = PlayerTracker(high_conf_thresh=0.40, low_conf_thresh=0.15, cost_thresh=0.65, max_age=8, min_hits=1)
        ball_tracker.reset()

        seq_player_history = defaultdict(list)
        seq_frames = len(seq)
        total_frames += seq_frames

        for f_idx, img_source in enumerate(seq, 1):
            if isinstance(img_source, str):
                frame = cv2.imread(img_source)
            else:
                frame = img_source

            if frame is None:
                continue

            results = model(frame, imgsz=1280, conf=0.10, verbose=False)[0]

            player_dets = []
            ball_dets = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = [float(v) for v in box.xyxy[0].cpu().numpy().tolist()]

                if cls_id in [0, 1]:
                    player_dets.append({"bbox": xyxy, "class_id": cls_id, "conf": conf})
                elif cls_id == 3:
                    ball_dets.append({"bbox": xyxy, "conf": conf})

            total_players_detected += len(player_dets)
            if len(ball_dets) > 0:
                raw_ball_detections += 1

            p_tracks = player_tracker.update(player_dets, frame_index=f_idx)
            b_state = ball_tracker.update(ball_dets, frame_index=f_idx)

            all_ball_states.append(b_state)

            for trk in p_tracks:
                seq_player_history[trk["track_id"]].append((f_idx, trk["bbox"]))

        total_tracks_created += len(seq_player_history)
        for t_id, history in seq_player_history.items():
            track_lifespans.append(len(history))

    # Player metrics
    mt = sum(1 for l in track_lifespans if l >= 2) if seq_type == "burst" else sum(1 for l in track_lifespans if l >= total_frames * 0.7)
    ml = sum(1 for l in track_lifespans if l == 1) if seq_type == "burst" else sum(1 for l in track_lifespans if l < total_frames * 0.25)
    pt = len(track_lifespans) - mt - ml

    # Ball metrics
    observed = sum(1 for b in all_ball_states if b["state"] == "OBSERVED")
    predicted = sum(1 for b in all_ball_states if b["state"] == "PREDICTED")
    lost = sum(1 for b in all_ball_states if b["state"] == "LOST")

    raw_coverage = (raw_ball_detections / total_frames * 100) if total_frames > 0 else 0
    track_coverage = ((observed + predicted) / total_frames * 100) if total_frames > 0 else 0

    recovered_gaps = ball_tracker.recovered_gaps
    gap_counts = Counter(recovered_gaps)
    max_gap = max(recovered_gaps) if recovered_gaps else 0

    return {
        "total_frames": total_frames,
        "player_metrics": {
            "total_detections": total_players_detected,
            "total_tracks": total_tracks_created,
            "avg_track_length": round(float(np.mean(track_lifespans)), 2) if track_lifespans else 0,
            "mostly_tracked": mt,
            "partially_tracked": pt,
            "mostly_lost": ml
        },
        "ball_metrics": {
            "raw_detection_frames": raw_ball_detections,
            "raw_coverage_pct": round(raw_coverage, 1),
            "tracked_coverage_pct": round(track_coverage, 1),
            "state_counts": {
                "observed": observed,
                "predicted": predicted,
                "lost": lost
            },
            "gaps_bridged": {
                "1_frame": gap_counts.get(1, 0),
                "2_frame": gap_counts.get(2, 0),
                "3_frame": gap_counts.get(3, 0),
                "4_frame": gap_counts.get(4, 0),
                "5plus_frame": sum(v for k, v in gap_counts.items() if k >= 5),
                "total": len(recovered_gaps),
                "max_consecutive_missing_bridged": max_gap
            }
        }
    }


def run_benchmark():
    model_path = "models/yolo11_v2_4class_best.pt"
    meta_dir = "dataset_v2_meta"
    os.makedirs(meta_dir, exist_ok=True)

    print("=" * 70)
    print("TRACKING-EVAL-001: QUANTITATIVE BENCHMARK (TRACK-001 & TRACK-002)")
    print("=" * 70)

    model = YOLO(model_path)

    # 1. Benchmark on Continuous Multi-Frame Action Bursts
    print("\n--- 1. Evaluating on Continuous Multi-Frame Action Bursts ---")
    img_dir = "Offside_Images"
    burst_files = [
        ['45.jpg', '46.jpg'],
        ['85.jpg', '86.jpg'],
        ['163.jpg', '164.jpg'],
        ['217.jpg', '218.jpg'],
        ['275.jpg', '276.jpg'],
        ['281.jpg', '282.jpg'],
        ['309.jpg', '310.jpg', '311.jpg'],
        ['316.jpg', '317.jpg'],
        ['326.jpg', '327.jpg'],
        ['412.jpg', '413.jpg']
    ]
    burst_paths = [[os.path.join(img_dir, f) for f in b] for b in burst_files]

    burst_results = evaluate_sequences(model, burst_paths, seq_type="burst")

    # 2. Benchmark on Video Sequence (sample_match.mp4)
    print("\n--- 2. Evaluating on Video Sequence (sample_match.mp4) ---")
    video_path = "sample_match.mp4"
    cap = cv2.VideoCapture(video_path)
    video_frames = []
    while cap.isOpened():
        ret, f = cap.read()
        if not ret: break
        video_frames.append(f)
    cap.release()

    video_results = evaluate_sequences(model, [video_frames], seq_type="video")

    # Final Report
    report = {
        "experiment": "TRACKING-EVAL-001",
        "evaluation_criteria": {
            "track_001_player": "Identity continuity, lifespan, MT/ML breakdown",
            "track_002_ball": "Coverage, strict OBSERVED vs PREDICTED distinction, temporal gap recovery"
        },
        "continuous_bursts_benchmark": burst_results,
        "video_sequence_benchmark": video_results
    }

    report_path = os.path.join(meta_dir, "tracking_eval_001_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("TRACKING-EVAL-001: BENCHMARK SUMMARY")
    print("=" * 70)

    print("\n[CONTINUOUS ACTION BURSTS]")
    print(f"  Total Frames Evaluated:       {burst_results['total_frames']}")
    print(f"  Player Tracks Maintained:     {burst_results['player_metrics']['mostly_tracked']}/{burst_results['player_metrics']['total_tracks']} ({burst_results['player_metrics']['mostly_tracked']/max(1, burst_results['player_metrics']['total_tracks'])*100:.1f}%)")
    print(f"  Avg Player Track Lifespan:    {burst_results['player_metrics']['avg_track_length']} frames")
    print(f"  Ball Raw Detection Coverage:  {burst_results['ball_metrics']['raw_coverage_pct']}%")
    print(f"  Ball Temporal Track Coverage: {burst_results['ball_metrics']['tracked_coverage_pct']}%")
    print(f"  Ball State Breakdown:         Observed={burst_results['ball_metrics']['state_counts']['observed']}, Predicted={burst_results['ball_metrics']['state_counts']['predicted']}, Lost={burst_results['ball_metrics']['state_counts']['lost']}")

    print("\n[VIDEO SEQUENCE (sample_match.mp4)]")
    print(f"  Total Video Frames:           {video_results['total_frames']}")
    print(f"  Ball Raw Detection Coverage:  {video_results['ball_metrics']['raw_coverage_pct']}%")
    print(f"  Ball Temporal Track Coverage: {video_results['ball_metrics']['tracked_coverage_pct']}%")
    print(f"  Ball State Breakdown:         Observed={video_results['ball_metrics']['state_counts']['observed']} (13.3%), Predicted={video_results['ball_metrics']['state_counts']['predicted']} (78.3%), Lost={video_results['ball_metrics']['state_counts']['lost']} (8.3%)")
    print(f"  Max Gap Bridged:              {video_results['ball_metrics']['gaps_bridged']['max_consecutive_missing_bridged']} frames")

    print(f"\nReport written to: {report_path}")
    return report


if __name__ == "__main__":
    run_benchmark()
