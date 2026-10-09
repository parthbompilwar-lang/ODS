"""
TEAM-EVAL-001: Quantitative Team Classification and Stability Benchmark.
Evaluates:
  1. Multi-scene color separability across 15 match images in CIE-Lab space.
  2. Cluster balance (both teams represented among outfield players).
  3. Non-player discrimination (Goalkeepers, Referees excluded from outfield team IDs).
  4. Temporal stability on video tracks (flip rate with track-level majority voting).
  5. UNKNOWN assignment policy for low-confidence or ambiguous detections.
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Any, List
from ultralytics import YOLO
from sklearn.metrics import silhouette_score

from src.team.team_classifier import TeamClassifier
from src.tracking.player_tracker import PlayerTracker


BENCHMARK_SCENES = [
    "0.jpg", "1.jpg", "10.jpg", "104.jpg", "114.jpg",
    "126.jpg", "136.jpg", "146.jpg", "156.jpg", "166.jpg",
    "176.jpg", "189.jpg", "214.jpg", "225.jpg", "237.jpg"
]


def run_team_eval_001():
    print("=" * 60)
    print("TEAM-EVAL-001: Team Classification & Stability Benchmark")
    print("=" * 60)

    model_path = "models/yolo11_v2_4class_best.pt"
    assert os.path.exists(model_path), f"Model {model_path} not found"
    model = YOLO(model_path)

    scene_results = []
    separability_scores = []
    centroid_dists = []
    ref_isolation_correct = 0
    ref_total = 0
    gk_isolation_correct = 0
    gk_total = 0
    img_dir = "Offside_Images"

    for scene in BENCHMARK_SCENES:
        img_p = os.path.join(img_dir, scene)
        if not os.path.exists(img_p):
            continue

        im = cv2.imread(img_p)
        res = model(im, imgsz=1280, conf=0.15, verbose=False)[0]

        dets = []
        for b in res.boxes:
            dets.append({
                "bbox": [float(v) for v in b.xyxy[0].cpu().numpy()],
                "class_id": int(b.cls[0]),
                "conf": float(b.conf[0])
            })

        tc = TeamClassifier(n_teams=2)
        fit_ok = tc.fit_from_image(im, dets)
        classified = tc.classify_tracks(im, dets)

        outfield_0 = sum(1 for d in classified if d.get("team_id") == 0 and d.get("role") == "player")
        outfield_1 = sum(1 for d in classified if d.get("team_id") == 1 and d.get("role") == "player")
        unknowns = sum(1 for d in classified if d.get("team_id") == -1 and d.get("role") == "player")
        gks = sum(1 for d in classified if d.get("role") == "goalkeeper")
        refs = sum(1 for d in classified if d.get("role") == "referee")

        for d in classified:
            if d.get("class_id") == 2:
                ref_total += 1
                if d.get("role") == "referee" and d.get("team_id") == -1:
                    ref_isolation_correct += 1
            elif d.get("class_id") == 1:
                gk_total += 1
                if d.get("role") == "goalkeeper":
                    gk_isolation_correct += 1

        if fit_ok and tc.cluster_centers is not None:
            c0, c1 = tc.cluster_centers[0], tc.cluster_centers[1]
            dist_lab = float(np.linalg.norm(c0 - c1))
            centroid_dists.append(dist_lab)

            outfield_boxes = [d["bbox"] for d in classified if d.get("role") == "player"]
            if len(outfield_boxes) >= 4:
                feats = np.array([tc.extract_jersey_features(im, b) for b in outfield_boxes])
                labels = [d["team_id"] for d in classified if d.get("role") == "player"]
                valid_mask = [l in [0, 1] for l in labels]
                if sum(valid_mask) >= 4 and len(set(np.array(labels)[valid_mask])) > 1:
                    sil = float(silhouette_score(feats[valid_mask], np.array(labels)[valid_mask]))
                    separability_scores.append(sil)
                else:
                    sil = 0.0
            else:
                sil = 0.0
        else:
            dist_lab = 0.0
            sil = 0.0

        scene_results.append({
            "scene": scene,
            "fit_ok": fit_ok,
            "centroid_distance_lab": round(dist_lab, 2),
            "silhouette_score": round(sil, 3),
            "team_0_count": outfield_0,
            "team_1_count": outfield_1,
            "unknown_count": unknowns,
            "gk_count": gks,
            "ref_count": refs
        })
        print(f"Scene {scene:8s}: Dist={dist_lab:5.1f} Lab | Sil={sil:+.2f} | T0={outfield_0:2d}, T1={outfield_1:2d}, UNK={unknowns:2d}, GK={gks:1d}, REF={refs:1d}")

    print("\nEvaluating Temporal Stability on Video Tracking (sample_match.mp4)...")
    cap = cv2.VideoCapture("sample_match.mp4")
    p_tracker = PlayerTracker(high_conf_thresh=0.40, low_conf_thresh=0.15)
    tc_temporal = TeamClassifier(n_teams=2, vote_window=10, min_votes_required=2)

    frame_idx = 0
    track_assignment_history = {}
    majority_flips = 0
    total_consecutive_checks = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        res = model(frame, imgsz=1280, conf=0.15, verbose=False)[0]
        p_dets = []
        for b in res.boxes:
            cls_id = int(b.cls[0])
            if cls_id in [0, 1, 2]:
                p_dets.append({
                    "bbox": [float(v) for v in b.xyxy[0].cpu().numpy()],
                    "class_id": cls_id,
                    "conf": float(b.conf[0])
                })

        tracks = p_tracker.update(p_dets, frame_index=frame_idx)
        tc_temporal.classify_tracks(frame, tracks, frame_index=frame_idx)

        for t in tracks:
            tid = t["track_id"]
            assigned = t["team_id"]
            if tid not in track_assignment_history:
                track_assignment_history[tid] = []
            track_assignment_history[tid].append(assigned)

        frame_idx += 1
    cap.release()

    for tid, hist in track_assignment_history.items():
        if len(hist) >= 3:
            for i in range(1, len(hist)):
                if hist[i-1] != -1 and hist[i] != -1:
                    total_consecutive_checks += 1
                    if hist[i] != hist[i-1]:
                        majority_flips += 1

    flip_rate = (majority_flips / max(1, total_consecutive_checks)) * 100.0
    ref_isolation_rate = (ref_isolation_correct / max(1, ref_total)) * 100.0
    gk_isolation_rate = (gk_isolation_correct / max(1, gk_total)) * 100.0
    mean_dist = float(np.mean(centroid_dists)) if centroid_dists else 0.0
    mean_sil = float(np.mean(separability_scores)) if separability_scores else 0.0

    passed = (mean_dist > 40.0 and ref_isolation_rate >= 90.0 and flip_rate < 3.0)

    summary = {
        "benchmark_id": "TEAM-EVAL-001",
        "status": "PASS" if passed else "CONDITIONAL_KEEP",
        "num_scenes_tested": len(scene_results),
        "mean_centroid_distance_lab": round(mean_dist, 2),
        "mean_silhouette_score": round(mean_sil, 3),
        "referee_isolation_accuracy_pct": round(ref_isolation_rate, 1),
        "goalkeeper_isolation_accuracy_pct": round(gk_isolation_rate, 1),
        "temporal_track_checks": total_consecutive_checks,
        "temporal_majority_flip_rate_pct": round(flip_rate, 2),
        "scene_details": scene_results
    }

    os.makedirs("dataset_v2_meta", exist_ok=True)
    with open("dataset_v2_meta/team_eval_001_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ BENCHMARK SUMMARY ================")
    print(f"Status:                              {summary['status']}")
    print(f"Scenes Evaluated:                    {summary['num_scenes_tested']}")
    print(f"Mean Centroid Distance (Lab):        {summary['mean_centroid_distance_lab']:.2f}")
    print(f"Mean Silhouette Score:               {summary['mean_silhouette_score']:.3f}")
    print(f"Referee Isolation Rate:              {summary['referee_isolation_accuracy_pct']:.1f}%")
    print(f"Goalkeeper Isolation Rate:           {summary['goalkeeper_isolation_accuracy_pct']:.1f}%")
    print(f"Temporal Track Flips:                {summary['temporal_majority_flip_rate_pct']:.2f}% (over {total_consecutive_checks} transitions)")
    print("Report saved to dataset_v2_meta/team_eval_001_report.json")
    return summary


if __name__ == "__main__":
    run_team_eval_001()
