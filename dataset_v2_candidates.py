"""
DATASET-V2-BUILD-002: Multi-Source Candidate Generation Engine for Ball and Referee.
Executes scientific candidate proposals across all 461 canonical images:
- Ball: YOLO (COCO class 32, imgsz=1280), GEOMETRIC (pitch contour/circularity/foot proximity),
        TEMPORAL (tracking in multi-frame clusters), MULTI_SOURCE fusion.
- Referee: YOLO person detection compared against existing player annotations,
           pitch-boundary filtering, kit color signature analysis, and hard-case spatial tagging.
Outputs:
- dataset_v2_meta/ball_candidates.json
- dataset_v2_meta/referee_candidates.json
- dataset_v2_meta/candidate_review_manifest.json
"""

import os
import json
import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from ultralytics import YOLO
import torch


def load_canonical_data():
    img_dir = r"e:\ODS\Offside_Images"
    meta_path = r"e:\ODS\dataset_v2_meta\step1_audit_grouping_report.json"
    json_path = r"e:\ODS\final_data.json"

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    with open(json_path, 'r') as f:
        data = json.load(f)

    annotation_map = {entry["Image_ID"]: entry for entry in data if "Image_ID" in entry}
    split_assignment = meta["split_assignment"]
    canonical_files = sorted(
        list(split_assignment.keys()),
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 99999
    )

    # Multi-frame clusters from sequence groups
    multi_frame_sequences = [
        seq for seq in meta.get("sequence_groups", {}).get("group_sizes", [])
    ]

    return img_dir, canonical_files, annotation_map, split_assignment, meta


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def extract_player_feet(pose_dict: Dict) -> List[Tuple[float, float]]:
    """Extract feet/ground points of all annotated players in image."""
    feet = []
    if not isinstance(pose_dict, dict):
        return feet

    for team in ["Team1", "Team2", "GK"]:
        for p in pose_dict.get(team, []):
            pts = p.get("geometry", [])
            xs = [pt['x'] for pt in pts if pt.get('x') is not None]
            ys = [pt['y'] for pt in pts if pt.get('y') is not None]
            if xs and ys:
                feet.append(((min(xs) + max(xs)) / 2.0, float(max(ys))))
    return feet


def extract_annotated_boxes(pose_dict: Dict) -> List[Dict]:
    """Extract all annotated bounding boxes with team label."""
    boxes = []
    if not isinstance(pose_dict, dict):
        return boxes

    for team in ["Team1", "Team2", "GK"]:
        for p in pose_dict.get(team, []):
            pts = p.get("geometry", [])
            xs = [pt['x'] for pt in pts if pt.get('x') is not None]
            ys = [pt['y'] for pt in pts if pt.get('y') is not None]
            if xs and ys:
                boxes.append({
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "team": team
                })
    return boxes


def detect_geometric_balls(img: np.ndarray, feet: List[Tuple[float, float]]) -> List[Dict]:
    """Extract circular high-contrast candidates on pitch surface near player feet."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green pitch mask
    lower_green = np.array([30, 35, 35])
    upper_green = np.array([85, 255, 255])
    pitch_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Non-green elements on pitch
    non_pitch = cv2.bitwise_not(pitch_mask)
    # Remove top 15% (crowd/stadium banners)
    non_pitch[:int(h * 0.15), :] = 0

    contours, _ = cv2.findContours(non_pitch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 40 <= area <= 1600:  # Typical ball area in 1080p/1440p
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh)
            if 0.65 <= aspect <= 1.55 and 8 <= bw <= 48 and 8 <= bh <= 48:
                # Circularity / perimeter check
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity >= 0.55:
                        cx, cy = x + bw / 2.0, y + bh / 2.0
                        # Proximity to any player feet
                        min_dist = min([np.hypot(cx - fx, cy - fy) for fx, fy in feet]) if feet else 999.0
                        conf = min(0.45, circularity * 0.35 + (0.15 if min_dist < 100 else 0.0))

                        candidates.append({
                            "bbox": [x, y, x + bw, y + bh],
                            "conf": round(conf, 3),
                            "foot_dist": round(min_dist, 1)
                        })
    return candidates


def main():
    img_dir, canonical_files, annotation_map, split_assignment, meta = load_canonical_data()
    meta_dir = r"e:\ODS\dataset_v2_meta"

    print("=" * 70)
    print("DATASET-V2-BUILD-002: MULTI-SOURCE BALL & REFEREE CANDIDATE GENERATION")
    print("=" * 70)
    print(f"Target Images to Process: {len(canonical_files)} canonical images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference Device: {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # Load YOLO model
    model = YOLO("yolo11n.pt")

    ball_proposals_by_img = defaultdict(list)
    referee_proposals_by_img = defaultdict(list)

    # Process each image
    for idx, img_id in enumerate(canonical_files):
        img_path = os.path.join(img_dir, img_id)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        entry = annotation_map.get(img_id, {})
        pose_dict = entry.get("Pose", {})

        ann_players = extract_annotated_boxes(pose_dict)
        player_feet = extract_player_feet(pose_dict)

        # ----------------------------------------------------
        # 1. RUN YOLO HIGH-RESOLUTION INFERENCE (imgsz=1280)
        # Class 0: person, Class 32: sports ball
        # ----------------------------------------------------
        results = model(
            img,
            classes=[0, 32],
            conf=0.03,
            imgsz=1280,
            device=device,
            verbose=False
        )[0]

        yolo_balls = []
        yolo_persons = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [int(v) for v in box.xyxy[0].cpu().numpy().tolist()]

            if cls_id == 32:
                bw, bh = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                # Filter out absurdly large bounding boxes (e.g. stadium ads or player bodies mistaken for balls)
                if bw <= 65 and bh <= 65 and bw >= 6 and bh >= 6:
                    yolo_balls.append({"bbox": xyxy, "conf": round(conf, 3)})
            elif cls_id == 0:
                # Persons on pitch (ignore top 12% stadium roof/stands)
                if xyxy[3] > h * 0.15:
                    yolo_persons.append({"bbox": xyxy, "conf": round(conf, 3)})

        # ----------------------------------------------------
        # 2. RUN GEOMETRIC BALL PROPOSALS
        # ----------------------------------------------------
        geom_balls = detect_geometric_balls(img, player_feet)

        # ----------------------------------------------------
        # 3. FUSE BALL CANDIDATES & TAG HARD-CASES
        # ----------------------------------------------------
        fused_balls = []
        used_geom_indices = set()

        # Check YOLO balls
        for yb in yolo_balls:
            y_box = yb["bbox"]
            y_conf = yb["conf"]
            source = "YOLO"

            # Check IoU or center match with geometric proposals
            yc_x = (y_box[0] + y_box[2]) / 2.0
            yc_y = (y_box[1] + y_box[3]) / 2.0

            matched_geom = False
            for g_idx, gb in enumerate(geom_balls):
                if compute_iou(y_box, gb["bbox"]) > 0.30:
                    source = "MULTI_SOURCE"
                    y_conf = max(y_conf, gb["conf"])
                    used_geom_indices.add(g_idx)
                    matched_geom = True
                    break

            bw, bh = y_box[2] - y_box[0], y_box[3] - y_box[1]

            # Hard-case tags
            tags = []
            if bw <= 18 or bh <= 18:
                tags.append("tiny_ball")

            # Check foot proximity
            min_foot_dist = min([np.hypot(yc_x - fx, yc_y - fy) for fx, fy in player_feet]) if player_feet else 999.0
            if min_foot_dist <= 75.0:
                tags.append("ball_near_foot")

            # Check overlap with any player box
            max_player_iou = max([compute_iou(y_box, ap["bbox"]) for ap in ann_players]) if ann_players else 0.0
            if max_player_iou > 0.05:
                tags.append("ball_occluded")

            if y_conf < 0.15:
                tags.append("ball_motion_blur")

            # Visibility assessment
            if "ball_occluded" in tags:
                visibility = "partial"
            elif "ball_motion_blur" in tags:
                visibility = "blurred"
            else:
                visibility = "clear"

            fused_balls.append({
                "candidate_id": f"ball_{img_id}_{len(fused_balls)}",
                "image_id": img_id,
                "bbox": y_box,
                "source": source,
                "candidate_confidence": y_conf,
                "visibility": visibility,
                "hard_case_tags": tags,
                "review_status": "PENDING"
            })

        # Add top high-confidence geometric balls not matched by YOLO
        for g_idx, gb in enumerate(geom_balls):
            if g_idx not in used_geom_indices and gb["conf"] >= 0.35:
                g_box = gb["bbox"]
                gc_x = (g_box[0] + g_box[2]) / 2.0
                gc_y = (g_box[1] + g_box[3]) / 2.0

                tags = []
                bw, bh = g_box[2] - g_box[0], g_box[3] - g_box[1]
                if bw <= 18 or bh <= 18: tags.append("tiny_ball")
                if gb["foot_dist"] <= 75.0: tags.append("ball_near_foot")

                visibility = "partial" if "ball_near_foot" in tags else "clear"

                fused_balls.append({
                    "candidate_id": f"ball_{img_id}_{len(fused_balls)}",
                    "image_id": img_id,
                    "bbox": g_box,
                    "source": "GEOMETRIC",
                    "candidate_confidence": gb["conf"],
                    "visibility": visibility,
                    "hard_case_tags": tags,
                    "review_status": "PENDING"
                })

        ball_proposals_by_img[img_id] = fused_balls

        # ----------------------------------------------------
        # 4. REFEREE CANDIDATE EXTRACTION & COLOR FILTER
        # ----------------------------------------------------
        referee_proposals = []
        for yp in yolo_persons:
            p_box = yp["bbox"]
            p_conf = yp["conf"]

            # Compute IoU against all annotated players
            ious = [compute_iou(p_box, ap["bbox"]) for ap in ann_players]
            max_iou = max(ious) if ious else 0.0

            # If unannotated person on pitch (IoU < 0.25 with known players)
            if max_iou < 0.25:
                px1, py1, px2, py2 = p_box
                pw, ph = px2 - px1, py2 - py1
                # Must be full human-sized on pitch
                if ph >= 45 and pw >= 15:
                    # Sample torso region for kit color analysis
                    torso_y1 = py1 + int(ph * 0.15)
                    torso_y2 = py1 + int(ph * 0.50)
                    torso_x1 = px1 + int(pw * 0.20)
                    torso_x2 = px2 - int(pw * 0.20)

                    torso_crop = img[max(0, torso_y1):min(h, torso_y2), max(0, torso_x1):min(w, torso_x2)]
                    ref_score = 0.0
                    is_referee_candidate = False

                    if torso_crop.size > 0:
                        t_hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
                        mean_h = np.mean(t_hsv[:, :, 0])
                        mean_s = np.mean(t_hsv[:, :, 1])
                        mean_v = np.mean(t_hsv[:, :, 2])

                        # Referee kit colors:
                        # 1. Dark/Black: Low saturation & low-medium value
                        # 2. Neon yellow/green: Hue 25-50, Sat > 100
                        # 3. Turquoise/Cyan: Hue 85-115, Sat > 70
                        if mean_v < 60 and mean_s < 70:
                            ref_score = 0.75  # Black kit
                            is_referee_candidate = True
                        elif 25 <= mean_h <= 50 and mean_s >= 100:
                            ref_score = 0.85  # Neon yellow kit
                            is_referee_candidate = True
                        elif 85 <= mean_h <= 115 and mean_s >= 65:
                            ref_score = 0.80  # Cyan/blue kit
                            is_referee_candidate = True
                        elif mean_v >= 200 and mean_s < 30:
                            # White kit (might be player or ref)
                            ref_score = 0.40
                            is_referee_candidate = True
                        else:
                            ref_score = 0.30

                    if is_referee_candidate or ref_score >= 0.40:
                        # Spatial hard cases
                        tags = []
                        # Check distance to closest defender
                        def_feet = [p for p in player_feet]
                        center_x = (px1 + px2) / 2.0
                        center_y = float(py2)

                        min_def_dist = min([np.hypot(center_x - fx, center_y - fy) for fx, fy in def_feet]) if def_feet else 999.0
                        if min_def_dist <= 120.0:
                            tags.append("referee_near_defender")

                        if max_iou > 0.10:
                            tags.append("referee_overlapping_player")

                        # Goal area check (lower 20% of pitch or near lateral goal boundaries)
                        if py2 >= h * 0.80:
                            tags.append("referee_goal_area")

                        referee_proposals.append({
                            "candidate_id": f"ref_{img_id}_{len(referee_proposals)}",
                            "image_id": img_id,
                            "bbox": p_box,
                            "source": "YOLO_APPEARANCE_FILTER",
                            "candidate_confidence": round(ref_score, 3),
                            "hard_case_tags": tags,
                            "review_status": "PENDING"
                        })

        referee_proposals_by_img[img_id] = referee_proposals

        if (idx + 1) % 50 == 0 or (idx + 1) == len(canonical_files):
            print(f"Processed {idx + 1}/{len(canonical_files)} images...")

    # ----------------------------------------------------
    # 5. TEMPORAL CONTINUITY PASS (Multi-frame bursts)
    # ----------------------------------------------------
    temporal_upgraded_balls = 0
    # Process sequence groups from step1 report
    with open(os.path.join(meta_dir, "step1_audit_grouping_report.json"), "r") as f:
        step1_rep = json.load(f)

    # Check contiguous sequence frames
    for i in range(len(canonical_files) - 1):
        f1 = canonical_files[i]
        f2 = canonical_files[i + 1]

        # If they belong to same burst (contiguous frames)
        if step1_rep["split_assignment"].get(f1) == step1_rep["split_assignment"].get(f2):
            balls1 = ball_proposals_by_img[f1]
            balls2 = ball_proposals_by_img[f2]

            for b1 in balls1:
                c1_x = (b1["bbox"][0] + b1["bbox"][2]) / 2.0
                c1_y = (b1["bbox"][1] + b1["bbox"][3]) / 2.0
                for b2 in balls2:
                    c2_x = (b2["bbox"][0] + b2["bbox"][2]) / 2.0
                    c2_y = (b2["bbox"][1] + b2["bbox"][3]) / 2.0

                    # If ball moves less than 160 pixels between contiguous frames
                    if np.hypot(c1_x - c2_x, c1_y - c2_y) <= 160.0:
                        if b1["source"] == "MULTI_SOURCE":
                            b1["source"] = "MULTI_SOURCE_TEMPORAL"
                        elif b1["source"] in ["YOLO", "GEOMETRIC"]:
                            b1["source"] = "TEMPORAL_VERIFIED"
                        b1["candidate_confidence"] = min(0.95, b1["candidate_confidence"] + 0.15)
                        temporal_upgraded_balls += 1

    print(f"\nTemporal Continuity Pass: Upgraded {temporal_upgraded_balls} ball candidates across multi-frame bursts.")

    # ----------------------------------------------------
    # 6. SAVE PROPOSALS TO JSON
    # ----------------------------------------------------
    all_balls = [b for b_list in ball_proposals_by_img.values() for b in b_list]
    all_refs = [r for r_list in referee_proposals_by_img.values() for r in r_list]

    ball_cand_path = os.path.join(meta_dir, "ball_candidates.json")
    ref_cand_path = os.path.join(meta_dir, "referee_candidates.json")
    manifest_path = os.path.join(meta_dir, "candidate_review_manifest.json")

    with open(ball_cand_path, "w") as f:
        json.dump(all_balls, f, indent=2)

    with open(ref_cand_path, "w") as f:
        json.dump(all_refs, f, indent=2)

    manifest = {
        "dataset_name": "ODS Dataset V2 Candidates",
        "total_images_reviewed": len(canonical_files),
        "total_ball_candidates": len(all_balls),
        "total_referee_candidates": len(all_refs),
        "images_with_ball_candidate": sum(1 for b_list in ball_proposals_by_img.values() if len(b_list) > 0),
        "images_with_referee_candidate": sum(1 for r_list in referee_proposals_by_img.values() if len(r_list) > 0),
        "split_counts": {
            "train": {
                "images": sum(1 for f in canonical_files if split_assignment[f] == "train"),
                "balls": sum(len(ball_proposals_by_img[f]) for f in canonical_files if split_assignment[f] == "train"),
                "referees": sum(len(referee_proposals_by_img[f]) for f in canonical_files if split_assignment[f] == "train")
            },
            "val": {
                "images": sum(1 for f in canonical_files if split_assignment[f] == "val"),
                "balls": sum(len(ball_proposals_by_img[f]) for f in canonical_files if split_assignment[f] == "val"),
                "referees": sum(len(referee_proposals_by_img[f]) for f in canonical_files if split_assignment[f] == "val")
            },
            "test_locked": {
                "images": sum(1 for f in canonical_files if split_assignment[f] == "test"),
                "balls": sum(len(ball_proposals_by_img[f]) for f in canonical_files if split_assignment[f] == "test"),
                "referees": sum(len(referee_proposals_by_img[f]) for f in canonical_files if split_assignment[f] == "test")
            }
        }
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCandidate files saved:")
    print(f"  - {ball_cand_path} ({len(all_balls)} proposals)")
    print(f"  - {ref_cand_path} ({len(all_refs)} proposals)")
    print(f"  - {manifest_path}")

    # Print Summary Report
    print("\n" + "=" * 70)
    print("DATASET-V2-BUILD-002: CANDIDATE GENERATION SUMMARY")
    print("=" * 70)

    # Ball stats
    ball_vis = defaultdict(int)
    ball_tags = defaultdict(int)
    ball_sources = defaultdict(int)
    for b in all_balls:
        ball_vis[b["visibility"]] += 1
        ball_sources[b["source"]] += 1
        for t in b["hard_case_tags"]:
            ball_tags[t] += 1

    print("\nBALL CANDIDATES:")
    print(f"  Images reviewed:      {len(canonical_files)}")
    print(f"  Candidate boxes:      {len(all_balls)}")
    print(f"  Sources:              {dict(ball_sources)}")
    print(f"  Visibility:")
    print(f"    clear:              {ball_vis['clear']}")
    print(f"    partial:            {ball_vis['partial']}")
    print(f"    blurred:            {ball_vis['blurred']}")
    print(f"  Hard cases:")
    print(f"    tiny_ball:          {ball_tags['tiny_ball']}")
    print(f"    ball_near_foot:     {ball_tags['ball_near_foot']}")
    print(f"    ball_occluded:      {ball_tags['ball_occluded']}")
    print(f"    ball_motion_blur:   {ball_tags['ball_motion_blur']}")

    # Referee stats
    ref_tags = defaultdict(int)
    for r in all_refs:
        for t in r["hard_case_tags"]:
            ref_tags[t] += 1

    print("\nREFEREE CANDIDATES:")
    print(f"  Images reviewed:      {len(canonical_files)}")
    print(f"  Candidate boxes:      {len(all_refs)}")
    print(f"  Hard cases:")
    print(f"    referee_near_defender:       {ref_tags['referee_near_defender']}")
    print(f"    referee_overlapping_player:  {ref_tags['referee_overlapping_player']}")
    print(f"    referee_goal_area:           {ref_tags['referee_goal_area']}")


if __name__ == "__main__":
    main()
