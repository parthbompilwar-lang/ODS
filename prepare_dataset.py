"""
Prepare YOLOv11 Training Dataset from Offside Dataset annotations.
Converts keypoints from final_data.json into normalized YOLO bounding box format:
Class 0: Not_Offside (Regular Outfield Player)
Class 1: Offside (Player in offside position)
Class 2: Goalkeeper
Class 3: Ball
"""

import os
import json
import shutil
import random
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


def get_player_bbox(points: List[Dict[str, int]], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Calculate normalized YOLO bbox (cx, cy, w, h) from keypoint list with anatomical padding."""
    xs = [p['x'] for p in points if p.get('x') is not None]
    ys = [p['y'] for p in points if p.get('y') is not None]

    if not xs or not ys:
        return 0, 0, 0, 0

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    w = max_x - min_x
    h = max_y - min_y

    # Add 10% horizontal padding and 15% vertical padding for head and boots
    pad_x = max(10, int(w * 0.12))
    pad_y = max(15, int(h * 0.15))

    x1 = max(0, min_x - pad_x)
    y1 = max(0, min_y - pad_y)
    x2 = min(img_w, max_x + pad_x)
    y2 = min(img_h, max_y + pad_y)

    box_w = (x2 - x1) / img_w
    box_h = (y2 - y1) / img_h
    cx = (x1 + x2) / (2.0 * img_w)
    cy = (y1 + y2) / (2.0 * img_h)

    return cx, cy, box_w, box_h


def prepare_yolo_dataset(
    json_path: str = "final_data.json",
    image_src_dir: str = "Offside_Images",
    target_dir: str = "yolo_offside_dataset",
    train_ratio: float = 0.8
):
    print("Preparing YOLOv11 Offside Dataset...")
    os.makedirs(os.path.join(target_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "labels", "val"), exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Filter out entries without valid pose dict
    valid_entries = [d for d in data if isinstance(d.get("Pose"), dict)]
    random.seed(42)
    random.shuffle(valid_entries)

    num_train = int(len(valid_entries) * train_ratio)
    train_entries = valid_entries[:num_train]
    val_entries = valid_entries[num_train:]

    print(f"Total valid samples: {len(valid_entries)} (Train: {len(train_entries)}, Val: {len(val_entries)})")

    splits = [("train", train_entries), ("val", val_entries)]

    total_boxes = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for split_name, entries in splits:
        img_out_dir = os.path.join(target_dir, "images", split_name)
        lbl_out_dir = os.path.join(target_dir, "labels", split_name)

        for item in entries:
            img_id = item["Image_ID"]
            src_img_path = os.path.join(image_src_dir, img_id)
            if not os.path.exists(src_img_path):
                continue

            img = cv2.imread(src_img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            pose_data = item["Pose"]

            t1_players = pose_data.get("Team1", [])
            t2_players = pose_data.get("Team2", [])
            gk_players = pose_data.get("GK", [])

            label_lines = []

            # Determine second-last defender for offside labeling
            # Assume team 1 attacking team 2 towards left
            t2_xs = []
            for p in t2_players:
                pts = p.get("geometry", [])
                if pts:
                    t2_xs.append(min(pt['x'] for pt in pts if pt.get('x') is not None))

            t2_xs_sorted = sorted(t2_xs)
            # Second-last defender threshold
            second_last_def_x = t2_xs_sorted[1] if len(t2_xs_sorted) >= 2 else (t2_xs_sorted[0] if t2_xs_sorted else w / 2)

            # Team 1 players
            for p in t1_players:
                pts = p.get("geometry", [])
                if pts:
                    cx, cy, bw, bh = get_player_bbox(pts, w, h)
                    p_min_x = min(pt['x'] for pt in pts if pt.get('x') is not None)
                    # Class: 1 if offside, 0 if not offside
                    cls_id = 1 if p_min_x < second_last_def_x else 0
                    label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    class_counts[cls_id] += 1
                    total_boxes += 1

            # Team 2 players (defenders) -> Class 0 (Not Offside)
            for p in t2_players:
                pts = p.get("geometry", [])
                if pts:
                    cx, cy, bw, bh = get_player_bbox(pts, w, h)
                    label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    class_counts[0] += 1
                    total_boxes += 1

            # Goalkeepers -> Class 2
            for p in gk_players:
                pts = p.get("geometry", [])
                if pts:
                    cx, cy, bw, bh = get_player_bbox(pts, w, h)
                    label_lines.append(f"2 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    class_counts[2] += 1
                    total_boxes += 1

            # Copy image and write label file
            dst_img_path = os.path.join(img_out_dir, img_id)
            if not os.path.exists(dst_img_path):
                shutil.copyfile(src_img_path, dst_img_path)

            lbl_name = os.path.splitext(img_id)[0] + ".txt"
            dst_lbl_path = os.path.join(lbl_out_dir, lbl_name)
            with open(dst_lbl_path, "w") as lf:
                lf.write("\n".join(label_lines) + "\n")

    # Create data.yaml
    yaml_content = f"""path: {os.path.abspath(target_dir)}
train: images/train
val: images/val

names:
  0: Not_Offside
  1: Offside
  2: Goalkeeper
  3: Ball
"""
    yaml_path = os.path.join(target_dir, "data.yaml")
    with open(yaml_path, "w") as yf:
        yf.write(yaml_content)

    print(f"Dataset generated at {target_dir}!")
    print(f"Total annotations: {total_boxes}")
    print(f"Class distribution: Not_Offside={class_counts[0]}, Offside={class_counts[1]}, Goalkeeper={class_counts[2]}")
    print(f"data.yaml created at {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    prepare_yolo_dataset()
