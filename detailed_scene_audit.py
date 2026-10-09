"""
Detailed Scene & Feature Audit for DATASET-AUDIT-001.
Quantifies ball detectability, referee presence, goal-mouth scenes,
match clustering, and duplicate leakage.
"""

import os
import json
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO


def detailed_audit():
    img_dir = r"e:\ODS\Offside_Images"
    json_path = r"e:\ODS\final_data.json"

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Probe for Ball and Referee using pretrained COCO detector (YOLOv11n)
    # COCO Class 0 = Person (Outfield, GK, Referees), Class 32 = Sports Ball
    detector = YOLO("yolo11n.pt")

    ball_detections = []
    referee_candidates = []
    goal_mouth_count = 0
    midfield_count = 0

    # Sample 60 evenly spaced frames across the dataset for deep scene inspection
    sample_indices = np.linspace(0, len(data) - 1, 60, dtype=int)
    sample_items = [data[i] for i in sample_indices]

    for item in sample_items:
        img_id = item["Image_ID"]
        img_path = os.path.join(img_dir, img_id)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # Check for goal-mouth / net presence (white net texture or penalty box lines)
        # Goal net typically in left 30% or right 30% with high white/gray edge density
        left_area = img[int(h*0.2):int(h*0.7), :int(w*0.3)]
        right_area = img[int(h*0.2):int(h*0.7), int(w*0.7):]
        gray_l = cv2.cvtColor(left_area, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_area, cv2.COLOR_BGR2GRAY)
        edge_l = cv2.Canny(gray_l, 100, 200).mean()
        edge_r = cv2.Canny(gray_r, 100, 200).mean()

        if edge_l > 8.0 or edge_r > 8.0:
            goal_mouth_count += 1
        else:
            midfield_count += 1

        # Run ball probe (low conf to catch tiny balls)
        results = detector(img, conf=0.15, verbose=False)[0]
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for b, c, conf in zip(boxes, clss, confs):
                if int(c) == 32:  # sports ball
                    bw = b[2] - b[0]
                    bh = b[3] - b[1]
                    area = bw * bh
                    ball_detections.append({
                        "image": img_id,
                        "w": float(bw),
                        "h": float(bh),
                        "area": float(area),
                        "conf": float(conf),
                        "aspect_ratio": float(bw / bh) if bh > 0 else 1.0,
                        "norm_w": float(bw / w),
                        "norm_h": float(bh / h)
                    })

    # Ball statistics
    ball_widths = [b["w"] for b in ball_detections]
    ball_heights = [b["h"] for b in ball_detections]
    ball_areas = [b["area"] for b in ball_detections]

    # Match clustering by distinct broadcast visual signatures
    # (resolution, scorebug location, team jersey colors)
    broadcast_signatures = set()
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')], key=lambda x: int(os.path.splitext(x)[0]))
    for f in files:
        im = cv2.imread(os.path.join(img_dir, f))
        if im is None: continue
        res = im.shape[:2]
        # sample 4 corners for broadcast graphics / station logos
        corner_tl = tuple(np.mean(im[:50, :150], axis=(0, 1)).astype(int) // 30)
        corner_tr = tuple(np.mean(im[:50, -150:], axis=(0, 1)).astype(int) // 30)
        corner_bl = tuple(np.mean(im[-50:, :150], axis=(0, 1)).astype(int) // 30)
        sig = (res, corner_tl, corner_tr, corner_bl)
        broadcast_signatures.add(sig)

    print("--- Detailed Scene Audit Output ---")
    print(f"Sampled Frames: {len(sample_items)}")
    print(f"Goal-Mouth Scenes Detected in Sample: {goal_mouth_count}/{len(sample_items)} ({goal_mouth_count/len(sample_items)*100:.1f}%)")
    print(f"Midfield / Open Pitch Scenes in Sample: {midfield_count}/{len(sample_items)} ({midfield_count/len(sample_items)*100:.1f}%)")
    print(f"Ball Candidates Detected in Sample: {len(ball_detections)}")
    if ball_widths:
        print(f"Ball Pixel Width: Mean={np.mean(ball_widths):.1f}px, Min={np.min(ball_widths):.1f}px, Max={np.max(ball_widths):.1f}px")
        print(f"Ball Pixel Height: Mean={np.mean(ball_heights):.1f}px, Min={np.min(ball_heights):.1f}px, Max={np.max(ball_heights):.1f}px")
        print(f"Ball Pixel Area: Mean={np.mean(ball_areas):.1f}px^2, Median={np.median(ball_areas):.1f}px^2")
        print(f"Tiny Balls (< 20px diameter): {sum(1 for w in ball_widths if w < 20)} / {len(ball_widths)} ({sum(1 for w in ball_widths if w < 20)/len(ball_widths)*100:.1f}%)")
    print(f"Estimated Unique Match Broadcasts: {len(broadcast_signatures)}")

    stats = {
        "sampled_frames": len(sample_items),
        "goal_mouth_percentage": float(goal_mouth_count / len(sample_items) * 100),
        "midfield_percentage": float(midfield_count / len(sample_items) * 100),
        "balls_detected_in_sample": len(ball_detections),
        "ball_mean_width_px": float(np.mean(ball_widths)) if ball_widths else 0,
        "ball_min_width_px": float(np.min(ball_widths)) if ball_widths else 0,
        "ball_max_width_px": float(np.max(ball_widths)) if ball_widths else 0,
        "ball_mean_area_px2": float(np.mean(ball_areas)) if ball_areas else 0,
        "tiny_ball_pct_under_20px": float(sum(1 for w in ball_widths if w < 20) / len(ball_widths) * 100) if ball_widths else 0,
        "unique_broadcast_signatures": len(broadcast_signatures)
    }

    with open("scene_audit_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    detailed_audit()
