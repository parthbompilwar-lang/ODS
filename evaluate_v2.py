"""
DETECTOR-EVAL-001: Benchmark Evaluation on VAL and LOCKED TEST sets.
Loads: models/yolo11_v2_4class_best.pt
Runs evaluation on:
1. VAL set (69 images)
2. LOCKED TEST set (69 images)
3. Small-ball breakdown across size categories on the locked test set.
Saves: dataset_v2_meta/train_001_evaluation_report.json
"""

import os
import json
import shutil
import cv2
import numpy as np
from collections import defaultdict
import torch
from ultralytics import YOLO


def evaluate_detector():
    meta_dir = os.path.abspath(r"dataset_v2_meta")
    yaml_path = os.path.abspath(r"dataset_v2/data.yaml")
    weights_src = r"E:\ODS\runs\detect\runs\train_v2\train_001_yolo11_1280\weights\best.pt"
    weights_dest = os.path.abspath(r"models/yolo11_v2_4class_best.pt")

    os.makedirs("models", exist_ok=True)
    if os.path.exists(weights_src):
        shutil.copy2(weights_src, weights_dest)
        print(f"Copied best.pt to: {weights_dest}")
    elif not os.path.exists(weights_dest):
        raise FileNotFoundError(f"Neither {weights_src} nor {weights_dest} exists.")

    device = "0" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("DETECTOR-EVAL-001: EVALUATING best.pt ON VAL & LOCKED TEST SETS")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    model = YOLO(weights_dest)

    # 1. Evaluate on Validation Set
    print("\n--- 1. Validation Set Evaluation ---")
    val_metrics = model.val(
        data=yaml_path,
        split="val",
        imgsz=1280,
        batch=8,
        device=device,
        verbose=True
    )

    # 2. Evaluate on Locked Test Set
    print("\n--- 2. Locked Test Set Evaluation ---")
    test_metrics = model.val(
        data=yaml_path,
        split="test",
        imgsz=1280,
        batch=8,
        device=device,
        verbose=True
    )

    # Helper function to extract per-class metrics
    def parse_metrics(m):
        names = m.names
        class_res = {}
        for idx, cls_id in enumerate(m.ap_class_index):
            c_name = names.get(cls_id, str(cls_id))
            p = float(m.box.p[idx]) if len(m.box.p) > idx else 0.0
            r = float(m.box.r[idx]) if len(m.box.r) > idx else 0.0
            ap50 = float(m.box.ap50[idx]) if len(m.box.ap50) > idx else 0.0
            ap = float(m.box.ap[idx]) if len(m.box.ap) > idx else 0.0
            class_res[c_name] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "mAP50": round(ap50, 4),
                "mAP50_95": round(ap, 4)
            }
        return {
            "overall": {
                "precision": round(float(m.box.mp), 4),
                "recall": round(float(m.box.mr), 4),
                "mAP50": round(float(m.box.map50), 4),
                "mAP50_95": round(float(m.box.map), 4)
            },
            "per_class": class_res
        }

    val_summary = parse_metrics(val_metrics)
    test_summary = parse_metrics(test_metrics)

    # 3. Small-Ball Resolution Analysis on Locked Test Set
    print("\n--- 3. Small-Ball Resolution Analysis (Locked Test Set) ---")
    test_img_dir = r"dataset_v2/images/test"
    test_lbl_dir = r"dataset_v2/labels/test"
    test_images = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    ball_stats = {
        "lt_8px": {"total": 0, "detected": 0},
        "8_16px": {"total": 0, "detected": 0},
        "16_32px": {"total": 0, "detected": 0},
        "32_64px": {"total": 0, "detected": 0}
    }

    for img_name in test_images:
        img_path = os.path.join(test_img_dir, img_name)
        lbl_path = os.path.join(test_lbl_dir, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        ih, iw = img.shape[:2]

        gt_balls = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5 and int(parts[0]) == 3:
                    cx, cy, nw, nh = map(float, parts[1:])
                    bw_px = nw * iw
                    bh_px = nh * ih
                    x1 = (cx - nw / 2) * iw
                    y1 = (cy - nh / 2) * ih
                    x2 = (cx + nw / 2) * iw
                    y2 = (cy + nh / 2) * ih
                    gt_balls.append({"bbox": [x1, y1, x2, y2], "width_px": bw_px})

        if not gt_balls:
            continue

        # Inference at conf=0.10
        preds = model(img, classes=[3], conf=0.10, imgsz=1280, device=device, verbose=False)[0]
        pred_boxes = [box.xyxy[0].cpu().numpy().tolist() for box in preds.boxes]

        for gt in gt_balls:
            w_px = gt["width_px"]
            if w_px < 8:
                category = "lt_8px"
            elif w_px < 16:
                category = "8_16px"
            elif w_px <= 32:
                category = "16_32px"
            else:
                category = "32_64px"

            ball_stats[category]["total"] += 1

            gx1, gy1, gx2, gy2 = gt["bbox"]
            gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2

            matched = False
            for pb in pred_boxes:
                px1, py1, px2, py2 = pb
                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                dist = np.hypot(gcx - pcx, gcy - pcy)
                if dist <= max(35.0, w_px * 1.5):
                    matched = True
                    break
            if matched:
                ball_stats[category]["detected"] += 1

    for cat, data in ball_stats.items():
        tot = data["total"]
        det = data["detected"]
        recall = (det / tot * 100) if tot > 0 else 0.0
        data["recall_pct"] = round(recall, 1)

    # Save final report
    report = {
        "experiment_name": "TRAIN-001 / DETECTOR-EVAL-001",
        "training_configuration": {
            "model_base": "yolo11n.pt",
            "imgsz": 1280,
            "epochs": 25,
            "batch": 8,
            "seed": 42,
            "device": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "dataset": "dataset_v2/data.yaml (323 train, 69 val, 69 locked test)",
            "training_time_seconds": 704.5
        },
        "best_weights": weights_dest,
        "validation_metrics": val_summary,
        "locked_test_metrics": test_summary,
        "small_ball_breakdown": ball_stats
    }

    report_path = os.path.join(meta_dir, "train_001_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("DETECTOR-EVAL-001: LOCKED TEST SET RESULTS")
    print("=" * 70)
    print(f"Overall Test mAP@0.5:      {test_summary['overall']['mAP50']:.4f}")
    print(f"Overall Test mAP@0.5:0.95: {test_summary['overall']['mAP50_95']:.4f}")
    print(f"Overall Test Precision:    {test_summary['overall']['precision']:.4f}")
    print(f"Overall Test Recall:       {test_summary['overall']['recall']:.4f}")

    print("\nPer-Class Breakdown (Locked Test Set):")
    for c_name, m in test_summary["per_class"].items():
        print(f"  {c_name:12s} - P: {m['precision']:.4f} | R: {m['recall']:.4f} | mAP50: {m['mAP50']:.4f} | mAP50-95: {m['mAP50_95']:.4f}")

    print("\nSmall-Ball Recall by Size (Locked Test Set):")
    for cat, data in ball_stats.items():
        print(f"  {cat:10s} - {data['detected']}/{data['total']} detected ({data['recall_pct']}%)")

    print(f"\nReport written to: {report_path}")
    return report


if __name__ == "__main__":
    evaluate_detector()
