"""
TRAIN-001 & DETECTOR-EVAL-001: First Controlled Retraining on Dataset V2.
Scientific setup:
- Model: YOLO11 (yolo11n.pt base)
- Input resolution: imgsz=1280
- Classes: 4 (0: Player, 1: Goalkeeper, 2: Referee, 3: Ball)
- Dataset: dataset_v2/data.yaml (323 train, 69 val, 69 locked test)
- Hardware: NVIDIA RTX 4050 Laptop GPU (CUDA)
- Controlled configuration: seed=42, batch=8, epochs=25, standard augmentations
- Post-training evaluation: Evaluates best.pt on both VAL and LOCKED TEST sets,
  computing per-class mAP50, mAP50-95, Precision, Recall, and small-ball performance.
"""

import os
import time
import json
import shutil
import cv2
import numpy as np
from collections import defaultdict
import torch
from ultralytics import YOLO


def run_train_001():
    yaml_path = os.path.abspath(r"dataset_v2/data.yaml")
    meta_dir = os.path.abspath(r"dataset_v2_meta")
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("=" * 70)
    print("TRAIN-001: CONTROLLED RETRAINING ON DATASET V2 (imgsz=1280)")
    print("=" * 70)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name:    {torch.cuda.get_device_name(0)}")

    # Configuration
    config = {
        "model_base": "yolo11n.pt",
        "data_yaml": yaml_path,
        "imgsz": 1280,
        "epochs": 25,
        "batch": 8,
        "seed": 42,
        "device": device,
        "project": "runs/train_v2",
        "name": "train_001_yolo11_1280",
        "classes": {0: "Player", 1: "Goalkeeper", 2: "Referee", 3: "Ball"},
        "hypothesis": "The Dataset V2 ball distribution motivates high-resolution training at imgsz=1280; the effectiveness of this choice will be evaluated experimentally."
    }

    start_time = time.time()

    # Load base model
    model = YOLO(config["model_base"])

    # Train
    train_results = model.train(
        data=config["data_yaml"],
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        seed=config["seed"],
        device=config["device"],
        project=config["project"],
        name=config["name"],
        plots=True,
        save=True,
        verbose=True
    )

    training_time_sec = round(time.time() - start_time, 2)
    print(f"\nTraining completed in {training_time_sec:.1f} seconds ({training_time_sec/60:.2f} minutes)!")

    # Locate best.pt
    run_dir = os.path.join(config["project"], config["name"])
    best_pt_path = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(best_pt_path):
        # Fallback search
        best_pt_path = os.path.join(config["project"], "train_001_yolo11_1280", "weights", "best.pt")

    # Copy to models directory
    v2_weights_dest = os.path.abspath(r"models/yolo11_v2_4class_best.pt")
    if os.path.exists(best_pt_path):
        shutil.copy2(best_pt_path, v2_weights_dest)
        print(f"Best weights copied to: {v2_weights_dest}")
    else:
        print(f"WARNING: best.pt not found at expected path: {best_pt_path}")

    # ----------------------------------------------------
    # DETECTOR-EVAL-001: BENCHMARK ON LOCKED TEST SET & VAL SET
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print("DETECTOR-EVAL-001: EVALUATING best.pt ON VAL & LOCKED TEST SETS")
    print("=" * 70)

    best_model = YOLO(v2_weights_dest if os.path.exists(v2_weights_dest) else best_pt_path)

    # 1. Evaluate on Validation Set
    print("\n--- 1. Validation Set Evaluation ---")
    val_metrics = best_model.val(
        data=yaml_path,
        split="val",
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=device,
        verbose=True
    )

    # 2. Evaluate on Locked Test Set
    print("\n--- 2. Locked Test Set Evaluation ---")
    test_metrics = best_model.val(
        data=yaml_path,
        split="test",
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=device,
        verbose=True
    )

    # Extract metrics helper
    def parse_metrics(m):
        names = m.names
        class_res = {}
        # Per class metrics
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

    # 3. Small-Ball Analysis on Locked Test Set
    print("\n--- 3. Small-Ball Resolution Analysis (Locked Test Set) ---")
    test_img_dir = r"dataset_v2/images/test"
    test_lbl_dir = r"dataset_v2/labels/test"
    
    test_images = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    ball_stats = {
        "lt_16px": {"total": 0, "detected": 0},
        "16_32px": {"total": 0, "detected": 0},
        "gt_32px": {"total": 0, "detected": 0}
    }

    for img_name in test_images:
        img_path = os.path.join(test_img_dir, img_name)
        lbl_path = os.path.join(test_lbl_dir, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(img_path)
        if img is None: continue
        ih, iw = img.shape[:2]

        # Read GT balls
        gt_balls = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5 and int(parts[0]) == 3:
                    cx, cy, nw, nh = map(float, parts[1:])
                    bw_px = nw * iw
                    bh_px = nh * ih
                    x1 = (cx - nw/2) * iw
                    y1 = (cy - nh/2) * ih
                    x2 = (cx + nw/2) * iw
                    y2 = (cy + nh/2) * ih
                    gt_balls.append({"bbox": [x1, y1, x2, y2], "width_px": bw_px})

        if not gt_balls:
            continue

        # Run inference
        preds = best_model(img, classes=[3], conf=0.15, imgsz=1280, device=device, verbose=False)[0]
        pred_boxes = [box.xyxy[0].cpu().numpy().tolist() for box in preds.boxes]

        for gt in gt_balls:
            w_px = gt["width_px"]
            category = "lt_16px" if w_px < 16 else ("16_32px" if w_px <= 32 else "gt_32px")
            ball_stats[category]["total"] += 1

            # Check IoU or center distance
            gx1, gy1, gx2, gy2 = gt["bbox"]
            gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
            
            matched = False
            for pb in pred_boxes:
                # Intersection over GT area or proximity
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

    # Compile Final Report
    report = {
        "experiment_name": "TRAIN-001 / DETECTOR-EVAL-001",
        "configuration": config,
        "training_time_seconds": training_time_sec,
        "best_weights": v2_weights_dest,
        "validation_metrics": val_summary,
        "locked_test_metrics": test_summary,
        "small_ball_breakdown": ball_stats
    }

    report_path = os.path.join(meta_dir, "train_001_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("DETECTOR-EVAL-001: FINAL LOCKED TEST BENCHMARK RESULTS")
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

    print(f"\nFull report written to: {report_path}")
    return report


if __name__ == "__main__":
    run_train_001()
