"""
DATASET-V2-ANNOTATION-001: Compile verified annotations into YOLO format.
Combines:
- Existing Player/GK labels from final_data.json (verified ground truth)
- Accepted Ball labels from ball_candidates.json (1 per image max)
- Accepted Referee labels from referee_candidates.json (1 per image max)

Output:
- dataset_v2/images/{train,val,test}/
- dataset_v2/labels/{train,val,test}/
- dataset_v2/data.yaml
- dataset_v2/manifest.json

Class mapping:
  0 = Player (Team1 + Team2 outfield)
  1 = Goalkeeper (GK)
  2 = Referee
  3 = Ball
"""

import os
import json
import shutil
import cv2
from collections import defaultdict


def compile_annotations():
    img_dir = r"e:\ODS\Offside_Images"
    json_path = r"e:\ODS\final_data.json"
    meta_dir = r"e:\ODS\dataset_v2_meta"
    output_dir = r"e:\ODS\dataset_v2"

    with open(json_path, 'r') as f:
        data = json.load(f)
    annotation_map = {entry["Image_ID"]: entry for entry in data if "Image_ID" in entry}

    with open(os.path.join(meta_dir, "step1_audit_grouping_report.json"), 'r') as f:
        step1 = json.load(f)
    split_assignment = step1["split_assignment"]

    with open(os.path.join(meta_dir, "ball_candidates.json"), 'r') as f:
        balls = json.load(f)
    accepted_balls = {}
    for b in balls:
        if b["review_status"] == "ACCEPTED":
            accepted_balls[b["image_id"]] = b

    with open(os.path.join(meta_dir, "referee_candidates.json"), 'r') as f:
        refs = json.load(f)
    accepted_refs = {}
    for r in refs:
        if r["review_status"] == "ACCEPTED":
            accepted_refs[r["image_id"]] = r

    # Create directory structure
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    canonical_files = sorted(
        list(split_assignment.keys()),
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 99999
    )

    manifest_entries = []
    stats = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int)
    }
    class_images = {
        "train": defaultdict(set),
        "val": defaultdict(set),
        "test": defaultdict(set)
    }
    ball_widths_all = []
    images_processed = 0
    images_skipped_no_annotation = 0

    for img_id in canonical_files:
        split = split_assignment[img_id]
        img_path = os.path.join(img_dir, img_id)
        img = cv2.imread(img_path)
        if img is None:
            images_skipped_no_annotation += 1
            continue

        h, w = img.shape[:2]
        entry = annotation_map.get(img_id, {})
        pose = entry.get("Pose", None)

        labels = []

        # Player and Goalkeeper from original annotations
        if isinstance(pose, dict):
            for team_key in ["Team1", "Team2"]:
                for p in pose.get(team_key, []):
                    pts = p.get("geometry", [])
                    xs = [pt['x'] for pt in pts if pt.get('x') is not None]
                    ys = [pt['y'] for pt in pts if pt.get('y') is not None]
                    if xs and ys:
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        bw = x_max - x_min
                        bh = y_max - y_min
                        if bw > 0 and bh > 0:
                            cx = (x_min + x_max) / 2.0 / w
                            cy = (y_min + y_max) / 2.0 / h
                            nw = bw / w
                            nh = bh / h
                            # Clamp to [0, 1]
                            cx = max(0.0, min(1.0, cx))
                            cy = max(0.0, min(1.0, cy))
                            nw = max(0.001, min(1.0, nw))
                            nh = max(0.001, min(1.0, nh))
                            labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                            stats[split]["player"] += 1
                            class_images[split]["player"].add(img_id)

            for p in pose.get("GK", []):
                pts = p.get("geometry", [])
                xs = [pt['x'] for pt in pts if pt.get('x') is not None]
                ys = [pt['y'] for pt in pts if pt.get('y') is not None]
                if xs and ys:
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    bw = x_max - x_min
                    bh = y_max - y_min
                    if bw > 0 and bh > 0:
                        cx = (x_min + x_max) / 2.0 / w
                        cy = (y_min + y_max) / 2.0 / h
                        nw = bw / w
                        nh = bh / h
                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))
                        nw = max(0.001, min(1.0, nw))
                        nh = max(0.001, min(1.0, nh))
                        labels.append(f"1 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                        stats[split]["goalkeeper"] += 1
                        class_images[split]["goalkeeper"].add(img_id)

        # Referee (accepted only, max 1 per image)
        if img_id in accepted_refs:
            ref = accepted_refs[img_id]
            bbox = ref["bbox"]
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            if bw > 0 and bh > 0:
                cx = (bbox[0] + bbox[2]) / 2.0 / w
                cy = (bbox[1] + bbox[3]) / 2.0 / h
                nw = bw / w
                nh = bh / h
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.001, min(1.0, nw))
                nh = max(0.001, min(1.0, nh))
                labels.append(f"2 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                stats[split]["referee"] += 1
                class_images[split]["referee"].add(img_id)

        # Ball (accepted only, max 1 per image; skip if not visible)
        if img_id in accepted_balls:
            ball = accepted_balls[img_id]
            bbox = ball["bbox"]
            bw_px = bbox[2] - bbox[0]
            bh_px = bbox[3] - bbox[1]
            if bw_px > 0 and bh_px > 0:
                cx = (bbox[0] + bbox[2]) / 2.0 / w
                cy = (bbox[1] + bbox[3]) / 2.0 / h
                nw = bw_px / w
                nh = bh_px / h
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.001, min(1.0, nw))
                nh = max(0.001, min(1.0, nh))
                labels.append(f"3 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                stats[split]["ball"] += 1
                class_images[split]["ball"].add(img_id)
                ball_widths_all.append(bw_px)

        # Copy image and write label
        dst_img = os.path.join(output_dir, "images", split, img_id)
        shutil.copy2(img_path, dst_img)

        label_name = os.path.splitext(img_id)[0] + ".txt"
        dst_label = os.path.join(output_dir, "labels", split, label_name)
        with open(dst_label, 'w') as f:
            f.write("\n".join(labels) + "\n" if labels else "")

        manifest_entries.append({
            "image_id": img_id,
            "split": split,
            "resolution": [w, h],
            "player_count": stats[split]["player"] - sum(1 for l in labels if l.startswith("0")) + sum(1 for l in labels if l.startswith("0")),
            "counts": {
                "player": sum(1 for l in labels if l.startswith("0 ")),
                "goalkeeper": sum(1 for l in labels if l.startswith("1 ")),
                "referee": sum(1 for l in labels if l.startswith("2 ")),
                "ball": sum(1 for l in labels if l.startswith("3 "))
            },
            "has_ball": img_id in accepted_balls,
            "has_referee": img_id in accepted_refs
        })
        images_processed += 1

    # Write data.yaml
    data_yaml = os.path.join(output_dir, "data.yaml")
    with open(data_yaml, 'w') as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("nc: 4\n")
        f.write("names:\n")
        f.write("  0: Player\n")
        f.write("  1: Goalkeeper\n")
        f.write("  2: Referee\n")
        f.write("  3: Ball\n")

    # Write manifest.json
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest_entries, f, indent=2)

    # Print summary
    print("=" * 70)
    print("DATASET-V2-ANNOTATION-001: COMPILATION COMPLETE")
    print("=" * 70)
    print(f"Images processed: {images_processed}")
    print(f"Images skipped (unreadable): {images_skipped_no_annotation}")
    print()

    for split in ["train", "val", "test"]:
        tag = "TEST (LOCKED)" if split == "test" else split.upper()
        img_count = sum(1 for e in manifest_entries if e["split"] == split)
        print(f"{tag}:")
        print(f"  Images:     {img_count}")
        print(f"  Player:     {stats[split]['player']} instances in {len(class_images[split]['player'])} images")
        print(f"  Goalkeeper: {stats[split]['goalkeeper']} instances in {len(class_images[split]['goalkeeper'])} images")
        print(f"  Referee:    {stats[split]['referee']} instances in {len(class_images[split]['referee'])} images")
        print(f"  Ball:       {stats[split]['ball']} instances in {len(class_images[split]['ball'])} images")
        print()

    # Ball width distribution
    print("Ball instances by width:")
    print(f"  < 8 px:    {sum(1 for w in ball_widths_all if w < 8)}")
    print(f"  8-16 px:   {sum(1 for w in ball_widths_all if 8 <= w < 16)}")
    print(f"  16-32 px:  {sum(1 for w in ball_widths_all if 16 <= w < 32)}")
    print(f"  32-64 px:  {sum(1 for w in ball_widths_all if 32 <= w < 64)}")
    print(f"  > 64 px:   {sum(1 for w in ball_widths_all if w >= 64)}")

    print(f"\nDataset V2 saved to: {output_dir}")
    print(f"data.yaml: {data_yaml}")
    print(f"manifest.json: {manifest_path}")


if __name__ == "__main__":
    compile_annotations()
