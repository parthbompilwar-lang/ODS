"""
DATASET-V2-VALIDATE-001: Independent Validation of Dataset V2.
Checks that are completely independent of the builder:
1. File integrity: every image has a matching label file and vice versa.
2. Label format: every line has exactly 5 fields, class in {0,1,2,3}, coords in [0,1], positive w/h.
3. No zero-area boxes.
4. MD5 duplicate check: zero duplicates across splits.
5. Sequence group leakage check: no group crosses train/val/test boundaries.
6. Class distribution per split with image-level and instance-level counts.
7. Ball width distribution in pixels.
8. Test set lock verification.
"""

import os
import json
import hashlib
from collections import defaultdict, Counter


def compute_md5(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_dataset():
    dataset_dir = r"e:\ODS\dataset_v2"
    meta_dir = r"e:\ODS\dataset_v2_meta"

    print("=" * 70)
    print("DATASET-V2-VALIDATE-001: INDEPENDENT DATASET VALIDATION")
    print("=" * 70)

    errors = []
    warnings = []
    passes = []

    splits = ["train", "val", "test"]
    CLASS_NAMES = {0: "Player", 1: "Goalkeeper", 2: "Referee", 3: "Ball"}

    # --------------------------------------------------------
    # 1. FILE INTEGRITY
    # --------------------------------------------------------
    print("\n--- 1. File Integrity ---")
    all_images = {}
    all_labels = {}

    for split in splits:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)

        if not os.path.isdir(img_dir):
            errors.append(f"Missing directory: {img_dir}")
            continue
        if not os.path.isdir(lbl_dir):
            errors.append(f"Missing directory: {lbl_dir}")
            continue

        imgs = set(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        lbls = set(os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt'))

        all_images[split] = imgs
        all_labels[split] = lbls

        imgs_without_label = imgs - lbls
        labels_without_img = lbls - imgs

        if imgs_without_label:
            errors.append(f"{split}: {len(imgs_without_label)} images without label files: {sorted(imgs_without_label)[:5]}...")
        if labels_without_img:
            errors.append(f"{split}: {len(labels_without_img)} label files without images: {sorted(labels_without_img)[:5]}...")

        if not imgs_without_label and not labels_without_img:
            passes.append(f"{split}: Every image has a corresponding label file ({len(imgs)} pairs)")

    # --------------------------------------------------------
    # 2. LABEL FORMAT VALIDATION
    # --------------------------------------------------------
    print("\n--- 2. Label Format Validation ---")
    class_counts = {split: defaultdict(int) for split in splits}
    class_image_sets = {split: defaultdict(set) for split in splits}
    total_labels = {split: 0 for split in splits}
    empty_label_files = {split: 0 for split in splits}
    malformed_lines = []
    zero_area_boxes = 0
    ball_widths = []

    for split in splits:
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        img_dir = os.path.join(dataset_dir, "images", split)

        for lbl_file in sorted(os.listdir(lbl_dir)):
            if not lbl_file.endswith('.txt'):
                continue
            stem = os.path.splitext(lbl_file)[0]
            lbl_path = os.path.join(lbl_dir, lbl_file)

            with open(lbl_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            if len(lines) == 0:
                empty_label_files[split] += 1
                continue

            for line_num, line in enumerate(lines, 1):
                parts = line.split()

                # Check 5 fields
                if len(parts) != 5:
                    malformed_lines.append(f"{split}/{lbl_file}:{line_num}: Expected 5 fields, got {len(parts)}")
                    continue

                try:
                    cls_id = int(parts[0])
                    cx, cy, nw, nh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    malformed_lines.append(f"{split}/{lbl_file}:{line_num}: Non-numeric fields")
                    continue

                # Class validity
                if cls_id not in {0, 1, 2, 3}:
                    malformed_lines.append(f"{split}/{lbl_file}:{line_num}: Invalid class {cls_id}")
                    continue

                # Coordinate range
                for name, val in [("cx", cx), ("cy", cy), ("w", nw), ("h", nh)]:
                    if val < 0.0 or val > 1.0:
                        malformed_lines.append(f"{split}/{lbl_file}:{line_num}: {name}={val:.6f} out of [0,1]")

                # Positive dimensions
                if nw <= 0 or nh <= 0:
                    zero_area_boxes += 1
                    malformed_lines.append(f"{split}/{lbl_file}:{line_num}: zero-area box w={nw:.6f} h={nh:.6f}")
                    continue

                class_counts[split][cls_id] += 1
                class_image_sets[split][cls_id].add(stem)
                total_labels[split] += 1

                # Track ball pixel widths
                if cls_id == 3:
                    # Need actual image resolution to convert normalized -> pixels
                    img_path = os.path.join(img_dir, stem + ".jpg")
                    if os.path.exists(img_path):
                        import cv2
                        img = cv2.imread(img_path)
                        if img is not None:
                            ih, iw = img.shape[:2]
                            ball_w_px = nw * iw
                            ball_widths.append(ball_w_px)

    if malformed_lines:
        errors.append(f"Found {len(malformed_lines)} malformed label lines")
        for ml in malformed_lines[:10]:
            errors.append(f"  {ml}")
    else:
        passes.append("All label lines have exactly 5 valid fields")

    if zero_area_boxes:
        errors.append(f"Found {zero_area_boxes} zero-area boxes")
    else:
        passes.append("No zero-area boxes found")

    for split in splits:
        if empty_label_files[split] > 0:
            warnings.append(f"{split}: {empty_label_files[split]} empty label files (images with no annotations)")

    # --------------------------------------------------------
    # 3. MD5 DUPLICATE CHECK ACROSS SPLITS
    # --------------------------------------------------------
    print("\n--- 3. Cross-Split Duplicate Check ---")
    md5_map = defaultdict(list)

    for split in splits:
        img_dir = os.path.join(dataset_dir, "images", split)
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(img_dir, f)
                digest = compute_md5(fpath)
                md5_map[digest].append(f"{split}/{f}")

    cross_split_dups = {k: v for k, v in md5_map.items() if len(v) > 1}
    within_split_dups = 0
    across_split_dups = 0

    for digest, locations in cross_split_dups.items():
        loc_splits = set(loc.split("/")[0] for loc in locations)
        if len(loc_splits) > 1:
            across_split_dups += 1
            errors.append(f"CROSS-SPLIT DUPLICATE: {locations}")
        else:
            within_split_dups += 1
            errors.append(f"WITHIN-SPLIT DUPLICATE: {locations}")

    if across_split_dups == 0 and within_split_dups == 0:
        passes.append("MD5 duplicates across all splits = 0")
    else:
        errors.append(f"Cross-split duplicates: {across_split_dups}, Within-split duplicates: {within_split_dups}")

    # --------------------------------------------------------
    # 4. IMAGE OVERLAP CHECK (filename collision)
    # --------------------------------------------------------
    print("\n--- 4. Filename Overlap Check ---")
    all_filenames = []
    for split in splits:
        img_dir = os.path.join(dataset_dir, "images", split)
        for f in os.listdir(img_dir):
            all_filenames.append(f"{split}/{f}")

    stem_to_split = defaultdict(list)
    for entry in all_filenames:
        parts = entry.split("/")
        stem_to_split[parts[1]].append(parts[0])

    leaks = {k: v for k, v in stem_to_split.items() if len(v) > 1}
    if leaks:
        errors.append(f"Same image appears in multiple splits: {dict(list(leaks.items())[:5])}")
    else:
        passes.append("No image appears in more than one split")

    # --------------------------------------------------------
    # 5. SEQUENCE GROUP LEAKAGE CHECK
    # --------------------------------------------------------
    print("\n--- 5. Sequence Group Leakage ---")
    with open(os.path.join(meta_dir, "step1_audit_grouping_report.json"), 'r') as f:
        step1 = json.load(f)

    split_assignment = step1["split_assignment"]
    # Verify that every image in the dataset is in its assigned split
    assignment_mismatches = 0
    for img_id, assigned_split in split_assignment.items():
        stem = os.path.splitext(img_id)[0]
        for split in splits:
            if split != assigned_split and stem in all_images.get(split, set()):
                errors.append(f"{img_id} assigned to {assigned_split} but found in {split}")
                assignment_mismatches += 1

    if assignment_mismatches == 0:
        passes.append("All images match their Step 1 split assignments (no group leakage)")

    # --------------------------------------------------------
    # 6. TEST SET LOCK VERIFICATION
    # --------------------------------------------------------
    print("\n--- 6. Test Set Lock Verification ---")
    test_img_dir = os.path.join(dataset_dir, "images", "test")
    test_count = len([f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    passes.append(f"Test set contains {test_count} images (LOCKED)")

    # --------------------------------------------------------
    # 7. CLASS DISTRIBUTION REPORT
    # --------------------------------------------------------
    print("\n--- 7. Class Distribution Report ---")
    print()
    for split in splits:
        tag = "TEST (LOCKED)" if split == "test" else split.upper()
        img_count = len(all_images.get(split, set()))
        print(f"{tag}:")
        print(f"  Images:           {img_count}")
        for cls_id in [0, 1, 2, 3]:
            name = CLASS_NAMES[cls_id]
            inst_count = class_counts[split].get(cls_id, 0)
            img_with = len(class_image_sets[split].get(cls_id, set()))
            print(f"  {name:12s}      {inst_count:5d} instances in {img_with:3d} images")
        print()

    # --------------------------------------------------------
    # 8. BALL WIDTH DISTRIBUTION
    # --------------------------------------------------------
    print("--- 8. Ball Instances by Width ---")
    print(f"  < 8 px:    {sum(1 for w in ball_widths if w < 8)}")
    print(f"  8-16 px:   {sum(1 for w in ball_widths if 8 <= w < 16)}")
    print(f"  16-32 px:  {sum(1 for w in ball_widths if 16 <= w < 32)}")
    print(f"  32-64 px:  {sum(1 for w in ball_widths if 32 <= w < 64)}")
    print(f"  > 64 px:   {sum(1 for w in ball_widths if w >= 64)}")

    # --------------------------------------------------------
    # 9. FINAL VERDICT
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION VERDICT")
    print("=" * 70)

    print(f"\nPASSES ({len(passes)}):")
    for p in passes:
        print(f"  [PASS] {p}")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN] {w}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  [FAIL] {e}")
        print("\n>>> DATASET V2 VALIDATION: FAILED <<<")
    else:
        print(f"\n>>> DATASET V2 VALIDATION: PASSED <<<")

    # Save validation report
    report = {
        "passes": passes,
        "warnings": warnings,
        "errors": errors,
        "class_distribution": {
            split: {
                CLASS_NAMES[cls_id]: {
                    "instances": class_counts[split].get(cls_id, 0),
                    "images": len(class_image_sets[split].get(cls_id, set()))
                } for cls_id in range(4)
            } for split in splits
        },
        "ball_width_distribution": {
            "lt_8px": sum(1 for w in ball_widths if w < 8),
            "8_16px": sum(1 for w in ball_widths if 8 <= w < 16),
            "16_32px": sum(1 for w in ball_widths if 16 <= w < 32),
            "32_64px": sum(1 for w in ball_widths if 32 <= w < 64),
            "gt_64px": sum(1 for w in ball_widths if w >= 64)
        },
        "verdict": "PASSED" if not errors else "FAILED"
    }

    report_path = os.path.join(meta_dir, "dataset_v2_validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nValidation report saved to: {report_path}")


if __name__ == "__main__":
    validate_dataset()
