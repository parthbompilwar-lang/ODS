"""
ODS Dataset V2 Builder - Step 1: Audit, Deduplication & Sequence Grouping.
Executes the scientific protocol:
1. Exact duplicate identification and resolution (MD5 hashing).
2. Systematic audit and triage of the 30 Skip frames.
3. Sequence and match burst clustering for leak-free group splitting.
4. Prepares candidate evaluation pipeline for Ball and Referee annotations.
"""

import os
import json
import hashlib
import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def compute_md5(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_and_group_dataset():
    img_dir = r"e:\ODS\Offside_Images"
    json_path = r"e:\ODS\final_data.json"
    output_meta_dir = r"e:\ODS\dataset_v2_meta"
    os.makedirs(output_meta_dir, exist_ok=True)

    print("=" * 70)
    print("DATASET-V2-BUILD-001: STEP 1 - AUDIT, DEDUPLICATION & GROUPING")
    print("=" * 70)

    with open(json_path, 'r') as f:
        data = json.load(f)

    annotation_map = {entry["Image_ID"]: entry for entry in data if "Image_ID" in entry}

    all_files = sorted(
        [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 99999
    )
    print(f"Total image files detected on disk: {len(all_files)}")

    # 1. MD5 Deduplication
    print("\n--- 1. MD5 Exact Duplicate Analysis ---")
    md5_to_files = defaultdict(list)
    file_to_md5 = {}
    for f in all_files:
        fpath = os.path.join(img_dir, f)
        digest = compute_md5(fpath)
        md5_to_files[digest].append(f)
        file_to_md5[f] = digest

    exact_duplicate_groups = [files for files in md5_to_files.values() if len(files) > 1]
    print(f"Found {len(exact_duplicate_groups)} exact duplicate groups ({sum(len(g) for g in exact_duplicate_groups)} total images).")

    canonical_images = set()
    removed_duplicates = {}

    for g in exact_duplicate_groups:
        g_sorted = sorted(g, key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 99999)
        canonical = g_sorted[0]
        canonical_images.add(canonical)
        for dup in g_sorted[1:]:
            removed_duplicates[dup] = {
                "canonical": canonical,
                "reason": "Exact bit-for-bit MD5 duplicate"
            }

    for digest, files in md5_to_files.items():
        if len(files) == 1:
            canonical_images.add(files[0])

    print(f"Canonical images after exact deduplication: {len(canonical_images)} (Removed {len(removed_duplicates)} files)")

    # 2. Audit and Triage of the 30 Skip Frames
    print("\n--- 2. Systematic Audit of the 30 Skip Frames ---")
    skip_frames = [f for f in all_files if annotation_map.get(f, {}).get("Pose") == "Skip"]
    print(f"Total Skip frames in final_data.json: {len(skip_frames)}")

    skip_triage = {}
    usable_skip_frames = []
    duplicate_skip_frames = []
    unusable_skip_frames = []

    for sf in skip_frames:
        fpath = os.path.join(img_dir, sf)
        if sf in removed_duplicates:
            duplicate_skip_frames.append(sf)
            skip_triage[sf] = {
                "decision": "REMOVE_DUPLICATE",
                "canonical": removed_duplicates[sf]["canonical"],
                "reason": "Exact duplicate of earlier canonical frame"
            }
            continue

        img = cv2.imread(fpath)
        if img is None:
            unusable_skip_frames.append(sf)
            skip_triage[sf] = {"decision": "EXCLUDE_UNUSABLE", "reason": "Image could not be decoded"}
            continue

        h, w = img.shape[:2]
        mean_val = float(np.mean(img))
        if mean_val < 15.0 or mean_val > 240.0:
            unusable_skip_frames.append(sf)
            skip_triage[sf] = {"decision": "EXCLUDE_UNUSABLE", "reason": f"Extreme brightness/darkness (mean={mean_val:.1f})"}
        else:
            usable_skip_frames.append(sf)
            skip_triage[sf] = {
                "decision": "USABLE_FOR_ANNOTATION",
                "resolution": [w, h],
                "mean_val": round(mean_val, 2),
                "reason": "Valid match frame; needs human-verified annotation"
            }

    print(f"Skip Frames Triage Results:")
    print(f"  - Duplicate of existing canonical: {len(duplicate_skip_frames)}")
    print(f"  - Unusable / corrupted:            {len(unusable_skip_frames)}")
    print(f"  - Usable match scenes retained:    {len(usable_skip_frames)}")

    # 3. Match / Sequence Burst Clustering
    print("\n--- 3. Sequence / Match Burst Clustering ---")
    active_dataset_files = sorted(
        [f for f in canonical_images if skip_triage.get(f, {}).get("decision") != "EXCLUDE_UNUSABLE"],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 99999
    )
    print(f"Total active candidate images for Dataset V2: {len(active_dataset_files)}")

    # Extract perceptual features (resolution, HSV histogram, downscaled thumbnail)
    features = {}
    for f in active_dataset_files:
        fpath = os.path.join(img_dir, f)
        img = cv2.imread(fpath)
        if img is None: continue
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        thumb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 32))
        features[f] = {"res": (w, h), "hist": hist, "thumb": thumb}

    sequences = []
    current_seq = [active_dataset_files[0]]

    for i in range(1, len(active_dataset_files)):
        prev_f = active_dataset_files[i - 1]
        curr_f = active_dataset_files[i]

        p_feat = features[prev_f]
        c_feat = features[curr_f]

        res_match = (p_feat["res"] == c_feat["res"])
        hist_corr = cv2.compareHist(p_feat["hist"], c_feat["hist"], cv2.HISTCMP_CORREL)
        thumb_diff = np.mean((p_feat["thumb"].astype(float) - c_feat["thumb"].astype(float)) ** 2)

        # Contiguous replay frames share resolution, high color correlation, and low visual MSE
        if res_match and hist_corr > 0.82 and thumb_diff < 1500:
            current_seq.append(curr_f)
        else:
            sequences.append(current_seq)
            current_seq = [curr_f]

    if current_seq:
        sequences.append(current_seq)

    print(f"Clustered active images into {len(sequences)} distinct sequence/match groups.")

    # 4. Sequence-Disjoint Split (70% Train, 15% Val, 15% Locked Test)
    np.random.seed(42)
    seq_indices = list(range(len(sequences)))
    np.random.shuffle(seq_indices)

    total_imgs = len(active_dataset_files)
    target_train = int(total_imgs * 0.70)
    target_val = int(total_imgs * 0.15)

    train_seqs, val_seqs, test_seqs = [], [], []
    train_count, val_count, test_count = 0, 0, 0

    for s_idx in seq_indices:
        seq = sequences[s_idx]
        seq_len = len(seq)
        if train_count + seq_len <= target_train or (val_count >= target_val and test_count >= target_val):
            train_seqs.append(seq)
            train_count += seq_len
        elif val_count + seq_len <= target_val or test_count >= target_val:
            val_seqs.append(seq)
            val_count += seq_len
        else:
            test_seqs.append(seq)
            test_count += seq_len

    print("\n--- 4. Final Group Split Partitioning ---")
    print(f"Train Set: {train_count} images ({len(train_seqs)} groups, {train_count/total_imgs*100:.1f}%)")
    print(f"Val Set:   {val_count} images ({len(val_seqs)} groups, {val_count/total_imgs*100:.1f}%)")
    print(f"Test Set (LOCKED): {test_count} images ({len(test_seqs)} groups, {test_count/total_imgs*100:.1f}%)")
    print(f"Total:     {train_count + val_count + test_count} images across {len(sequences)} sequence groups.")

    split_assignment = {}
    for seq in train_seqs:
        for f in seq: split_assignment[f] = "train"
    for seq in val_seqs:
        for f in seq: split_assignment[f] = "val"
    for seq in test_seqs:
        for f in seq: split_assignment[f] = "test"

    step1_report = {
        "total_source_images": len(all_files),
        "exact_duplicates_removed": len(removed_duplicates),
        "removed_duplicates_detail": removed_duplicates,
        "skip_frames_audit": {
            "total_skip": len(skip_frames),
            "duplicate_skip": len(duplicate_skip_frames),
            "unusable_skip": len(unusable_skip_frames),
            "usable_skip_retained": len(usable_skip_frames),
            "triage_detail": skip_triage
        },
        "sequence_groups": {
            "total_groups": len(sequences),
            "group_sizes": [len(s) for s in sequences]
        },
        "splits": {
            "train": {"count": train_count, "groups": len(train_seqs)},
            "val": {"count": val_count, "groups": len(val_seqs)},
            "test_locked": {"count": test_count, "groups": len(test_seqs)}
        },
        "split_assignment": split_assignment
    }

    report_path = os.path.join(output_meta_dir, "step1_audit_grouping_report.json")
    with open(report_path, 'w') as f:
        json.dump(step1_report, f, indent=2)

    print(f"\nStep 1 Report saved to: {report_path}")
    return step1_report


if __name__ == "__main__":
    audit_and_group_dataset()
