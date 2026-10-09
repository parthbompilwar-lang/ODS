"""
DATASET-AUDIT-001: Comprehensive Scientific Audit of the Offside Detection Dataset.
Performs exhaustive empirical analysis of raw dataset files, annotations,
class distributions, resolution, match sequences, duplicates, and omissions.
"""

import os
import json
import hashlib
import cv2
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Computes difference hash for perceptual image deduplication."""
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    diff = gray[:, 1:] > gray[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def audit_dataset():
    img_dir = r"e:\ODS\Offside_Images"
    json_path = r"e:\ODS\final_data.json"

    print("=" * 70)
    print("DATASET-AUDIT-001: STARTING EMPIRICAL AUDIT")
    print("=" * 70)

    # 1. File Inventory
    all_files = os.listdir(img_dir)
    non_images = [f for f in all_files if not f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Sort numerically by filename stem if digits
    image_files_sorted = sorted(
        image_files,
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 99999
    )

    with open(json_path, 'r') as f:
        data = json.load(f)

    json_ids = [entry.get("Image_ID") for entry in data]
    json_id_set = set(json_ids)
    disk_id_set = set(image_files)

    # 2. Annotation Structure Inspection
    valid_entries = []
    skip_entries = []
    other_entries = []

    for idx, entry in enumerate(data):
        img_id = entry.get("Image_ID")
        pose = entry.get("Pose")
        if isinstance(pose, dict):
            valid_entries.append((idx, img_id, pose))
        elif pose == "Skip":
            skip_entries.append((idx, img_id, pose))
        else:
            other_entries.append((idx, img_id, pose))

    # 3. Class and Instance Inventory in Annotations
    all_annotated_keys = set()
    for _, _, pose in valid_entries:
        all_annotated_keys.update(pose.keys())

    team1_instances = 0
    team2_instances = 0
    gk_instances = 0
    referee_instances = 0
    ball_instances = 0

    team1_imgs = 0
    team2_imgs = 0
    gk_imgs = 0

    players_per_image = []
    box_widths = []
    box_heights = []
    box_areas = []

    # Check keypoint count per player
    kpt_counts = Counter()

    # 4. Image Resolution & Integrity Audit
    resolutions = Counter()
    corrupted_images = []
    dhashes = {}
    duplicate_pairs = []

    for img_name in image_files_sorted:
        img_path = os.path.join(img_dir, img_name)
        try:
            img = cv2.imread(img_path)
            if img is None:
                corrupted_images.append(img_name)
                continue
            h, w, c = img.shape
            resolutions[(w, h)] += 1

            # Deduplication hash
            h_val = compute_dhash(img)
            dhashes[img_name] = (h_val, (w, h))
        except Exception as e:
            corrupted_images.append((img_name, str(e)))

    # Find near-duplicate frames (hamming distance <= 2 on 64-bit dhash)
    img_names = list(dhashes.keys())
    near_duplicates = []
    for i in range(len(img_names)):
        for j in range(i + 1, min(i + 20, len(img_names))):  # sequential window
            n1, n2 = img_names[i], img_names[j]
            h1, res1 = dhashes[n1]
            h2, res2 = dhashes[n2]
            if res1 == res2:
                hamming = bin(h1 ^ h2).count('1')
                if hamming <= 3:
                    near_duplicates.append((n1, n2, hamming))

    # Detailed instance metrics
    for idx, img_id, pose in valid_entries:
        t1 = pose.get("Team1", [])
        t2 = pose.get("Team2", [])
        gk = pose.get("GK", [])

        if t1: team1_imgs += 1
        if t2: team2_imgs += 1
        if gk: gk_imgs += 1

        team1_instances += len(t1)
        team2_instances += len(t2)
        gk_instances += len(gk)

        # Check other keys (Referee, Ball, etc.)
        for k in pose.keys():
            if k not in ["Team1", "Team2", "GK"]:
                if "ref" in k.lower(): referee_instances += len(pose[k])
                if "ball" in k.lower(): ball_instances += len(pose[k])

        img_player_count = len(t1) + len(t2) + len(gk)
        players_per_image.append(img_player_count)

        # Measure bboxes
        img_path = os.path.join(img_dir, img_id)
        img = cv2.imread(img_path)
        if img is not None:
            img_h, img_w = img.shape[:2]
            for p_list in [t1, t2, gk]:
                for p in p_list:
                    pts = p.get("geometry", [])
                    kpt_counts[len(pts)] += 1
                    xs = [pt['x'] for pt in pts if pt.get('x') is not None]
                    ys = [pt['y'] for pt in pts if pt.get('y') is not None]
                    if xs and ys:
                        bw = max(xs) - min(xs)
                        bh = max(ys) - min(ys)
                        box_widths.append(bw)
                        box_heights.append(bh)
                        box_areas.append(bw * bh)

    # 5. Scene / Broadcast Cluster Analysis
    # Let's inspect sequences: group consecutive filenames by resolution and color palette
    sequences = []
    current_seq = [image_files_sorted[0]]
    for i in range(1, len(image_files_sorted)):
        prev_f = image_files_sorted[i-1]
        curr_f = image_files_sorted[i]
        prev_res = dhashes.get(prev_f, (0, (0, 0)))[1]
        curr_res = dhashes.get(curr_f, (0, (0, 0)))[1]
        prev_h = dhashes.get(prev_f, (0, (0, 0)))[0]
        curr_h = dhashes.get(curr_f, (0, (0, 0)))[0]
        ham = bin(prev_h ^ curr_h).count('1')

        # If same resolution and small distance or contiguous index
        if prev_res == curr_res and ham <= 12:
            current_seq.append(curr_f)
        else:
            sequences.append(current_seq)
            current_seq = [curr_f]
    if current_seq:
        sequences.append(current_seq)

    # Compile Structured Audit Dictionary
    report = {
        "file_inventory": {
            "total_files_in_folder": len(all_files),
            "image_files_on_disk": len(image_files),
            "non_image_files": non_images,
            "json_entries": len(data),
            "json_images_matching_disk": len(json_id_set.intersection(disk_id_set)),
            "images_on_disk_not_in_json": list(disk_id_set - json_id_set),
            "json_entries_not_on_disk": list(json_id_set - disk_id_set),
        },
        "annotations": {
            "valid_pose_dict_entries": len(valid_entries),
            "skip_entries": len(skip_entries),
            "skip_image_ids": [img_id for _, img_id, _ in skip_entries],
            "other_entries": len(other_entries),
            "annotated_dictionary_keys": list(all_annotated_keys),
            "instances": {
                "Team1_player_instances": team1_instances,
                "Team2_player_instances": team2_instances,
                "total_outfield_player_instances": team1_instances + team2_instances,
                "goalkeeper_instances": gk_instances,
                "referee_instances": referee_instances,
                "ball_instances": ball_instances,
                "total_instances": team1_instances + team2_instances + gk_instances + referee_instances + ball_instances
            },
            "images_containing_class": {
                "Team1_images": team1_imgs,
                "Team2_images": team2_imgs,
                "goalkeeper_images": gk_imgs,
                "referee_images": 0,
                "ball_images": 0
            },
            "keypoints_distribution": dict(kpt_counts),
        },
        "image_properties": {
            "corrupted_images": corrupted_images,
            "resolutions": {f"{w}x{h}": count for (w, h), count in resolutions.items()},
            "players_per_image": {
                "min": int(np.min(players_per_image)) if players_per_image else 0,
                "max": int(np.max(players_per_image)) if players_per_image else 0,
                "mean": float(np.mean(players_per_image)) if players_per_image else 0,
                "median": float(np.median(players_per_image)) if players_per_image else 0
            },
            "bounding_boxes_px": {
                "width_mean": float(np.mean(box_widths)) if box_widths else 0,
                "width_min": float(np.min(box_widths)) if box_widths else 0,
                "width_max": float(np.max(box_widths)) if box_widths else 0,
                "height_mean": float(np.mean(box_heights)) if box_heights else 0,
                "height_min": float(np.min(box_heights)) if box_heights else 0,
                "height_max": float(np.max(box_heights)) if box_heights else 0,
            }
        },
        "leakage_and_sequences": {
            "estimated_video_clips_sequences": len(sequences),
            "sequence_lengths_sample": [len(s) for s in sequences[:10]],
            "near_duplicate_sequential_pairs_count": len(near_duplicates),
            "sample_near_duplicates": near_duplicates[:10]
        }
    }

    # Save full audit json
    with open("dataset_audit_report.json", "w") as out_f:
        json.dump(report, out_f, indent=2)

    print("\nAUDIT COMPLETED. Summary:")
    print(f"- Disk Images: {len(image_files)} (Non-images: {non_images})")
    print(f"- JSON Entries: {len(data)}")
    print(f"  * Valid Annotated Scenes: {len(valid_entries)}")
    print(f"  * Skipped Scenes: {len(skip_entries)}")
    print(f"- Keys Annotated in JSON: {all_annotated_keys}")
    print(f"- Instances: Outfield={team1_instances + team2_instances} (T1={team1_instances}, T2={team2_instances}), GK={gk_instances}")
    print(f"- Referees Annotated: {referee_instances}")
    print(f"- Balls Annotated: {ball_instances}")
    print(f"- Resolutions: {resolutions}")
    print(f"- Estimated Independent Sequences/Clips: {len(sequences)}")
    print(f"- Near-duplicate sequential frames: {len(near_duplicates)}")


if __name__ == "__main__":
    audit_dataset()
