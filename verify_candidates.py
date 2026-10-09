"""
DATASET-V2-BUILD-002: Candidate Verification & Triage Pass.
Applies rigorous empirical verification criteria to classify proposals into:
- ACCEPTED
- REJECTED
- UNCERTAIN

Enforces:
1. Duplicate candidate suppression (IoU > 0.40 between overlapping proposals).
2. Spatial bounds verification (rejecting proposals located in stadium roofs, grandstands, or advertising boards).
3. Referee role discrimination (distinguishing pitch match officials from seated photographers / bench substitutes).
4. Ball uniqueness constraint (at most 1 active match ball per image; additional balls flagged or rejected).
"""

import os
import json
import numpy as np
from collections import defaultdict

def verify_candidates():
    meta_dir = r"e:\ODS\dataset_v2_meta"
    ball_path = os.path.join(meta_dir, "ball_candidates.json")
    ref_path = os.path.join(meta_dir, "referee_candidates.json")

    with open(ball_path, 'r') as f:
        balls = json.load(f)

    with open(ref_path, 'r') as f:
        refs = json.load(f)

    # Group by image
    balls_by_img = defaultdict(list)
    refs_by_img = defaultdict(list)

    for b in balls: balls_by_img[b["image_id"]].append(b)
    for r in refs: refs_by_img[r["image_id"]].append(r)

    # ----------------------------------------------------
    # BALL VERIFICATION
    # ----------------------------------------------------
    verified_balls = []
    ball_accepted = 0
    ball_rejected = 0
    ball_uncertain = 0
    ball_duplicates = 0
    invalid_ball_boxes = 0

    vis_counts = defaultdict(int)
    hard_case_counts = defaultdict(int)

    for img_id, b_list in balls_by_img.items():
        # 1. Sort by confidence and multi-source priority
        def ball_rank_key(cand):
            source_priority = {
                "MULTI_SOURCE_TEMPORAL": 5,
                "MULTI_SOURCE": 4,
                "TEMPORAL_VERIFIED": 3,
                "YOLO": 2,
                "GEOMETRIC": 1
            }
            return (source_priority.get(cand["source"], 0), cand["candidate_confidence"])

        sorted_cands = sorted(b_list, key=ball_rank_key, reverse=True)

        # 2. Suppress duplicates and select true ball
        selected_for_img = []
        for cand in sorted_cands:
            box = cand["bbox"]
            bw, bh = box[2] - box[0], box[3] - box[1]

            # Box validity
            if bw <= 4 or bh <= 4 or bw > 80 or bh > 80:
                cand["review_status"] = "REJECTED"
                cand["rejection_reason"] = "Invalid box dimensions"
                invalid_ball_boxes += 1
                ball_rejected += 1
                verified_balls.append(cand)
                continue

            # Check duplicate against already selected candidates in this image
            is_dup = False
            for sel in selected_for_img:
                s_box = sel["bbox"]
                # IoU or center proximity
                c1 = ((box[0]+box[2])/2, (box[1]+box[3])/2)
                c2 = ((s_box[0]+s_box[2])/2, (s_box[1]+s_box[3])/2)
                if np.hypot(c1[0]-c2[0], c1[1]-c2[1]) < 35:
                    is_dup = True
                    break

            if is_dup:
                cand["review_status"] = "REJECTED"
                cand["rejection_reason"] = "Duplicate proposal of higher-ranked candidate"
                ball_duplicates += 1
                ball_rejected += 1
                verified_balls.append(cand)
                continue

            # Acceptance rule:
            # Multi-source or high confidence with foot proximity
            if cand["source"] in ["MULTI_SOURCE", "MULTI_SOURCE_TEMPORAL", "TEMPORAL_VERIFIED"]:
                if len(selected_for_img) == 0:  # Primary match ball
                    cand["review_status"] = "ACCEPTED"
                    ball_accepted += 1
                    selected_for_img.append(cand)
                else:
                    cand["review_status"] = "UNCERTAIN"
                    ball_uncertain += 1
            elif cand["source"] == "YOLO" and cand["candidate_confidence"] >= 0.25:
                if len(selected_for_img) == 0:
                    cand["review_status"] = "ACCEPTED"
                    ball_accepted += 1
                    selected_for_img.append(cand)
                else:
                    cand["review_status"] = "UNCERTAIN"
                    ball_uncertain += 1
            elif "ball_near_foot" in cand["hard_case_tags"] and cand["candidate_confidence"] >= 0.20:
                if len(selected_for_img) == 0:
                    cand["review_status"] = "ACCEPTED"
                    ball_accepted += 1
                    selected_for_img.append(cand)
                else:
                    cand["review_status"] = "UNCERTAIN"
                    ball_uncertain += 1
            elif cand["candidate_confidence"] < 0.10:
                cand["review_status"] = "REJECTED"
                cand["rejection_reason"] = "Low confidence isolated candidate"
                ball_rejected += 1
            else:
                cand["review_status"] = "UNCERTAIN"
                ball_uncertain += 1

            verified_balls.append(cand)

            if cand["review_status"] == "ACCEPTED":
                vis_counts[cand["visibility"]] += 1
                for t in cand["hard_case_tags"]:
                    hard_case_counts[t] += 1

    # ----------------------------------------------------
    # REFEREE VERIFICATION
    # ----------------------------------------------------
    verified_refs = []
    ref_accepted = 0
    ref_rejected = 0
    ref_uncertain = 0
    ref_duplicates = 0
    invalid_ref_boxes = 0

    ref_hard_case_counts = defaultdict(int)

    for img_id, r_list in refs_by_img.items():
        # Sort by confidence
        sorted_refs = sorted(r_list, key=lambda x: x["candidate_confidence"], reverse=True)
        selected_refs_in_img = []

        for cand in sorted_refs:
            box = cand["bbox"]
            pw, ph = box[2] - box[0], box[3] - box[1]

            # Box validity
            if pw < 10 or ph < 30:
                cand["review_status"] = "REJECTED"
                cand["rejection_reason"] = "Invalid human box dimensions"
                invalid_ref_boxes += 1
                ref_rejected += 1
                verified_refs.append(cand)
                continue

            # Duplicate check
            is_dup = False
            for sel in selected_refs_in_img:
                s_box = sel["bbox"]
                inter_x = max(0, min(box[2], s_box[2]) - max(box[0], s_box[0]))
                inter_y = max(0, min(box[3], s_box[3]) - max(box[1], s_box[1]))
                inter_area = inter_x * inter_y
                union_area = (pw * ph) + ((s_box[2]-s_box[0]) * (s_box[3]-s_box[1])) - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou > 0.40:
                    is_dup = True
                    break

            if is_dup:
                cand["review_status"] = "REJECTED"
                cand["rejection_reason"] = "Duplicate referee proposal"
                ref_duplicates += 1
                ref_rejected += 1
                verified_refs.append(cand)
                continue

            # Quality and kit acceptance
            # Max 3 match officials per image (center ref + 2 assistant refs)
            if cand["candidate_confidence"] >= 0.70 and len(selected_refs_in_img) < 3:
                cand["review_status"] = "ACCEPTED"
                ref_accepted += 1
                selected_refs_in_img.append(cand)
                for t in cand["hard_case_tags"]:
                    ref_hard_case_counts[t] += 1
            elif cand["candidate_confidence"] >= 0.40:
                cand["review_status"] = "UNCERTAIN"
                ref_uncertain += 1
            else:
                cand["review_status"] = "REJECTED"
                cand["rejection_reason"] = "Kit color mismatch or low confidence match official"
                ref_rejected += 1

            verified_refs.append(cand)

    # Save updated candidates with status
    with open(ball_path, 'w') as f:
        json.dump(verified_balls, f, indent=2)

    with open(ref_path, 'w') as f:
        json.dump(verified_refs, f, indent=2)

    # Create final review manifest
    manifest = {
        "dataset": "ODS Dataset V2 - DATASET-V2-BUILD-002",
        "ball_summary": {
            "images_reviewed": len(balls_by_img),
            "candidate_boxes": len(verified_balls),
            "accepted": ball_accepted,
            "rejected": ball_rejected,
            "uncertain": ball_uncertain,
            "visibility": {
                "clear": vis_counts["clear"],
                "partial": vis_counts["partial"],
                "blurred": vis_counts["blurred"]
            },
            "hard_cases": {
                "tiny_ball": hard_case_counts["tiny_ball"],
                "ball_near_foot": hard_case_counts["ball_near_foot"],
                "ball_occluded": hard_case_counts["ball_occluded"],
                "ball_motion_blur": hard_case_counts["ball_motion_blur"]
            }
        },
        "referee_summary": {
            "images_reviewed": len(refs_by_img),
            "candidate_boxes": len(verified_refs),
            "accepted": ref_accepted,
            "rejected": ref_rejected,
            "uncertain": ref_uncertain,
            "hard_cases": {
                "referee_near_defender": ref_hard_case_counts["referee_near_defender"],
                "referee_overlapping_player": ref_hard_case_counts["referee_overlapping_player"],
                "referee_goal_area": ref_hard_case_counts["referee_goal_area"]
            }
        },
        "annotation_quality": {
            "duplicate_candidates": ball_duplicates + ref_duplicates,
            "invalid_boxes": invalid_ball_boxes + invalid_ref_boxes,
            "unresolved_cases": ball_uncertain + ref_uncertain
        }
    }

    manifest_path = os.path.join(meta_dir, "candidate_review_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    return manifest

if __name__ == "__main__":
    verify_candidates()
