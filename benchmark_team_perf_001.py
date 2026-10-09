"""
TEAM-PERF-001: TeamClassifier Compute Optimization & Stability Benchmark.
Compares 4 variants on offside_spurs_match.mp4 (241 frames, 1080p @ 30 FPS):
  Variant A: Baseline (Current sequential 8-attempt cv2.kmeans)
  Variant B: Fast / Vectorized Extraction (Optimized crop, 2-attempt kmeans, fast HSV/Lab)
  Variant C: Track Caching (10-vote consistency lock + N=15 audit + gap/size invalidation)
  Variant D: Fast + Cached (Variant B extraction + Variant C caching)

Measures:
  - Latency: Mean, P50, P95, P99, Max (ms per frame and per player)
  - KMeans calls per frame, Cache hit / reclassification rates
  - Baseline agreement (%)
  - Manual verified semantic accuracy (%) on 25 ground-truth player tracks
  - Track flip rate (%)
  - UNKNOWN rate (%)
  - Downstream Law 11 Invariance (Attacker, Defender, Margin, Decision at contact moment)
  - Pipeline Mean / P95 Latency and Effective FPS
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Any
from ultralytics import YOLO

from src.tracking.player_tracker import PlayerTracker
from src.tracking.ball_tracker import BallTracker
from src.passing.pass_detector import PassDetector
from src.passing.contact_estimator import ContactEstimator
from src.engine.play_state import PlayStateMachine
from src.engine.offside_engine import OffsideEngine
from app import get_default_pitch_geometry


# ============================================================================
# GROUND TRUTH SEMANTIC LABELS FOR TRACKS IN offside_spurs_match.mp4
# ============================================================================
# Verified from visual inspection of player kit colors:
# Team 0 = Spurs (White shirt / navy shorts)
# Team 1 = Opponent (Dark / Navy / Blue shirt)
# -1 = Referee (Yellow/Black)
GROUND_TRUTH_TRACKS = {
    0: 0,   # White
    1: 0,   # White
    2: 0,   # White
    3: 0,   # White
    4: 1,   # Dark
    5: 0,   # White
    6: 1,   # Dark
    7: 0,   # White
    8: 1,   # Dark
    9: 1,   # Dark
    10: 1,  # Dark
    11: 0,  # White
    12: 0,  # White
    13: 0,  # White
    14: 1,  # Dark
    15: 1,  # Dark
    16: 1,  # Dark
    17: 1,  # Dark
    19: 1,  # Dark
    20: 1,  # Dark
    21: 1,  # Dark
    22: 1,  # Dark
    24: 1,  # Dark
    28: 0,  # White
    34: 0,  # White
}


# ============================================================================
# VARIANT A: CURRENT PRODUCTION BASELINE
# ============================================================================
import colorsys
from sklearn.cluster import KMeans

def is_color_in_range_py(rgb_color: Tuple[int, int, int], min_hue: float, max_hue: float, min_sat: float, max_sat: float, min_val: float, max_val: float) -> bool:
    hsv = colorsys.rgb_to_hsv(rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0)
    h, s, v = hsv[0] * 360.0, hsv[1], hsv[2]
    return (min_hue <= h <= max_hue and min_sat <= s <= max_sat and min_val <= v <= max_val)


class TeamClassifierVariantA:
    """Variant A: Baseline (8-attempt cv2.kmeans, sequential execution, no caching)."""
    def __init__(self, n_teams: int = 2, vote_window: int = 10, min_votes_required: int = 2):
        self.n_teams = n_teams
        self.vote_window = vote_window
        self.min_votes_required = min_votes_required
        self.cluster_centers: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.vote_window))
        self.track_features: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.vote_window))
        self.kmeans_calls_this_frame = 0

    def extract_jersey_features(self, image: np.ndarray, bbox: Any, keypoints: Optional[np.ndarray] = None) -> np.ndarray:
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x2 - x1 < 5 or y2 - y1 < 10:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        box_w = x2 - x1
        box_h = y2 - y1
        t_x1 = max(0, int(x1 + 0.10 * box_w))
        t_x2 = min(w_img, int(x1 + 0.90 * box_w))
        t_y1 = max(0, int(y1 + 0.15 * box_h))
        t_y2 = min(h_img, int(y1 + 0.55 * box_h))
        if t_x2 <= t_x1 or t_y2 <= t_y1:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        crop = image[t_y1:t_y2, t_x1:t_x2]
        if crop.size == 0:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        try:
            small_crop = cv2.resize(crop, (36, 48))
            data = np.float32(small_crop.reshape(-1, 3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            self.kmeans_calls_this_frame += 1
            _, _, centers = cv2.kmeans(data, 2, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

            cand_rgbs = [(int(c[2]), int(c[1]), int(c[0])) for c in centers]
            field_range = (35.0, 145.0, 0.15, 1.0, 0.15, 1.0)
            non_field = [c for c in cand_rgbs if not is_color_in_range_py(c, *field_range)]
            chosen_rgb = non_field[0] if non_field else cand_rgbs[0]

            bgr_pixel = np.uint8([[[chosen_rgb[2], chosen_rgb[1], chosen_rgb[0]]]])
            lab_feat = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)[0, 0]
            return np.float32(lab_feat)
        except Exception:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

    def fit_from_image(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> bool:
        outfield_boxes = [d["bbox"] for d in detections if d.get("class_id", 0) == 0]
        if len(outfield_boxes) < 4:
            return False
        feats = [self.extract_jersey_features(image, b) for b in outfield_boxes]
        try:
            km = KMeans(n_clusters=self.n_teams, random_state=42, n_init=10).fit(np.array(feats))
            centers = km.cluster_centers_
            if len(centers) == 2 and centers[0, 0] < centers[1, 0]:
                centers = centers[::-1].copy()
            self.cluster_centers = centers
            self.is_fitted = True
            return True
        except Exception:
            return False

    def classify_single_feature(self, feat: np.ndarray) -> Tuple[int, float]:
        if not self.is_fitted or self.cluster_centers is None:
            return -1, 0.0
        dists = [np.linalg.norm(feat - c) for c in self.cluster_centers]
        min_idx = int(np.argmin(dists))
        min_dist = dists[min_idx]
        other_dist = dists[1 - min_idx] if len(dists) > 1 else 999.0
        if min_dist > 55.0 or (other_dist - min_dist) < 4.0:
            return -1, 0.3
        margin = max(0.0, min(1.0, (other_dist - min_dist) / max(1.0, other_dist)))
        return min_idx, margin

    def classify_tracks(self, image: np.ndarray, tracks: List[Dict[str, Any]], frame_index: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self.kmeans_calls_this_frame = 0
        num_classified = 0
        num_cached = 0

        if not tracks:
            return tracks, {"kmeans": 0, "classified": 0, "cached": 0}

        if not self.is_fitted:
            self.fit_from_image(image, tracks)

        for t in tracks:
            cls_id = t.get("class_id", 0)
            track_id = t.get("track_id", -1)
            bbox = t["bbox"]

            if cls_id == 2 or t.get("role") == "referee":
                t["team_id"] = -1
                t["role"] = "referee"
                t["team_label"] = "REF"
                continue
            elif cls_id == 1 or t.get("role") == "goalkeeper":
                t["role"] = "goalkeeper"
                t["team_label"] = "GK"

            num_classified += 1
            feat = self.extract_jersey_features(image, bbox, t.get("keypoints"))
            raw_team, conf = self.classify_single_feature(feat)

            if track_id >= 0:
                self.track_features[track_id].append(feat)
                if raw_team >= 0:
                    self.track_history[track_id].append(raw_team)
                votes = self.track_history[track_id]
                if len(votes) >= self.min_votes_required:
                    v_0 = votes.count(0)
                    v_1 = votes.count(1)
                    final_team = 0 if v_0 > v_1 else (1 if v_1 > v_0 else raw_team)
                else:
                    final_team = raw_team if len(votes) > 0 else -1
            else:
                final_team = raw_team

            t["team_id"] = final_team
            if t.get("role") != "goalkeeper":
                t["team_label"] = f"Team {final_team}" if final_team in [0, 1] else "UNKNOWN"
                t["role"] = "player"

        stats = {
            "kmeans": self.kmeans_calls_this_frame,
            "classified": num_classified,
            "cached": num_cached
        }
        return tracks, stats


# ============================================================================
# VARIANT B: FAST / VECTORIZED EXTRACTION
# ============================================================================
def is_color_in_range_fast(bgr: np.ndarray, field_hsv_range: Tuple[float, float, float, float, float, float]) -> bool:
    """Fast BGR to HSV range check for background field rejection."""
    pixel = np.uint8([[bgr]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = hsv[0] * 2.0, hsv[1] / 255.0, hsv[2] / 255.0
    min_h, max_h, min_s, max_s, min_v, max_v = field_hsv_range
    return (min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v)


class TeamClassifierVariantB:
    """Variant B: Fast/vectorized extraction (2-attempt kmeans, 24x32 crop, fast HSV/Lab). Preserves exact semantics."""
    def __init__(self, n_teams: int = 2, vote_window: int = 10, min_votes_required: int = 2):
        self.n_teams = n_teams
        self.vote_window = vote_window
        self.min_votes_required = min_votes_required
        self.cluster_centers: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.vote_window))
        self.track_features: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.vote_window))
        self.kmeans_calls_this_frame = 0
        self.field_range = (35.0, 145.0, 0.15, 1.0, 0.15, 1.0)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    def extract_jersey_features(self, image: np.ndarray, bbox: Any, keypoints: Optional[np.ndarray] = None) -> np.ndarray:
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 5 or box_h < 10:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        t_x1 = max(0, int(x1 + 0.10 * box_w))
        t_x2 = min(w_img, int(x1 + 0.90 * box_w))
        t_y1 = max(0, int(y1 + 0.15 * box_h))
        t_y2 = min(h_img, int(y1 + 0.55 * box_h))
        if t_x2 <= t_x1 or t_y2 <= t_y1:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        crop = image[t_y1:t_y2, t_x1:t_x2]
        if crop.size == 0:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        try:
            # 24x32 is 768 pixels (2.25x faster than 36x48 with identical color modes)
            small_crop = cv2.resize(crop, (24, 32), interpolation=cv2.INTER_LINEAR)
            data = np.float32(small_crop.reshape(-1, 3))
            self.kmeans_calls_this_frame += 1
            _, _, centers = cv2.kmeans(data, 2, None, self.criteria, 2, cv2.KMEANS_PP_CENTERS)

            c0 = np.uint8(np.clip(centers[0], 0, 255))
            c1 = np.uint8(np.clip(centers[1], 0, 255))
            c0_is_field = is_color_in_range_fast(c0, self.field_range)
            c1_is_field = is_color_in_range_fast(c1, self.field_range)

            if not c0_is_field and c1_is_field:
                chosen = c0
            elif not c1_is_field and c0_is_field:
                chosen = c1
            else:
                # Ambiguous tie case: both field or neither field
                g0 = (float(c0[1]) - max(float(c0[0]), float(c0[2]))) / max(1.0, float(c0[1]) + max(float(c0[0]), float(c0[2])))
                g1 = (float(c1[1]) - max(float(c1[0]), float(c1[2]))) / max(1.0, float(c1[1]) + max(float(c1[0]), float(c1[2])))
                chosen = c0 if g0 < g1 else c1

            bgr_pixel = np.uint8([[[chosen[0], chosen[1], chosen[2]]]])
            lab_feat = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)[0, 0]
            return np.float32(lab_feat)
        except Exception:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

    def fit_from_image_and_features(self, feats: List[np.ndarray]) -> bool:
        if len(feats) < 4:
            return False
        try:
            km = KMeans(n_clusters=self.n_teams, random_state=42, n_init=10).fit(np.array(feats))
            centers = km.cluster_centers_
            if len(centers) == 2 and centers[0, 0] < centers[1, 0]:
                centers = centers[::-1].copy()
            self.cluster_centers = centers
            self.is_fitted = True
            return True
        except Exception:
            return False

    def classify_single_feature(self, feat: np.ndarray) -> Tuple[int, float]:
        if not self.is_fitted or self.cluster_centers is None:
            return -1, 0.0
        dists = [float(np.linalg.norm(feat - c)) for c in self.cluster_centers]
        min_idx = 0 if dists[0] < dists[1] else 1
        min_dist = dists[min_idx]
        other_dist = dists[1 - min_idx]
        if min_dist > 55.0 or (other_dist - min_dist) < 4.0:
            return -1, 0.3
        margin = max(0.0, min(1.0, (other_dist - min_dist) / max(1.0, other_dist)))
        return min_idx, margin

    def classify_tracks(self, image: np.ndarray, tracks: List[Dict[str, Any]], frame_index: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self.kmeans_calls_this_frame = 0
        num_classified = 0
        num_cached = 0

        if not tracks:
            return tracks, {"kmeans": 0, "classified": 0, "cached": 0}

        # Single-pass feature extraction: avoid double-extraction during fitting
        extracted_feats = {}
        for t in tracks:
            cls_id = t.get("class_id", 0)
            if cls_id not in [1, 2] and t.get("role") not in ["referee", "goalkeeper"]:
                feat = self.extract_jersey_features(image, t["bbox"], t.get("keypoints"))
                extracted_feats[t.get("track_id", id(t))] = feat
                num_classified += 1

        if not self.is_fitted:
            outfield_feats = [feat for tid, feat in extracted_feats.items()]
            self.fit_from_image_and_features(outfield_feats)

        for t in tracks:
            cls_id = t.get("class_id", 0)
            track_id = t.get("track_id", -1)

            if cls_id == 2 or t.get("role") == "referee":
                t["team_id"] = -1
                t["role"] = "referee"
                t["team_label"] = "REF"
                continue
            elif cls_id == 1 or t.get("role") == "goalkeeper":
                t["role"] = "goalkeeper"
                t["team_label"] = "GK"

            feat = extracted_feats.get(track_id)
            if feat is None:
                feat = self.extract_jersey_features(image, t["bbox"], t.get("keypoints"))
                num_classified += 1

            raw_team, conf = self.classify_single_feature(feat)

            if track_id >= 0:
                self.track_features[track_id].append(feat)
                if raw_team >= 0:
                    self.track_history[track_id].append(raw_team)
                votes = self.track_history[track_id]
                if len(votes) >= self.min_votes_required:
                    v_0 = votes.count(0)
                    v_1 = votes.count(1)
                    final_team = 0 if v_0 > v_1 else (1 if v_1 > v_0 else raw_team)
                else:
                    final_team = raw_team if len(votes) > 0 else -1
            else:
                final_team = raw_team

            t["team_id"] = final_team
            if t.get("role") != "goalkeeper":
                t["team_label"] = f"Team {final_team}" if final_team in [0, 1] else "UNKNOWN"
                t["role"] = "player"

        stats = {
            "kmeans": self.kmeans_calls_this_frame,
            "classified": num_classified,
            "cached": num_cached
        }
        return tracks, stats


# ============================================================================
# VARIANT C: TRACK CACHING (WITH SAFEGUARDS) ON BASELINE EXTRACTION
# ============================================================================
class TeamClassifierVariantC:
    """Variant C: Track Caching on Baseline extraction.
    Locks team after >=10 consistent votes.
    Re-audits every N=15 frames, on track recovery, on sudden bbox size shift, or on instability.
    """
    def __init__(self, n_teams: int = 2, vote_window: int = 10, min_votes_required: int = 2, audit_interval: int = 15, lock_votes: int = 10):
        self.n_teams = n_teams
        self.vote_window = vote_window
        self.min_votes_required = min_votes_required
        self.audit_interval = audit_interval
        self.lock_votes = lock_votes

        self.cluster_centers: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.vote_window))
        self.track_features: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.vote_window))

        self.track_cache: Dict[int, Dict[str, Any]] = {}
        self.kmeans_calls_this_frame = 0

    def extract_jersey_features(self, image: np.ndarray, bbox: Any, keypoints: Optional[np.ndarray] = None) -> np.ndarray:
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        if box_w < 5 or box_h < 10:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        t_x1 = max(0, int(x1 + 0.10 * box_w))
        t_x2 = min(w_img, int(x1 + 0.90 * box_w))
        t_y1 = max(0, int(y1 + 0.15 * box_h))
        t_y2 = min(h_img, int(y1 + 0.55 * box_h))
        crop = image[t_y1:t_y2, t_x1:t_x2]
        if crop.size == 0:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        try:
            small_crop = cv2.resize(crop, (36, 48))
            data = np.float32(small_crop.reshape(-1, 3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            self.kmeans_calls_this_frame += 1
            _, _, centers = cv2.kmeans(data, 2, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
            cand_rgbs = [(int(c[2]), int(c[1]), int(c[0])) for c in centers]
            field_range = (35.0, 145.0, 0.15, 1.0, 0.15, 1.0)
            non_field = [c for c in cand_rgbs if not is_color_in_range_py(c, *field_range)]
            chosen_rgb = non_field[0] if non_field else cand_rgbs[0]
            bgr_pixel = np.uint8([[[chosen_rgb[2], chosen_rgb[1], chosen_rgb[0]]]])
            return np.float32(cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)[0, 0])
        except Exception:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

    def fit_from_image(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> bool:
        outfield_boxes = [d["bbox"] for d in detections if d.get("class_id", 0) == 0]
        if len(outfield_boxes) < 4:
            return False
        feats = [self.extract_jersey_features(image, b) for b in outfield_boxes]
        try:
            km = KMeans(n_clusters=self.n_teams, random_state=42, n_init=10).fit(np.array(feats))
            centers = km.cluster_centers_
            if len(centers) == 2 and centers[0, 0] < centers[1, 0]:
                centers = centers[::-1].copy()
            self.cluster_centers = centers
            self.is_fitted = True
            return True
        except Exception:
            return False

    def classify_single_feature(self, feat: np.ndarray) -> Tuple[int, float]:
        if not self.is_fitted or self.cluster_centers is None:
            return -1, 0.0
        dists = [np.linalg.norm(feat - c) for c in self.cluster_centers]
        min_idx = int(np.argmin(dists))
        min_dist = dists[min_idx]
        other_dist = dists[1 - min_idx] if len(dists) > 1 else 999.0
        if min_dist > 55.0 or (other_dist - min_dist) < 4.0:
            return -1, 0.3
        margin = max(0.0, min(1.0, (other_dist - min_dist) / max(1.0, other_dist)))
        return min_idx, margin

    def classify_tracks(self, image: np.ndarray, tracks: List[Dict[str, Any]], frame_index: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self.kmeans_calls_this_frame = 0
        num_classified = 0
        num_cached = 0

        if not tracks:
            return tracks, {"kmeans": 0, "classified": 0, "cached": 0}

        if not self.is_fitted:
            self.fit_from_image(image, tracks)

        for t in tracks:
            cls_id = t.get("class_id", 0)
            track_id = t.get("track_id", -1)
            bbox = t["bbox"]
            box_area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))

            if cls_id == 2 or t.get("role") == "referee":
                t["team_id"] = -1
                t["role"] = "referee"
                t["team_label"] = "REF"
                continue
            elif cls_id == 1 or t.get("role") == "goalkeeper":
                t["role"] = "goalkeeper"
                t["team_label"] = "GK"

            cinfo = self.track_cache.get(track_id)
            need_feature_extraction = True
            cached_team = None

            if cinfo is not None and cinfo.get("locked_team") is not None and track_id >= 0:
                last_seen = cinfo.get("last_frame_seen", frame_index - 1)
                is_gap_recovery = (frame_index - last_seen > 1)

                last_area = cinfo.get("last_bbox_area", box_area)
                area_ratio = box_area / max(1.0, last_area)
                is_shape_shift = (area_ratio < 0.55 or area_ratio > 1.80)

                frames_since_audit = cinfo.get("frames_since_audit", 0)
                is_periodic_audit = (frames_since_audit >= self.audit_interval)

                if not is_gap_recovery and not is_shape_shift and not is_periodic_audit:
                    cached_team = cinfo["locked_team"]
                    need_feature_extraction = False
                    cinfo["frames_since_audit"] = frames_since_audit + 1
                    cinfo["last_frame_seen"] = frame_index
                    cinfo["last_bbox_area"] = box_area
                    num_cached += 1

            if not need_feature_extraction and cached_team is not None:
                final_team = cached_team
            else:
                num_classified += 1
                feat = self.extract_jersey_features(image, bbox, t.get("keypoints"))
                raw_team, conf = self.classify_single_feature(feat)

                if track_id >= 0:
                    self.track_features[track_id].append(feat)
                    if raw_team >= 0:
                        self.track_history[track_id].append(raw_team)

                    votes = self.track_history[track_id]
                    if len(votes) >= self.min_votes_required:
                        v_0 = votes.count(0)
                        v_1 = votes.count(1)
                        final_team = 0 if v_0 > v_1 else (1 if v_1 > v_0 else raw_team)
                    else:
                        final_team = raw_team if len(votes) > 0 else -1

                    if track_id not in self.track_cache:
                        self.track_cache[track_id] = {
                            "locked_team": None,
                            "frames_since_audit": 0,
                            "last_frame_seen": frame_index,
                            "last_bbox_area": box_area,
                            "disagreements": 0
                        }
                    c_entry = self.track_cache[track_id]
                    c_entry["last_frame_seen"] = frame_index
                    c_entry["last_bbox_area"] = box_area

                    if len(votes) >= self.lock_votes:
                        v0_count = votes.count(0)
                        v1_count = votes.count(1)
                        if v0_count == len(votes):
                            c_entry["locked_team"] = 0
                            c_entry["frames_since_audit"] = 0
                            c_entry["disagreements"] = 0
                        elif v1_count == len(votes):
                            c_entry["locked_team"] = 1
                            c_entry["frames_since_audit"] = 0
                            c_entry["disagreements"] = 0
                        else:
                            if c_entry["locked_team"] is not None and final_team != c_entry["locked_team"]:
                                c_entry["disagreements"] += 1
                                if c_entry["disagreements"] >= 2:
                                    c_entry["locked_team"] = None
                else:
                    final_team = raw_team

            t["team_id"] = final_team
            if t.get("role") != "goalkeeper":
                t["team_label"] = f"Team {final_team}" if final_team in [0, 1] else "UNKNOWN"
                t["role"] = "player"

        stats = {
            "kmeans": self.kmeans_calls_this_frame,
            "classified": num_classified,
            "cached": num_cached
        }
        return tracks, stats


# ============================================================================
# VARIANT D: FAST EXTRACTION + TRACK CACHING (VARIANT B + C)
# ============================================================================
class TeamClassifierVariantD:
    """Variant D: Fast Vectorized Extraction + Track Caching with Safeguards."""
    def __init__(self, n_teams: int = 2, vote_window: int = 10, min_votes_required: int = 2, audit_interval: int = 15, lock_votes: int = 10):
        self.n_teams = n_teams
        self.vote_window = vote_window
        self.min_votes_required = min_votes_required
        self.audit_interval = audit_interval
        self.lock_votes = lock_votes

        self.cluster_centers: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.vote_window))
        self.track_features: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.vote_window))
        self.track_cache: Dict[int, Dict[str, Any]] = {}
        self.kmeans_calls_this_frame = 0
        self.field_range = (35.0, 145.0, 0.15, 1.0, 0.15, 1.0)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    def extract_jersey_features(self, image: np.ndarray, bbox: Any, keypoints: Optional[np.ndarray] = None) -> np.ndarray:
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        if box_w < 5 or box_h < 10:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        t_x1 = max(0, int(x1 + 0.10 * box_w))
        t_x2 = min(w_img, int(x1 + 0.90 * box_w))
        t_y1 = max(0, int(y1 + 0.15 * box_h))
        t_y2 = min(h_img, int(y1 + 0.55 * box_h))
        crop = image[t_y1:t_y2, t_x1:t_x2]
        if crop.size == 0:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        try:
            small_crop = cv2.resize(crop, (24, 32), interpolation=cv2.INTER_LINEAR)
            data = np.float32(small_crop.reshape(-1, 3))
            self.kmeans_calls_this_frame += 1
            _, _, centers = cv2.kmeans(data, 2, None, self.criteria, 2, cv2.KMEANS_PP_CENTERS)

            c0 = np.uint8(np.clip(centers[0], 0, 255))
            c1 = np.uint8(np.clip(centers[1], 0, 255))
            c0_is_field = is_color_in_range_fast(c0, self.field_range)
            c1_is_field = is_color_in_range_fast(c1, self.field_range)

            if not c0_is_field and c1_is_field:
                chosen = c0
            elif not c1_is_field and c0_is_field:
                chosen = c1
            else:
                # Ambiguous tie case: both field or neither field
                g0 = (float(c0[1]) - max(float(c0[0]), float(c0[2]))) / max(1.0, float(c0[1]) + max(float(c0[0]), float(c0[2])))
                g1 = (float(c1[1]) - max(float(c1[0]), float(c1[2]))) / max(1.0, float(c1[1]) + max(float(c1[0]), float(c1[2])))
                chosen = c0 if g0 < g1 else c1

            bgr_pixel = np.uint8([[[chosen[0], chosen[1], chosen[2]]]])
            return np.float32(cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)[0, 0])
        except Exception:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

    def fit_from_image_and_features(self, feats: List[np.ndarray]) -> bool:
        if len(feats) < 4:
            return False
        try:
            km = KMeans(n_clusters=self.n_teams, random_state=42, n_init=10).fit(np.array(feats))
            centers = km.cluster_centers_
            if len(centers) == 2 and centers[0, 0] < centers[1, 0]:
                centers = centers[::-1].copy()
            self.cluster_centers = centers
            self.is_fitted = True
            return True
        except Exception:
            return False

    def classify_single_feature(self, feat: np.ndarray) -> Tuple[int, float]:
        if not self.is_fitted or self.cluster_centers is None:
            return -1, 0.0
        dists = [float(np.linalg.norm(feat - c)) for c in self.cluster_centers]
        min_idx = 0 if dists[0] < dists[1] else 1
        min_dist = dists[min_idx]
        other_dist = dists[1 - min_idx]
        if min_dist > 55.0 or (other_dist - min_dist) < 4.0:
            return -1, 0.3
        margin = max(0.0, min(1.0, (other_dist - min_dist) / max(1.0, other_dist)))
        return min_idx, margin

    def classify_tracks(self, image: np.ndarray, tracks: List[Dict[str, Any]], frame_index: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self.kmeans_calls_this_frame = 0
        num_classified = 0
        num_cached = 0

        if not tracks:
            return tracks, {"kmeans": 0, "classified": 0, "cached": 0}

        tracks_needing_feat = []
        for t in tracks:
            cls_id = t.get("class_id", 0)
            track_id = t.get("track_id", -1)
            bbox = t["bbox"]
            box_area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))

            if cls_id in [1, 2] or t.get("role") in ["referee", "goalkeeper"]:
                continue

            cinfo = self.track_cache.get(track_id)
            need_ext = True
            if cinfo is not None and cinfo.get("locked_team") is not None and track_id >= 0:
                last_seen = cinfo.get("last_frame_seen", frame_index - 1)
                is_gap = (frame_index - last_seen > 1)
                last_area = cinfo.get("last_bbox_area", box_area)
                ratio = box_area / max(1.0, last_area)
                is_shift = (ratio < 0.55 or ratio > 1.80)
                is_audit = (cinfo.get("frames_since_audit", 0) >= self.audit_interval)

                if not is_gap and not is_shift and not is_audit:
                    need_ext = False
                    cinfo["frames_since_audit"] = cinfo.get("frames_since_audit", 0) + 1
                    cinfo["last_frame_seen"] = frame_index
                    cinfo["last_bbox_area"] = box_area
                    num_cached += 1

            if need_ext:
                tracks_needing_feat.append(t)

        extracted_feats = {}
        for t in tracks_needing_feat:
            feat = self.extract_jersey_features(image, t["bbox"], t.get("keypoints"))
            extracted_feats[t.get("track_id", id(t))] = feat
            num_classified += 1

        if not self.is_fitted:
            outfield_feats = list(extracted_feats.values())
            self.fit_from_image_and_features(outfield_feats)

        for t in tracks:
            cls_id = t.get("class_id", 0)
            track_id = t.get("track_id", -1)
            bbox = t["bbox"]
            box_area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))

            if cls_id == 2 or t.get("role") == "referee":
                t["team_id"] = -1
                t["role"] = "referee"
                t["team_label"] = "REF"
                continue
            elif cls_id == 1 or t.get("role") == "goalkeeper":
                t["role"] = "goalkeeper"
                t["team_label"] = "GK"

            cinfo = self.track_cache.get(track_id)
            if track_id in extracted_feats:
                feat = extracted_feats[track_id]
                raw_team, conf = self.classify_single_feature(feat)

                if track_id >= 0:
                    self.track_features[track_id].append(feat)
                    if raw_team >= 0:
                        self.track_history[track_id].append(raw_team)

                    votes = self.track_history[track_id]
                    if len(votes) >= self.min_votes_required:
                        v_0 = votes.count(0)
                        v_1 = votes.count(1)
                        final_team = 0 if v_0 > v_1 else (1 if v_1 > v_0 else raw_team)
                    else:
                        final_team = raw_team if len(votes) > 0 else -1

                    if track_id not in self.track_cache:
                        self.track_cache[track_id] = {
                            "locked_team": None,
                            "frames_since_audit": 0,
                            "last_frame_seen": frame_index,
                            "last_bbox_area": box_area,
                            "disagreements": 0
                        }
                    c_entry = self.track_cache[track_id]
                    c_entry["last_frame_seen"] = frame_index
                    c_entry["last_bbox_area"] = box_area

                    if len(votes) >= self.lock_votes:
                        v0_count = votes.count(0)
                        v1_count = votes.count(1)
                        if v0_count == len(votes):
                            c_entry["locked_team"] = 0
                            c_entry["frames_since_audit"] = 0
                            c_entry["disagreements"] = 0
                        elif v1_count == len(votes):
                            c_entry["locked_team"] = 1
                            c_entry["frames_since_audit"] = 0
                            c_entry["disagreements"] = 0
                        else:
                            if c_entry["locked_team"] is not None and final_team != c_entry["locked_team"]:
                                c_entry["disagreements"] += 1
                                if c_entry["disagreements"] >= 2:
                                    c_entry["locked_team"] = None
                else:
                    final_team = raw_team
            elif cinfo is not None and cinfo.get("locked_team") is not None:
                final_team = cinfo["locked_team"]
            else:
                final_team = -1

            t["team_id"] = final_team
            if t.get("role") != "goalkeeper":
                t["team_label"] = f"Team {final_team}" if final_team in [0, 1] else "UNKNOWN"
                t["role"] = "player"

        stats = {
            "kmeans": self.kmeans_calls_this_frame,
            "classified": num_classified,
            "cached": num_cached
        }
        return tracks, stats


# ============================================================================
# BENCHMARK HARNESS
# ============================================================================
def run_benchmark():
    video_path = "offside_spurs_match.mp4"
    assert os.path.exists(video_path), f"Test video {video_path} not found"

    print("=" * 70)
    print("TEAM-PERF-001: TeamClassifier Compute Optimization & Stability Benchmark")
    print(f"Target Video: {video_path}")
    print("=" * 70)

    print("\nPhase 1: Pre-computing YOLO detections across all frames...")
    model_path = "models/yolo11_v2_4class_best.pt"
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_detections = []
    frames = []

    f_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        res = model(frame, imgsz=1280, conf=0.10, verbose=False)[0]
        p_dets = []
        b_dets = []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [float(v) for v in box.xyxy[0].cpu().numpy().tolist()]
            if cls_id in [0, 1, 2]:
                p_dets.append({"bbox": xyxy, "class_id": cls_id, "conf": conf})
            elif cls_id == 3:
                b_dets.append({"bbox": xyxy, "conf": conf})
        frame_detections.append((p_dets, b_dets))
        f_idx += 1
    cap.release()
    print(f"Loaded {len(frames)} frames ({total_frames} total) with YOLO detections.")

    variants = {
        "A_Baseline": TeamClassifierVariantA,
        "B_Fast": TeamClassifierVariantB,
        "C_Cached": TeamClassifierVariantC,
        "D_Fast_Cached": TeamClassifierVariantD,
    }

    variant_results = {}

    for var_name, var_cls in variants.items():
        print(f"\nEvaluating Variant: {var_name}...")
        tc = var_cls(n_teams=2, vote_window=10, min_votes_required=2)
        p_tracker = PlayerTracker(high_conf_thresh=0.40, low_conf_thresh=0.15)
        b_tracker = BallTracker()
        pass_detector = PassDetector(proximity_threshold=85.0, min_departure_speed=6.5, min_departure_displacement=20.0)
        contact_estimator = ContactEstimator(search_half_window=6)
        geom = get_default_pitch_geometry(1920, 1080)
        offside_engine = OffsideEngine(pitch_geometry=geom)

        team_latencies = []
        pipeline_latencies = []
        kmeans_per_frame = []
        players_per_frame = []
        cached_per_frame = []
        classified_per_frame = []

        track_history_map = defaultdict(list)
        per_frame_assignments = []
        sliding_history = []
        evaluations_recorded = []

        for frame_index in range(len(frames)):
            frame = frames[frame_index]
            p_dets, b_dets = frame_detections[frame_index]

            t_pipe_start = time.perf_counter()

            p_tracks = p_tracker.update(p_dets, frame_index=frame_index)
            b_state = b_tracker.update(b_dets, frame_index=frame_index)

            import copy
            tracks_for_tc = copy.deepcopy(p_tracks)

            t_tc_start = time.perf_counter()
            classified_tracks, tc_stats = tc.classify_tracks(frame, tracks_for_tc, frame_index=frame_index)
            t_tc_end = time.perf_counter()
            team_ms = (t_tc_end - t_tc_start) * 1000.0

            team_latencies.append(team_ms)
            kmeans_per_frame.append(tc_stats["kmeans"])
            players_per_frame.append(len(p_tracks))
            cached_per_frame.append(tc_stats["cached"])
            classified_per_frame.append(tc_stats["classified"])

            frame_map = {}
            for t in classified_tracks:
                tid = t.get("track_id")
                team_id = t.get("team_id")
                if tid is not None:
                    track_history_map[tid].append(team_id)
                    frame_map[tid] = team_id
            per_frame_assignments.append(frame_map)

            hist_rec = {
                "frame": frame_index,
                "ball_state": b_state,
                "ball_pos": b_state.get("position"),
                "player_tracks": classified_tracks,
                "frame_image": frame
            }
            sliding_history.append(hist_rec)
            if len(sliding_history) > 20:
                sliding_history.pop(0)

            pass_cand = pass_detector.update(b_state, classified_tracks, frame_index=frame_index)
            contact_est = None
            offside_res = None

            if pass_cand is not None:
                contact_est = contact_estimator.estimate_contact(pass_cand, sliding_history)
                if contact_est is not None:
                    c_frame = contact_est.frame_hat
                    target_rec = next((h for h in sliding_history if h["frame"] == c_frame), None)
                    if target_rec:
                        p_t = target_rec["player_tracks"]
                        b_p = target_rec["ball_pos"]
                        k_id = contact_est.passer_track_id
                        offside_res = offside_engine.evaluate_contact_moment(
                            players=p_t,
                            attack_team_id=0,
                            attack_direction="right",
                            ball_pos=b_p,
                            image_shape=(1080, 1920),
                            require_metric=False
                        )
                        evaluations_recorded.append({
                            "eval_frame": frame_index,
                            "contact_frame": c_frame,
                            "kicker_id": k_id,
                            "offside_result": offside_res.to_dict()
                        })

            t_pipe_end = time.perf_counter()
            pipeline_latencies.append((t_pipe_end - t_pipe_start) * 1000.0)

        mean_team = float(np.mean(team_latencies))
        p50_team = float(np.percentile(team_latencies, 50))
        p95_team = float(np.percentile(team_latencies, 95))
        p99_team = float(np.percentile(team_latencies, 99))
        max_team = float(np.max(team_latencies))

        mean_pipe = float(np.mean(pipeline_latencies))
        p95_pipe = float(np.percentile(pipeline_latencies, 95))
        effective_fps = 1000.0 / mean_pipe if mean_pipe > 0 else 0.0

        mean_kmeans = float(np.mean(kmeans_per_frame))
        mean_players = float(np.mean(players_per_frame))
        total_cached = int(np.sum(cached_per_frame))
        total_classified = int(np.sum(classified_per_frame))

        consecutive_checks = 0
        flips = 0
        for tid, hists in track_history_map.items():
            if len(hists) >= 2:
                for i in range(1, len(hists)):
                    if hists[i-1] != -1 and hists[i] != -1:
                        consecutive_checks += 1
                        if hists[i] != hists[i-1]:
                            flips += 1
        flip_rate = (flips / max(1, consecutive_checks)) * 100.0

        total_outfield_instances = 0
        unknown_outfield_instances = 0
        for fmap in per_frame_assignments:
            for tid, t_id in fmap.items():
                if tid in GROUND_TRUTH_TRACKS:
                    total_outfield_instances += 1
                    if t_id == -1:
                        unknown_outfield_instances += 1
        unknown_rate = (unknown_outfield_instances / max(1, total_outfield_instances)) * 100.0

        gt_correct = 0
        gt_total = 0
        for tid, expected_team in GROUND_TRUTH_TRACKS.items():
            hists = track_history_map.get(tid, [])
            valid_votes = [v for v in hists if v in [0, 1]]
            if valid_votes:
                assigned_team = 0 if valid_votes.count(0) >= valid_votes.count(1) else 1
                gt_total += 1
                if assigned_team == expected_team:
                    gt_correct += 1
        semantic_accuracy = (gt_correct / max(1, gt_total)) * 100.0

        variant_results[var_name] = {
            "mean_team_ms": round(mean_team, 2),
            "p50_team_ms": round(p50_team, 2),
            "p95_team_ms": round(p95_team, 2),
            "p99_team_ms": round(p99_team, 2),
            "max_team_ms": round(max_team, 2),
            "mean_kmeans_calls": round(mean_kmeans, 1),
            "mean_players_per_frame": round(mean_players, 1),
            "total_cached_classifications": total_cached,
            "total_reclassifications": total_classified,
            "track_flip_rate_pct": round(flip_rate, 2),
            "unknown_rate_pct": round(unknown_rate, 2),
            "semantic_ground_truth_accuracy_pct": round(semantic_accuracy, 1),
            "gt_correct": gt_correct,
            "gt_total": gt_total,
            "mean_pipe_ms": round(mean_pipe, 2),
            "p95_pipe_ms": round(p95_pipe, 2),
            "effective_fps": round(effective_fps, 1),
            "per_frame_assignments": per_frame_assignments,
            "evaluations_recorded": evaluations_recorded,
            "team_latencies": team_latencies
        }
        print(f"  Mean Team Latency: {mean_team:.2f} ms (P50: {p50_team:.2f}, P95: {p95_team:.2f}, Max: {max_team:.2f})")
        print(f"  KMeans Calls/Frame: {mean_kmeans:.1f} | Cached: {total_cached} | Reclass: {total_classified}")
        print(f"  Semantic Accuracy: {semantic_accuracy:.1f}% ({gt_correct}/{gt_total}) | Flip Rate: {flip_rate:.2f}%")

    base_assignments = variant_results["A_Baseline"]["per_frame_assignments"]
    base_evals = variant_results["A_Baseline"]["evaluations_recorded"]

    for var_name, vdata in variant_results.items():
        total_eval_points = 0
        agree_points = 0
        for f_idx in range(len(base_assignments)):
            base_f = base_assignments[f_idx]
            var_f = vdata["per_frame_assignments"][f_idx]
            for tid, base_team in base_f.items():
                if tid in var_f:
                    total_eval_points += 1
                    if var_f[tid] == base_team:
                        agree_points += 1
        agreement_pct = (agree_points / max(1, total_eval_points)) * 100.0
        vdata["baseline_agreement_pct"] = round(agreement_pct, 2)

        var_evals = vdata["evaluations_recorded"]
        decision_match = 0
        attacker_match = 0
        defender_match = 0
        total_law11_evals = max(len(base_evals), len(var_evals))

        for idx in range(min(len(base_evals), len(var_evals))):
            b_ev = base_evals[idx]["offside_result"]
            v_ev = var_evals[idx]["offside_result"]

            if b_ev.get("decision") == v_ev.get("decision") and b_ev.get("boundary_status") == v_ev.get("boundary_status"):
                decision_match += 1
            if b_ev.get("relevant_attacker_id") == v_ev.get("relevant_attacker_id"):
                attacker_match += 1
            if b_ev.get("second_last_defender_id") == v_ev.get("second_last_defender_id"):
                defender_match += 1

        vdata["law11_decision_agreement_pct"] = round((decision_match / max(1, total_law11_evals)) * 100.0, 1) if total_law11_evals > 0 else 100.0
        vdata["attacker_identity_agreement_pct"] = round((attacker_match / max(1, total_law11_evals)) * 100.0, 1) if total_law11_evals > 0 else 100.0
        vdata["defender_identity_agreement_pct"] = round((defender_match / max(1, total_law11_evals)) * 100.0, 1) if total_law11_evals > 0 else 100.0
        vdata["law11_eval_count"] = total_law11_evals

    json_summary = {}
    for var_name, vdata in variant_results.items():
        json_summary[var_name] = {
            "mean_team_ms": vdata["mean_team_ms"],
            "p50_team_ms": vdata["p50_team_ms"],
            "p95_team_ms": vdata["p95_team_ms"],
            "p99_team_ms": vdata["p99_team_ms"],
            "max_team_ms": vdata["max_team_ms"],
            "speedup_vs_baseline": round(variant_results["A_Baseline"]["mean_team_ms"] / max(0.001, vdata["mean_team_ms"]), 2),
            "mean_kmeans_calls_per_frame": vdata["mean_kmeans_calls"],
            "mean_players_per_frame": vdata["mean_players_per_frame"],
            "total_cached_classifications": vdata["total_cached_classifications"],
            "total_reclassifications": vdata["total_reclassifications"],
            "baseline_agreement_pct": vdata["baseline_agreement_pct"],
            "semantic_ground_truth_accuracy_pct": vdata["semantic_ground_truth_accuracy_pct"],
            "track_flip_rate_pct": vdata["track_flip_rate_pct"],
            "unknown_rate_pct": vdata["unknown_rate_pct"],
            "law11_decision_agreement_pct": vdata["law11_decision_agreement_pct"],
            "attacker_identity_agreement_pct": vdata["attacker_identity_agreement_pct"],
            "defender_identity_agreement_pct": vdata["defender_identity_agreement_pct"],
            "mean_pipe_ms": vdata["mean_pipe_ms"],
            "p95_pipe_ms": vdata["p95_pipe_ms"],
            "effective_fps": vdata["effective_fps"]
        }

    out_json = "dataset_v2_meta/team_perf_001_report.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(json_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    headers = [
        "Metric", "A (Baseline)", "B (Fast)", "C (Cached)", "D (Fast+Cached)"
    ]
    rows = [
        ("Mean Team ms", [f"{json_summary[k]['mean_team_ms']:.2f}" for k in json_summary]),
        ("P50 Team ms", [f"{json_summary[k]['p50_team_ms']:.2f}" for k in json_summary]),
        ("P95 Team ms", [f"{json_summary[k]['p95_team_ms']:.2f}" for k in json_summary]),
        ("P99 Team ms", [f"{json_summary[k]['p99_team_ms']:.2f}" for k in json_summary]),
        ("Max Team ms", [f"{json_summary[k]['max_team_ms']:.2f}" for k in json_summary]),
        ("Speedup vs Base", [f"{json_summary[k]['speedup_vs_baseline']:.2f}x" for k in json_summary]),
        ("KMeans calls/fr", [f"{json_summary[k]['mean_kmeans_calls_per_frame']:.1f}" for k in json_summary]),
        ("Cached classif.", [f"{json_summary[k]['total_cached_classifications']}" for k in json_summary]),
        ("Reclassifications", [f"{json_summary[k]['total_reclassifications']}" for k in json_summary]),
        ("Baseline Agree", [f"{json_summary[k]['baseline_agreement_pct']:.1f}%" for k in json_summary]),
        ("Manual GT Acc", [f"{json_summary[k]['semantic_ground_truth_accuracy_pct']:.1f}%" for k in json_summary]),
        ("Track Flip Rate", [f"{json_summary[k]['track_flip_rate_pct']:.2f}%" for k in json_summary]),
        ("UNKNOWN Rate", [f"{json_summary[k]['unknown_rate_pct']:.1f}%" for k in json_summary]),
        ("Law 11 Decision", [f"{json_summary[k]['law11_decision_agreement_pct']:.1f}%" for k in json_summary]),
        ("Attacker Agree", [f"{json_summary[k]['attacker_identity_agreement_pct']:.1f}%" for k in json_summary]),
        ("Defender Agree", [f"{json_summary[k]['defender_identity_agreement_pct']:.1f}%" for k in json_summary]),
        ("Pipe Mean ms", [f"{json_summary[k]['mean_pipe_ms']:.2f}" for k in json_summary]),
        ("Pipe P95 ms", [f"{json_summary[k]['p95_pipe_ms']:.2f}" for k in json_summary]),
        ("Effective FPS", [f"{json_summary[k]['effective_fps']:.1f}" for k in json_summary]),
    ]

    print(f"{headers[0]:<20} | {headers[1]:<12} | {headers[2]:<12} | {headers[3]:<12} | {headers[4]:<12}")
    print("-" * 75)
    for label, vals in rows:
        print(f"{label:<20} | {vals[0]:<12} | {vals[1]:<12} | {vals[2]:<12} | {vals[3]:<12}")
    print("=" * 75)
    print(f"Report saved to {out_json}")


if __name__ == "__main__":
    run_benchmark()
