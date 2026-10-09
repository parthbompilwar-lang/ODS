"""
Team Classifier using Unsupervised Color Clustering in CIE-Lab Color Space.
Separates players into Team 0, Team 1, Goalkeepers, Referees, or UNKNOWN based on jersey ROI.
Includes track-level temporal majority voting and guarded identity caching for high-speed,
temporally stable execution (TEAM-PERF-001 Variant D architecture).
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict, deque
from sklearn.cluster import KMeans


def is_color_in_range_fast(bgr: np.ndarray, field_hsv_range: Tuple[float, float, float, float, float, float]) -> bool:
    """
    Fast BGR to HSV range check for background field rejection.
    Operates on a 3-element uint8 BGR array.
    """
    pixel = np.uint8([[bgr]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = hsv[0] * 2.0, hsv[1] / 255.0, hsv[2] / 255.0
    min_h, max_h, min_s, max_s, min_v, max_v = field_hsv_range
    return (min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v)


class TeamClassifier:
    """
    High-performance, temporally stable Team Classifier.
    - Uses 2-cluster KMeans in CIE-Lab with background pitch grass rejection.
    - Caches team identity for stable tracks (>=10 consistent votes) with periodic N=15 audits,
      gap recovery invalidation, and bbox morphology shift detection (TEAM-PERF-001).
    - Enforces canonical luminance ordering (Cluster 0 = lighter kit, Cluster 1 = darker kit).
    """

    def __init__(
        self,
        n_teams: int = 2,
        vote_window: int = 12,
        min_votes_required: int = 2,
        groups_color_filters: Optional[Dict[str, Tuple]] = None,
        field_color_range: Tuple[float, float, float, float, float, float] = (35.0, 145.0, 0.15, 1.0, 0.15, 1.0),
        lock_votes: int = 10,
        audit_interval: int = 15,
        crop_size: Tuple[int, int] = (24, 32),
        kmeans_attempts: int = 2
    ):
        self.n_teams = n_teams
        self.vote_window = vote_window
        self.min_votes_required = min_votes_required
        self.groups_color_filters = groups_color_filters
        self.field_color_range = field_color_range
        self.lock_votes = lock_votes
        self.audit_interval = audit_interval
        self.crop_size = crop_size
        self.kmeans_attempts = kmeans_attempts

        # Cluster centroids in CIE-Lab space
        self.cluster_centers: Optional[np.ndarray] = None
        self.is_fitted: bool = False

        # Visual BGR colors for overlay rendering
        self.team_colors: Dict[int, Tuple[int, int, int]] = {
            0: (235, 120, 30),   # Team 0 (Cyan/Blue default)
            1: (40, 60, 220),    # Team 1 (Orange/Red default)
            -1: (180, 180, 180)  # UNKNOWN (Neutral Gray)
        }

        # Track-level history for temporal majority voting: track_id -> deque of recent votes
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.vote_window))
        self.track_features: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.vote_window))

        # Track-level identity cache: track_id -> cache metadata
        # {"locked_team": int, "frames_since_audit": int, "last_frame_seen": int, "last_bbox_area": float, "disagreements": int}
        self.track_cache: Dict[int, Dict[str, Any]] = {}

        # Telemetry stats for the most recent classify_tracks invocation
        self.last_frame_stats: Dict[str, Any] = {"kmeans": 0, "classified": 0, "cached": 0}

        self.kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    def reset(self):
        """Resets online tracking buffers, cache, and learned centroids."""
        self.cluster_centers = None
        self.is_fitted = False
        self.track_history.clear()
        self.track_features.clear()
        self.track_cache.clear()
        self.last_frame_stats = {"kmeans": 0, "classified": 0, "cached": 0}

    def extract_jersey_features(
        self,
        image: np.ndarray,
        bbox: Any,
        keypoints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extracts dominant jersey color descriptor in CIE-Lab space using 2-cluster
        background grass rejection.
        """
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 5 or box_h < 10:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        # Isolate body / torso crop
        crop = None
        if keypoints is not None and len(keypoints) >= 13:
            ls, rs = keypoints[5][:2], keypoints[6][:2]
            lh, rh = keypoints[11][:2], keypoints[12][:2]
            if all(pt[0] > 0 and pt[1] > 0 for pt in [ls, rs, lh, rh]):
                min_x = max(0, int(min(ls[0], rs[0], lh[0], rh[0])))
                max_x = min(w_img, int(max(ls[0], rs[0], lh[0], rh[0])))
                min_y = max(0, int(min(ls[1], rs[1])))
                max_y = min(h_img, int(max(lh[1], rh[1])))
                if max_x - min_x > 4 and max_y - min_y > 4:
                    crop = image[min_y:max_y, min_x:max_x]

        if crop is None:
            t_x1 = max(0, int(x1 + 0.10 * box_w))
            t_x2 = min(w_img, int(x1 + 0.90 * box_w))
            t_y1 = max(0, int(y1 + 0.15 * box_h))
            t_y2 = min(h_img, int(y1 + 0.55 * box_h))
            if t_x2 > t_x1 and t_y2 > t_y1:
                crop = image[t_y1:t_y2, t_x1:t_x2]

        if crop is None or crop.size == 0:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        # 2-cluster KMeans to isolate field grass from jersey pixels
        try:
            small_crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)
            data = np.float32(small_crop.reshape(-1, 3))
            self.last_frame_stats["kmeans"] += 1
            _, _, centers = cv2.kmeans(
                data,
                2,
                None,
                self.kmeans_criteria,
                self.kmeans_attempts,
                cv2.KMEANS_PP_CENTERS
            )

            c0 = np.uint8(np.clip(centers[0], 0, 255))
            c1 = np.uint8(np.clip(centers[1], 0, 255))
            c0_is_field = is_color_in_range_fast(c0, self.field_color_range)
            c1_is_field = is_color_in_range_fast(c1, self.field_color_range)

            if not c0_is_field and c1_is_field:
                chosen = c0
            elif not c1_is_field and c0_is_field:
                chosen = c1
            else:
                # Ambiguous tie case (both field or neither field):
                # Discard pitch grass by picking cluster with lowest greenness
                g0 = (float(c0[1]) - max(float(c0[0]), float(c0[2]))) / max(1.0, float(c0[1]) + max(float(c0[0]), float(c0[2])))
                g1 = (float(c1[1]) - max(float(c1[0]), float(c1[2]))) / max(1.0, float(c1[1]) + max(float(c1[0]), float(c1[2])))
                chosen = c0 if g0 < g1 else c1

            bgr_pixel = np.uint8([[[chosen[0], chosen[1], chosen[2]]]])
            lab_feat = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)[0, 0]
            return np.float32(lab_feat)
        except Exception:
            # Fallback: simple HSV grass mask & Lab median
            try:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                lower_green = np.array([30, 40, 40])
                upper_green = np.array([85, 255, 255])
                grass_mask = cv2.inRange(hsv, lower_green, upper_green)
                jersey_mask = cv2.bitwise_not(grass_mask)
                lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                pixels = lab[jersey_mask > 0]
                if len(pixels) < 10:
                    pixels = lab.reshape(-1, 3)
                return np.float32(np.median(pixels, axis=0))
            except Exception:
                return np.array([128.0, 128.0, 128.0], dtype=np.float32)

    def fit_from_features(self, feats: List[np.ndarray]) -> bool:
        """
        Fits initial KMeans clusters on pre-extracted outfield player features.
        Enforces canonical luminance ordering: Cluster 0 = lighter kit, Cluster 1 = darker kit.
        """
        if len(feats) < 4:
            return False

        try:
            kmeans = KMeans(n_clusters=self.n_teams, random_state=42, n_init=10).fit(np.array(feats))
            centers = kmeans.cluster_centers_

            # Canonical luminance ordering: ensure Cluster 0 is always the lighter kit (higher L*)
            if len(centers) == 2 and centers[0, 0] < centers[1, 0]:
                centers = centers[::-1].copy()

            self.cluster_centers = centers
            self.is_fitted = True

            # Calculate representative BGR colors for overlay rendering
            for t_idx in range(self.n_teams):
                lab_center = np.uint8([[self.cluster_centers[t_idx]]])
                bgr_center = cv2.cvtColor(lab_center, cv2.COLOR_LAB2BGR)[0][0]
                self.team_colors[t_idx] = (int(bgr_center[0]), int(bgr_center[1]), int(bgr_center[2]))

            return True
        except Exception:
            return False

    def fit_from_image(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> bool:
        """
        Fits initial KMeans clusters on outfield player detections from a frame.
        """
        outfield_boxes = []
        for det in detections:
            cls_id = det.get("class_id", 0)
            if cls_id == 0:  # Outfield player
                outfield_boxes.append(det["bbox"])

        if len(outfield_boxes) < 4:
            return False

        feats = [self.extract_jersey_features(image, b) for b in outfield_boxes]
        return self.fit_from_features(feats)

    def classify_single_feature(self, feat: np.ndarray) -> Tuple[int, float]:
        """
        Assigns feature to nearest cluster center in CIE-Lab space.
        Returns: (team_id, confidence)
        """
        if not self.is_fitted or self.cluster_centers is None:
            return -1, 0.0

        dists = [float(np.linalg.norm(feat - c)) for c in self.cluster_centers]
        min_idx = 0 if dists[0] < dists[1] else 1
        min_dist = dists[min_idx]
        other_dist = dists[1 - min_idx] if len(dists) > 1 else 999.0

        # If too ambiguous (both distances very close) or too far from both clusters
        if min_dist > 55.0 or (other_dist - min_dist) < 4.0:
            return -1, 0.3

        margin = max(0.0, min(1.0, (other_dist - min_dist) / max(1.0, other_dist)))
        return min_idx, margin

    def classify_tracks(
        self,
        image: np.ndarray,
        tracks: List[Dict[str, Any]],
        frame_index: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Assigns team IDs to tracked players using temporal majority voting and guarded caching.
        Modifies tracks in place and returns them.
        """
        self.last_frame_stats = {"kmeans": 0, "classified": 0, "cached": 0}
        if not tracks:
            return tracks

        # 1. Determine which tracks need feature extraction
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
                    self.last_frame_stats["cached"] += 1

            if need_ext:
                tracks_needing_feat.append(t)

        # 2. Extract features only for tracks needing it (single pass)
        extracted_feats = {}
        for t in tracks_needing_feat:
            feat = self.extract_jersey_features(image, t["bbox"], t.get("keypoints"))
            extracted_feats[t.get("track_id", id(t))] = feat
            self.last_frame_stats["classified"] += 1

        # 3. If not fitted yet, fit on newly extracted outfield features
        if not self.is_fitted:
            outfield_feats = list(extracted_feats.values())
            self.fit_from_features(outfield_feats)

        # 4. Classify, vote, and update cache
        for t in tracks:
            cls_id = t.get("class_id", 0)
            track_id = t.get("track_id", -1)
            bbox = t["bbox"]
            box_area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))

            # Role differentiation
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

                    # Majority voting over temporal window
                    votes = self.track_history[track_id]
                    if len(votes) >= self.min_votes_required:
                        v_0 = votes.count(0)
                        v_1 = votes.count(1)
                        if v_0 > v_1:
                            final_team = 0
                        elif v_1 > v_0:
                            final_team = 1
                        else:
                            final_team = raw_team
                    else:
                        final_team = raw_team if len(votes) > 0 else -1

                    # Cache state management
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

                    # Lock condition: >= lock_votes consistent votes
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
                            # Mixed votes: check for disagreement with locked team
                            if c_entry["locked_team"] is not None and final_team != c_entry["locked_team"]:
                                c_entry["disagreements"] += 1
                                if c_entry["disagreements"] >= 2:
                                    c_entry["locked_team"] = None  # Unlock
                else:
                    final_team = raw_team
            elif cinfo is not None and cinfo.get("locked_team") is not None:
                final_team = cinfo["locked_team"]
            else:
                final_team = -1

            t["team_id"] = final_team
            if t.get("role") != "goalkeeper":
                if final_team == 0:
                    t["team_label"] = "Team 0"
                    t["role"] = "player"
                elif final_team == 1:
                    t["team_label"] = "Team 1"
                    t["role"] = "player"
                else:
                    t["team_label"] = "UNKNOWN"
                    t["role"] = "player"

        return tracks

    def classify_players(self, image: np.ndarray, players: List[Any], goal_direction: str = 'right') -> List[Any]:
        """
        Legacy batch classification interface for object instances.
        Maintains backward compatibility with src/pipeline.py.
        """
        if not players:
            return players

        features = []
        for p in players:
            kpts = getattr(p, 'keypoints', None)
            bbox = getattr(p, 'bbox', None)
            feat = self.extract_jersey_features(image, bbox, kpts)
            features.append(feat)

        features = np.array(features)

        if len(players) < 3:
            for i, p in enumerate(players):
                p.team_id = i % 2
            return players

        kmeans = KMeans(n_clusters=self.n_teams, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_
        if len(centers) == 2 and centers[0, 0] < centers[1, 0]:
            centers = centers[::-1].copy()
            labels = 1 - labels
        self.cluster_centers = centers
        self.is_fitted = True

        for i, p in enumerate(players):
            p.team_id = int(labels[i])

        return players
