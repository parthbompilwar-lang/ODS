"""
TRACK-001: Player and Goalkeeper Multi-Object Tracker.
Maintains identity continuity across video frames using a two-stage association strategy
(ByteTrack-style IoU matching with a constant-velocity Kalman motion model).
Does NOT perform team classification at this stage, keeping tracking variables isolated.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    """
    Kalman filter for tracking bounding boxes in image space [cx, cy, s, r],
    where s = area (w*h) and r = aspect ratio (w/h).
    """
    count = 0

    def __init__(self, bbox: List[float], class_id: int, confidence: float, frame_index: int):
        # State: [cx, cy, s, r, v_cx, v_cy, v_s]
        self.dim_z = 4
        self.dim_x = 7

        self.x = np.zeros((7, 1))
        self._bbox_to_z(bbox)

        # State transition matrix F
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        # Covariance matrices
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.class_id = class_id
        self.confidence = confidence
        self.frame_index = frame_index
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history = [bbox]

    def _bbox_to_z(self, bbox: List[float]):
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        self.x[:4, 0] = [cx, cy, s, r]

    def predict(self) -> List[float]:
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1
        pred_box = self.get_state()
        self.history.append(pred_box)
        return pred_box

    def update(self, bbox: List[float], class_id: int, confidence: float, frame_index: int):
        self.time_since_update = 0
        self.hits += 1
        self.confidence = confidence
        self.class_id = class_id
        self.frame_index = frame_index

        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        z = np.array([[bbox[0] + w / 2.0], [bbox[1] + h / 2.0], [w * h], [w / float(h)]])

        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        self.history[-1] = self.get_state()

    def get_state(self) -> List[float]:
        cx = self.x[0, 0]
        cy = self.x[1, 0]
        s = max(1.0, self.x[2, 0])
        r = max(0.01, self.x[3, 0])
        w = np.sqrt(s * r)
        h = s / w
        return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def calculate_hybrid_cost_matrix(
    boxes_a: List[List[float]],
    boxes_b: List[List[float]],
    iou_weight: float = 0.50,
    dist_weight: float = 0.50
) -> np.ndarray:
    """
    Computes hybrid association cost combining (1 - IoU) and normalized Euclidean distance.
    Robust against broadcast camera pan, tilt, and rapid player running.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)))

    cost_matrix = np.zeros((len(boxes_a), len(boxes_b)), dtype=float)

    for i, b1 in enumerate(boxes_a):
        c1_x = (b1[0] + b1[2]) / 2.0
        c1_y = (b1[1] + b1[3]) / 2.0
        h1 = max(1.0, b1[3] - b1[1])
        w1 = max(1.0, b1[2] - b1[0])

        for j, b2 in enumerate(boxes_b):
            c2_x = (b2[0] + b2[2]) / 2.0
            c2_y = (b2[1] + b2[3]) / 2.0
            h2 = max(1.0, b2[3] - b2[1])
            w2 = max(1.0, b2[2] - b2[0])

            # 1. IoU calculation
            inter_x1 = max(b1[0], b2[0])
            inter_y1 = max(b1[1], b2[1])
            inter_x2 = min(b1[2], b2[2])
            inter_y2 = min(b1[3], b2[3])

            inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
            union_area = (w1 * h1) + (w2 * h2) - inter_area
            iou = inter_area / union_area if union_area > 0 else 0.0

            # 2. Normalized Center Distance
            dist = np.hypot(c1_x - c2_x, c1_y - c2_y)
            norm_dist = min(1.0, dist / max(h1, h2, 1.0))

            cost_matrix[i, j] = iou_weight * (1.0 - iou) + dist_weight * norm_dist

    return cost_matrix


class PlayerTracker:
    """
    Multi-Player and Goalkeeper Tracker with camera-motion robust hybrid association.
    Maintains track identity for outfield players (class 0) and goalkeepers (class 1).
    """

    def __init__(
        self,
        high_conf_thresh: float = 0.45,
        low_conf_thresh: float = 0.15,
        cost_thresh: float = 0.65,
        max_age: int = 15,
        min_hits: int = 2
    ):
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.cost_thresh = cost_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, detections: List[Dict[str, Any]], frame_index: int) -> List[Dict[str, Any]]:
        """
        Updates tracks with detections from frame_index.
        Each detection: {'bbox': [x1, y1, x2, y2], 'class_id': int, 'conf': float}
        """
        self.frame_count = frame_index

        # Predict positions for all existing trackers
        for trk in self.trackers:
            trk.predict()

        # Partition detections into high and low confidence
        high_dets = []
        low_dets = []
        for det in detections:
            cls_id = det.get("class_id", 0)
            if cls_id in [0, 1]:  # Player or Goalkeeper
                conf = det.get("conf", 0.0)
                if conf >= self.high_conf_thresh:
                    high_dets.append(det)
                elif conf >= self.low_conf_thresh:
                    low_dets.append(det)

        # ----------------------------------------------------
        # Stage 1: Associate high-confidence detections via Hybrid Cost
        # ----------------------------------------------------
        trks_boxes = [t.get_state() for t in self.trackers]
        high_boxes = [d["bbox"] for d in high_dets]

        matched_trks = []
        unmatched_trks = list(range(len(self.trackers)))
        unmatched_high_dets = list(range(len(high_dets)))

        if len(trks_boxes) > 0 and len(high_boxes) > 0:
            cost_matrix = calculate_hybrid_cost_matrix(trks_boxes, high_boxes)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.cost_thresh:
                    self.trackers[r].update(
                        high_dets[c]["bbox"],
                        high_dets[c]["class_id"],
                        high_dets[c]["conf"],
                        frame_index
                    )
                    matched_trks.append(r)
                    if r in unmatched_trks: unmatched_trks.remove(r)
                    if c in unmatched_high_dets: unmatched_high_dets.remove(c)

        # ----------------------------------------------------
        # Stage 2: Associate remaining tracks with low-confidence detections
        # ----------------------------------------------------
        low_boxes = [d["bbox"] for d in low_dets]
        if len(unmatched_trks) > 0 and len(low_boxes) > 0:
            rem_trk_boxes = [self.trackers[i].get_state() for i in unmatched_trks]
            cost_matrix_low = calculate_hybrid_cost_matrix(rem_trk_boxes, low_boxes)
            row_ind, col_ind = linear_sum_assignment(cost_matrix_low)

            for r_idx, c in zip(row_ind, col_ind):
                real_trk_idx = unmatched_trks[r_idx]
                if cost_matrix_low[r_idx, c] <= 0.70:
                    self.trackers[real_trk_idx].update(
                        low_dets[c]["bbox"],
                        low_dets[c]["class_id"],
                        low_dets[c]["conf"],
                        frame_index
                    )
                    matched_trks.append(real_trk_idx)

        # Update remaining unmatched trackers
        final_unmatched = [t for i, t in enumerate(self.trackers) if i not in matched_trks]

        # ----------------------------------------------------
        # Initialize new trackers from unmatched high-confidence detections
        # ----------------------------------------------------
        for c in unmatched_high_dets:
            new_trk = KalmanBoxTracker(
                high_dets[c]["bbox"],
                high_dets[c]["class_id"],
                high_dets[c]["conf"],
                frame_index
            )
            self.trackers.append(new_trk)

        # Remove dead tracks
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # Active output tracks
        active_tracks = []
        for t in self.trackers:
            if (t.hits >= self.min_hits or self.frame_count <= self.min_hits) and t.time_since_update <= 1:
                active_tracks.append({
                    "track_id": t.id,
                    "class_id": t.class_id,
                    "bbox": [round(v, 2) for v in t.get_state()],
                    "confidence": round(t.confidence, 3),
                    "frame_index": frame_index,
                    "hits": t.hits,
                    "time_since_update": t.time_since_update
                })

        return active_tracks
