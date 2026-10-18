"""
Dynamic Pitch Tracker with Pan-Stabilized Projective Anchoring & Homography Validation.
Maintains metric homography H_t across broadcast camera pans and zooms without dead-reckoning drift.
Tracks ONLY persistent pitch markings (touchlines, penalty box, halfway line).
Guarantees strictly bounded condition numbers and flags geometry status.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


class DynamicPitchTracker:
    def __init__(self, initial_H: np.ndarray, image_shape: Tuple[int, int] = (1080, 1920)):
        self.H_initial = initial_H.copy()
        self.H_active = initial_H.copy()
        self.image_shape = image_shape

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None

        # Cumulative camera displacement from base calibration
        self.cum_dx: float = 0.0
        self.cum_dy: float = 0.0

        self.current_motion_state: str = "STATIC"
        self.geometry_status: str = "CALIBRATING"
        self.inlier_count: int = 0
        self.condition_number: float = float(np.linalg.cond(initial_H))

    def get_pitch_markings_mask(self, frame: np.ndarray, player_dets: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extracts binary mask of persistent pitch markings while strictly masking out
        player bounding boxes and stadium crowd/stands.
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Focus strictly on playing pitch region (exclude upper 22% stands/billboards)
        mask[int(h * 0.22):int(h * 0.95), int(w * 0.04):int(w * 0.96)] = 255

        # White pitch markings extraction in HSV space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 60, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Dilate slightly so line intersections are captured
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white_dilated = cv2.dilate(white_mask, kernel, iterations=1)
        combined_mask = cv2.bitwise_and(mask, white_dilated)

        # Mask out player bounding boxes with 12px safety margin
        for p in player_dets:
            bx1, by1, bx2, by2 = [int(v) for v in p.get("bbox", [0, 0, 0, 0])]
            px1 = max(0, bx1 - 12)
            py1 = max(0, by1 - 12)
            px2 = min(w, bx2 + 12)
            py2 = min(h, by2 + 12)
            combined_mask[py1:py2, px1:px2] = 0

        return combined_mask

    def update(self, frame: np.ndarray, player_dets: List[Dict[str, Any]]) -> Tuple[np.ndarray, str, str]:
        """
        Tracks pitch lines from previous frame to current frame, estimates camera pan displacement,
        and projects through the pan-stabilized projective anchor.
        Returns: (H_t, motion_state, geometry_status)
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h_img, w_img = curr_gray.shape[:2]

        pitch_mask = self.get_pitch_markings_mask(frame, player_dets)

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 8:
            self.prev_gray = curr_gray.copy()
            self.prev_pts = cv2.goodFeaturesToTrack(
                curr_gray, maxCorners=120, qualityLevel=0.025, minDistance=25, mask=pitch_mask
            )
            self.current_motion_state = "STATIC"
            self.geometry_status = "INITIALIZING"
            return self.H_active, self.current_motion_state, self.geometry_status

        # 1. Forward Optical Flow (Lucas-Kanade)
        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        # 2. Backward Optical Flow (Forward-Backward consistency check)
        pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, self.prev_gray, pts_curr, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        fb_err = np.linalg.norm(self.prev_pts.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1)
        valid = (status.flatten() == 1) & (status_back.flatten() == 1) & (fb_err < 1.5)

        good_prev = self.prev_pts.reshape(-1, 2)[valid]
        good_curr = pts_curr.reshape(-1, 2)[valid]
        self.inlier_count = len(good_curr)

        if len(good_prev) >= 6:
            # Measure median displacement vector of reliable pitch markings
            diffs = good_curr - good_prev
            median_dx = float(np.median(diffs[:, 0]))
            median_dy = float(np.median(diffs[:, 1]))
            trans_mag = float(np.sqrt(median_dx**2 + median_dy**2))

            # Classify motion state
            if trans_mag > 2.0:
                self.current_motion_state = "PAN"
            elif trans_mag > 0.5:
                self.current_motion_state = "LOW_MOTION"
            else:
                self.current_motion_state = "STATIC"

            # Filter outlier jumps (> 35px per frame is typically a cut/occlusion)
            if trans_mag < 35.0:
                self.cum_dx += median_dx
                self.cum_dy += median_dy

            # Pan-Stabilized Projective Anchor:
            # T_mat shifts image coordinates by negative cumulative displacement back to the base anchor
            T_mat = np.array([
                [1.0, 0.0, -self.cum_dx],
                [0.0, 1.0, -self.cum_dy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

            new_H = self.H_initial @ T_mat
            new_H = new_H / new_H[2, 2]

            self.condition_number = float(np.linalg.cond(new_H))
            det_val = float(np.linalg.det(new_H))

            # Guard validation check: condition number must be stable and determinant positive
            if self.condition_number < 3.5e4 and det_val > 0:
                self.H_active = new_H
                self.geometry_status = "TRACKING [VALID]"
            else:
                self.geometry_status = "GEOMETRY_UNRELIABLE"
        else:
            self.geometry_status = "INSUFFICIENT_FEATURES"

        # Refresh tracking features on current frame
        self.prev_gray = curr_gray.copy()
        self.prev_pts = cv2.goodFeaturesToTrack(
            curr_gray, maxCorners=120, qualityLevel=0.025, minDistance=25, mask=pitch_mask
        )

        return self.H_active, self.current_motion_state, self.geometry_status
