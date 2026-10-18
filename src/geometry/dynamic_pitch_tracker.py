"""
Dynamic Pitch Tracker with Guarded Optical-Flow Homography.
Maintains metric homography H_t across broadcast camera pans and zooms.
Tracks ONLY persistent pitch markings (touchlines, penalty box, halfway line).
Strictly masks out players, ball, referee, crowd, and billboards.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


class DynamicPitchTracker:
    def __init__(self, initial_H: np.ndarray, image_shape: Tuple[int, int] = (1080, 1920)):
        self.H_active = initial_H.copy()
        self.H_initial = initial_H.copy()
        self.image_shape = image_shape
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.current_motion_state: str = "STATIC"

    def get_pitch_markings_mask(self, frame: np.ndarray, player_dets: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extracts binary mask of persistent pitch markings while strictly masking out
        player bounding boxes and stadium crowd/stands.
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Focus on pitch region (exclude upper 18% stands/billboards)
        mask[int(h * 0.18):h, 0:w] = 255

        # White pitch markings extraction in HSV space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 155])
        upper_white = np.array([180, 55, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Dilate slightly so line intersections are captured
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white_dilated = cv2.dilate(white_mask, kernel, iterations=1)
        combined_mask = cv2.bitwise_and(mask, white_dilated)

        # Mask out player bounding boxes with 10px safety margin
        for p in player_dets:
            bx1, by1, bx2, by2 = [int(v) for v in p.get("bbox", [0, 0, 0, 0])]
            px1 = max(0, bx1 - 10)
            py1 = max(0, by1 - 10)
            px2 = min(w, bx2 + 10)
            py2 = min(h, by2 + 10)
            combined_mask[py1:py2, px1:px2] = 0

        return combined_mask

    def update(self, frame: np.ndarray, player_dets: List[Dict[str, Any]]) -> Tuple[np.ndarray, str]:
        """
        Tracks pitch lines from previous frame to current frame, estimates camera motion,
        and safely updates projective homography H_t.
        Returns: (H_t, motion_state)
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h_img, w_img = curr_gray.shape[:2]

        pitch_mask = self.get_pitch_markings_mask(frame, player_dets)

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 8:
            self.prev_gray = curr_gray.copy()
            self.prev_pts = cv2.goodFeaturesToTrack(curr_gray, maxCorners=100, qualityLevel=0.025, minDistance=25, mask=pitch_mask)
            self.current_motion_state = "STATIC"
            return self.H_active, self.current_motion_state

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

        if len(good_prev) >= 8:
            # Check translation magnitude to classify motion
            diffs = good_curr - good_prev
            median_dx = float(np.median(diffs[:, 0]))
            median_dy = float(np.median(diffs[:, 1]))
            trans_mag = float(np.sqrt(median_dx**2 + median_dy**2))

            if trans_mag > 2.5:
                self.current_motion_state = "PAN"
            elif trans_mag > 0.8:
                self.current_motion_state = "LOW_MOTION"
            else:
                self.current_motion_state = "STATIC"

            # Estimate incremental affine / projective transformation between frames
            M, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=2.5)
            if M is not None and inliers is not None and np.sum(inliers) >= 6:
                # Convert 2x3 affine matrix to 3x3 projective transformation Delta H
                delta_H = np.vstack([M, [0.0, 0.0, 1.0]])
                det_d = np.linalg.det(delta_H)
                if abs(det_d) > 1e-4:
                    try:
                        delta_H_inv = np.linalg.inv(delta_H)
                        new_H = self.H_active @ delta_H_inv
                        new_H = new_H / new_H[2, 2]

                        # Guard: Check condition number and determinant stability
                        cond = float(np.linalg.cond(new_H))
                        det_new = float(np.linalg.det(new_H))

                        if cond < 1e5 and det_new > 0:
                            self.H_active = new_H
                    except Exception:
                        pass

        # Refresh tracking features on the current frame
        self.prev_gray = curr_gray.copy()
        self.prev_pts = cv2.goodFeaturesToTrack(curr_gray, maxCorners=100, qualityLevel=0.025, minDistance=25, mask=pitch_mask)

        return self.H_active, self.current_motion_state
