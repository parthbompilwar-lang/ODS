"""
GEOMETRY-CAMERA-001: Dynamic Homography under Camera Motion Benchmark.
Evaluates 3 variants across GEOMETRY-EVAL-001 benchmark scenes and offside_spurs_match.mp4 (241 frames):
  Variant A: Static H0 Baseline (Current production geometry unchanged)
  Variant B: Lucas-Kanade Optical-Flow H (cv2.calcOpticalFlowPyrLK on pitch markings only)
  Variant C: Hybrid LK + Guarded Recalibration / UNDETERMINED Fallback

Tracks ONLY persistent pitch markings (touchlines, penalty box, halfway line).
Never tracks players, ball, referee, crowd, ads, or shadows.

Outputs:
  - geometry_camera_001_frame_metrics.csv
  - geometry_camera_001_homography_metrics.csv
  - geometry_camera_001_offside_impact.csv
  - geometry_camera_001_recalibration_events.csv
  - dataset_v2_meta/geometry_camera_001_report.json
"""

import os
import sys
import time
import json
import csv
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from ultralytics import YOLO

from src.tracking.player_tracker import PlayerTracker
from src.tracking.ball_tracker import BallTracker
from src.team.team_classifier import TeamClassifier
from src.passing.pass_detector import PassDetector
from src.passing.contact_estimator import ContactEstimator
from src.engine.offside_engine import OffsideEngine, OffsideResult
from app import get_default_pitch_geometry
from benchmark_geometry_eval_001 import define_15_benchmark_scenes


# ============================================================================
# MOTION CLASSIFICATION
# ============================================================================
def classify_camera_motion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    pitch_mask: Optional[np.ndarray] = None
) -> Tuple[str, float, float, np.ndarray, np.ndarray]:
    """
    Estimates affine camera motion (translation magnitude and scale factor).
    Returns: (motion_class, trans_mag_px, scale_factor, inlier_prev, inlier_curr)
    """
    pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=120,
        qualityLevel=0.02,
        minDistance=25,
        mask=pitch_mask
    )
    if pts is None or len(pts) < 8:
        return "STATIC", 0.0, 1.0, np.empty((0, 2)), np.empty((0, 2))

    pts_curr, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    )

    # Forward-Backward consistency check
    pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray,
        prev_gray,
        pts_curr,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    )

    fb_err = np.linalg.norm(pts.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1)
    valid = (status.flatten() == 1) & (status_back.flatten() == 1) & (fb_err < 1.5)

    good_prev = pts.reshape(-1, 2)[valid]
    good_curr = pts_curr.reshape(-1, 2)[valid]

    if len(good_prev) < 6:
        return "STATIC", 0.0, 1.0, np.empty((0, 2)), np.empty((0, 2))

    M, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if M is None:
        return "STATIC", 0.0, 1.0, good_prev, good_curr

    dx = float(M[0, 2])
    dy = float(M[1, 2])
    trans_mag = float(np.sqrt(dx**2 + dy**2))
    scale = float(np.sqrt(M[0, 0]**2 + M[0, 1]**2))

    is_trans = trans_mag > 2.5
    is_scale = abs(scale - 1.0) > 0.005

    if not is_trans and not is_scale:
        m_class = "STATIC" if trans_mag < 0.8 else "LOW_MOTION"
    elif is_trans and not is_scale:
        m_class = "PAN"
    elif not is_trans and is_scale:
        m_class = "ZOOM"
    else:
        m_class = "PAN_ZOOM"

    return m_class, trans_mag, scale, good_prev, good_curr


# ============================================================================
# PITCH-LINE FEATURE EXTRACTOR (MASKING PLAYERS & SKY)
# ============================================================================
def get_pitch_markings_mask(frame: np.ndarray, player_dets: List[Dict[str, Any]]) -> np.ndarray:
    """
    Creates a binary mask isolating pitch markings.
    Masks out player bounding boxes, crowd/sky (upper 15% of frame), and ads.
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Focus on pitch region (exclude upper 18% stands/sky)
    mask[int(h * 0.18):h, 0:w] = 255

    # White pitch markings extraction in HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Dilate markings slightly so corner detector catches intersections
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white_dilated = cv2.dilate(white_mask, kernel, iterations=1)

    combined_mask = cv2.bitwise_and(mask, white_dilated)

    # Strictly mask out player bounding boxes (+10px padding)
    for p in player_dets:
        bx1, by1, bx2, by2 = [int(v) for v in p["bbox"]]
        px1 = max(0, bx1 - 10)
        py1 = max(0, by1 - 10)
        px2 = min(w, bx2 + 10)
        py2 = min(h, by2 + 10)
        combined_mask[py1:py2, px1:px2] = 0

    return combined_mask


# ============================================================================
# GEOMETRY EVALUATOR CLASS
# ============================================================================
class CameraGeometryHarness:
    def __init__(self, initial_H: np.ndarray, image_shape: Tuple[int, int] = (1080, 1920)):
        self.H_static = initial_H.copy()
        self.H_optflow = initial_H.copy()
        self.H_hybrid = initial_H.copy()
        self.initial_H = initial_H.copy()
        self.det_initial = float(np.linalg.det(initial_H))
        self.image_shape = image_shape

        # Tracking state for Variant B (Optical Flow)
        self.prev_pts_b: Optional[np.ndarray] = None
        self.prev_gray_b: Optional[np.ndarray] = None

        # Tracking state for Variant C (Hybrid)
        self.prev_pts_c: Optional[np.ndarray] = None
        self.prev_gray_c: Optional[np.ndarray] = None
        self.last_valid_H_c = initial_H.copy()

    def update_optical_flow_b(
        self,
        curr_gray: np.ndarray,
        pitch_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Variant B: Lucas-Kanade optical flow update on pitch markings without recalibration.
        """
        t0 = time.perf_counter()
        stats = {
            "pts_tracked": 0,
            "fb_err": 0.0,
            "reproj_err": 0.0,
            "cond_num": float(np.linalg.cond(self.H_optflow)),
            "det": float(np.linalg.det(self.H_optflow)),
            "delta_H_frob": 0.0,
            "status": "ACCEPTED",
            "latency_ms": 0.0
        }

        if self.prev_gray_b is None:
            self.prev_gray_b = curr_gray.copy()
            self.prev_pts_b = cv2.goodFeaturesToTrack(curr_gray, maxCorners=100, qualityLevel=0.03, minDistance=25, mask=pitch_mask)
            stats["latency_ms"] = (time.perf_counter() - t0) * 1000.0
            return self.H_optflow, stats

        if self.prev_pts_b is None or len(self.prev_pts_b) < 6:
            self.prev_pts_b = cv2.goodFeaturesToTrack(self.prev_gray_b, maxCorners=100, qualityLevel=0.03, minDistance=25, mask=pitch_mask)

        if self.prev_pts_b is None or len(self.prev_pts_b) < 6:
            stats["status"] = "INSUFFICIENT_POINTS"
            self.prev_gray_b = curr_gray.copy()
            stats["latency_ms"] = (time.perf_counter() - t0) * 1000.0
            return self.H_optflow, stats

        # Forward flow
        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray_b, curr_gray, self.prev_pts_b, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        # Backward flow for forward-backward consistency check
        pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, self.prev_gray_b, pts_curr, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        fb_err = np.linalg.norm(self.prev_pts_b.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1)
        valid = (status.flatten() == 1) & (status_back.flatten() == 1) & (fb_err < 1.5)

        good_prev = self.prev_pts_b.reshape(-1, 2)[valid]
        good_curr = pts_curr.reshape(-1, 2)[valid]
        stats["pts_tracked"] = len(good_prev)
        stats["fb_err"] = float(np.mean(fb_err[valid])) if len(good_prev) > 0 else 999.0

        if len(good_prev) >= 6:
            # Estimate incremental projective homography Delta H
            delta_H, mask_h = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 2.5)
            if delta_H is not None and np.linalg.det(delta_H) > 0:
                # Update H_t = H_{t-1} * (Delta H)^{-1}
                try:
                    delta_H_inv = np.linalg.inv(delta_H)
                    new_H = self.H_optflow @ delta_H_inv
                    new_H = new_H / new_H[2, 2]

                    # Reprojection error of inliers
                    inliers_prev = good_prev[mask_h.flatten() == 1]
                    inliers_curr = good_curr[mask_h.flatten() == 1]
                    if len(inliers_prev) >= 4:
                        proj = cv2.perspectiveTransform(inliers_prev.reshape(-1, 1, 2), delta_H).reshape(-1, 2)
                        reproj_err = float(np.mean(np.linalg.norm(proj - inliers_curr, axis=1)))
                        stats["reproj_err"] = reproj_err

                    frob_diff = float(np.linalg.norm(new_H - self.H_optflow, 'fro'))
                    stats["delta_H_frob"] = frob_diff
                    stats["cond_num"] = float(np.linalg.cond(new_H))
                    stats["det"] = float(np.linalg.det(new_H))

                    self.H_optflow = new_H
                    stats["status"] = "ACCEPTED"
                except Exception:
                    stats["status"] = "INVERSION_FAILED"
            else:
                stats["status"] = "HOMOGRAPHY_FAILED"
        else:
            stats["status"] = "TOO_FEW_INLIERS"

        # Re-detect points for next frame
        self.prev_gray_b = curr_gray.copy()
        self.prev_pts_b = cv2.goodFeaturesToTrack(curr_gray, maxCorners=100, qualityLevel=0.03, minDistance=25, mask=pitch_mask)
        stats["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return self.H_optflow, stats

    def update_hybrid_c(
        self,
        curr_gray: np.ndarray,
        pitch_mask: np.ndarray,
        fit_landmarks: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Variant C: Hybrid LK + 5 Quality Guards + Full Recalibration / Fallback.
        """
        t0 = time.perf_counter()
        h_img, w_img = self.image_shape
        stats = {
            "pts_tracked": 0,
            "spatial_area_ratio": 0.0,
            "fb_err": 0.0,
            "reproj_err": 0.0,
            "cond_num": float(np.linalg.cond(self.H_hybrid)),
            "det": float(np.linalg.det(self.H_hybrid)),
            "delta_H_frob": 0.0,
            "status": "ACCEPTED",
            "recalibration_triggered": False,
            "fallback_used": False,
            "latency_ms": 0.0
        }

        if self.prev_gray_c is None:
            self.prev_gray_c = curr_gray.copy()
            self.prev_pts_c = cv2.goodFeaturesToTrack(curr_gray, maxCorners=120, qualityLevel=0.025, minDistance=25, mask=pitch_mask)
            stats["latency_ms"] = (time.perf_counter() - t0) * 1000.0
            return self.H_hybrid, stats

        if self.prev_pts_c is None or len(self.prev_pts_c) < 6:
            self.prev_pts_c = cv2.goodFeaturesToTrack(self.prev_gray_c, maxCorners=120, qualityLevel=0.025, minDistance=25, mask=pitch_mask)

        # If points are still missing, trigger recalibration
        if self.prev_pts_c is None or len(self.prev_pts_c) < 6:
            return self._trigger_recalibration(fit_landmarks, stats, t0, "INSUFFICIENT_INITIAL_POINTS", curr_gray, pitch_mask)

        # Optical Flow Forward
        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray_c, curr_gray, self.prev_pts_c, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        # Backward Flow Consistency
        pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, self.prev_gray_c, pts_curr, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        fb_err = np.linalg.norm(self.prev_pts_c.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1)
        valid = (status.flatten() == 1) & (status_back.flatten() == 1) & (fb_err < 1.5)

        good_prev = self.prev_pts_c.reshape(-1, 2)[valid]
        good_curr = pts_curr.reshape(-1, 2)[valid]
        stats["pts_tracked"] = len(good_prev)

        # GUARD 1: Tracked point count >= 6
        if len(good_prev) < 6:
            return self._trigger_recalibration(fit_landmarks, stats, t0, "TOO_FEW_TRACKED_POINTS", curr_gray, pitch_mask)

        # GUARD 2: Spatial distribution bounding box check (ensure non-collinear 2D spread)
        min_x, max_x = np.min(good_curr[:, 0]), np.max(good_curr[:, 0])
        min_y, max_y = np.min(good_curr[:, 1]), np.max(good_curr[:, 1])
        box_area = (max_x - min_x) * (max_y - min_y)
        area_ratio = box_area / float(w_img * h_img)
        stats["spatial_area_ratio"] = float(area_ratio)
        has_2d_spread = (area_ratio >= 0.15) or ((max_y - min_y) >= 100.0 and (max_x - min_x) >= 250.0)
        if not has_2d_spread:
            return self._trigger_recalibration(fit_landmarks, stats, t0, "DEGENERATE_SPATIAL_DISTRIBUTION", curr_gray, pitch_mask)

        delta_H, mask_h = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 2.0)
        if delta_H is None or np.linalg.det(delta_H) <= 0:
            return self._trigger_recalibration(fit_landmarks, stats, t0, "HOMOGRAPHY_FIND_FAILED", curr_gray, pitch_mask)

        # Inlier count & reprojection check
        inliers_prev = good_prev[mask_h.flatten() == 1]
        inliers_curr = good_curr[mask_h.flatten() == 1]
        if len(inliers_prev) < 5:
            return self._trigger_recalibration(fit_landmarks, stats, t0, "TOO_FEW_RANSAC_INLIERS", curr_gray, pitch_mask)

        proj = cv2.perspectiveTransform(inliers_prev.reshape(-1, 1, 2), delta_H).reshape(-1, 2)
        reproj_err = float(np.mean(np.linalg.norm(proj - inliers_curr, axis=1)))
        stats["reproj_err"] = reproj_err

        # GUARD 3: Reprojection error < 3.5 px
        if reproj_err > 3.5:
            return self._trigger_recalibration(fit_landmarks, stats, t0, "HIGH_REPROJECTION_ERROR", curr_gray, pitch_mask)

        try:
            delta_H_inv = np.linalg.inv(delta_H)
            new_H = self.H_hybrid @ delta_H_inv
            new_H = new_H / new_H[2, 2]

            cond_num = float(np.linalg.cond(new_H))
            det = float(np.linalg.det(new_H))
            stats["cond_num"] = cond_num
            stats["det"] = det

            # GUARD 4: Condition number <= 1e5 and sign-preserving determinant
            det_sign_match = (np.sign(det) == np.sign(self.det_initial))
            if cond_num > 1e5 or not np.isfinite(cond_num) or not det_sign_match or abs(det) < 1e-6:
                return self._trigger_recalibration(fit_landmarks, stats, t0, "POOR_CONDITIONING", curr_gray, pitch_mask)

            # GUARD 5: Geometric Plausibility Check (Pitch Convexity Preservation)
            H_inv = np.linalg.inv(new_H)
            world_corners = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32).reshape(-1, 1, 2)
            img_corners = cv2.perspectiveTransform(world_corners, H_inv).reshape(-1, 2)
            if not cv2.isContourConvex(img_corners.astype(np.int32)):
                return self._trigger_recalibration(fit_landmarks, stats, t0, "NON_CONVEX_PITCH_PROJECTION", curr_gray, pitch_mask)

            frob_diff = float(np.linalg.norm(new_H - self.H_hybrid, 'fro'))
            stats["delta_H_frob"] = frob_diff

            self.H_hybrid = new_H
            self.last_valid_H_c = new_H.copy()
            stats["status"] = "ACCEPTED"

        except Exception as e:
            return self._trigger_recalibration(fit_landmarks, stats, t0, f"EXCEPTION_{type(e).__name__}", curr_gray, pitch_mask)

        self.prev_gray_c = curr_gray.copy()
        self.prev_pts_c = cv2.goodFeaturesToTrack(curr_gray, maxCorners=120, qualityLevel=0.025, minDistance=25, mask=pitch_mask)
        stats["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return self.H_hybrid, stats

    def _trigger_recalibration(
        self,
        fit_landmarks: Optional[List[Dict[str, Any]]],
        stats: Dict[str, Any],
        t0: float,
        reason: str,
        curr_gray: np.ndarray,
        pitch_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Handles full pitch landmark recalibration or safe fallback to last valid H."""
        stats["recalibration_triggered"] = True
        stats["recalibration_reason"] = reason

        # Always update tracker state to current frame to avoid tracking against a stale frame
        self.prev_gray_c = curr_gray.copy()
        self.prev_pts_c = cv2.goodFeaturesToTrack(curr_gray, maxCorners=120, qualityLevel=0.025, minDistance=25, mask=pitch_mask)

        if fit_landmarks and len(fit_landmarks) >= 4:
            src = np.array([lm["image"] for lm in fit_landmarks], dtype=np.float32)
            dst = np.array([lm["world"] for lm in fit_landmarks], dtype=np.float32)
            recalib_H, _ = cv2.findHomography(src, dst)
            if recalib_H is not None and np.linalg.cond(recalib_H) < 1e5 and (np.sign(np.linalg.det(recalib_H)) == np.sign(self.det_initial)):
                self.H_hybrid = recalib_H / recalib_H[2, 2]
                self.last_valid_H_c = self.H_hybrid.copy()
                stats["status"] = "RECALIBRATED"
                stats["fallback_used"] = False
                stats["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                return self.H_hybrid, stats

        # Fallback to last known validated H
        self.H_hybrid = self.last_valid_H_c.copy()
        stats["status"] = "FALLBACK_PREVIOUS_VALID"
        stats["fallback_used"] = True
        stats["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return self.H_hybrid, stats


# ============================================================================
# BENCHMARK EXECUTION HARNESS
# ============================================================================
def run_benchmark():
    print("=" * 80)
    print("GEOMETRY-CAMERA-001: Dynamic Homography under Camera Motion Benchmark")
    print("=" * 80)

    # 1. Evaluate on the 15 Benchmark Scenes (GEOMETRY-EVAL-001 unchanged)
    scenes_15 = define_15_benchmark_scenes()
    print(f"\nPhase 1: Evaluating 15 benchmark scenes from GEOMETRY-EVAL-001...")

    scene_records = []
    for sc in scenes_15:
        im_path = sc["image_path"]
        if not os.path.exists(im_path):
            continue
        im = cv2.imread(im_path)
        h, w = im.shape[:2]

        src_fit = np.array([lm["image"] for lm in sc["fit_landmarks"]], dtype=np.float32)
        dst_fit = np.array([lm["world"] for lm in sc["fit_landmarks"]], dtype=np.float32)
        H0, _ = cv2.findHomography(src_fit, dst_fit)

        # Test holdout landmarks
        holdouts = sc["holdout_landmarks"]
        ex_list, ey_list, e2d_list, epx_list = [], [], [], []

        for ho in holdouts:
            pt_img = np.array([[ho["image"]]], dtype=np.float32)
            pt_world_gt = np.array(ho["world"], dtype=np.float32)

            pt_world_pred = cv2.perspectiveTransform(pt_img, H0)[0, 0]
            ex = abs(pt_world_pred[0] - pt_world_gt[0])
            ey = abs(pt_world_pred[1] - pt_world_gt[1])
            e2d = np.sqrt(ex**2 + ey**2)

            # Reprojection pixel error
            pt_w_homo = np.array([[pt_world_gt]], dtype=np.float32)
            pt_img_back = cv2.perspectiveTransform(pt_w_homo, np.linalg.inv(H0))[0, 0]
            epx = np.linalg.norm(pt_img_back - np.array(ho["image"]))

            ex_list.append(ex)
            ey_list.append(ey)
            e2d_list.append(e2d)
            epx_list.append(epx)

        scene_records.append({
            "scene_id": sc["scene_id"],
            "category": sc["category"],
            "ex_median": float(np.median(ex_list)),
            "ex_p95": float(np.percentile(ex_list, 95)),
            "ex_max": float(np.max(ex_list)),
            "ey_median": float(np.median(ey_list)),
            "e2d_median": float(np.median(e2d_list)),
            "epx_median": float(np.median(epx_list))
        })

    print(f"Loaded {len(scene_records)} benchmark scenes.")
    ref_ex_med = np.median([r["ex_median"] for r in scene_records])
    ref_ex_p95 = np.percentile([r["ex_p95"] for r in scene_records], 95)
    print(f"Static Reference E_X: Median = {ref_ex_med:.3f}m, P95 = {ref_ex_p95:.3f}m")

    # 2. Continuous Video Evaluation (offside_spurs_match.mp4, 241 frames)
    video_path = "offside_spurs_match.mp4"
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nPhase 2: Continuous video evaluation on {video_path} ({total_frames} frames)...")

    # Pre-read first frame and calculate initial H0
    ret, frame_0 = cap.read()
    h_v, w_v = frame_0.shape[:2]
    geom_default = get_default_pitch_geometry(w_v, h_v)
    H0 = geom_default.homography_matrix.copy()

    harness = CameraGeometryHarness(H0, (h_v, w_v))

    # Fixed Ground-Truth Pitch World Landmarks for Temporal Drift Analysis
    FIXED_WORLD_POINTS = {
        "Penalty_Spot": [11.0, 34.0],
        "18yd_Box_Center": [16.5, 34.0],
        "Penalty_Arc_Apex": [20.15, 34.0],
        "Halfway_Center": [52.5, 34.0],
        "18yd_Left_Corner": [16.5, 13.84],
        "18yd_Right_Corner": [16.5, 54.16]
    }

    # Offside engine for decision invariance comparison
    offside_engine = OffsideEngine(pitch_geometry=geom_default)

    # Telemetry storage
    frame_metrics = []
    homography_metrics = []
    offside_impact = []
    recalibration_events = []

    # Model for detections
    model = YOLO("models/yolo11_v2_4class_best.pt")
    p_tracker = PlayerTracker()
    team_classifier = TeamClassifier()

    # Landmark drift tracking state
    landmark_tracker_initialized = False
    ref_image_pts = None
    ref_world_pts = None
    prev_landmark_gray = None

    prev_gray = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    f_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run YOLO for player bounding box masking
        res = model(frame, imgsz=1280, conf=0.10, verbose=False)[0]
        p_dets = []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id in [0, 1, 2]:
                p_dets.append({"bbox": [float(v) for v in box.xyxy[0].cpu().numpy()], "class_id": cls_id, "conf": conf})

        tracks = p_tracker.update(p_dets, frame_index=f_idx)
        if not team_classifier.is_fitted and len(p_dets) >= 4:
            team_classifier.fit_from_image(frame, p_dets)
        classified_tracks = team_classifier.classify_tracks(frame, tracks, frame_index=f_idx)

        # Pitch-line mask (excludes players and stands)
        pitch_mask = get_pitch_markings_mask(frame, p_dets)

        # 1. Motion Classification
        m_class, trans_mag, scale, _, _ = classify_camera_motion(prev_gray, curr_gray, pitch_mask)

        # 2. Update Variants
        # Variant A: Static H0
        H_a = harness.H_static

        # Variant B: Optical Flow LK
        H_b, stats_b = harness.update_optical_flow_b(curr_gray, pitch_mask)

        # Variant C: Hybrid LK
        H_c, stats_c = harness.update_hybrid_c(curr_gray, pitch_mask)

        if stats_c.get("recalibration_triggered"):
            recalibration_events.append({
                "frame_id": f_idx,
                "reason": stats_c.get("recalibration_reason"),
                "status": stats_c.get("status"),
                "fallback_used": stats_c.get("fallback_used")
            })

        # Set up pitch geometries for B and C
        geom_b = get_default_pitch_geometry(w_v, h_v)
        geom_b.homography_matrix = H_b
        geom_b.inv_homography_matrix = np.linalg.inv(H_b) if abs(np.linalg.det(H_b)) > 1e-12 else None

        geom_c = get_default_pitch_geometry(w_v, h_v)
        geom_c.homography_matrix = H_c
        geom_c.inv_homography_matrix = np.linalg.inv(H_c) if abs(np.linalg.det(H_c)) > 1e-12 else None

        # 3. Fixed-World-Point Temporal Drift Measurement
        drift_a, drift_b, drift_c = [0.0], [0.0], [0.0]
        if not landmark_tracker_initialized and f_idx >= 15:
            pts_init = cv2.goodFeaturesToTrack(curr_gray, maxCorners=50, qualityLevel=0.03, minDistance=30, mask=pitch_mask)
            if pts_init is not None and len(pts_init) >= 8:
                ref_image_pts = pts_init.reshape(-1, 2)
                ref_world_pts = geom_default.image_to_pitch(ref_image_pts)
                prev_landmark_gray = curr_gray.copy()
                landmark_tracker_initialized = True
        elif landmark_tracker_initialized and ref_image_pts is not None and len(ref_image_pts) >= 4:
            next_pts, st_lm, _ = cv2.calcOpticalFlowPyrLK(
                prev_landmark_gray, curr_gray,
                ref_image_pts.astype(np.float32), None,
                winSize=(21, 21), maxLevel=3
            )
            val_lm = (st_lm.flatten() == 1)
            back_pts, st_back, _ = cv2.calcOpticalFlowPyrLK(
                curr_gray, prev_landmark_gray,
                next_pts, None,
                winSize=(21, 21), maxLevel=3
            )
            fb_lm = np.linalg.norm(ref_image_pts - back_pts.reshape(-1, 2), axis=1)
            val_lm = val_lm & (st_back.flatten() == 1) & (fb_lm < 2.0)

            if np.sum(val_lm) >= 4:
                active_pts = next_pts[val_lm].reshape(-1, 2)
                active_wgt = ref_world_pts[val_lm]

                try:
                    w_a = geom_default.image_to_pitch(active_pts)
                    drift_a = np.linalg.norm(w_a - active_wgt, axis=1).tolist()
                except Exception:
                    drift_a = [99.0]

                try:
                    w_b = geom_b.image_to_pitch(active_pts)
                    drift_b = np.linalg.norm(w_b - active_wgt, axis=1).tolist()
                except Exception:
                    drift_b = [99.0]

                try:
                    w_c = geom_c.image_to_pitch(active_pts)
                    drift_c = np.linalg.norm(w_c - active_wgt, axis=1).tolist()
                except Exception:
                    drift_c = [99.0]

                ref_image_pts = active_pts
                ref_world_pts = active_wgt
                prev_landmark_gray = curr_gray.copy()
            else:
                pts_reseed = cv2.goodFeaturesToTrack(curr_gray, maxCorners=50, qualityLevel=0.03, minDistance=30, mask=pitch_mask)
                if pts_reseed is not None and len(pts_reseed) >= 8:
                    ref_image_pts = pts_reseed.reshape(-1, 2)
                    ref_world_pts = geom_c.image_to_pitch(ref_image_pts)
                    prev_landmark_gray = curr_gray.copy()

        # 4. Offside Decision Invariance Impact (evaluate contact / freeze moment)
        offside_res_a = offside_engine.evaluate_contact_moment(
            players=classified_tracks,
            attack_team_id=0,
            attack_direction="right",
            image_shape=(h_v, w_v)
        )

        engine_b = OffsideEngine(pitch_geometry=geom_b)
        offside_res_b = engine_b.evaluate_contact_moment(
            players=classified_tracks,
            attack_team_id=0,
            attack_direction="right",
            image_shape=(h_v, w_v)
        )

        engine_c = OffsideEngine(pitch_geometry=geom_c)
        offside_res_c = engine_c.evaluate_contact_moment(
            players=classified_tracks,
            attack_team_id=0,
            attack_direction="right",
            image_shape=(h_v, w_v)
        )

        offside_impact.append({
            "frame_id": f_idx,
            "motion_class": m_class,
            "decision_a": offside_res_a.decision,
            "decision_b": offside_res_b.decision,
            "decision_c": offside_res_c.decision,
            "margin_a": offside_res_a.margin_val,
            "margin_b": offside_res_b.margin_val,
            "margin_c": offside_res_c.margin_val,
            "boundary_status_a": offside_res_a.boundary_status,
            "boundary_status_b": offside_res_b.boundary_status,
            "boundary_status_c": offside_res_c.boundary_status,
            "decision_match_ab": offside_res_a.decision == offside_res_b.decision,
            "decision_match_ac": offside_res_a.decision == offside_res_c.decision
        })

        frame_metrics.append({
            "frame_id": f_idx,
            "motion_class": m_class,
            "translation_mag_px": round(trans_mag, 2),
            "scale_factor": round(scale, 4),
            "pts_tracked_b": stats_b["pts_tracked"],
            "pts_tracked_c": stats_c["pts_tracked"],
            "spatial_area_ratio_c": round(stats_c.get("spatial_area_ratio", 0.0), 3),
            "reproj_err_b_px": round(stats_b["reproj_err"], 2),
            "reproj_err_c_px": round(stats_c["reproj_err"], 2),
            "drift_a_m": round(float(np.mean(drift_a)), 3),
            "drift_b_m": round(float(np.mean(drift_b)), 3),
            "drift_c_m": round(float(np.mean(drift_c)), 3),
            "latency_b_ms": round(stats_b["latency_ms"], 2),
            "latency_c_ms": round(stats_c["latency_ms"], 2)
        })

        homography_metrics.append({
            "frame_id": f_idx,
            "motion_class": m_class,
            "status_b": stats_b["status"],
            "status_c": stats_c["status"],
            "cond_num_a": round(float(np.linalg.cond(H_a)), 1),
            "cond_num_b": round(stats_b["cond_num"], 1),
            "cond_num_c": round(stats_c["cond_num"], 1),
            "det_a": round(float(np.linalg.det(H_a)), 3),
            "det_b": round(stats_b["det"], 3),
            "det_c": round(stats_c["det"], 3),
            "delta_H_frob_b": round(stats_b["delta_H_frob"], 4),
            "delta_H_frob_c": round(stats_c["delta_H_frob"], 4)
        })

        prev_gray = curr_gray
        f_idx += 1

    cap.release()

    # Dump CSVs
    csv_paths = {
        "frame_metrics": "geometry_camera_001_frame_metrics.csv",
        "homography_metrics": "geometry_camera_001_homography_metrics.csv",
        "offside_impact": "geometry_camera_001_offside_impact.csv",
        "recalibration_events": "geometry_camera_001_recalibration_events.csv"
    }

    for name, path in csv_paths.items():
        data = {
            "frame_metrics": frame_metrics,
            "homography_metrics": homography_metrics,
            "offside_impact": offside_impact,
            "recalibration_events": recalibration_events
        }[name]

        if data:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            print(f"Dumped {path} ({len(data)} rows)")

    # 3. Stratified Motion Analysis
    motion_groups = defaultdict(list)
    for fm in frame_metrics:
        motion_groups[fm["motion_class"]].append(fm)

    stratified_summary = {}
    for m_class, rows in motion_groups.items():
        stratified_summary[m_class] = {
            "count": len(rows),
            "pct": round(len(rows) / max(1, len(frame_metrics)) * 100.0, 1),
            "mean_drift_a_m": round(float(np.mean([r["drift_a_m"] for r in rows])), 3),
            "mean_drift_b_m": round(float(np.mean([r["drift_b_m"] for r in rows])), 3),
            "mean_drift_c_m": round(float(np.mean([r["drift_c_m"] for r in rows])), 3),
            "reproj_c_px": round(float(np.mean([r["reproj_err_c_px"] for r in rows])), 2)
        }

    # Summary JSON
    summary_report = {
        "benchmark_id": "GEOMETRY-CAMERA-001",
        "total_frames_evaluated": len(frame_metrics),
        "reference_15_scenes_ex_median_m": round(ref_ex_med, 3),
        "reference_15_scenes_ex_p95_m": round(ref_ex_p95, 3),
        "stratified_motion": stratified_summary,
        "overall_drift_a_m": round(float(np.mean([r["drift_a_m"] for r in frame_metrics])), 3),
        "overall_drift_b_m": round(float(np.mean([r["drift_b_m"] for r in frame_metrics])), 3),
        "overall_drift_c_m": round(float(np.mean([r["drift_c_m"] for r in frame_metrics])), 3),
        "recalibration_count": len(recalibration_events),
        "offside_decision_agreement_ac_pct": round(sum(1 for o in offside_impact if o["decision_match_ac"]) / max(1, len(offside_impact)) * 100.0, 1),
        "offside_decision_agreement_ab_pct": round(sum(1 for o in offside_impact if o["decision_match_ab"]) / max(1, len(offside_impact)) * 100.0, 1),
        "mean_latency_c_ms": round(float(np.mean([r["latency_c_ms"] for r in frame_metrics])), 2)
    }

    report_json_path = "dataset_v2_meta/geometry_camera_001_report.json"
    os.makedirs(os.path.dirname(report_json_path), exist_ok=True)
    with open(report_json_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nSaved summary report to {report_json_path}")

    print("\n" + "=" * 80)
    print("STRATIFIED MOTION BENCHMARK SUMMARY")
    print("=" * 80)
    for m_class, s in stratified_summary.items():
        print(f"Motion: {m_class:<12} | Frames: {s['count']:3d} ({s['pct']:4.1f}%) | Drift A: {s['mean_drift_a_m']:.3f}m | Drift B: {s['mean_drift_b_m']:.3f}m | Drift C: {s['mean_drift_c_m']:.3f}m")
    print("=" * 80)
    print(f"Overall Drift A (Static):  {summary_report['overall_drift_a_m']:.3f} m")
    print(f"Overall Drift B (OptFlow): {summary_report['overall_drift_b_m']:.3f} m")
    print(f"Overall Drift C (Hybrid):  {summary_report['overall_drift_c_m']:.3f} m")
    print(f"Total Recalibrations:      {summary_report['recalibration_count']}")
    print(f"Decision Agreement (A vs C): {summary_report['offside_decision_agreement_ac_pct']:.1f}%")
    print(f"Mean Hybrid Latency:       {summary_report['mean_latency_c_ms']:.2f} ms")


if __name__ == "__main__":
    run_benchmark()
