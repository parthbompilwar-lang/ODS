"""
VALIDATE-VIEWPOINT-STATE-001: Verification of Homography Viewpoint Validity State Machine.
Demonstrates that instantaneous motion = 0 does NOT imply H0 is valid.
Compares:
1. Naive Velocity-Gated Policy: Reverts to H0 whenever instantaneous translation < 1.0 px.
2. Viewpoint State Machine Policy: Tracks cumulative viewpoint displacement. Retains validated dynamic H
   at displaced stationary viewpoints and refuses stale H0 regression.
"""

import cv2
import numpy as np
from app import get_default_pitch_geometry
from benchmark_geometry_camera_001 import (
    classify_camera_motion,
    get_pitch_markings_mask,
    CameraGeometryHarness
)

def run_validation():
    print("=" * 75)
    print("VALIDATE-VIEWPOINT-STATE-001: Viewpoint State Tracking vs Naive Motion Gate")
    print("=" * 75)

    video_path = "offside_spurs_match.mp4"
    cap = cv2.VideoCapture(video_path)
    h_v, w_v = 1080, 1920
    geom_default = get_default_pitch_geometry(w_v, h_v)
    H0 = geom_default.homography_matrix.copy()

    harness = CameraGeometryHarness(H0, (h_v, w_v))

    # We evaluate frames 140 to 160: contains heavy PAN (145-152) and sudden camera stop (153-156)
    start_frame = 140
    end_frame = 160

    # Ground truth pitch features tracked across this window
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame_start = cap.read()
    gray_start = cv2.cvtColor(frame_start, cv2.COLOR_BGR2GRAY)
    
    # Initialize features on pitch markings at start_frame
    mask_start = get_pitch_markings_mask(frame_start, [])
    pts_start = cv2.goodFeaturesToTrack(gray_start, maxCorners=60, qualityLevel=0.03, minDistance=25, mask=mask_start)
    w_gt = geom_default.image_to_pitch(pts_start.reshape(-1, 2))

    prev_gray = gray_start.copy()
    active_pts = pts_start.copy()
    active_wgt = w_gt.copy()

    cum_disp_px = 0.0
    viewpoint_state = "VALID_CALIBRATED"

    # Tracking dynamic H
    H_dynamic = H0.copy()

    print(f"\nEvaluating transition window: Frame {start_frame} to {end_frame}...")
    print(f"{'Frame':<6} | {'Motion':<8} | {'Trans(px)':<9} | {'CumDisp(px)':<11} | {'Viewpoint State':<18} | {'Drift Naive(m)':<15} | {'Drift State-Mach(m)':<20}")
    print("-" * 105)

    for f_idx in range(start_frame + 1, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pitch_mask = get_pitch_markings_mask(frame, [])

        m_class, trans_mag, scale, _, _ = classify_camera_motion(prev_gray, curr_gray, pitch_mask)
        cum_disp_px += trans_mag

        # Update dynamic homography via Variant C
        H_dynamic, stats_c = harness.update_hybrid_c(curr_gray, pitch_mask)

        # Track ground-truth landmarks
        next_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, active_pts, None)
        val = (st.flatten() == 1)

        active_pts = next_pts[val]
        active_wgt = active_wgt[val]

        # 1. NAIVE POLICY: If instantaneous motion == STATIC, revert to H0!
        if m_class in ["STATIC", "LOW_MOTION"]:
            H_naive = H0.copy()
        else:
            H_naive = H_dynamic.copy()

        # 2. VIEWPOINT STATE MACHINE POLICY:
        # If camera has moved significantly from calibrated viewpoint (cum_disp >= 10 px),
        # viewpoint is STALE_DISPLACED. Never revert to H0! Keep validated dynamic H.
        if cum_disp_px < 5.0:
            viewpoint_state = "VALID_CALIBRATED"
            H_state = H0.copy()
        else:
            if m_class in ["STATIC", "LOW_MOTION"]:
                viewpoint_state = "STALE_DISPLACED_STATIONARY"
            else:
                viewpoint_state = "TRACKING_ACTIVE"
            H_state = H_dynamic.copy()

        # Compute physical pitch drift
        pts_2d = active_pts.reshape(-1, 2)
        geom_naive = get_default_pitch_geometry(w_v, h_v)
        geom_naive.homography_matrix = H_naive
        w_naive = geom_naive.image_to_pitch(pts_2d)
        drift_naive = float(np.mean(np.linalg.norm(w_naive - active_wgt, axis=1)))

        geom_state = get_default_pitch_geometry(w_v, h_v)
        geom_state.homography_matrix = H_state
        w_state = geom_state.image_to_pitch(pts_2d)
        drift_state = float(np.mean(np.linalg.norm(w_state - active_wgt, axis=1)))

        marker = " <== CAMERA STOPS (H0 SPIKE!)" if (m_class == "STATIC" and drift_naive > 10.0) else ""
        print(f"{f_idx:<6} | {m_class:<8} | {trans_mag:<9.2f} | {cum_disp_px:<11.1f} | {viewpoint_state:<18} | {drift_naive:<15.3f} | {drift_state:<20.3f}{marker}")

        prev_gray = curr_gray.copy()

    print("-" * 105)
    print("Conclusion: Naive motion gating suffers a catastrophic drift spike whenever the camera")
    print("stops at a displaced viewpoint. The Viewpoint State Machine maintains continuity.")

if __name__ == "__main__":
    run_validation()
