"""
TRACK-002: Ball Tracker with Explicit State and Uncertainty Modeling.
Strictly distinguishes OBSERVED frames from PREDICTED frames.
Never treats Kalman predictions as ground-truth observations.
Measures temporal gap recovery (1-frame, 2-frame, 3-frame, etc.) and decays
confidence exponentially during observation dropouts.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any


class BallKalmanFilter:
    """
    2D Constant Velocity / Acceleration Kalman Filter for soccer ball motion.
    State: [x, y, vx, vy]
    """

    def __init__(self, x: float, y: float):
        self.state = np.array([[x], [y], [0.0], [0.0]], dtype=float)
        # Transition matrix F (dt = 1 frame)
        self.F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)

        # Measurement matrix H (observes x, y)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=float)

        # Covariances
        self.P = np.diag([25.0, 25.0, 100.0, 100.0])
        self.Q = np.diag([2.0, 2.0, 15.0, 15.0])
        self.R = np.diag([8.0, 8.0])

    def predict(self) -> Tuple[float, float]:
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.state[0, 0]), float(self.state[1, 0])

    def update(self, x: float, y: float):
        z = np.array([[x], [y]], dtype=float)
        y_residual = z - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.state = self.state + np.dot(K, y_residual)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_position(self) -> Tuple[float, float]:
        return float(self.state[0, 0]), float(self.state[1, 0])

    def get_velocity(self) -> Tuple[float, float]:
        return float(self.state[2, 0]), float(self.state[3, 0])


class BallTracker:
    """
    Temporal Ball Tracker.
    Produces state machine transitions:
      OBSERVED -> PREDICTED (confidence decaying) -> LOST (if gap > max_gap)
      PREDICTED -> OBSERVED (track recovered, gap length logged)
    """

    def __init__(
        self,
        max_missing_frames: int = 5,
        initial_gating_dist: float = 75.0,
        decay_factor: float = 0.72
    ):
        self.max_missing_frames = max_missing_frames
        self.initial_gating_dist = initial_gating_dist
        self.decay_factor = decay_factor

        self.kf: Optional[BallKalmanFilter] = None
        self.track_state = "LOST"  # "OBSERVED", "PREDICTED", "LOST"
        self.consecutive_missing = 0
        self.last_observation_conf = 0.0
        self.last_bbox: Optional[List[float]] = None
        self.trajectory_history: List[Dict[str, Any]] = []

        # Gap statistics
        self.recovered_gaps: List[int] = []

    def reset(self):
        self.kf = None
        self.track_state = "LOST"
        self.consecutive_missing = 0
        self.last_observation_conf = 0.0
        self.last_bbox = None
        self.trajectory_history.clear()
        self.recovered_gaps.clear()

    def update(self, ball_detections: List[Dict[str, Any]], frame_index: int) -> Dict[str, Any]:
        """
        Updates the ball track for a single frame.
        ball_detections: list of {'bbox': [x1, y1, x2, y2], 'conf': float}
        """
        # If tracker is currently LOST, attempt to initialize from highest-confidence detection
        if self.track_state == "LOST" or self.kf is None:
            if len(ball_detections) > 0:
                best_det = max(ball_detections, key=lambda d: d.get("conf", 0.0))
                bx1, by1, bx2, by2 = best_det["bbox"]
                cx = (bx1 + bx2) / 2.0
                cy = (by1 + by2) / 2.0
                conf = best_det.get("conf", 0.0)

                self.kf = BallKalmanFilter(cx, cy)
                self.track_state = "OBSERVED"
                self.consecutive_missing = 0
                self.last_observation_conf = conf
                self.last_bbox = best_det["bbox"]

                step_info = {
                    "frame_index": frame_index,
                    "state": "OBSERVED",
                    "is_observed": True,
                    "position": (round(cx, 1), round(cy, 1)),
                    "bbox": [round(v, 1) for v in best_det["bbox"]],
                    "confidence": round(conf, 3),
                    "velocity": (0.0, 0.0),
                    "speed": 0.0,
                    "consecutive_missing": 0,
                    "recovered_gap": None
                }
                self.trajectory_history.append(step_info)
                return step_info
            else:
                step_info = {
                    "frame_index": frame_index,
                    "state": "LOST",
                    "is_observed": False,
                    "position": None,
                    "bbox": None,
                    "confidence": 0.0,
                    "velocity": (0.0, 0.0),
                    "speed": 0.0,
                    "consecutive_missing": 999,
                    "recovered_gap": None
                }
                self.trajectory_history.append(step_info)
                return step_info

        # Tracker is ACTIVE (OBSERVED or PREDICTED)
        # Step 1: Predict forward
        pred_x, pred_y = self.kf.predict()
        vx, vy = self.kf.get_velocity()
        speed = float(np.hypot(vx, vy))

        # Dynamic gating distance based on current speed
        gating_radius = max(self.initial_gating_dist, speed * 2.5)

        # Step 2: Associate with candidates
        matched_det = None
        min_dist = float('inf')

        for det in ball_detections:
            bx1, by1, bx2, by2 = det["bbox"]
            det_cx = (bx1 + bx2) / 2.0
            det_cy = (by1 + by2) / 2.0
            dist = float(np.hypot(det_cx - pred_x, det_cy - pred_y))

            if dist <= gating_radius and dist < min_dist:
                min_dist = dist
                matched_det = det

        # Step 3: Handle Measurement vs Prediction
        recovered_gap = None

        if matched_det is not None:
            # Observation matched
            bx1, by1, bx2, by2 = matched_det["bbox"]
            det_cx = (bx1 + bx2) / 2.0
            det_cy = (by1 + by2) / 2.0
            conf = matched_det.get("conf", 0.0)

            if self.consecutive_missing > 0:
                recovered_gap = self.consecutive_missing
                self.recovered_gaps.append(recovered_gap)

            self.kf.update(det_cx, det_cy)
            self.track_state = "OBSERVED"
            self.consecutive_missing = 0
            self.last_observation_conf = conf
            self.last_bbox = matched_det["bbox"]

            curr_pos = self.kf.get_position()
            curr_vel = self.kf.get_velocity()

            step_info = {
                "frame_index": frame_index,
                "state": "OBSERVED",
                "is_observed": True,
                "position": (round(curr_pos[0], 1), round(curr_pos[1], 1)),
                "bbox": [round(v, 1) for v in matched_det["bbox"]],
                "confidence": round(conf, 3),
                "velocity": (round(curr_vel[0], 1), round(curr_vel[1], 1)),
                "speed": round(float(np.hypot(curr_vel[0], curr_vel[1])), 1),
                "consecutive_missing": 0,
                "recovered_gap": recovered_gap
            }
        else:
            # Check for confident unassociated observation to prevent ghost/lagging track
            if len(ball_detections) > 0 and self.consecutive_missing >= 2:
                best_unmatched = max(ball_detections, key=lambda d: d.get("conf", 0.0))
                if best_unmatched.get("conf", 0.0) >= 0.35:
                    bx1, by1, bx2, by2 = best_unmatched["bbox"]
                    det_cx = (bx1 + bx2) / 2.0
                    det_cy = (by1 + by2) / 2.0
                    conf = best_unmatched.get("conf", 0.0)

                    recovered_gap = self.consecutive_missing
                    self.recovered_gaps.append(recovered_gap)

                    self.kf = BallKalmanFilter(det_cx, det_cy)
                    self.track_state = "OBSERVED"
                    self.consecutive_missing = 0
                    self.last_observation_conf = conf
                    self.last_bbox = best_unmatched["bbox"]

                    curr_pos = self.kf.get_position()
                    curr_vel = self.kf.get_velocity()

                    step_info = {
                        "frame_index": frame_index,
                        "state": "OBSERVED",
                        "is_observed": True,
                        "position": (round(curr_pos[0], 1), round(curr_pos[1], 1)),
                        "bbox": [round(v, 1) for v in best_unmatched["bbox"]],
                        "confidence": round(conf, 3),
                        "velocity": (round(curr_vel[0], 1), round(curr_vel[1], 1)),
                        "speed": round(float(np.hypot(curr_vel[0], curr_vel[1])), 1),
                        "consecutive_missing": 0,
                        "recovered_gap": recovered_gap
                    }
                    self.trajectory_history.append(step_info)
                    return step_info

            # Missing observation: Pure prediction with decaying confidence
            self.consecutive_missing += 1

            # Prevent stale static predictions: if speed is near-zero, cut off prediction early
            effective_max_missing = 3 if speed < 4.0 else self.max_missing_frames

            if self.consecutive_missing > effective_max_missing:
                self.track_state = "LOST"
                step_info = {
                    "frame_index": frame_index,
                    "state": "LOST",
                    "is_observed": False,
                    "position": None,
                    "bbox": None,
                    "confidence": 0.0,
                    "velocity": (0.0, 0.0),
                    "speed": 0.0,
                    "consecutive_missing": self.consecutive_missing,
                    "recovered_gap": None
                }
            else:
                self.track_state = "PREDICTED"
                decayed_conf = self.last_observation_conf * (self.decay_factor ** self.consecutive_missing)

                # Synthesize predicted bounding box around predicted center using last known dimensions
                if self.last_bbox is not None:
                    bw = self.last_bbox[2] - self.last_bbox[0]
                    bh = self.last_bbox[3] - self.last_bbox[1]
                    synth_bbox = [pred_x - bw / 2, pred_y - bh / 2, pred_x + bw / 2, pred_y + bh / 2]
                else:
                    synth_bbox = [pred_x - 10, pred_y - 10, pred_x + 10, pred_y + 10]

                step_info = {
                    "frame_index": frame_index,
                    "state": "PREDICTED",
                    "is_observed": False,  # STRICT: Never claim observation on prediction
                    "position": (round(pred_x, 1), round(pred_y, 1)),
                    "bbox": [round(v, 1) for v in synth_bbox],
                    "confidence": round(decayed_conf, 3),
                    "velocity": (round(vx, 1), round(vy, 1)),
                    "speed": round(speed, 1),
                    "consecutive_missing": self.consecutive_missing,
                    "recovered_gap": None
                }

        self.trajectory_history.append(step_info)
        return step_info
