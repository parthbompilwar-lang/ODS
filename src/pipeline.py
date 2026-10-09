"""
End-to-End Offside Detection Pipeline Coordinator.
Coordinates detection, pose estimation, pitch geometry, team classification,
offside evaluation, and VAR graphics generation.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

from src.geometry.pitch_geometry import PitchGeometry
from src.detection.pose_detector import PoseEstimator, PlayerPose
from src.detection.player_detector import ObjectDetector
from src.team.team_classifier import TeamClassifier
from src.engine.offside_engine import OffsideEngine, OffsideResult
from src.visualization.var_renderer import VARRenderer


class OffsideDetectorPipeline:
    def __init__(
        self,
        pose_model: str = "yolo11n-pose.pt",
        det_model: str = "yolo11n.pt",
        device: str = "cuda"
    ):
        self.pitch_geom = PitchGeometry()
        self.pose_estimator = PoseEstimator(model_name=pose_model, device=device)
        self.object_detector = ObjectDetector(model_name=det_model, device=device)
        self.team_classifier = TeamClassifier(n_teams=2)
        self.offside_engine = OffsideEngine(pitch_geometry=self.pitch_geom)
        self.renderer = VARRenderer()

    def calibrate_pitch(
        self,
        image: np.ndarray,
        goal_direction: str = 'right',
        homography_src_pts: Optional[np.ndarray] = None,
        manual_vp: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float]:
        """
        Calibrate pitch perspective for any ground condition:
        1. If manual_vp provided: use it directly.
        2. If homography_src_pts provided: compute metric homography matrix H.
        3. Else: automatically estimate vanishing point of goal-parallel lines.
        """
        if manual_vp is not None:
            self.pitch_geom.vanishing_point = manual_vp
            return manual_vp

        if homography_src_pts is not None and len(homography_src_pts) >= 4:
            self.pitch_geom.set_homography(homography_src_pts)

        vp = self.pitch_geom.estimate_vanishing_point(image, goal_direction=goal_direction)
        return vp

    def process_frame(
        self,
        image: np.ndarray,
        attack_team_id: int = 0,
        attack_direction: str = 'right',
        manual_vp: Optional[Tuple[float, float]] = None,
        homography_pts: Optional[np.ndarray] = None,
        draw_radar: bool = True,
        draw_skeletons: bool = True
    ) -> Tuple[np.ndarray, OffsideResult]:
        """
        Full end-to-end offside detection on a single frame.
        Returns:
            annotated_image: Broadcast VAR overlay
            offside_result: Complete structured decision and measurements
        """
        h, w = image.shape[:2]

        # 1. Pitch Perspective Calibration (Invariant to ground conditions)
        self.calibrate_pitch(
            image,
            goal_direction=attack_direction,
            homography_src_pts=homography_pts,
            manual_vp=manual_vp
        )

        # 2. Player Pose Detection (FIFA Law 11 Legal Keypoints)
        players = self.pose_estimator.estimate(image)

        # 3. Ball Detection
        _, ball_pos = self.object_detector.detect(image)

        # Fallback if no pose detected by YOLO model: use detector bounding boxes
        if not players:
            det_persons, _ = self.object_detector.detect(image)
            for i, d in enumerate(det_persons):
                x1, y1, x2, y2 = d['bbox']
                dummy_kpts = np.zeros((17, 3))
                # Head, hips, feet approximations
                dummy_kpts[0] = [(x1 + x2) / 2, y1 + (y2 - y1) * 0.1, 1.0]
                dummy_kpts[15] = [x1 + (x2 - x1) * 0.3, y2, 1.0]
                dummy_kpts[16] = [x1 + (x2 - x1) * 0.7, y2, 1.0]
                p = PlayerPose(player_id=i, bbox=d['bbox'], keypoints=dummy_kpts, confidence=d['confidence'])
                players.append(p)

        # 4. Team Classification via CIE-Lab Color Clustering
        players = self.team_classifier.classify_players(image, players, goal_direction=attack_direction)

        # 5. Core Offside Evaluation (FIFA Law 11 Rules Engine)
        offside_result = self.offside_engine.evaluate_frame(
            players=players,
            attack_team_id=attack_team_id,
            attack_direction=attack_direction,
            ball_pos=ball_pos,
            image_shape=(h, w)
        )

        # 6. Render Premier League Broadcast Overlays
        annotated_frame = self.renderer.render_var_overlay(
            image=image,
            offside_result=offside_result,
            pitch_geom=self.pitch_geom,
            draw_skeletons=draw_skeletons,
            draw_radar=draw_radar
        )

        return annotated_frame, offside_result
