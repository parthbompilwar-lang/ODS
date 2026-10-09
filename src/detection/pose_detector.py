"""
Pose Estimator and FIFA Law 11 Legal Keypoint Extractor.
Extracts player skeletons and identifies the leading playable body part
while strictly excluding arms and hands per official FIFA Law 11 rules.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


class PlayerPose:
    """Represents detected keypoints and offside metrics for a single player."""

    # COCO Keypoint Index Mapping
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7     # EXCLUDED
    RIGHT_ELBOW = 8    # EXCLUDED
    LEFT_WRIST = 9     # EXCLUDED
    RIGHT_WRIST = 10   # EXCLUDED
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    # Playable keypoints per FIFA Law 11 (Head, Torso, Legs, Feet)
    PLAYABLE_KEYPOINT_INDICES = [
        NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE,
        LEFT_ANKLE, RIGHT_ANKLE
    ]

    def __init__(
        self,
        player_id: int,
        bbox: Tuple[int, int, int, int],
        keypoints: np.ndarray,
        confidence: float = 1.0,
        team_id: int = 0
    ):
        """
        player_id: Unique identifier for player
        bbox: (x1, y1, x2, y2)
        keypoints: (17, 3) or (17, 2) array of (x, y, [conf])
        """
        self.player_id = player_id
        self.bbox = bbox
        self.keypoints = np.array(keypoints)
        self.confidence = confidence
        self.team_id = team_id
        self.role: str = "outfield"  # "outfield", "goalkeeper", "referee"
        self.offside_decision: str = "ONSIDE"  # "ONSIDE", "OFFSIDE", "DEFENDER"

        # Leading playable point & ground projection
        self.leading_point: Tuple[float, float] = (0.0, 0.0)
        self.ground_point: Tuple[float, float] = (0.0, 0.0)
        self.leading_part_name: str = "foot"

    def compute_leading_point(self, attack_direction: str = 'right') -> Tuple[float, float]:
        """
        Finds the playable body part nearest to the opponent goal line.
        Arms (elbows, wrists, hands) are strictly ignored per FIFA rules.
        """
        best_x = -1e9 if attack_direction == 'right' else 1e9
        best_pt = (float((self.bbox[0] + self.bbox[2]) / 2), float(self.bbox[3]))
        best_name = "foot"

        names = {
            self.NOSE: "head", self.LEFT_EYE: "head", self.RIGHT_EYE: "head",
            self.LEFT_EAR: "head", self.RIGHT_EAR: "head",
            self.LEFT_SHOULDER: "shoulder", self.RIGHT_SHOULDER: "shoulder",
            self.LEFT_HIP: "torso", self.RIGHT_HIP: "torso",
            self.LEFT_KNEE: "knee", self.RIGHT_KNEE: "knee",
            self.LEFT_ANKLE: "foot", self.RIGHT_ANKLE: "foot"
        }

        for idx in self.PLAYABLE_KEYPOINT_INDICES:
            if idx < len(self.keypoints):
                pt = self.keypoints[idx]
                # Check confidence if available
                conf = pt[2] if len(pt) > 2 else 1.0
                if conf > 0.25 and pt[0] > 0 and pt[1] > 0:
                    x, y = float(pt[0]), float(pt[1])
                    if attack_direction == 'right':
                        if x > best_x:
                            best_x = x
                            best_pt = (x, y)
                            best_name = names.get(idx, "body")
                    else:
                        if x < best_x:
                            best_x = x
                            best_pt = (x, y)
                            best_name = names.get(idx, "body")

        self.leading_point = best_pt
        self.leading_part_name = best_name

        # Compute ground point (feet or base of bounding box)
        ankles = []
        for a_idx in [self.LEFT_ANKLE, self.RIGHT_ANKLE]:
            if a_idx < len(self.keypoints):
                pt = self.keypoints[a_idx]
                conf = pt[2] if len(pt) > 2 else 1.0
                if conf > 0.25 and pt[0] > 0 and pt[1] > 0:
                    ankles.append((float(pt[0]), float(pt[1])))

        if ankles:
            # Lowest y is physically highest on ground
            avg_x = sum(p[0] for p in ankles) / len(ankles)
            max_y = max(p[1] for p in ankles)
            self.ground_point = (avg_x, max_y)
        else:
            self.ground_point = (float((self.bbox[0] + self.bbox[2]) / 2), float(self.bbox[3]))

        return self.leading_point


class PoseEstimator:
    """YOLO-Pose loader and batch keypoint estimator."""

    def __init__(self, model_name: str = "yolo11n-pose.pt", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from ultralytics import YOLO
            import torch
            actual_device = "cuda" if torch.cuda.is_available() and self.device == "cuda" else "cpu"
            self.model = YOLO(self.model_name)
            self.model.to(actual_device)
        except Exception as e:
            # Model will be loaded dynamically or fallback to annotations
            self.model = None

    def estimate(self, image: np.ndarray, conf_threshold: float = 0.35) -> List[PlayerPose]:
        """
        Run pose estimation on image, returning list of PlayerPose objects.
        """
        if self.model is None:
            self._init_model()

        if self.model is None:
            return []

        results = self.model(image, conf=conf_threshold, verbose=False)
        players = []
        if len(results) > 0 and results[0].boxes is not None and results[0].keypoints is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            kpts = results[0].keypoints.data.cpu().numpy()

            for i in range(len(boxes)):
                bbox = (int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3]))
                pose = PlayerPose(
                    player_id=i,
                    bbox=bbox,
                    keypoints=kpts[i],
                    confidence=float(confs[i])
                )
                players.append(pose)
        return players
