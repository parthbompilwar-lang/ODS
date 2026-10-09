"""
Player, Referee, and Ball Detector using YOLO (YOLOv11 / YOLOv8).
Supports real-time inference and dataset annotation loading.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any


class ObjectDetector:
    def __init__(self, model_name: str = "yolo11n.pt", device: str = "cuda"):
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
        except Exception:
            self.model = None

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.3
    ) -> Tuple[List[Dict[str, Any]], Optional[Tuple[float, float]]]:
        """
        Detects persons (players/referees) and sports ball in image.
        Returns:
            persons: list of dicts with 'bbox' and 'conf'
            ball_pos: (x, y) coordinates of ball center if detected, else None
        """
        if self.model is None:
            self._init_model()

        if self.model is None:
            return [], None

        results = self.model(image, conf=conf_threshold, verbose=False)
        persons = []
        ball_pos = None
        ball_conf = 0.0

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for i in range(len(boxes)):
                cls_id = int(clss[i])
                # COCO classes: 0 = person, 32 = sports ball
                if cls_id == 0:
                    bbox = (int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3]))
                    persons.append({
                        "bbox": bbox,
                        "confidence": float(confs[i]),
                        "class_id": 0
                    })
                elif cls_id == 32:
                    if confs[i] > ball_conf:
                        ball_conf = confs[i]
                        bx = float((boxes[i][0] + boxes[i][2]) / 2)
                        by = float((boxes[i][1] + boxes[i][3]) / 2)
                        ball_pos = (bx, by)

        return persons, ball_pos
