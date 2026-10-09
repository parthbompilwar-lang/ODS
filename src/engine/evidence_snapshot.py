"""
Evidence Snapshot Module.
Enforces the architectural invariant:
  The evidence frame, decision, and metric parameters must be an immutable snapshot
  captured strictly at the physical contact frame t*.
Eliminates all frame counter desynchronization between engine, renderer, and UI.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


class EvidenceSnapshot:
    """
    Immutable Evidence Snapshot captured strictly at contact moment t*.
    All downstream visualization, banners, metric cards, and file exports
    must read strictly from this single object.
    """

    def __init__(
        self,
        frame_index: int,
        image: np.ndarray,
        decision: str,
        margin_val: float,
        margin_unit: str,
        confidence: float,
        ball_evidence_state: str,
        offside_line_endpoints: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
        attacker_line_endpoints: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
        passer_track_id: Optional[int],
        offside_boundary_world_x: Optional[float],
        explanation: str,
        homography_valid: bool,
        law11_state: str = "OFFSIDE_POSITION",
        receiver_id: Optional[int] = None,
        boundary_status: str = "DECISIVE",
        empirical_error_budget_m: float = 0.59,
        geometry_status: str = "NORMAL"
    ):
        # Strict validation
        if not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError(f"Invalid frame_index: {frame_index}")
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("EvidenceSnapshot requires a valid numpy image.")

        self.frame_index = frame_index
        self.image = image.copy()  # Immutable copy
        self.decision = decision
        self.margin_val = margin_val
        self.margin_unit = margin_unit
        self.confidence = confidence
        self.ball_evidence_state = ball_evidence_state
        self.offside_line_endpoints = offside_line_endpoints
        self.attacker_line_endpoints = attacker_line_endpoints
        self.passer_track_id = passer_track_id
        self.offside_boundary_world_x = offside_boundary_world_x
        self.explanation = explanation
        self.homography_valid = homography_valid
        self.law11_state = law11_state
        self.receiver_id = receiver_id
        self.boundary_status = boundary_status
        self.empirical_error_budget_m = empirical_error_budget_m
        self.geometry_status = geometry_status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "decision": self.decision,
            "boundary_status": self.boundary_status,
            "empirical_error_budget_m": self.empirical_error_budget_m,
            "geometry_status": self.geometry_status,
            "law11_state": self.law11_state,
            "receiver_id": self.receiver_id,
            "margin_val": round(self.margin_val, 2),
            "margin_unit": self.margin_unit,
            "confidence": round(self.confidence, 3),
            "ball_evidence_state": self.ball_evidence_state,
            "offside_boundary_world_x": round(self.offside_boundary_world_x, 2) if self.offside_boundary_world_x else None,
            "offside_line_endpoints": self.offside_line_endpoints,
            "attacker_line_endpoints": self.attacker_line_endpoints,
            "passer_track_id": self.passer_track_id,
            "explanation": self.explanation,
            "homography_valid": self.homography_valid
        }
