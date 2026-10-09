"""
PASS-001: Interpretable Deterministic Pass Candidate Detector.
Operates on temporal ball trajectory and player tracks without deep learning.
Detects when the ball departs a nearby player with significant velocity
and directional persistence, generating pass candidates for contact estimation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any


class PassCandidate:
    """Represents an identified pass event candidate."""

    def __init__(
        self,
        candidate_id: str,
        start_frame: int,
        end_frame: int,
        passer_track_id: Optional[int],
        passer_bbox: Optional[List[float]],
        passer_distance: float,
        initial_speed: float,
        direction_vector: Tuple[float, float],
        observed_ratio: float,
        confidence: float
    ):
        self.candidate_id = candidate_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.passer_track_id = passer_track_id
        self.passer_bbox = passer_bbox
        self.passer_distance = passer_distance
        self.initial_speed = initial_speed
        self.direction_vector = direction_vector
        self.observed_ratio = observed_ratio
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "passer_track_id": self.passer_track_id,
            "passer_bbox": [round(v, 1) for v in self.passer_bbox] if self.passer_bbox else None,
            "passer_distance_px": round(self.passer_distance, 1),
            "initial_speed_px_per_frame": round(self.initial_speed, 1),
            "direction_vector": (round(self.direction_vector[0], 2), round(self.direction_vector[1], 2)),
            "observed_ratio": round(self.observed_ratio, 2),
            "confidence": round(self.confidence, 3)
        }


class PassDetector:
    """
    Deterministic Pass Candidate Generator.
    Monitors ball trajectory kinematics and spatial proximity to player tracks.
    """

    def __init__(
        self,
        proximity_threshold: float = 90.0,
        min_departure_speed: float = 8.0,
        min_departure_displacement: float = 30.0,
        history_window: int = 15,
        min_cooldown_frames: int = 8
    ):
        self.proximity_threshold = proximity_threshold
        self.min_departure_speed = min_departure_speed
        self.min_departure_displacement = min_departure_displacement
        self.history_window = history_window
        self.min_cooldown_frames = min_cooldown_frames

        # Ring buffer of recent frames
        self.history: List[Dict[str, Any]] = []
        self.last_candidate_frame = -999
        self.candidate_count = 0

    def reset(self):
        self.history.clear()
        self.last_candidate_frame = -999
        self.candidate_count = 0

    def update(
        self,
        ball_state: Dict[str, Any],
        player_tracks: List[Dict[str, Any]],
        frame_index: int
    ) -> Optional[PassCandidate]:
        """
        Processes one frame of tracking data.
        Returns a PassCandidate if a departure event is triggered, else None.
        """
        # Store frame record
        record = {
            "frame": frame_index,
            "ball_state": ball_state,
            "ball_pos": ball_state.get("position"),
            "ball_is_observed": ball_state.get("is_observed", False),
            "ball_speed": ball_state.get("speed", 0.0),
            "ball_vel": ball_state.get("velocity", (0.0, 0.0)),
            "player_tracks": player_tracks
        }
        self.history.append(record)

        # Maintain sliding window size
        if len(self.history) > self.history_window:
            self.history.pop(0)

        # Need at least 5 frames of history to detect kinematic departure
        if len(self.history) < 5:
            return None

        # Check cooldown to prevent duplicate triggers on the same kick
        if frame_index - self.last_candidate_frame < self.min_cooldown_frames:
            return None

        # Check if ball has position in current frame
        curr_ball_pos = record["ball_pos"]
        if curr_ball_pos is None:
            return None

        # ----------------------------------------------------
        # Kinematic Feature Analysis across history window
        # ----------------------------------------------------
        # Look back 3 to 7 frames to find if ball was close to a player
        cand = self._evaluate_departure(frame_index)
        if cand is not None:
            self.last_candidate_frame = frame_index
            self.candidate_count += 1
            return cand

        return None

    def _evaluate_departure(self, curr_frame: int) -> Optional[PassCandidate]:
        """Examines the window for ball departure from a player."""
        n = len(self.history)
        curr_rec = self.history[-1]
        curr_pos = curr_rec["ball_pos"]
        curr_speed = curr_rec["ball_speed"]
        curr_vel = curr_rec["ball_vel"]

        # Current ball speed must exceed minimum departure velocity
        if curr_speed < self.min_departure_speed:
            return None

        # Scan backwards from the most recent frame before departure (n-2 down to 0)
        # The first frame near a player's foot is the physical release/departure point
        best_passer = None
        best_contact_idx = -1
        min_dist_at_contact = float('inf')

        for idx in range(n - 2, max(-1, n - 8), -1):
            past_rec = self.history[idx]
            past_ball_pos = past_rec["ball_pos"]
            if past_ball_pos is None:
                continue

            # Compare against all player tracks in that past frame
            for p in past_rec["player_tracks"]:
                bx1, by1, bx2, by2 = p["bbox"]
                foot_x = (bx1 + bx2) / 2.0
                foot_y = float(by2)

                dist = np.hypot(past_ball_pos[0] - foot_x, past_ball_pos[1] - foot_y)
                if dist <= self.proximity_threshold:
                    min_dist_at_contact = dist
                    best_passer = p
                    best_contact_idx = idx
                    break

            if best_passer is not None:
                break

        if best_passer is None or best_contact_idx < 0:
            return None

        contact_rec = self.history[best_contact_idx]
        contact_pos = contact_rec["ball_pos"]
        contact_frame = contact_rec["frame"]

        # Displacement from contact point to current position
        displacement = np.hypot(curr_pos[0] - contact_pos[0], curr_pos[1] - contact_pos[1])
        if displacement < self.min_departure_displacement:
            return None

        # Direction vector of pass
        dx = curr_pos[0] - contact_pos[0]
        dy = curr_pos[1] - contact_pos[1]
        norm = np.hypot(dx, dy)
        dir_vec = (dx / norm, dy / norm) if norm > 0 else (0.0, 0.0)

        # Vector from passer to ball must align with movement direction
        passer_foot_x = (best_passer["bbox"][0] + best_passer["bbox"][2]) / 2.0
        passer_foot_y = float(best_passer["bbox"][3])
        from_passer_dx = curr_pos[0] - passer_foot_x
        from_passer_dy = curr_pos[1] - passer_foot_y
        from_passer_norm = np.hypot(from_passer_dx, from_passer_dy)

        if from_passer_norm > 0:
            cos_align = (dx * from_passer_dx + dy * from_passer_dy) / (norm * from_passer_norm)
            if cos_align < 0.20:
                # Ball is moving towards the player, not departing
                return None

        # Evidence quality: fraction of frames with OBSERVED ball in departure window
        departure_window = self.history[best_contact_idx:]
        observed_count = sum(1 for r in departure_window if r["ball_is_observed"])
        observed_ratio = observed_count / max(1, len(departure_window))

        # Require at least one directly observed ball frame in window
        if observed_count < 1:
            return None

        # Confidence calculation based on speed, proximity, and observation quality
        prox_score = max(0.0, 1.0 - (min_dist_at_contact / self.proximity_threshold))
        speed_score = min(1.0, curr_speed / 25.0)
        conf = 0.40 * prox_score + 0.35 * speed_score + 0.25 * observed_ratio

        candidate = PassCandidate(
            candidate_id=f"pass_{contact_frame}_{curr_frame}",
            start_frame=contact_frame,
            end_frame=curr_frame,
            passer_track_id=best_passer.get("track_id"),
            passer_bbox=best_passer.get("bbox"),
            passer_distance=min_dist_at_contact,
            initial_speed=curr_speed,
            direction_vector=dir_vec,
            observed_ratio=observed_ratio,
            confidence=conf
        )
        return candidate
