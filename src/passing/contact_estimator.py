"""
CONTACT-001: Contact Moment Estimator (t_hat*).
Isolates the exact physical contact frame t_hat* where a teammate plays or touches the ball.
CRITICAL SCIENTIFIC CONSTRAINT:
- t_hat* MUST be backed by direct OBSERVED ball evidence.
- Never designates a purely PREDICTED Kalman frame as the contact moment.
- Reports temporal uncertainty search window and observation confidence.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from src.passing.pass_detector import PassCandidate


class ContactEstimate:
    """Represents the estimated contact moment t_hat* with uncertainty bounds."""

    def __init__(
        self,
        frame_hat: int,
        search_window: Tuple[int, int],
        confidence: float,
        observed_ball: bool,
        passer_track_id: Optional[int],
        passer_bbox: Optional[List[float]],
        ball_position: Tuple[float, float],
        distance_to_foot_px: float,
        candidate_id: str
    ):
        self.frame_hat = frame_hat
        self.search_window = search_window
        self.confidence = confidence
        self.observed_ball = observed_ball
        self.passer_track_id = passer_track_id
        self.passer_bbox = passer_bbox
        self.ball_position = ball_position
        self.distance_to_foot_px = distance_to_foot_px
        self.candidate_id = candidate_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_hat_star": self.frame_hat,
            "search_window": list(self.search_window),
            "temporal_uncertainty_frames": self.search_window[1] - self.search_window[0],
            "confidence": round(self.confidence, 3),
            "observed_ball": self.observed_ball,
            "passer_track_id": self.passer_track_id,
            "passer_bbox": [round(v, 1) for v in self.passer_bbox] if self.passer_bbox else None,
            "ball_position": (round(self.ball_position[0], 1), round(self.ball_position[1], 1)),
            "distance_to_foot_px": round(self.distance_to_foot_px, 1),
            "candidate_id": self.candidate_id
        }


class ContactEstimator:
    """
    Estimates the exact contact frame t_hat* for a given PassCandidate.
    Searches a local temporal window around candidate initiation for the physical
    inflection point (minimum ball-to-foot distance + onset of velocity) with the strict
    requirement that t_hat* must be a directly OBSERVED ball frame.
    """

    def __init__(self, search_half_window: int = 6):
        self.search_half_window = search_half_window

    def estimate_contact(
        self,
        candidate: PassCandidate,
        tracking_history: List[Dict[str, Any]]
    ) -> Optional[ContactEstimate]:
        """
        Estimates t_hat* given a pass candidate and the sliding tracking history.
        tracking_history: list of dicts containing:
          - 'frame': int
          - 'ball_state': dict with 'state' ('OBSERVED'/'PREDICTED'), 'position', 'confidence'
          - 'player_tracks': list of player track dicts
        """
        nominal_frame = candidate.start_frame
        t_min = max(tracking_history[0]["frame"], nominal_frame - self.search_half_window)
        t_max = min(tracking_history[-1]["frame"], nominal_frame + self.search_half_window)

        # Filter window to candidate frames
        window_records = [
            r for r in tracking_history
            if t_min <= r["frame"] <= t_max
        ]

        if not window_records:
            return None

        # ----------------------------------------------------
        # Rule 1: Find OBSERVED ball frames in the search window
        # ----------------------------------------------------
        observed_records = [
            r for r in window_records
            if r.get("ball_state", {}).get("state") == "OBSERVED" and r.get("ball_pos") is not None
        ]

        if not observed_records:
            # STRICT CONSTRAINT: If no directly observed ball frame exists in the
            # local search window, do NOT manufacture a contact estimate from predictions!
            return None

        # ----------------------------------------------------
        # Rule 2: Evaluate physical proximity to passer foot
        # ----------------------------------------------------
        scored_candidates = []
        passer_track_id = candidate.passer_track_id

        for rec in observed_records:
            f = rec["frame"]
            b_pos = rec["ball_pos"]
            b_conf = rec.get("ball_state", {}).get("confidence", 0.5)

            # Find matching passer track in this frame
            passer_in_frame = None
            if passer_track_id is not None:
                for p in rec.get("player_tracks", []):
                    if p.get("track_id") == passer_track_id:
                        passer_in_frame = p
                        break

            # If track ID lost or not found, fall back to nearest player in frame
            if passer_in_frame is None and rec.get("player_tracks"):
                # Find player with minimum foot distance
                min_d = float('inf')
                for p in rec["player_tracks"]:
                    bx1, by1, bx2, by2 = p["bbox"]
                    foot_pt = ((bx1 + bx2) / 2.0, float(by2))
                    d = np.hypot(b_pos[0] - foot_pt[0], b_pos[1] - foot_pt[1])
                    if d < min_d:
                        min_d = d
                        passer_in_frame = p

            if passer_in_frame is not None:
                bx1, by1, bx2, by2 = passer_in_frame["bbox"]
                foot_x = (bx1 + bx2) / 2.0
                foot_y = float(by2)
                dist_to_foot = float(np.hypot(b_pos[0] - foot_x, b_pos[1] - foot_y))
            else:
                dist_to_foot = candidate.passer_distance
                passer_in_frame = {"bbox": candidate.passer_bbox, "track_id": passer_track_id}

            # Proximity score (closer to foot = higher probability of contact)
            prox_score = max(0.0, 1.0 - (dist_to_foot / 120.0))

            # Velocity acceleration inflection score:
            # Physical contact t* is the moment immediately BEFORE or AT ball departure
            ball_spd = rec.get("ball_state", {}).get("speed", 0.0)
            
            # Find next frame speed to detect acceleration onset
            next_spd = 0.0
            for r_next in window_records:
                if r_next["frame"] == f + 1:
                    next_spd = r_next.get("ball_state", {}).get("speed", 0.0)
                    break
            
            # Contact inflection: low speed or close to foot at frame f, high acceleration at f+1
            accel_onset = max(0.0, min(1.0, (next_spd - ball_spd) / 10.0))

            # Temporal alignment score: prefers the latest contact frame before departure
            temporal_dist = abs(f - nominal_frame)
            temp_score = max(0.0, 1.0 - (temporal_dist / (self.search_half_window + 1)))

            # Composite contact likelihood:
            # Weights physical proximity to foot and the acceleration transition
            contact_score = 0.45 * prox_score + 0.30 * accel_onset + 0.15 * b_conf + 0.10 * temp_score

            scored_candidates.append({
                "frame": f,
                "score": contact_score,
                "ball_pos": b_pos,
                "dist_to_foot": dist_to_foot,
                "passer": passer_in_frame
            })

        if not scored_candidates:
            return None

        # Select highest-scoring OBSERVED frame as t_hat*
        best = max(scored_candidates, key=lambda x: x["score"])

        # Determine actual uncertainty bounds around best frame
        hat_f = best["frame"]
        window_bounds = (max(t_min, hat_f - 1), min(t_max, hat_f + 1))

        return ContactEstimate(
            frame_hat=hat_f,
            search_window=window_bounds,
            confidence=best["score"],
            observed_ball=True,  # STRICT: Guaranteed True by observed_records filter
            passer_track_id=best["passer"].get("track_id") if best["passer"] else None,
            passer_bbox=best["passer"].get("bbox") if best["passer"] else None,
            ball_position=best["ball_pos"],
            distance_to_foot_px=best["dist_to_foot"],
            candidate_id=candidate.candidate_id
        )
