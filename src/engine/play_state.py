"""
Law 11 Play State Machine with Immutable Evidence Snapshot.
Formal IFAB Law 11 Distinction:
  Offside Position != Offside Offence.
  It is not an offence in itself to be in an offside position.
  An offside offence is only committed when a player in an offside position
  becomes actively involved in play (interfering with play, opponent, or gaining an advantage).

State Progression:
  PLAYING
    -> PASS_CANDIDATE
    -> CONTACT_ESTIMATED
    -> POSITION_EVALUATED
         ├── If ONSIDE -> PLAYING (uninterrupted)
         └── If OFFSIDE -> OFFSIDE_POSITION_DETECTED
                             ├── If PASSIVE (no involvement) -> PLAYING
                             └── If ACTIVE_INVOLVEMENT -> OFFSIDE_OFFENCE -> PLAY_STOPPED
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from src.engine.evidence_snapshot import EvidenceSnapshot


class PlayStateEnum(Enum):
    PLAYING = "PLAYING"
    PASS_CANDIDATE = "PASS_CANDIDATE"
    CONTACT_ESTIMATED = "CONTACT_ESTIMATED"
    POSITION_EVALUATED = "POSITION_EVALUATED"
    OFFSIDE_POSITION_DETECTED = "OFFSIDE_POSITION_DETECTED"
    ACTIVE_INVOLVEMENT = "ACTIVE_INVOLVEMENT"
    OFFSIDE_OFFENCE = "OFFSIDE_OFFENCE"
    PLAY_STOPPED = "PLAY_STOPPED"


class PlayStateMachine:
    def __init__(self, freeze_duration_frames: int = 45):  # 3 seconds @ 15 FPS
        self.state = PlayStateEnum.PLAYING
        self.freeze_duration_frames = freeze_duration_frames
        self.frozen_frames_remaining = 0
        self.active_snapshot: Optional[EvidenceSnapshot] = None

        self.last_contact_estimate: Optional[Any] = None
        self.last_offside_result: Optional[Any] = None
        self.frozen_image: Optional[np.ndarray] = None

    def reset(self):
        self.state = PlayStateEnum.PLAYING
        self.frozen_frames_remaining = 0
        self.active_snapshot = None
        self.last_contact_estimate = None
        self.last_offside_result = None
        self.frozen_image = None

    def signal_active_involvement(self) -> Tuple[PlayStateEnum, bool, Optional[EvidenceSnapshot]]:
        """
        Signals that the player previously identified in an offside position has become
        actively involved in play (e.g., touched the ball or interfered with opponent).
        Transitions: OFFSIDE_POSITION_DETECTED -> ACTIVE_INVOLVEMENT -> OFFSIDE_OFFENCE -> PLAY_STOPPED.
        """
        if self.state in [PlayStateEnum.OFFSIDE_POSITION_DETECTED, PlayStateEnum.POSITION_EVALUATED]:
            self.state = PlayStateEnum.ACTIVE_INVOLVEMENT
            self.state = PlayStateEnum.OFFSIDE_OFFENCE
            self.state = PlayStateEnum.PLAY_STOPPED
            self.frozen_frames_remaining = self.freeze_duration_frames

            if self.last_contact_estimate is not None and self.last_offside_result is not None:
                contact_f = self.last_contact_estimate.frame_hat
                self.active_snapshot = EvidenceSnapshot(
                    frame_index=contact_f,
                    image=self.frozen_image if self.frozen_image is not None else np.zeros((100, 100, 3), dtype=np.uint8),
                    decision=self.last_offside_result.decision,
                    margin_val=self.last_offside_result.margin_val,
                    margin_unit=self.last_offside_result.margin_unit,
                    confidence=getattr(self.last_contact_estimate, "confidence", 1.0),
                    ball_evidence_state="OBSERVED" if getattr(self.last_contact_estimate, "observed_ball", True) else "PREDICTED",
                    offside_line_endpoints=self.last_offside_result.offside_line,
                    attacker_line_endpoints=self.last_offside_result.attacker_line,
                    passer_track_id=getattr(self.last_contact_estimate, "passer_track_id", None),
                    offside_boundary_world_x=getattr(self.last_offside_result, "offside_boundary_world_x", None),
                    explanation=self.last_offside_result.explanation,
                    homography_valid=self.last_offside_result.homography_valid,
                    boundary_status=getattr(self.last_offside_result, "boundary_status", "DECISIVE"),
                    empirical_error_budget_m=getattr(self.last_offside_result, "empirical_error_budget_m", 0.59),
                    geometry_status=getattr(self.last_offside_result, "geometry_status", "NORMAL")
                )
            return PlayStateEnum.PLAY_STOPPED, True, self.active_snapshot

        return self.state, False, None

    def signal_passive_play(self) -> Tuple[PlayStateEnum, bool, Optional[EvidenceSnapshot]]:
        """
        Signals that the player in an offside position remained passive and never became involved.
        Under Law 11, no offence occurred; play continues uninterrupted.
        """
        self.state = PlayStateEnum.PLAYING
        self.active_snapshot = None
        return PlayStateEnum.PLAYING, False, None

    def update(
        self,
        frame_index: int,
        frame_image: np.ndarray,
        pass_candidate: Optional[Any],
        contact_estimate: Optional[Any],
        offside_result: Optional[Any],
        active_involvement: bool = True
    ) -> Tuple[PlayStateEnum, bool, Optional[EvidenceSnapshot]]:
        """
        Updates the match play state machine.
        Returns:
          (current_state, should_freeze_video, active_snapshot)
        """
        # If currently holding frozen evidence
        if self.state == PlayStateEnum.PLAY_STOPPED:
            if self.frozen_frames_remaining > 0:
                self.frozen_frames_remaining -= 1
                return PlayStateEnum.PLAY_STOPPED, True, self.active_snapshot
            else:
                # Freeze duration expired; return to PLAYING for subsequent gameplay
                self.state = PlayStateEnum.PLAYING
                return PlayStateEnum.PLAYING, False, None

        # 1. State transition: PASS_CANDIDATE
        if pass_candidate is not None and self.state == PlayStateEnum.PLAYING:
            self.state = PlayStateEnum.PASS_CANDIDATE

        # 2. State transition: CONTACT_ESTIMATED
        if contact_estimate is not None:
            self.state = PlayStateEnum.CONTACT_ESTIMATED
            self.last_contact_estimate = contact_estimate
            self.frozen_image = frame_image.copy() if frame_image is not None else None

        # 3. State transition: POSITION_EVALUATED at t*
        if offside_result is not None and self.last_contact_estimate is not None:
            self.last_offside_result = offside_result
            self.state = PlayStateEnum.POSITION_EVALUATED

            # Check if an offside position was determined
            if offside_result.decision == "OFFSIDE":
                # Distinct State: OFFSIDE_POSITION_DETECTED
                self.state = PlayStateEnum.OFFSIDE_POSITION_DETECTED

                # Law 11: Offside position is not an offence unless actively involved
                if active_involvement:
                    return self.signal_active_involvement()
                else:
                    return self.signal_passive_play()
            else:
                # ONSIDE position -> play continues uninterrupted
                self.state = PlayStateEnum.PLAYING
                return PlayStateEnum.PLAYING, False, None

        return self.state, False, None
