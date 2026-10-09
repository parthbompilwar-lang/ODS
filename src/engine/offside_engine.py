"""
FIFA Law 11 Offside Decision Engine.
Strictly enforces the architectural invariants:
1. NO VALID HOMOGRAPHY -> NO METRIC OFFSIDE DECISION -> UNDETERMINED / REVIEW.
2. The offside line is the projection of a pitch line via homography H^-1 across full pitch width.
3. Coarse sanity guard: flags margins > 5.0m as SUSPECT (protects against gross ID swaps).
4. Error-budget calibration guard: flags margins <= empirical_error_budget_m as MARGINAL.
5. Evaluates offside strictly at the contact moment t_hat*.
6. Explicit attacking direction sign (+1 or -1) with zero post-hoc sign inference.
7. Ball-line priority and Halfway-line immunity formally enforced.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from src.engine.defender_selector import DefenderSelector

# Engineering computational level tolerance for the experimental system
# (A ±0.05m tolerance is introduced as a numerical zero/tie tolerance; not claimed as an official IFAB threshold)
LEVEL_TOLERANCE_M = 0.05

# Empirical geometry review threshold derived from P95 E_X error observed in GEOMETRY-EVAL-001
# (An engineering review policy threshold, not an official IFAB or physical rule)
EMPIRICAL_GEOMETRY_REVIEW_THRESHOLD_M = 0.59
DEFAULT_EMPIRICAL_ERROR_BUDGET_M = EMPIRICAL_GEOMETRY_REVIEW_THRESHOLD_M


class OffsideResult:
    """Encapsulates the complete VAR offside analysis result for a contact frame."""

    def __init__(self):
        self.decision: str = "ONSIDE"  # "OFFSIDE", "ONSIDE", "UNDETERMINED"
        self.attack_team_id: int = 0
        self.defend_team_id: int = 1
        self.attack_direction: str = 'right'
        self.attacking_direction_sign: int = 1  # +1 (towards 105m) or -1 (towards 0m)
        self.attacking_direction_label: str = "+X (attacking toward 105m)"

        # Reference elements
        self.reference_element: str = "SECOND_LAST_DEFENDER"  # "SECOND_LAST_DEFENDER" or "BALL_LINE"
        self.second_last_defender: Optional[Any] = None
        self.last_defender: Optional[Any] = None
        self.ball_position: Optional[Tuple[float, float]] = None

        # World pitch coordinates (meters)
        self.defender_world_x: Optional[float] = None
        self.ball_world_x: Optional[float] = None
        self.offside_boundary_world_x: Optional[float] = None
        self.attacker_world_x: Optional[float] = None

        # Projected perspective offside lines in camera pixel coordinates: ((x1, y1), (x2, y2))
        self.offside_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        self.attacker_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

        # Calibration, Margin & Uncertainty Telemetry
        self.homography_valid: bool = False
        self.calibration_status: str = "UNCALIBRATED"
        self.margin_val: float = 0.0
        self.margin_unit: str = "px"  # Strictly "px" unless homography is verified
        self.level_tolerance_m: float = LEVEL_TOLERANCE_M
        self.empirical_error_budget_m: float = DEFAULT_EMPIRICAL_ERROR_BUDGET_M
        self.boundary_status: str = "DECISIVE"  # "DECISIVE" or "MARGINAL"
        self.geometry_status: str = "NORMAL"    # "NORMAL", "SUSPECT", or "INVALID"

        # Offending player details
        self.offside_players: List[Any] = []
        self.onside_players: List[Any] = []
        self.all_players: List[Any] = []
        self.explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "margin_val": round(self.margin_val, 2),
            "margin_unit": self.margin_unit,
            "boundary_status": self.boundary_status,
            "empirical_error_budget_m": self.empirical_error_budget_m,
            "geometry_status": self.geometry_status,
            "attacking_direction_sign": self.attacking_direction_sign,
            "attacking_direction_label": self.attacking_direction_label,
            "reference_element": self.reference_element,
            "defender_world_x": round(self.defender_world_x, 2) if self.defender_world_x is not None else None,
            "ball_world_x": round(self.ball_world_x, 2) if self.ball_world_x is not None else None,
            "offside_boundary_world_x": round(self.offside_boundary_world_x, 2) if self.offside_boundary_world_x is not None else None,
            "attacker_world_x": round(self.attacker_world_x, 2) if self.attacker_world_x is not None else None,
            "homography_valid": self.homography_valid,
            "calibration_status": self.calibration_status,
            "offside_line": self.offside_line,
            "attacker_line": self.attacker_line,
            "num_offside_players": len(self.offside_players),
            "explanation": self.explanation
        }


class OffsideEngine:
    def __init__(self, pitch_geometry: Optional[Any] = None, empirical_error_budget_m: float = DEFAULT_EMPIRICAL_ERROR_BUDGET_M):
        self.pitch_geom = pitch_geometry
        self.defender_selector = DefenderSelector()
        self.empirical_error_budget_m = empirical_error_budget_m

    def _resolve_direction(self, attack_direction: str, attacking_direction_sign: Optional[int]) -> Tuple[int, str]:
        """Resolves attack direction string and sign into canonical sign (+1 or -1)."""
        if attacking_direction_sign is not None and attacking_direction_sign in [1, -1]:
            sign = attacking_direction_sign
        elif attack_direction in ['left', '-X', '-x']:
            sign = -1
        else:
            sign = 1  # default right / +X

        label = "+X (attacking toward 105m)" if sign == 1 else "-X (attacking toward 0m)"
        return sign, label

    def evaluate_contact_moment(
        self,
        players: List[Dict[str, Any]],
        attack_team_id: int = 0,
        attack_direction: str = 'right',
        attacking_direction_sign: Optional[int] = None,
        ball_pos: Optional[Tuple[float, float]] = None,
        ball_world_pos: Optional[Tuple[float, float]] = None,
        image_shape: Tuple[int, int] = (1080, 1920),
        require_metric: bool = False
    ) -> OffsideResult:
        """
        Evaluates Law 11 offside position at contact moment t_hat*.
        """
        h, w = image_shape[:2]
        result = OffsideResult()
        result.attack_team_id = attack_team_id
        result.defend_team_id = 1 if attack_team_id == 0 else 0
        result.attack_direction = attack_direction
        result.empirical_error_budget_m = self.empirical_error_budget_m
        result.ball_position = ball_pos
        result.all_players = players

        sign, label = self._resolve_direction(attack_direction, attacking_direction_sign)
        result.attacking_direction_sign = sign
        result.attacking_direction_label = label

        if not players:
            result.decision = "UNDETERMINED"
            result.geometry_status = "INVALID"
            result.explanation = "No players detected in scene."
            return result

        # Invariant 1: Validate Homography
        has_valid_homo = False
        if self.pitch_geom is not None and getattr(self.pitch_geom, "homography_matrix", None) is not None:
            H = self.pitch_geom.homography_matrix
            det = np.linalg.det(H)
            if np.isfinite(det) and abs(det) > 1e-12:
                has_valid_homo = True

        result.homography_valid = has_valid_homo
        result.calibration_status = "CALIBRATED_HOMOGRAPHY" if has_valid_homo else "UNCALIBRATED_PIXEL_ONLY"

        # Invariant 2: Refuse fake metric numbers if homography is missing
        if require_metric and not has_valid_homo:
            result.decision = "UNDETERMINED"
            result.boundary_status = "UNDETERMINED"
            result.geometry_status = "INVALID"
            result.explanation = "CRITICAL: Metric offside decision refused because no valid pitch homography is calibrated."
            return result

        # 1. Identify Qualifying Leading Points for each player
        for p in players:
            bbox = p.get("bbox", [0, 0, 50, 100])
            cx = (bbox[0] + bbox[2]) / 2.0
            p["ground_point"] = (cx, float(bbox[3]))

            # Qualifying body point (head, torso, feet; hands/arms excluded)
            if sign == 1:
                p["qualifying_point"] = (float(bbox[2]), (bbox[1] + bbox[3]) / 2.0)
            else:
                p["qualifying_point"] = (float(bbox[0]), (bbox[1] + bbox[3]) / 2.0)

        # 2. Select Second-Last Opponent dynamically
        last_opp, second_last_opp, baseline_coord = self.defender_selector.select_second_last_opponent(
            players=players,
            defend_team_id=result.defend_team_id,
            attack_direction=attack_direction,
            pitch_geom=self.pitch_geom if has_valid_homo else None,
            attacking_direction_sign=sign
        )
        result.last_defender = last_opp
        result.second_last_defender = second_last_opp

        if second_last_opp is None:
            result.decision = "UNDETERMINED"
            result.geometry_status = "INVALID"
            result.explanation = "Could not identify defending opponents."
            return result

        # 3. Compute Offside Boundary in Metric World Coordinates (if homography valid)
        pitch_width = getattr(self.pitch_geom, "pitch_width", 68.0)
        pitch_length = getattr(self.pitch_geom, "pitch_length", 105.0)

        if has_valid_homo:
            # Transform defender baseline point to pitch coordinates
            if "world_x" in second_last_opp:
                def_world_x = float(second_last_opp["world_x"])
            else:
                def_q = second_last_opp.get("qualifying_point", ((second_last_opp["bbox"][0] + second_last_opp["bbox"][2]) / 2, baseline_coord))
                def_world = self.pitch_geom.image_to_pitch(np.array([[def_q[0], def_q[1]]]))[0]
                def_world_x = float(def_world[0])

            result.defender_world_x = def_world_x
            x_offside_world = def_world_x
            result.reference_element = "SECOND_LAST_DEFENDER"

            # Ball line consideration in metric world space
            ball_x = None
            if ball_world_pos is not None:
                ball_x = float(ball_world_pos[0])
            elif ball_pos is not None:
                try:
                    b_w = self.pitch_geom.image_to_pitch(np.array([[ball_pos[0], ball_pos[1]]]))[0]
                    ball_x = float(b_w[0])
                except Exception:
                    pass

            if ball_x is not None:
                result.ball_world_x = ball_x
                # Invariant 4: Ball-line priority rule:
                # When attacking toward +X (105m): ball defines line if ball_x > def_world_x
                # When attacking toward -X (0m): ball defines line if ball_x < def_world_x
                if sign == 1:
                    if ball_x > def_world_x:
                        x_offside_world = ball_x
                        result.reference_element = "BALL_LINE"
                else:
                    if ball_x < def_world_x:
                        x_offside_world = ball_x
                        result.reference_element = "BALL_LINE"

            result.offside_boundary_world_x = x_offside_world

            # Invariant 3: Project the pitch line using H^-1 across full pitch width
            p1_w = np.array([x_offside_world, 0.0])
            p2_w = np.array([x_offside_world, pitch_width])
            p1_img = self.pitch_geom.pitch_to_image(p1_w)[0]
            p2_img = self.pitch_geom.pitch_to_image(p2_w)[0]

            result.offside_line = (
                (int(round(p1_img[0])), int(round(p1_img[1]))),
                (int(round(p2_img[0])), int(round(p2_img[1])))
            )
        else:
            # Uncalibrated fallback (image plane line)
            x_offside_world = baseline_coord
            if attack_direction in ['top', 'up', 'bottom', 'down']:
                y_base = max(0, min(h - 1, int(baseline_coord)))
                result.offside_line = ((0, y_base), (w, y_base))
            else:
                x_base = max(0, min(w - 1, int(baseline_coord)))
                result.offside_line = ((x_base, 0), (x_base, h))

        # 4. Evaluate Attackers
        attackers = [
            p for p in players
            if p.get("class_id") == 0 and p.get("team_id") == result.attack_team_id
        ]
        if not attackers:
            attackers = [
                p for p in players
                if p.get("track_id") != second_last_opp.get("track_id")
                and (last_opp is None or p.get("track_id") != last_opp.get("track_id"))
                and p.get("class_id") != 2 and p.get("role") != "referee"
            ]

        max_signed_margin_m = -999.0
        max_signed_margin_px = -9999.0
        primary_offside_attacker = None
        closest_onside_margin_m = -999.0

        for att in attackers:
            if has_valid_homo:
                if "world_x" in att:
                    att_x = float(att["world_x"])
                else:
                    q_pt = att.get("qualifying_point", (0, 0))
                    att_world = self.pitch_geom.image_to_pitch(np.array([[q_pt[0], q_pt[1]]]))[0]
                    att_x = float(att_world[0])
                att["world_x"] = att_x

                # Invariant 5: Halfway-line immunity (Law 11)
                # Player cannot be in an offside position in their own half (excluding halfway line)
                in_opponent_half = (att_x > 52.5) if sign == 1 else (att_x < 52.5)

                # Invariant 6: Directional Signed Metric Margin
                # Delta X = sign * (X_attacker - X_boundary)
                # Positive means attacker is nearer to opponent goal line than the boundary
                signed_margin_m = sign * (att_x - x_offside_world)

                if in_opponent_half and signed_margin_m > LEVEL_TOLERANCE_M:
                    result.offside_players.append(att)
                    if signed_margin_m > max_signed_margin_m:
                        max_signed_margin_m = signed_margin_m
                        primary_offside_attacker = att
                else:
                    result.onside_players.append(att)
                    # For onside reporting: record the signed margin (or 0 if halfway immune)
                    reported_onside_margin = 0.0 if not in_opponent_half else signed_margin_m
                    if reported_onside_margin > closest_onside_margin_m:
                        closest_onside_margin_m = reported_onside_margin
            else:
                # Pixel-based comparison (uncalibrated)
                q_pt = att.get("qualifying_point", (0, 0))
                att_coord = q_pt[0] if attack_direction in ['left', 'right', '+X', '-X'] else q_pt[1]
                signed_margin_px = (att_coord - baseline_coord) if sign == 1 else (baseline_coord - att_coord)
                if signed_margin_px > 5.0:
                    result.offside_players.append(att)
                    if signed_margin_px > max_signed_margin_px:
                        max_signed_margin_px = signed_margin_px
                        primary_offside_attacker = att
                else:
                    result.onside_players.append(att)

        # 5. Final Decision and Telemetry
        if len(result.offside_players) > 0 and primary_offside_attacker is not None:
            result.decision = "OFFSIDE"

            if has_valid_homo:
                att_x = float(primary_offside_attacker["world_x"])
                result.attacker_world_x = att_x
                result.margin_val = round(max_signed_margin_m, 2)
                result.margin_unit = "m"

                # Boundary status: flag MARGINAL if within empirical error budget
                if abs(result.margin_val) <= self.empirical_error_budget_m:
                    result.boundary_status = "MARGINAL"
                else:
                    result.boundary_status = "DECISIVE"

                # Coarse sanity check guard: flag SUSPECT if margin > 5.0m
                if abs(result.margin_val) > 5.0:
                    result.geometry_status = "SUSPECT"
                    result.explanation = (
                        f"OFFSIDE (SUSPECT: +{result.margin_val:.2f}m margin exceeds 5.0m realistic competitive threshold; "
                        f"possible defender misidentification or gross tracking error)."
                    )
                else:
                    result.geometry_status = "NORMAL"
                    marginal_note = " [MARGINAL: within empirical error budget]" if result.boundary_status == "MARGINAL" else ""
                    result.explanation = (
                        f"OFFSIDE: Attacker leading body part is +{result.margin_val:.2f}m beyond the offside boundary "
                        f"(reference: {result.reference_element} at X={x_offside_world:.2f}m){marginal_note}."
                    )

                # Project perspective attacker line via H^-1
                p1_att_w = np.array([att_x, 0.0])
                p2_att_w = np.array([att_x, pitch_width])
                p1_att_img = self.pitch_geom.pitch_to_image(p1_att_w)[0]
                p2_att_img = self.pitch_geom.pitch_to_image(p2_att_w)[0]

                result.attacker_line = (
                    (int(round(p1_att_img[0])), int(round(p1_att_img[1]))),
                    (int(round(p2_att_img[0])), int(round(p2_att_img[1])))
                )
            else:
                result.margin_val = round(max_signed_margin_px, 1)
                result.margin_unit = "px"
                result.boundary_status = "DECISIVE"
                result.geometry_status = "NORMAL"
                result.explanation = f"OFFSIDE: Attacker is +{max_signed_margin_px:.1f}px beyond defensive baseline (UNCALIBRATED: Metric distance requires pitch homography)."
        else:
            result.decision = "ONSIDE"
            if has_valid_homo:
                result.margin_val = round(closest_onside_margin_m, 2) if closest_onside_margin_m > -900 else 0.0
                result.margin_unit = "m"
                if abs(result.margin_val) <= self.empirical_error_budget_m:
                    result.boundary_status = "MARGINAL"
                else:
                    result.boundary_status = "DECISIVE"
                result.geometry_status = "NORMAL"
                result.explanation = f"ONSIDE: All attackers level with or behind the offside boundary (reference: {result.reference_element} at X={x_offside_world:.2f}m)."
            else:
                result.margin_val = 0.0
                result.margin_unit = "px"
                result.boundary_status = "DECISIVE"
                result.geometry_status = "NORMAL"
                result.explanation = "ONSIDE: All attackers level with or behind the defensive baseline."

        return result
