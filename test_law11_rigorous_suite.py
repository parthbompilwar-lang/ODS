"""
Comprehensive Law 11 Rigorous Verification and Benchmark Suite.
Divided into three distinct layers as required:
  Layer 1: LAW11-UNIT-001    - Pure mathematical edge cases (symmetry, halfway immunity, ball-line, GK as opponent, level tolerance, invalid geometry)
  Layer 2: LAW11-EVAL-001    - 25 manually verified scenarios covering seven project-defined geometric/state cases
  Layer 3: LAW11-STATE-001   - State machine transition integrity (offside position vs offside offence, passive play, freeze duration)
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple

from src.geometry.pitch_geometry import PitchGeometry
from src.engine.offside_engine import OffsideEngine, OffsideResult, LEVEL_TOLERANCE_M
from src.engine.defender_selector import DefenderSelector
from src.engine.play_state import PlayStateMachine, PlayStateEnum
from src.engine.evidence_snapshot import EvidenceSnapshot


def create_calibrated_pitch_geometry() -> PitchGeometry:
    """Creates a validated PitchGeometry using the 4 fit landmarks from GEOMETRY-DEBUG-002."""
    pitch_geom = PitchGeometry(pitch_length=105.0, pitch_width=68.0)
    src_pixels = np.array([
        [560.0, 480.0],
        [1980.0, 480.0],
        [2480.0, 1180.0],
        [120.0, 1180.0]
    ], dtype=np.float32)

    dst_world = np.array([
        [16.5, 13.84],
        [16.5, 54.16],
        [52.5, 68.00],
        [52.5, 0.00]
    ], dtype=np.float32)

    pitch_geom.set_homography(src_pixels, dst_world)
    return pitch_geom


# =============================================================================
# LAYER 1: LAW11-UNIT-001 (Pure Mathematical Invariant Tests)
# =============================================================================
def run_law11_unit_001(pitch_geom: PitchGeometry) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("LAYER 1: LAW11-UNIT-001 (PURE MATHEMATICAL INVARIANT TESTS)")
    print("=" * 75)

    engine = OffsideEngine(pitch_geometry=pitch_geom, empirical_error_budget_m=0.27)
    passed_tests = 0
    total_tests = 7
    unit_results = []

    # Test 1: +X / -X Mathematical Symmetry
    # Attacking +X: Attacker at 75m, Def at 70m -> +5m offside
    # Attacking -X: Attacker at 30m, Def at 35m -> +5m offside
    p_att_plus = {"track_id": 1, "class_id": 0, "team_id": 0, "world_x": 75.0, "bbox": [0, 0, 50, 100]}
    p_def1_plus = {"track_id": 2, "class_id": 0, "team_id": 1, "world_x": 70.0, "bbox": [0, 0, 50, 100]}
    p_def2_plus = {"track_id": 3, "class_id": 1, "team_id": 1, "world_x": 95.0, "bbox": [0, 0, 50, 100]}
    res_plus = engine.evaluate_contact_moment([p_att_plus, p_def1_plus, p_def2_plus], attack_team_id=0, attacking_direction_sign=1)

    p_att_minus = {"track_id": 4, "class_id": 0, "team_id": 0, "world_x": 30.0, "bbox": [0, 0, 50, 100]}
    p_def1_minus = {"track_id": 5, "class_id": 0, "team_id": 1, "world_x": 35.0, "bbox": [0, 0, 50, 100]}
    p_def2_minus = {"track_id": 6, "class_id": 1, "team_id": 1, "world_x": 10.0, "bbox": [0, 0, 50, 100]}
    res_minus = engine.evaluate_contact_moment([p_att_minus, p_def1_minus, p_def2_minus], attack_team_id=0, attacking_direction_sign=-1)

    sym_passed = (res_plus.decision == "OFFSIDE" and res_plus.margin_val == 5.0 and
                  res_minus.decision == "OFFSIDE" and res_minus.margin_val == 5.0)
    passed_tests += int(sym_passed)
    unit_results.append({"test": "Directional Symmetry (+X vs -X)", "passed": sym_passed, "res_plus": res_plus.margin_val, "res_minus": res_minus.margin_val})
    print(f"  [1/7] Directional Symmetry:          {'PASSED' if sym_passed else 'FAILED'} (+X: {res_plus.margin_val:+.2f}m, -X: {res_minus.margin_val:+.2f}m)")

    # Test 2: Halfway-Line Immunity
    # Attacking +X: Attacker at 48m (own half), Def at 45m -> Margin would be +3m, but must be ONSIDE
    p_att_half = {"track_id": 10, "class_id": 0, "team_id": 0, "world_x": 48.0, "bbox": [0, 0, 50, 100]}
    p_def_half1 = {"track_id": 11, "class_id": 0, "team_id": 1, "world_x": 45.0, "bbox": [0, 0, 50, 100]}
    p_def_half2 = {"track_id": 12, "class_id": 1, "team_id": 1, "world_x": 80.0, "bbox": [0, 0, 50, 100]}
    res_half = engine.evaluate_contact_moment([p_att_half, p_def_half1, p_def_half2], attack_team_id=0, attacking_direction_sign=1)
    half_passed = (res_half.decision == "ONSIDE")
    passed_tests += int(half_passed)
    unit_results.append({"test": "Halfway-Line Immunity (Own Half)", "passed": half_passed, "decision": res_half.decision})
    print(f"  [2/7] Halfway-Line Immunity:         {'PASSED' if half_passed else 'FAILED'} (Decision: {res_half.decision})")

    # Test 3: Ball-Line Priority
    # Ball is ahead of 2nd-last defender (X_ball = 85m, X_def = 80m). Offside boundary must be 85m.
    # Attacker at 87m -> Margin = +2m OFFSIDE. Reference must be BALL_LINE.
    p_att_ball = {"track_id": 20, "class_id": 0, "team_id": 0, "world_x": 87.0, "bbox": [0, 0, 50, 100]}
    p_def_ball1 = {"track_id": 21, "class_id": 0, "team_id": 1, "world_x": 80.0, "bbox": [0, 0, 50, 100]}
    p_def_ball2 = {"track_id": 22, "class_id": 1, "team_id": 1, "world_x": 95.0, "bbox": [0, 0, 50, 100]}
    res_ball = engine.evaluate_contact_moment(
        [p_att_ball, p_def_ball1, p_def_ball2],
        attack_team_id=0, attacking_direction_sign=1,
        ball_world_pos=(85.0, 34.0)
    )
    ball_passed = (res_ball.decision == "OFFSIDE" and res_ball.reference_element == "BALL_LINE" and res_ball.margin_val == 2.0)
    passed_tests += int(ball_passed)
    unit_results.append({"test": "Ball-Line Priority", "passed": ball_passed, "ref": res_ball.reference_element, "margin": res_ball.margin_val})
    print(f"  [3/7] Ball-Line Priority:            {'PASSED' if ball_passed else 'FAILED'} (Ref: {res_ball.reference_element}, Margin: {res_ball.margin_val:+.2f}m)")

    # Test 4: Second-Last Defender Dynamic Selection (Non-hardcoded GK)
    # Goalkeeper (class 1) is advanced at 65m. Two outfield defenders at 90m and 98m.
    # 2nd-last defender must be the outfield defender at 90m (X_boundary = 90.0m).
    p_att_gk = {"track_id": 30, "class_id": 0, "team_id": 0, "world_x": 88.0, "bbox": [0, 0, 50, 100]}
    p_gk_adv = {"track_id": 31, "class_id": 1, "team_id": 1, "world_x": 65.0, "bbox": [0, 0, 50, 100]}
    p_out_def1 = {"track_id": 32, "class_id": 0, "team_id": 1, "world_x": 90.0, "bbox": [0, 0, 50, 100]}
    p_out_def2 = {"track_id": 33, "class_id": 0, "team_id": 1, "world_x": 98.0, "bbox": [0, 0, 50, 100]}
    res_gk = engine.evaluate_contact_moment([p_att_gk, p_gk_adv, p_out_def1, p_out_def2], attack_team_id=0, attacking_direction_sign=1)
    gk_passed = (res_gk.decision == "ONSIDE" and res_gk.second_last_defender["track_id"] == 32 and res_gk.offside_boundary_world_x == 90.0)
    passed_tests += int(gk_passed)
    unit_results.append({"test": "Dynamic 2nd-Last Defender (Advanced GK)", "passed": gk_passed, "2nd_last_id": getattr(res_gk.second_last_defender, "get", lambda k: None)("track_id")})
    print(f"  [4/7] Advanced GK / 2nd-Last Def:    {'PASSED' if gk_passed else 'FAILED'} (Boundary: {res_gk.offside_boundary_world_x:.1f}m, 2nd-last track ID: 32)")

    # Test 5: Computational Level Tolerance (LEVEL_TOLERANCE_M = 0.05m)
    # Attacker at 80.03m, Def at 80.00m -> Margin = +0.03m <= 0.05m -> Must be ONSIDE
    p_att_lvl = {"track_id": 40, "class_id": 0, "team_id": 0, "world_x": 80.03, "bbox": [0, 0, 50, 100]}
    p_def_lvl1 = {"track_id": 41, "class_id": 0, "team_id": 1, "world_x": 80.00, "bbox": [0, 0, 50, 100]}
    p_def_lvl2 = {"track_id": 42, "class_id": 1, "team_id": 1, "world_x": 95.00, "bbox": [0, 0, 50, 100]}
    res_lvl = engine.evaluate_contact_moment([p_att_lvl, p_def_lvl1, p_def_lvl2], attack_team_id=0, attacking_direction_sign=1)
    lvl_passed = (res_lvl.decision == "ONSIDE")
    passed_tests += int(lvl_passed)
    unit_results.append({"test": "Computational Level Tolerance (0.05m)", "passed": lvl_passed, "decision": res_lvl.decision})
    print(f"  [5/7] Level Tolerance (+0.03m):     {'PASSED' if lvl_passed else 'FAILED'} (Decision: {res_lvl.decision})")

    # Test 6: Invariant 1 - Missing Homography Refuses Metric Output
    engine_uncalib = OffsideEngine(pitch_geometry=None)
    res_uncalib = engine_uncalib.evaluate_contact_moment(
        [p_att_plus, p_def1_plus, p_def2_plus],
        attack_team_id=0, attacking_direction_sign=1, require_metric=True
    )
    inv_passed = (res_uncalib.decision == "UNDETERMINED" and res_uncalib.geometry_status == "INVALID")
    passed_tests += int(inv_passed)
    unit_results.append({"test": "Invariant 1: Uncalibrated Refuses Metric", "passed": inv_passed, "decision": res_uncalib.decision})
    print(f"  [6/7] Invariant: No Homo Refusal:    {'PASSED' if inv_passed else 'FAILED'} (Decision: {res_uncalib.decision})")

    # Test 7: Coarse Sanity Guard (SUSPECT on Margin > 5.0m)
    p_att_gross = {"track_id": 50, "class_id": 0, "team_id": 0, "world_x": 92.0, "bbox": [0, 0, 50, 100]}
    p_def_gross1 = {"track_id": 51, "class_id": 0, "team_id": 1, "world_x": 80.0, "bbox": [0, 0, 50, 100]}
    p_def_gross2 = {"track_id": 52, "class_id": 1, "team_id": 1, "world_x": 98.0, "bbox": [0, 0, 50, 100]}
    res_gross = engine.evaluate_contact_moment([p_att_gross, p_def_gross1, p_def_gross2], attack_team_id=0, attacking_direction_sign=1)
    sanity_passed = (res_gross.decision == "OFFSIDE" and res_gross.geometry_status == "SUSPECT" and res_gross.margin_val == 12.0)
    passed_tests += int(sanity_passed)
    unit_results.append({"test": "Sanity Guard (Margin > 5m SUSPECT)", "passed": sanity_passed, "status": res_gross.geometry_status})
    print(f"  [7/7] Sanity Guard (Margin +12m):    {'PASSED' if sanity_passed else 'FAILED'} (Status: {res_gross.geometry_status})")

    pass_rate_pct = round(100.0 * passed_tests / total_tests, 1)
    print(f"\nLAYER 1 SUMMARY: {passed_tests}/{total_tests} ({pass_rate_pct}%) unit tests passed.")

    return {
        "benchmark": "LAW11-UNIT-001",
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "pass_rate_pct": pass_rate_pct,
        "details": unit_results
    }


# =============================================================================
# LAYER 2: LAW11-EVAL-001 (25 Manually Verified Scenes)
# =============================================================================
def run_law11_eval_001(pitch_geom: PitchGeometry) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("LAYER 2: LAW11-EVAL-001 (25 MANUALLY VERIFIED BENCHMARK SCENES)")
    print("Covering seven project-defined geometric/state cases")
    print("=" * 75)

    engine = OffsideEngine(pitch_geometry=pitch_geom, empirical_error_budget_m=0.59)

    # Define 25 manually verified scenarios covering 7 project-defined categories
    scenarios = [
        # --- Category A: Clearly Onside (4 scenes) ---
        {
            "id": "SCENE-A01", "category": "A_CLEARLY_ONSIDE", "sign": 1,
            "att_x": 65.0, "def_x": 75.0, "gk_x": 95.0, "ball_x": 60.0,
            "gt_decision": "ONSIDE", "gt_margin_m": -10.0, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "DECISIVE"
        },
        {
            "id": "SCENE-A02", "category": "A_CLEARLY_ONSIDE", "sign": 1,
            "att_x": 80.0, "def_x": 82.5, "gk_x": 98.0, "ball_x": 70.0,
            "gt_decision": "ONSIDE", "gt_margin_m": -2.5, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "DECISIVE"
        },
        {
            "id": "SCENE-A03", "category": "A_CLEARLY_ONSIDE", "sign": -1,
            "att_x": 38.0, "def_x": 25.0, "gk_x": 5.0, "ball_x": 42.0,
            "gt_decision": "ONSIDE", "gt_margin_m": -13.0, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "DECISIVE"
        },
        {
            "id": "SCENE-A04", "category": "A_CLEARLY_ONSIDE", "sign": -1,
            "att_x": 30.0, "def_x": 28.0, "gk_x": 4.0, "ball_x": 35.0,
            "gt_decision": "ONSIDE", "gt_margin_m": -2.0, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "DECISIVE"
        },

        # --- Category B: Clearly Offside (4 scenes) ---
        {
            "id": "SCENE-B01", "category": "B_CLEARLY_OFFSIDE", "sign": 1,
            "att_x": 85.0, "def_x": 82.0, "gk_x": 96.0, "ball_x": 75.0,
            "gt_decision": "OFFSIDE", "gt_margin_m": 3.0, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "DECISIVE"
        },
        {
            "id": "SCENE-B02", "category": "B_CLEARLY_OFFSIDE", "sign": 1,
            "att_x": 92.0, "def_x": 90.5, "gk_x": 100.0, "ball_x": 80.0,
            "gt_decision": "OFFSIDE", "gt_margin_m": 1.5, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "DECISIVE"
        },
        {
            "id": "SCENE-B03", "category": "B_CLEARLY_OFFSIDE", "sign": -1,
            "att_x": 18.0, "def_x": 22.0, "gk_x": 4.0, "ball_x": 25.0,
            "gt_decision": "OFFSIDE", "gt_margin_m": 4.0, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "DECISIVE"
        },
        {
            "id": "SCENE-B04", "category": "B_CLEARLY_OFFSIDE", "sign": -1,
            "att_x": 12.0, "def_x": 14.2, "gk_x": 3.5, "ball_x": 18.0,
            "gt_decision": "OFFSIDE", "gt_margin_m": 2.2, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "DECISIVE"
        },

        # --- Category C: Level Onside (Within Engineering Level Tolerance) (3 scenes) ---
        {
            "id": "SCENE-C01", "category": "C_LEVEL_ONSIDE", "sign": 1,
            "att_x": 78.02, "def_x": 78.00, "gk_x": 95.0, "ball_x": 70.0,
            "gt_decision": "ONSIDE", "gt_margin_m": 0.02, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "MARGINAL"
        },
        {
            "id": "SCENE-C02", "category": "C_LEVEL_ONSIDE", "sign": 1,
            "att_x": 84.00, "def_x": 84.00, "gk_x": 98.0, "ball_x": 75.0,
            "gt_decision": "ONSIDE", "gt_margin_m": 0.00, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "MARGINAL"
        },
        {
            "id": "SCENE-C03", "category": "C_LEVEL_ONSIDE", "sign": -1,
            "att_x": 21.97, "def_x": 22.00, "gk_x": 5.0, "ball_x": 30.0,
            "gt_decision": "ONSIDE", "gt_margin_m": 0.03, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "MARGINAL"
        },

        # --- Category D: Ball-Line Priority (4 scenes) ---
        {
            "id": "SCENE-D01", "category": "D_BALL_LINE_PRIORITY", "sign": 1,
            "att_x": 88.0, "def_x": 80.0, "gk_x": 96.0, "ball_x": 86.0,
            "gt_decision": "OFFSIDE", "gt_margin_m": 2.0, "gt_ref": "BALL_LINE", "gt_boundary_status": "DECISIVE"
        },
        {
            "id": "SCENE-D02", "category": "D_BALL_LINE_PRIORITY", "sign": 1,
            "att_x": 84.0, "def_x": 75.0, "gk_x": 98.0, "ball_x": 86.0,
            "gt_decision": "ONSIDE", "gt_margin_m": -2.0, "gt_ref": "BALL_LINE", "gt_boundary_status": "DECISIVE"
        },
        {
            "id": "SCENE-D03", "category": "D_BALL_LINE_PRIORITY", "sign": -1,
            "att_x": 15.0, "def_x": 25.0, "gk_x": 4.0, "ball_x": 18.0,
            "gt_decision": "OFFSIDE", "gt_margin_m": 3.0, "gt_ref": "BALL_LINE", "gt_boundary_status": "DECISIVE"
        },
        {
            "id": "SCENE-D04", "category": "D_BALL_LINE_PRIORITY", "sign": -1,
            "att_x": 20.0, "def_x": 28.0, "gk_x": 4.0, "ball_x": 18.0,
            "gt_decision": "ONSIDE", "gt_margin_m": -2.0, "gt_ref": "BALL_LINE", "gt_boundary_status": "DECISIVE"
        },

        # --- Category E: Advanced Goalkeeper / Non-GK Last Opponent (3 scenes) ---
        {
            "id": "SCENE-E01", "category": "E_ADVANCED_GOALKEEPER", "sign": 1,
            "att_x": 90.0, "def_x": 88.0, "gk_x": 75.0, "extra_def_x": 96.0, "ball_x": 70.0,
            "gt_decision": "OFFSIDE", "gt_margin_m": 2.0, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "DECISIVE"
        },
        {
            "id": "SCENE-E02", "category": "E_ADVANCED_GOALKEEPER", "sign": 1,
            "att_x": 82.0, "def_x": 70.0, "gk_x": 85.0, "extra_def_x": 98.0, "ball_x": 65.0,
            "gt_decision": "ONSIDE", "gt_margin_m": -3.0, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "DECISIVE"
        },
        {
            "id": "SCENE-E03", "category": "E_ADVANCED_GOALKEEPER", "sign": -1,
            "att_x": 12.0, "def_x": 15.0, "gk_x": 28.0, "extra_def_x": 4.0, "ball_x": 35.0,
            "gt_decision": "OFFSIDE", "gt_margin_m": 3.0, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "DECISIVE"
        },

        # --- Category F: Reversed Attacking Direction & Halfway (4 scenes) ---
        {
            "id": "SCENE-F01", "category": "F_DIRECTION_SYMMETRY", "sign": 1,
            "att_x": 75.0, "def_x": 70.0, "gk_x": 95.0, "ball_x": 60.0,
            "gt_decision": "OFFSIDE", "gt_margin_m": 5.0, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "DECISIVE"
        },
        {
            "id": "SCENE-F02", "category": "F_DIRECTION_SYMMETRY", "sign": -1,
            "att_x": 30.0, "def_x": 35.0, "gk_x": 10.0, "ball_x": 45.0,
            "gt_decision": "OFFSIDE", "gt_margin_m": 5.0, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "DECISIVE"
        },
        {
            "id": "SCENE-F03", "category": "F_DIRECTION_SYMMETRY", "sign": -1,
            "att_x": 75.0, "def_x": 70.0, "gk_x": 95.0, "ball_x": 60.0,
            "gt_decision": "ONSIDE", "gt_margin_m": -5.0, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "DECISIVE"
        },
        {
            "id": "SCENE-F04", "category": "F_DIRECTION_SYMMETRY", "sign": 1,
            "att_x": 48.0, "def_x": 45.0, "gk_x": 80.0, "ball_x": 40.0,
            "gt_decision": "ONSIDE", "gt_margin_m": 0.0, "gt_ref": "SECOND_LAST_DEFENDER", "gt_boundary_status": "MARGINAL"
        },

        # --- Category G: Invalid Homography Invariants (3 scenes) ---
        {
            "id": "SCENE-G01", "category": "G_INVALID_HOMOGRAPHY", "sign": 1,
            "att_x": 80.0, "def_x": 75.0, "gk_x": 95.0, "ball_x": 60.0,
            "use_homo": False, "require_metric": True,
            "gt_decision": "UNDETERMINED", "gt_status": "INVALID"
        },
        {
            "id": "SCENE-G02", "category": "G_INVALID_HOMOGRAPHY", "sign": 1,
            "att_x": 80.0, "def_x": 75.0, "gk_x": 95.0, "ball_x": 60.0,
            "singular_homo": True, "require_metric": True,
            "gt_decision": "UNDETERMINED", "gt_status": "INVALID"
        },
        {
            "id": "SCENE-G03", "category": "G_INVALID_HOMOGRAPHY", "sign": 1,
            "att_x": 80.0, "def_x": 75.0, "gk_x": 95.0, "ball_x": 60.0,
            "use_homo": False, "require_metric": False,
            "gt_decision": "OFFSIDE", "gt_unit": "px"
        }
    ]

    total_scenes = len(scenarios)
    correct_decisions = 0
    margin_errors = []
    scene_eval_records = []

    for sc in scenarios:
        # Build scene players
        players = [
            {"track_id": 100, "class_id": 0, "team_id": 0, "world_x": sc["att_x"], "bbox": [500, 500, 550, 600]},
            {"track_id": 200, "class_id": 0, "team_id": 1, "world_x": sc["def_x"], "bbox": [400, 500, 450, 600]},
            {"track_id": 300, "class_id": 1, "team_id": 1, "world_x": sc["gk_x"], "bbox": [300, 500, 350, 600]}
        ]
        if "extra_def_x" in sc:
            players.append({"track_id": 201, "class_id": 0, "team_id": 1, "world_x": sc["extra_def_x"], "bbox": [350, 500, 400, 600]})

        # Pick engine based on scenario invariants
        if sc.get("use_homo") is False:
            cur_engine = OffsideEngine(pitch_geometry=None, empirical_error_budget_m=0.27)
        elif sc.get("singular_homo"):
            sing_geom = PitchGeometry()
            sing_geom.homography_matrix = np.zeros((3, 3))
            cur_engine = OffsideEngine(pitch_geometry=sing_geom, empirical_error_budget_m=0.27)
        else:
            cur_engine = engine

        ball_world = (sc["ball_x"], 34.0) if "ball_x" in sc else None

        res = cur_engine.evaluate_contact_moment(
            players=players,
            attack_team_id=0,
            attacking_direction_sign=sc["sign"],
            ball_world_pos=ball_world,
            require_metric=sc.get("require_metric", False)
        )

        is_dec_correct = (res.decision == sc["gt_decision"])
        if is_dec_correct:
            correct_decisions += 1

        margin_err = None
        if "gt_margin_m" in sc and res.decision in ["OFFSIDE", "ONSIDE"] and res.margin_unit == "m":
            margin_err = abs(round(res.margin_val - sc["gt_margin_m"], 2))
            margin_errors.append(margin_err)

        rec = {
            "id": sc["id"],
            "category": sc["category"],
            "direction": "+X" if sc["sign"] == 1 else "-X",
            "gt_decision": sc["gt_decision"],
            "pred_decision": res.decision,
            "decision_correct": is_dec_correct,
            "pred_margin": res.margin_val,
            "pred_margin_unit": res.margin_unit,
            "boundary_status": res.boundary_status,
            "geometry_status": res.geometry_status,
            "reference_element": res.reference_element,
            "margin_error_m": margin_err
        }
        scene_eval_records.append(rec)

        status_flag = "PASS" if is_dec_correct else "FAIL"
        margin_str = f"{res.margin_val:+.2f}{res.margin_unit}" if res.decision != "UNDETERMINED" else "N/A"
        print(f"  [{status_flag}] {sc['id']:<10} | Cat: {sc['category']:<24} | GT: {sc['gt_decision']:<12} | Pred: {res.decision:<12} | Margin: {margin_str:<8} | Ref: {res.reference_element}")

    accuracy_pct = round(100.0 * correct_decisions / total_scenes, 1)
    mean_margin_err = float(np.mean(margin_errors)) if margin_errors else 0.0

    print(f"\nLAYER 2 SUMMARY:")
    print(f"  Total Scenarios:     {total_scenes}")
    print(f"  Decision Accuracy:   {correct_decisions}/{total_scenes} ({accuracy_pct}%)")
    print(f"  Mean Margin Error:   {mean_margin_err:.3f} m")

    return {
        "benchmark": "LAW11-EVAL-001",
        "total_scenarios": total_scenes,
        "correct_decisions": correct_decisions,
        "decision_accuracy_pct": accuracy_pct,
        "mean_margin_error_m": round(mean_margin_err, 3),
        "scene_results": scene_eval_records
    }


# =============================================================================
# LAYER 3: LAW11-STATE-001 (State Machine Distinction Tests)
# =============================================================================
def run_law11_state_001() -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("LAYER 3: LAW11-STATE-001 (STATE MACHINE DISTINCTION TESTS)")
    print("Verifying Offside Position != Offside Offence and Active vs Passive Play")
    print("=" * 75)

    sm = PlayStateMachine(freeze_duration_frames=45)
    tests_passed = 0
    total_tests = 4

    class MockContact:
        def __init__(self, f):
            self.frame_hat = f
            self.confidence = 0.95
            self.observed_ball = True
            self.passer_track_id = 7

    # Mock offside result
    res_off = OffsideResult()
    res_off.decision = "OFFSIDE"
    res_off.margin_val = 1.85
    res_off.margin_unit = "m"
    res_off.homography_valid = True
    res_off.explanation = "OFFSIDE: Attacker +1.85m beyond line."

    # Test 1: Full Sequence to Offside Offence and Play Stopped
    sm.reset()
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Step 1: PLAYING
    t1_s1 = (sm.state == PlayStateEnum.PLAYING)

    # Step 2: PASS_CANDIDATE
    sm.update(100, dummy_img, pass_candidate={"onset": True}, contact_estimate=None, offside_result=None)
    t1_s2 = (sm.state == PlayStateEnum.PASS_CANDIDATE)

    # Step 3: CONTACT_ESTIMATED
    contact = MockContact(105)
    sm.update(105, dummy_img, pass_candidate=None, contact_estimate=contact, offside_result=None)
    t1_s3 = (sm.state == PlayStateEnum.CONTACT_ESTIMATED)

    # Step 4: POSITION_EVALUATED -> ACTIVE_INVOLVEMENT -> OFFSIDE_OFFENCE -> PLAY_STOPPED
    st, freeze, snap = sm.update(105, dummy_img, pass_candidate=None, contact_estimate=None, offside_result=res_off, active_involvement=True)
    t1_s4 = (st == PlayStateEnum.PLAY_STOPPED and freeze is True and snap is not None and snap.frame_index == 105)

    seq_passed = (t1_s1 and t1_s2 and t1_s3 and t1_s4)
    tests_passed += int(seq_passed)
    print(f"  [1/4] Active Involvement Progression: {'PASSED' if seq_passed else 'FAILED'} (End State: {st.value}, Freeze: {freeze})")

    # Test 2: Passive Play Distinction (Offside Position without Offence)
    sm.reset()
    sm.update(100, dummy_img, pass_candidate={"onset": True}, contact_estimate=None, offside_result=None)
    sm.update(105, dummy_img, pass_candidate=None, contact_estimate=contact, offside_result=None)
    # Update with active_involvement = False (passive player)
    st_pass, freeze_pass, snap_pass = sm.update(105, dummy_img, pass_candidate=None, contact_estimate=None, offside_result=res_off, active_involvement=False)
    passive_passed = (st_pass == PlayStateEnum.PLAYING and freeze_pass is False and snap_pass is None)
    tests_passed += int(passive_passed)
    print(f"  [2/4] Passive Play (No Offence):       {'PASSED' if passive_passed else 'FAILED'} (End State: {st_pass.value}, Freeze: {freeze_pass})")

    # Test 3: ONSIDE Position Causes No Stoppage
    res_on = OffsideResult()
    res_on.decision = "ONSIDE"
    res_on.margin_val = 0.0
    res_on.margin_unit = "m"
    res_on.homography_valid = True

    sm.reset()
    sm.update(100, dummy_img, pass_candidate={"onset": True}, contact_estimate=None, offside_result=None)
    sm.update(105, dummy_img, pass_candidate=None, contact_estimate=contact, offside_result=None)
    st_on, freeze_on, snap_on = sm.update(105, dummy_img, pass_candidate=None, contact_estimate=None, offside_result=res_on)
    onside_passed = (st_on == PlayStateEnum.PLAYING and freeze_on is False and snap_on is None)
    tests_passed += int(onside_passed)
    print(f"  [3/4] Onside Uninterrupted Play:      {'PASSED' if onside_passed else 'FAILED'} (End State: {st_on.value})")

    # Test 4: Freeze Duration Expiration and Recovery
    sm.reset()
    sm.freeze_duration_frames = 2  # Set small duration for testing
    sm.update(100, dummy_img, pass_candidate={"onset": True}, contact_estimate=None, offside_result=None)
    sm.update(105, dummy_img, pass_candidate=None, contact_estimate=contact, offside_result=None)
    sm.update(105, dummy_img, pass_candidate=None, contact_estimate=None, offside_result=res_off, active_involvement=True)

    # Frame 1 of freeze
    st_f1, freeze_f1, _ = sm.update(106, dummy_img, None, None, None)
    # Frame 2 of freeze
    st_f2, freeze_f2, _ = sm.update(107, dummy_img, None, None, None)
    # Frame 3: Expired -> Resumes PLAYING
    st_f3, freeze_f3, _ = sm.update(108, dummy_img, None, None, None)
    freeze_passed = (freeze_f1 is True and freeze_f2 is True and freeze_f3 is False and st_f3 == PlayStateEnum.PLAYING)
    tests_passed += int(freeze_passed)
    print(f"  [4/4] Freeze Expiration & Resume:     {'PASSED' if freeze_passed else 'FAILED'} (Resumed State: {st_f3.value})")

    pass_rate_pct = round(100.0 * tests_passed / total_tests, 1)
    print(f"\nLAYER 3 SUMMARY: {tests_passed}/{total_tests} ({pass_rate_pct}%) state machine tests passed.")

    return {
        "benchmark": "LAW11-STATE-001",
        "total_tests": total_tests,
        "passed_tests": tests_passed,
        "pass_rate_pct": pass_rate_pct
    }


def main():
    os.makedirs("dataset_v2_meta", exist_ok=True)
    pitch_geom = create_calibrated_pitch_geometry()

    report_unit = run_law11_unit_001(pitch_geom)
    report_eval = run_law11_eval_001(pitch_geom)
    report_state = run_law11_state_001()

    overall_report = {
        "evaluation_name": "LAW11-RIGOROUS-EVAL-002",
        "timestamp": "2026-09-12",
        "layer_1_unit_tests": report_unit,
        "layer_2_benchmark_eval": report_eval,
        "layer_3_state_machine": report_state,
        "overall_verdict": "PASS" if (report_unit["pass_rate_pct"] == 100.0 and
                                      report_eval["decision_accuracy_pct"] == 100.0 and
                                      report_state["pass_rate_pct"] == 100.0) else "FAIL"
    }

    report_path = "dataset_v2_meta/law11_eval_002_report.json"
    with open(report_path, "w") as f:
        json.dump(overall_report, f, indent=2)

    print("\n" + "=" * 75)
    print(f"OVERALL EVALUATION VERDICT: {overall_report['overall_verdict']}")
    print(f"Comprehensive evaluation report saved to: {report_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
