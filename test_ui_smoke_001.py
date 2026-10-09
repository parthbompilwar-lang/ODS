"""
UI-SMOKE-001: Automated Smoke Tests for Streamlit VAR App & Broadcast Renderer.
Tests:
  1. test_model_loading: Validates YOLO11 4-class model loading.
  2. test_single_incident_frame_pipeline: Runs complete pipeline on benchmark scene.
  3. test_pitch_clipped_shading_invariant: Verifies crowd/stands are NOT shaded.
  4. test_undetermined_state_handling: Verifies explicit refusal under invalid calibration.
  5. test_team_classifier_integration: Verifies team classification in track pipeline.
"""

import os
import cv2
import numpy as np
from app import load_yolo_v2_model, analyze_single_incident_frame, get_default_pitch_geometry
from src.engine.offside_engine import OffsideEngine, OffsideResult
from src.geometry.pitch_geometry import PitchGeometry
from src.team.team_classifier import TeamClassifier


def test_model_loading():
    model = load_yolo_v2_model()
    assert model is not None
    assert hasattr(model, "predict")
    print("PASS: test_model_loading")


def test_single_incident_frame_pipeline():
    test_img_path = "Offside_Images/0.jpg"
    assert os.path.exists(test_img_path), f"Missing {test_img_path}"

    raw_img = cv2.imread(test_img_path)
    annotated, result, stats = analyze_single_incident_frame(
        raw_img,
        attack_direction="right",
        attack_team_id=0
    )

    assert annotated is not None
    assert annotated.shape == raw_img.shape
    assert isinstance(result, OffsideResult)
    assert result.decision in ["OFFSIDE", "ONSIDE", "UNDETERMINED"]
    assert "team_0_count" in stats and "team_1_count" in stats
    assert stats["team_0_count"] + stats["team_1_count"] > 0
    print(f"PASS: test_single_incident_frame_pipeline (Decision: {result.decision}, Margin: {result.margin_val}{result.margin_unit})")


def test_pitch_clipped_shading_invariant():
    """
    CRITICAL REGRESSION TEST:
    Verifies that create_offside_shaded_overlay does NOT darken the top canvas
    (crowd, stands, scoreboard).
    """
    w, h = 1920, 1080
    test_img = np.ones((h, w, 3), dtype=np.uint8) * 200  # Uniform gray image

    geom = get_default_pitch_geometry(w, h)
    offside_line = ((600, 300), (400, 900))

    shaded = geom.create_offside_shaded_overlay(
        test_img,
        offside_line,
        goal_direction="right",
        boundary_world_x=50.0,
        alpha=0.42
    )

    # Top-left corner (0:100, 0:100) and top-right corner (0:100, w-100:w) must be UNTOUCHED
    top_left_diff = np.max(np.abs(shaded[0:100, 0:100].astype(float) - test_img[0:100, 0:100].astype(float)))
    top_right_diff = np.max(np.abs(shaded[0:100, w-100:w].astype(float) - test_img[0:100, w-100:w].astype(float)))

    assert top_left_diff == 0.0, f"FAILED: Top-left image region was darkened (diff={top_left_diff})!"
    assert top_right_diff == 0.0, f"FAILED: Top-right image region was darkened (diff={top_right_diff})!"
    print("PASS: test_pitch_clipped_shading_invariant (Crowd/sky areas strictly untouched)")


def test_undetermined_state_handling():
    """
    Verifies that missing or singular homography produces UNDETERMINED with explicit refusal.
    """
    bad_geom = PitchGeometry()
    bad_geom.homography_matrix = None  # Uncalibrated

    engine = OffsideEngine(pitch_geometry=bad_geom)
    dummy_players = [
        {"bbox": [100, 100, 150, 200], "class_id": 0, "team_id": 0, "track_id": 1},
        {"bbox": [300, 100, 350, 200], "class_id": 0, "team_id": 1, "track_id": 2},
        {"bbox": [400, 100, 450, 200], "class_id": 1, "team_id": 1, "track_id": 3}
    ]

    res = engine.evaluate_contact_moment(
        players=dummy_players,
        attack_team_id=0,
        attack_direction="right",
        require_metric=True  # Require metric calibration
    )

    assert res.decision == "UNDETERMINED", f"Expected UNDETERMINED, got {res.decision}"
    assert res.geometry_status == "INVALID"
    assert "refused" in res.explanation.lower() or "no valid" in res.explanation.lower()
    print("PASS: test_undetermined_state_handling (Metric offside correctly refused under missing H)")


def test_team_classifier_integration():
    tc = TeamClassifier(n_teams=2, vote_window=5, min_votes_required=1)
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)

    dummy_tracks = [
        {"bbox": [100, 100, 140, 220], "class_id": 0, "track_id": 1},
        {"bbox": [200, 100, 240, 220], "class_id": 0, "track_id": 2},
        {"bbox": [300, 100, 340, 220], "class_id": 1, "track_id": 3},  # GK
        {"bbox": [400, 100, 440, 220], "class_id": 2, "track_id": 4}   # REF
    ]

    classified = tc.classify_tracks(dummy_img, dummy_tracks, frame_index=0)
    assert len(classified) == 4
    ref = [t for t in classified if t.get("role") == "referee"][0]
    gk = [t for t in classified if t.get("role") == "goalkeeper"][0]

    assert ref["team_id"] == -1
    assert ref["team_label"] == "REF"
    assert gk["role"] == "goalkeeper"
    print("PASS: test_team_classifier_integration")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING UI-SMOKE-001 SUITE")
    print("=" * 60)
    test_model_loading()
    test_single_incident_frame_pipeline()
    test_pitch_clipped_shading_invariant()
    test_undetermined_state_handling()
    test_team_classifier_integration()
    print("=" * 60)
    print("ALL 5 UI-SMOKE-001 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
