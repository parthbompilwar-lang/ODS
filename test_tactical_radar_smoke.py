"""
Smoke test for Football Tactical Radar & Heatmap Engine.
"""
import os
import cv2
import numpy as np
from src.visualization.tactical_radar_heatmap import TacticalRadarEngine
from radar_app import get_pitch_geometry, load_yolo_model


def test_radar_smoke():
    print("Running Tactical Radar Smoke Tests...")

    # 1. Test model loader
    model = load_yolo_model()
    assert model is not None
    print("PASS: YOLO model loaded.")

    # 2. Test pitch geometry
    geom = get_pitch_geometry(1920, 1080)
    assert geom.homography_matrix is not None
    print("PASS: Pitch geometry homography valid.")

    # 3. Test radar drawing
    engine = TacticalRadarEngine()
    test_players = [
        {"track_id": 1, "team_id": 0, "role": "player", "world_x": 30.0, "world_y": 25.0},
        {"track_id": 2, "team_id": 0, "role": "player", "world_x": 35.0, "world_y": 45.0},
        {"track_id": 3, "team_id": 0, "role": "player", "world_x": 40.0, "world_y": 30.0},
        {"track_id": 4, "team_id": 1, "role": "player", "world_x": 65.0, "world_y": 20.0},
        {"track_id": 5, "team_id": 1, "role": "player", "world_x": 70.0, "world_y": 35.0},
        {"track_id": 6, "team_id": 1, "role": "player", "world_x": 75.0, "world_y": 50.0},
    ]
    engine.update_frame(test_players, ball_pos=(960, 540), pitch_geom=geom)
    radar = engine.render_tactical_radar(test_players, width=640, height=380)
    assert radar.shape == (380, 640, 3)
    print("PASS: 2D tactical radar rendered.")

    # 4. Test heatmap generation
    heat = engine.generate_heatmap(engine.team0_positions, width=640, height=380)
    assert heat.shape == (380, 640, 3)
    print("PASS: 2D spatial heatmap rendered.")

    # 5. Test metrics computation
    metrics = engine.compute_tactical_metrics()
    assert "team0_control_pct" in metrics
    assert "team0_compactness_m2" in metrics
    print(f"PASS: Tactical metrics computed (Control: T1={metrics['team0_control_pct']}%, T2={metrics['team1_control_pct']}%)")
    print("\nALL TACTICAL RADAR SMOKE TESTS PASSED!")


if __name__ == "__main__":
    test_radar_smoke()
