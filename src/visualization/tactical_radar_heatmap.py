"""
AI Football Tactical Radar & Spatial Heatmap Engine.
Implements:
1. Standard FIFA 2D Pitch Diagram Projection (105m x 68m).
2. Live Multi-Team Tactical Minimap with Convex Hull Shape & Centroids.
3. 2D Gaussian Kernel Density Estimation (KDE) Team & Player Heatmaps.
4. Pitch Control & Spatial Dominance Analytics (Territory %, Defensive Line Height, Compactness).
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy.spatial import ConvexHull


class TacticalRadarEngine:
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # Color palette (BGR format)
        self.COLOR_TURF = (28, 58, 28)           # Dark tactical green
        self.COLOR_TURF_LIGHT = (34, 68, 34)     # Alternating lawn stripe
        self.COLOR_LINES = (220, 220, 220)       # White markings
        self.COLOR_TEAM0 = (240, 160, 40)        # Team 0 (Cyan/Sky Blue)
        self.COLOR_TEAM1 = (40, 70, 235)         # Team 1 (Red/Orange)
        self.COLOR_GK = (0, 220, 255)            # Goalkeeper (Yellow)
        self.COLOR_REF = (180, 50, 180)          # Referee (Purple)
        self.COLOR_BALL = (0, 0, 255)            # Ball (Red marker)

        # Spatial tracking history for cumulative heatmaps
        # Dict[track_id -> List[(world_x, world_y)]]
        self.player_trajectories: Dict[int, List[Tuple[float, float]]] = {}
        self.team0_positions: List[Tuple[float, float]] = []
        self.team1_positions: List[Tuple[float, float]] = []
        self.ball_positions: List[Tuple[float, float]] = []

    def reset_history(self):
        """Clears spatial tracking accumulator."""
        self.player_trajectories.clear()
        self.team0_positions.clear()
        self.team1_positions.clear()
        self.ball_positions.clear()

    def update_frame(
        self,
        players: List[Dict[str, Any]],
        ball_pos: Optional[Tuple[float, float]] = None,
        pitch_geom: Optional[Any] = None
    ):
        """
        Ingests detected player ground positions and updates trajectory history.
        """
        for p in players:
            tid = p.get("track_id", -1)
            t_id = p.get("team_id", -1)
            role = p.get("role", "player")

            # Extract or compute pitch world coordinates
            wx = p.get("world_x")
            wy = p.get("world_y")
            if (wx is None or wy is None) and pitch_geom is not None and getattr(pitch_geom, "homography_matrix", None) is not None:
                bbox = p.get("bbox", [0, 0, 50, 50])
                foot_pt = np.array([[(bbox[0] + bbox[2]) / 2.0, float(bbox[3])]], dtype=np.float32)
                transformed = pitch_geom.image_to_pitch(foot_pt)[0]
                wx, wy = float(transformed[0]), float(transformed[1])
                p["world_x"] = wx
                p["world_y"] = wy

            if wx is not None and wy is not None:
                # Clamp within pitch boundaries with slight margin
                wx = max(0.0, min(self.pitch_length, wx))
                wy = max(0.0, min(self.pitch_width, wy))

                if tid >= 0:
                    if tid not in self.player_trajectories:
                        self.player_trajectories[tid] = []
                    self.player_trajectories[tid].append((wx, wy))

                if t_id == 0 and role != "referee":
                    self.team0_positions.append((wx, wy))
                elif t_id == 1 and role != "referee":
                    self.team1_positions.append((wx, wy))

        if ball_pos is not None and pitch_geom is not None and getattr(pitch_geom, "homography_matrix", None) is not None:
            ball_pt = np.array([[ball_pos[0], ball_pos[1]]], dtype=np.float32)
            trans_ball = pitch_geom.image_to_pitch(ball_pt)[0]
            b_wx, b_wy = float(trans_ball[0]), float(trans_ball[1])
            if 0.0 <= b_wx <= self.pitch_length and 0.0 <= b_wy <= self.pitch_width:
                self.ball_positions.append((b_wx, b_wy))

    def draw_pitch_diagram(
        self,
        width: int = 800,
        height: int = 520,
        margin: int = 30
    ) -> np.ndarray:
        """
        Draws an accurate, top-down FIFA regulation 2D football pitch.
        """
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)  # Dark background border

        # Inner playable turf area
        x0 = margin
        y0 = margin
        pw = width - 2 * margin
        ph = height - 2 * margin

        # Fill turf base with alternating stripes
        canvas[y0:y0 + ph, x0:x0 + pw] = self.COLOR_TURF
        n_stripes = 10
        stripe_w = pw / n_stripes
        for i in range(0, n_stripes, 2):
            sx1 = int(x0 + i * stripe_w)
            sx2 = int(x0 + (i + 1) * stripe_w)
            canvas[y0:y0 + ph, sx1:sx2] = self.COLOR_TURF_LIGHT

        # Boundary touchlines and goal lines
        cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), self.COLOR_LINES, 2, cv2.LINE_AA)

        # Halfway line
        mid_x = x0 + pw // 2
        cv2.line(canvas, (mid_x, y0), (mid_x, y0 + ph), self.COLOR_LINES, 2, cv2.LINE_AA)

        # Center circle (9.15m radius)
        center_y = y0 + ph // 2
        r_center = int(9.15 * (pw / self.pitch_length))
        cv2.circle(canvas, (mid_x, center_y), r_center, self.COLOR_LINES, 2, cv2.LINE_AA)
        cv2.circle(canvas, (mid_x, center_y), 3, self.COLOR_LINES, -1, cv2.LINE_AA)

        # Scale factors: pixels per meter
        sx = pw / self.pitch_length
        sy = ph / self.pitch_width

        # Penalty Box Dimensions (16.5m length, 40.32m width)
        pen_len = int(16.5 * sx)
        pen_top = int(y0 + ((self.pitch_width - 40.32) / 2.0) * sy)
        pen_bot = int(y0 + ((self.pitch_width + 40.32) / 2.0) * sy)

        # Left Penalty Box (X=0)
        cv2.rectangle(canvas, (x0, pen_top), (x0 + pen_len, pen_bot), self.COLOR_LINES, 2, cv2.LINE_AA)
        # Right Penalty Box (X=105)
        cv2.rectangle(canvas, (x0 + pw - pen_len, pen_top), (x0 + pw, pen_bot), self.COLOR_LINES, 2, cv2.LINE_AA)

        # Goal Area Dimensions (5.5m length, 18.32m width)
        ga_len = int(5.5 * sx)
        ga_top = int(y0 + ((self.pitch_width - 18.32) / 2.0) * sy)
        ga_bot = int(y0 + ((self.pitch_width + 18.32) / 2.0) * sy)

        # Left Goal Area
        cv2.rectangle(canvas, (x0, ga_top), (x0 + ga_len, ga_bot), self.COLOR_LINES, 2, cv2.LINE_AA)
        # Right Goal Area
        cv2.rectangle(canvas, (x0 + pw - ga_len, ga_top), (x0 + pw, ga_bot), self.COLOR_LINES, 2, cv2.LINE_AA)

        # Penalty Spots (11.0m from goal line)
        spot_left_x = int(x0 + 11.0 * sx)
        spot_right_x = int(x0 + pw - 11.0 * sx)
        cv2.circle(canvas, (spot_left_x, center_y), 3, self.COLOR_LINES, -1, cv2.LINE_AA)
        cv2.circle(canvas, (spot_right_x, center_y), 3, self.COLOR_LINES, -1, cv2.LINE_AA)

        # Goal nets (drawn outside goal line for TV presentation)
        net_w = max(4, int(2.0 * sx))
        net_top = int(y0 + ((self.pitch_width - 7.32) / 2.0) * sy)
        net_bot = int(y0 + ((self.pitch_width + 7.32) / 2.0) * sy)
        cv2.rectangle(canvas, (x0 - net_w, net_top), (x0, net_bot), (160, 160, 160), 1)
        cv2.rectangle(canvas, (x0 + pw, net_top), (x0 + pw + net_w, net_bot), (160, 160, 160), 1)

        return canvas

    def world_to_pitch_px(
        self,
        wx: float,
        wy: float,
        width: int = 800,
        height: int = 520,
        margin: int = 30
    ) -> Tuple[int, int]:
        """Converts physical pitch (X, Y) meters into pixel coordinates on the 2D diagram."""
        pw = width - 2 * margin
        ph = height - 2 * margin
        px = int(margin + (wx / self.pitch_length) * pw)
        py = int(margin + (wy / self.pitch_width) * ph)
        return px, py

    def render_tactical_radar(
        self,
        players: List[Dict[str, Any]],
        ball_pos: Optional[Tuple[float, float]] = None,
        pitch_geom: Optional[Any] = None,
        width: int = 800,
        height: int = 520,
        margin: int = 30,
        show_convex_hulls: bool = True,
        show_centroids: bool = True
    ) -> np.ndarray:
        """
        Generates real-time 2D tactical radar with live player dots, team shape polygons,
        and team centroids.
        """
        radar = self.draw_pitch_diagram(width=width, height=height, margin=margin)

        team0_pts = []
        team1_pts = []

        # 1. Collect and plot players
        for p in players:
            wx = p.get("world_x")
            wy = p.get("world_y")
            if (wx is None or wy is None) and pitch_geom is not None and getattr(pitch_geom, "homography_matrix", None) is not None:
                bbox = p.get("bbox", [0, 0, 50, 50])
                foot = np.array([[(bbox[0] + bbox[2]) / 2.0, float(bbox[3])]], dtype=np.float32)
                w_pt = pitch_geom.image_to_pitch(foot)[0]
                wx, wy = float(w_pt[0]), float(w_pt[1])

            if wx is None or wy is None:
                continue

            rx, ry = self.world_to_pitch_px(wx, wy, width, height, margin)
            role = p.get("role", "player")
            t_id = p.get("team_id", -1)

            if role == "referee":
                col = self.COLOR_REF
                lbl = "R"
            elif role == "goalkeeper":
                col = self.COLOR_GK
                lbl = "GK"
            elif t_id == 0:
                col = self.COLOR_TEAM0
                lbl = str(p.get("track_id", ""))
                team0_pts.append((rx, ry))
            elif t_id == 1:
                col = self.COLOR_TEAM1
                lbl = str(p.get("track_id", ""))
                team1_pts.append((rx, ry))
            else:
                col = (160, 160, 160)
                lbl = "?"

            # Player circle with black outline
            cv2.circle(radar, (rx, ry), 9, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(radar, (rx, ry), 7, col, -1, cv2.LINE_AA)
            cv2.circle(radar, (rx, ry), 7, (255, 255, 255), 1, cv2.LINE_AA)

            if len(lbl) <= 2:
                cv2.putText(radar, lbl, (rx - 5, ry + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw Convex Hulls (Tactical Compactness Shape)
        if show_convex_hulls:
            overlay = radar.copy()
            for pts, col in [(team0_pts, self.COLOR_TEAM0), (team1_pts, self.COLOR_TEAM1)]:
                if len(pts) >= 3:
                    pts_arr = np.array(pts)
                    try:
                        hull = ConvexHull(pts_arr)
                        hull_pts = pts_arr[hull.vertices]
                        cv2.fillPoly(overlay, [hull_pts], col, cv2.LINE_AA)
                        cv2.polylines(radar, [hull_pts], True, col, 2, cv2.LINE_AA)
                    except Exception:
                        pass
            # 25% opacity for hull fill
            cv2.addWeighted(overlay, 0.25, radar, 0.75, 0, radar)

        # 3. Draw Team Centroids (Team Center of Mass)
        if show_centroids:
            for pts, col, label in [(team0_pts, self.COLOR_TEAM0, "TEAM 1 CENTROID"), (team1_pts, self.COLOR_TEAM1, "TEAM 2 CENTROID")]:
                if len(pts) > 0:
                    cx = int(np.mean([pt[0] for pt in pts]))
                    cy = int(np.mean([pt[1] for pt in pts]))
                    # Crosshair marker
                    cv2.drawMarker(radar, (cx, cy), col, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
                    cv2.circle(radar, (cx, cy), 12, col, 1, cv2.LINE_AA)

        # 4. Draw Ball marker
        if ball_pos is not None and pitch_geom is not None and getattr(pitch_geom, "homography_matrix", None) is not None:
            ball_pt = np.array([[ball_pos[0], ball_pos[1]]], dtype=np.float32)
            trans_ball = pitch_geom.image_to_pitch(ball_pt)[0]
            b_wx, b_wy = float(trans_ball[0]), float(trans_ball[1])
            if 0.0 <= b_wx <= self.pitch_length and 0.0 <= b_wy <= self.pitch_width:
                bx, by = self.world_to_pitch_px(b_wx, b_wy, width, height, margin)
                cv2.circle(radar, (bx, by), 8, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(radar, (bx, by), 10, (255, 255, 255), 2, cv2.LINE_AA)

        # Top title card
        cv2.rectangle(radar, (0, 0), (width, 24), (15, 15, 15), -1)
        cv2.putText(radar, "TACTICAL 2D PITCH RADAR (TOP-DOWN METRIC PROJECTION)", (margin, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        return radar

    def generate_heatmap(
        self,
        positions: List[Tuple[float, float]],
        width: int = 800,
        height: int = 520,
        margin: int = 30,
        colormap: int = cv2.COLORMAP_JET,
        kernel_size: int = 51,
        title: str = "PITCH HEATMAP"
    ) -> np.ndarray:
        """
        Computes 2D spatial Gaussian density heatmap over physical pitch coordinates.
        """
        pitch_base = self.draw_pitch_diagram(width=width, height=height, margin=margin)
        pw = width - 2 * margin
        ph = height - 2 * margin

        if not positions:
            cv2.putText(pitch_base, f"{title}: NO DATA ACCUMULATED", (margin + 20, margin + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            return pitch_base

        # Accumulator grid inside playable turf
        grid = np.zeros((ph, pw), dtype=np.float32)

        for wx, wy in positions:
            gx = int((wx / self.pitch_length) * (pw - 1))
            gy = int((wy / self.pitch_width) * (ph - 1))
            if 0 <= gx < pw and 0 <= gy < ph:
                grid[gy, gx] += 1.0

        if np.max(grid) > 0:
            # 2D Gaussian Kernel Smoothing
            k_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            blurred = cv2.GaussianBlur(grid, (k_size, k_size), 0)
            norm_map = blurred / (np.max(blurred) + 1e-6)
            norm_u8 = (norm_map * 255).astype(np.uint8)

            # Apply False-Color Thermal Colormap
            heat_color = cv2.applyColorMap(norm_u8, colormap)

            # Alpha blend onto playable turf
            turf_roi = pitch_base[margin:margin + ph, margin:margin + pw]
            mask = (norm_map > 0.08).astype(np.float32)[:, :, np.newaxis]
            blended_roi = (heat_color * mask * 0.65 + turf_roi * (1.0 - mask * 0.65)).astype(np.uint8)
            pitch_base[margin:margin + ph, margin:margin + pw] = blended_roi

        # Top title banner
        cv2.rectangle(pitch_base, (0, 0), (width, 24), (15, 15, 15), -1)
        cv2.putText(pitch_base, f"{title} | ACCUMULATED SAMPLES: {len(positions)}", (margin, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        return pitch_base

    def compute_tactical_metrics(self) -> Dict[str, Any]:
        """
        Computes key sports science & tactical analytics:
        - Pitch territory control %
        - Average defensive line depth
        - Team compactness (convex hull area in m^2)
        - High-press index (% in attacking third)
        """
        t0_arr = np.array(self.team0_positions) if len(self.team0_positions) > 0 else np.empty((0, 2))
        t1_arr = np.array(self.team1_positions) if len(self.team1_positions) > 0 else np.empty((0, 2))

        # Territory split by total observations
        tot = len(self.team0_positions) + len(self.team1_positions)
        t0_pct = (len(self.team0_positions) / tot * 100.0) if tot > 0 else 50.0
        t1_pct = (len(self.team1_positions) / tot * 100.0) if tot > 0 else 50.0

        # Centroids
        t0_centroid = (float(np.mean(t0_arr[:, 0])), float(np.mean(t0_arr[:, 1]))) if len(t0_arr) > 0 else (52.5, 34.0)
        t1_centroid = (float(np.mean(t1_arr[:, 0])), float(np.mean(t1_arr[:, 1]))) if len(t1_arr) > 0 else (52.5, 34.0)

        # Defensive line height (10th percentile closest to respective goal)
        t0_def_line = float(np.percentile(t0_arr[:, 0], 15)) if len(t0_arr) >= 5 else 20.0
        t1_def_line = float(np.percentile(t1_arr[:, 0], 85)) if len(t1_arr) >= 5 else 85.0

        # Attacking third penetration (% samples in opponent final 35m)
        t0_att_third = (float(np.sum(t0_arr[:, 0] > 70.0)) / len(t0_arr) * 100.0) if len(t0_arr) > 0 else 0.0
        t1_att_third = (float(np.sum(t1_arr[:, 0] < 35.0)) / len(t1_arr) * 100.0) if len(t1_arr) > 0 else 0.0

        # Compactness (Hull Area in m^2)
        def get_hull_area(arr):
            if len(arr) >= 3:
                try:
                    return float(ConvexHull(arr).volume)  # 2D volume = area
                except Exception:
                    return 0.0
            return 0.0

        t0_area = get_hull_area(t0_arr)
        t1_area = get_hull_area(t1_arr)

        return {
            "team0_samples": len(self.team0_positions),
            "team1_samples": len(self.team1_positions),
            "team0_control_pct": round(t0_pct, 1),
            "team1_control_pct": round(t1_pct, 1),
            "team0_centroid": (round(t0_centroid[0], 2), round(t0_centroid[1], 2)),
            "team1_centroid": (round(t1_centroid[0], 2), round(t1_centroid[1], 2)),
            "team0_def_line_m": round(t0_def_line, 2),
            "team1_def_line_m": round(t1_def_line, 2),
            "team0_attacking_third_pct": round(t0_att_third, 1),
            "team1_attacking_third_pct": round(t1_att_third, 1),
            "team0_compactness_m2": round(t0_area, 1),
            "team1_compactness_m2": round(t1_area, 1),
            "tracked_players_count": len(self.player_trajectories)
        }
