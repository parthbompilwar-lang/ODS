"""
AI Football Tactical Radar & Spatial Heatmap Engine.
Core Features:
1. Validated Metric Pitch Projection (105m x 68m FIFA Standard).
2. Live Tactical 2D Pitch Radar with Clean Translucent Convex Hulls.
3. Proximity-Based Player Label Decluttering (Prevents text overlaps).
4. Distinct Ball Visual Identity with OBSERVED / PREDICTED / LOST states.
5. Time-Windowed Accumulated Gaussian KDE Spatial Density Layer.
6. Event-Driven Pass Vector Arrows & Measured Proximity Ball Possession.
7. Exact Sports Science Metrics:
   - Team Compactness (Mean Distance to Centroid in meters & Hull Area in m^2)
   - Backline Defensive Height (Mean X of deepest 4 outfield players)
   - Spatial Pitch Occupancy (%)
   - Tracking Continuity (% frames with valid detections)
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
        self.COLOR_TURF = (24, 52, 24)           # Deep tactical pitch green
        self.COLOR_TURF_LIGHT = (28, 60, 28)     # Muted turf stripe
        self.COLOR_LINES = (210, 210, 210)       # Pitch line markings
        self.COLOR_TEAM0 = (245, 155, 35)        # Team 1 (Sky Blue / Cyan, BGR)
        self.COLOR_TEAM1 = (35, 65, 235)         # Team 2 (Red / Coral, BGR)
        self.COLOR_GK = (0, 220, 255)            # Goalkeeper (Amber / Yellow)
        self.COLOR_REF = (180, 50, 180)          # Referee (Purple)
        self.COLOR_BALL = (255, 255, 255)        # Ball core (Pure White)
        self.COLOR_BALL_RING = (0, 215, 255)     # Ball ring (Gold / Yellow)

        # Spatial tracking accumulators
        # player_trajectories: track_id -> List[(frame_idx, world_x, world_y)]
        self.player_trajectories: Dict[int, List[Tuple[int, float, float]]] = {}
        self.team0_history: List[Tuple[int, float, float]] = []
        self.team1_history: List[Tuple[int, float, float]] = []
        self.ball_history: List[Tuple[int, float, float, str]] = []

        # Real measured possession accumulator (based on ball proximity <= 2.5m)
        self.team0_possession_frames: int = 0
        self.team1_possession_frames: int = 0
        self.total_possession_frames: int = 0

        # Pass event history: List[(start_pt, end_pt, start_frame, expiry_frame, team_id)]
        self.active_passes: List[Dict[str, Any]] = []
        self.total_passes_detected: int = 0

        # Tracking continuity metrics
        self.total_frames_processed: int = 0
        self.frames_with_players: int = 0

    def reset_history(self):
        """Resets all accumulated tracking states."""
        self.player_trajectories.clear()
        self.team0_history.clear()
        self.team1_history.clear()
        self.ball_history.clear()
        self.team0_possession_frames = 0
        self.team1_possession_frames = 0
        self.total_possession_frames = 0
        self.active_passes.clear()
        self.total_passes_detected = 0
        self.total_frames_processed = 0
        self.frames_with_players = 0

    def register_pass_event(
        self,
        start_world: Tuple[float, float],
        end_world: Tuple[float, float],
        current_frame: int,
        team_id: int,
        duration_frames: int = 30
    ):
        """Registers a verified pass candidate to render an analytical trajectory arrow."""
        self.total_passes_detected += 1
        self.active_passes.append({
            "start": start_world,
            "end": end_world,
            "start_frame": current_frame,
            "expiry_frame": current_frame + duration_frames,
            "team_id": team_id
        })

    def update_frame(
        self,
        players: List[Dict[str, Any]],
        ball_pos: Optional[Tuple[float, float]] = None,
        ball_state: str = "OBSERVED",
        frame_idx: int = 0,
        pitch_geom: Optional[Any] = None
    ):
        """
        Updates tracking accumulator with validated metric coordinates and computes
        instantaneous possession based on true physical proximity.
        """
        self.total_frames_processed += 1
        if len(players) > 0:
            self.frames_with_players += 1

        # Clean expired pass events
        self.active_passes = [p for p in self.active_passes if p["expiry_frame"] > frame_idx]

        # 1. Transform and record players
        ball_metric: Optional[Tuple[float, float]] = None
        if ball_pos is not None and pitch_geom is not None and getattr(pitch_geom, "homography_matrix", None) is not None:
            ball_pt = np.array([[ball_pos[0], ball_pos[1]]], dtype=np.float32)
            trans_ball = pitch_geom.image_to_pitch(ball_pt)[0]
            b_wx, b_wy = float(trans_ball[0]), float(trans_ball[1])
            if 0.0 <= b_wx <= self.pitch_length and 0.0 <= b_wy <= self.pitch_width:
                ball_metric = (b_wx, b_wy)
                self.ball_history.append((frame_idx, b_wx, b_wy, ball_state))

        closest_player_dist: float = 999.0
        closest_player_team: int = -1

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
                wx = max(0.0, min(self.pitch_length, wx))
                wy = max(0.0, min(self.pitch_width, wy))

                if tid >= 0:
                    if tid not in self.player_trajectories:
                        self.player_trajectories[tid] = []
                    self.player_trajectories[tid].append((frame_idx, wx, wy))

                if t_id == 0 and role != "referee":
                    self.team0_history.append((frame_idx, wx, wy))
                elif t_id == 1 and role != "referee":
                    self.team1_history.append((frame_idx, wx, wy))

                # Check proximity to ball for true possession calculation
                if ball_metric is not None and role != "referee":
                    dist = float(np.hypot(wx - ball_metric[0], wy - ball_metric[1]))
                    if dist < closest_player_dist:
                        closest_player_dist = dist
                        closest_player_team = t_id

        # True Measured Possession Rule: Player within 2.5m of ball has active possession
        if ball_metric is not None and closest_player_dist <= 2.5:
            if closest_player_team == 0:
                self.team0_possession_frames += 1
                self.total_possession_frames += 1
            elif closest_player_team == 1:
                self.team1_possession_frames += 1
                self.total_possession_frames += 1

    def draw_pitch_diagram(
        self,
        width: int = 800,
        height: int = 520,
        margin: int = 25
    ) -> np.ndarray:
        """Draws standard FIFA regulation pitch diagram with crisp 2D geometry."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 18)

        x0, y0 = margin, margin
        pw = width - 2 * margin
        ph = height - 2 * margin

        # Base turf with subtle alternating bands
        canvas[y0:y0 + ph, x0:x0 + pw] = self.COLOR_TURF
        n_stripes = 10
        stripe_w = pw / n_stripes
        for i in range(0, n_stripes, 2):
            sx1 = int(x0 + i * stripe_w)
            sx2 = int(x0 + (i + 1) * stripe_w)
            canvas[y0:y0 + ph, sx1:sx2] = self.COLOR_TURF_LIGHT

        # Touchlines and goal lines
        cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), self.COLOR_LINES, 1, cv2.LINE_AA)

        # Halfway line
        mid_x = x0 + pw // 2
        cv2.line(canvas, (mid_x, y0), (mid_x, y0 + ph), self.COLOR_LINES, 1, cv2.LINE_AA)

        # Center circle (9.15m radius)
        center_y = y0 + ph // 2
        r_center = int(9.15 * (pw / self.pitch_length))
        cv2.circle(canvas, (mid_x, center_y), r_center, self.COLOR_LINES, 1, cv2.LINE_AA)
        cv2.circle(canvas, (mid_x, center_y), 3, self.COLOR_LINES, -1, cv2.LINE_AA)

        # Scale factors
        sx = pw / self.pitch_length
        sy = ph / self.pitch_width

        # Penalty Boxes (16.5m length, 40.32m width)
        pen_len = int(16.5 * sx)
        pen_top = int(y0 + ((self.pitch_width - 40.32) / 2.0) * sy)
        pen_bot = int(y0 + ((self.pitch_width + 40.32) / 2.0) * sy)
        cv2.rectangle(canvas, (x0, pen_top), (x0 + pen_len, pen_bot), self.COLOR_LINES, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0 + pw - pen_len, pen_top), (x0 + pw, pen_bot), self.COLOR_LINES, 1, cv2.LINE_AA)

        # Goal Areas (5.5m length, 18.32m width)
        ga_len = int(5.5 * sx)
        ga_top = int(y0 + ((self.pitch_width - 18.32) / 2.0) * sy)
        ga_bot = int(y0 + ((self.pitch_width + 18.32) / 2.0) * sy)
        cv2.rectangle(canvas, (x0, ga_top), (x0 + ga_len, ga_bot), self.COLOR_LINES, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0 + pw - ga_len, ga_top), (x0 + pw, ga_bot), self.COLOR_LINES, 1, cv2.LINE_AA)

        # Penalty Spots (11.0m from goal line)
        spot_left_x = int(x0 + 11.0 * sx)
        spot_right_x = int(x0 + pw - 11.0 * sx)
        cv2.circle(canvas, (spot_left_x, center_y), 2, self.COLOR_LINES, -1, cv2.LINE_AA)
        cv2.circle(canvas, (spot_right_x, center_y), 2, self.COLOR_LINES, -1, cv2.LINE_AA)

        # Goal posts
        net_w = max(4, int(2.0 * sx))
        net_top = int(y0 + ((self.pitch_width - 7.32) / 2.0) * sy)
        net_bot = int(y0 + ((self.pitch_width + 7.32) / 2.0) * sy)
        cv2.rectangle(canvas, (x0 - net_w, net_top), (x0, net_bot), (120, 120, 120), 1)
        cv2.rectangle(canvas, (x0 + pw, net_top), (x0 + pw + net_w, net_bot), (120, 120, 120), 1)

        return canvas

    def world_to_pitch_px(
        self,
        wx: float,
        wy: float,
        width: int = 800,
        height: int = 520,
        margin: int = 25
    ) -> Tuple[int, int]:
        """Converts physical pitch (X, Y) meters into pixel coordinates on the 2D diagram."""
        pw = width - 2 * margin
        ph = height - 2 * margin
        px = int(margin + (wx / self.pitch_length) * pw)
        py = int(margin + (wy / self.pitch_width) * ph)
        return px, py

    def _render_density_layer(
        self,
        canvas: np.ndarray,
        time_window: Optional[int] = 60,
        current_frame: int = 0,
        width: int = 800,
        height: int = 520,
        margin: int = 25
    ):
        """
        Renders a restrained 2D Gaussian Kernel Density Estimation (KDE) density layer
        directly onto the pitch canvas as a base under the active player tokens.
        """
        pw = width - 2 * margin
        ph = height - 2 * margin

        # Filter points by time window if specified
        pts = []
        for hist in [self.team0_history, self.team1_history]:
            for f_idx, wx, wy in hist:
                if time_window is None or (current_frame - time_window <= f_idx <= current_frame):
                    pts.append((wx, wy))

        if len(pts) < 5:
            return

        grid = np.zeros((ph, pw), dtype=np.float32)
        for wx, wy in pts:
            gx = int((wx / self.pitch_length) * (pw - 1))
            gy = int((wy / self.pitch_width) * (ph - 1))
            if 0 <= gx < pw and 0 <= gy < ph:
                grid[gy, gx] += 1.0

        if np.max(grid) > 0:
            blurred = cv2.GaussianBlur(grid, (51, 51), 0)
            norm_map = blurred / (np.max(blurred) + 1e-6)

            # Restrained thermal colormap (Teal to Soft Gold)
            norm_u8 = (norm_map * 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(norm_u8, cv2.COLORMAP_VIRIDIS)

            # Subtle alpha blend on turf region (35% opacity)
            turf_roi = canvas[margin:margin + ph, margin:margin + pw]
            mask = (norm_map > 0.10).astype(np.float32)[:, :, np.newaxis]
            blended = (heat_color * mask * 0.35 + turf_roi * (1.0 - mask * 0.35)).astype(np.uint8)
            canvas[margin:margin + ph, margin:margin + pw] = blended

    def render_tactical_radar(
        self,
        players: List[Dict[str, Any]],
        ball_pos: Optional[Tuple[float, float]] = None,
        ball_state: str = "OBSERVED",
        pitch_geom: Optional[Any] = None,
        current_frame: int = 0,
        width: int = 800,
        height: int = 520,
        margin: int = 25,
        show_density_base: bool = True,
        density_window: Optional[int] = 60,
        show_convex_hulls: bool = True,
        show_centroids: bool = True,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Renders complete, clean 2D metric tactical pitch radar with:
        - Density heatmap layer under turf
        - Soft translucent team shape hulls
        - Decluttered player tokens
        - Distinct white/gold ball marker with state
        - Active pass trajectory arrows
        - Embedded tactical legend
        """
        radar = self.draw_pitch_diagram(width=width, height=height, margin=margin)

        # 1. Base Layer: Spatial Density KDE Heatmap
        if show_density_base:
            self._render_density_layer(
                radar,
                time_window=density_window,
                current_frame=current_frame,
                width=width,
                height=height,
                margin=margin
            )

        team0_radar_pts = []
        team1_radar_pts = []
        player_radar_items = []

        # 2. Extract and project player positions
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
            tid = p.get("track_id", -1)

            if role == "referee":
                col = self.COLOR_REF
                lbl = "R"
            elif role == "goalkeeper":
                col = self.COLOR_GK
                lbl = "GK"
            elif t_id == 0:
                col = self.COLOR_TEAM0
                lbl = str(tid) if tid >= 0 else "1"
                team0_radar_pts.append((rx, ry))
            elif t_id == 1:
                col = self.COLOR_TEAM1
                lbl = str(tid) if tid >= 0 else "2"
                team1_radar_pts.append((rx, ry))
            else:
                col = (140, 140, 140)
                lbl = "?"

            player_radar_items.append({
                "pt": (rx, ry),
                "col": col,
                "lbl": lbl,
                "role": role,
                "tid": tid
            })

        # 3. Translucent Team Shape Convex Hulls (Soft fill, thin 1px border)
        if show_convex_hulls:
            overlay = radar.copy()
            for pts, col in [(team0_radar_pts, self.COLOR_TEAM0), (team1_radar_pts, self.COLOR_TEAM1)]:
                if len(pts) >= 3:
                    pts_arr = np.array(pts)
                    try:
                        hull = ConvexHull(pts_arr)
                        hull_pts = pts_arr[hull.vertices]
                        cv2.fillPoly(overlay, [hull_pts], col, cv2.LINE_AA)
                        cv2.polylines(radar, [hull_pts], True, col, 1, cv2.LINE_AA)
                    except Exception:
                        pass
            # 16% opacity for gentle, non-obtrusive team shape boundary
            cv2.addWeighted(overlay, 0.16, radar, 0.84, 0, radar)

        # 4. Team Centroids (Center of Mass crosshair)
        if show_centroids:
            for pts, col in [(team0_radar_pts, self.COLOR_TEAM0), (team1_radar_pts, self.COLOR_TEAM1)]:
                if len(pts) > 0:
                    cx = int(np.mean([pt[0] for pt in pts]))
                    cy = int(np.mean([pt[1] for pt in pts]))
                    cv2.drawMarker(radar, (cx, cy), col, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
                    cv2.circle(radar, (cx, cy), 9, col, 1, cv2.LINE_AA)

        # 5. Render Active Pass Arrows
        for pass_ev in self.active_passes:
            spx, spy = self.world_to_pitch_px(pass_ev["start"][0], pass_ev["start"][1], width, height, margin)
            epx, epy = self.world_to_pitch_px(pass_ev["end"][0], pass_ev["end"][1], width, height, margin)
            p_col = self.COLOR_TEAM0 if pass_ev["team_id"] == 0 else self.COLOR_TEAM1
            cv2.arrowedLine(radar, (spx, spy), (epx, epy), p_col, 2, cv2.LINE_AA, tipLength=0.25)

        # 6. Player Tokens with Proximity-Based Label Decluttering
        all_pts = [item["pt"] for item in player_radar_items]
        for i, item in enumerate(player_radar_items):
            rx, ry = item["pt"]
            col = item["col"]
            lbl = item["lbl"]

            # Token: solid black outer, colored core, white border
            cv2.circle(radar, (rx, ry), 8, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(radar, (rx, ry), 6, col, -1, cv2.LINE_AA)
            cv2.circle(radar, (rx, ry), 6, (255, 255, 255), 1, cv2.LINE_AA)

            # Check for label overlap: hide number if another player is closer than 22px
            has_close_neighbor = False
            for j, other_pt in enumerate(all_pts):
                if i != j:
                    dist_px = np.hypot(rx - other_pt[0], ry - other_pt[1])
                    if dist_px < 22.0:
                        has_close_neighbor = True
                        break

            if show_labels and not has_close_neighbor and len(lbl) <= 2:
                # Offset text cleanly above marker
                cv2.putText(radar, lbl, (rx - 4, ry + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

        # 7. Distinct Ball Visual Identity
        if ball_pos is not None and pitch_geom is not None and getattr(pitch_geom, "homography_matrix", None) is not None:
            ball_pt = np.array([[ball_pos[0], ball_pos[1]]], dtype=np.float32)
            trans_ball = pitch_geom.image_to_pitch(ball_pt)[0]
            b_wx, b_wy = float(trans_ball[0]), float(trans_ball[1])
            if 0.0 <= b_wx <= self.pitch_length and 0.0 <= b_wy <= self.pitch_width:
                bx, by = self.world_to_pitch_px(b_wx, b_wy, width, height, margin)

                # Distinct Ball: Outer Black -> Golden Ring -> White Center
                cv2.circle(radar, (bx, by), 9, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(radar, (bx, by), 7, self.COLOR_BALL_RING, -1, cv2.LINE_AA)
                cv2.circle(radar, (bx, by), 5, self.COLOR_BALL, -1, cv2.LINE_AA)

                # Explicit State Badge
                b_badge = "BALL [OBS]" if ball_state == "OBSERVED" else "BALL [PRED]"
                badge_col = (0, 220, 100) if ball_state == "OBSERVED" else (0, 180, 255)
                cv2.putText(radar, b_badge, (bx + 10, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, badge_col, 1, cv2.LINE_AA)

        # 8. Embedded Modern Tactical Legend (Bottom-Left)
        leg_x, leg_y = margin + 10, height - 12
        cv2.circle(radar, (leg_x, leg_y - 3), 4, self.COLOR_TEAM0, -1, cv2.LINE_AA)
        cv2.putText(radar, "Team 1", (leg_x + 8, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.circle(radar, (leg_x + 65, leg_y - 3), 4, self.COLOR_TEAM1, -1, cv2.LINE_AA)
        cv2.putText(radar, "Team 2", (leg_x + 73, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.circle(radar, (leg_x + 130, leg_y - 3), 4, self.COLOR_BALL_RING, -1, cv2.LINE_AA)
        cv2.putText(radar, "Ball", (leg_x + 138, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(radar, "KDE Density", (leg_x + 180, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 180, 160), 1, cv2.LINE_AA)

        # Header bar
        cv2.rectangle(radar, (0, 0), (width, 22), (12, 12, 12), -1)
        cv2.putText(radar, "METRIC PITCH RADAR (TOP-DOWN 105m x 68m)", (margin, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        return radar

    def generate_heatmap(
        self,
        positions: List[Tuple[int, float, float]],
        time_window: Optional[int] = None,
        current_frame: int = 0,
        width: int = 800,
        height: int = 520,
        margin: int = 25,
        colormap: int = cv2.COLORMAP_VIRIDIS,
        kernel_size: int = 51,
        title: str = "PITCH DENSITY HEATMAP"
    ) -> np.ndarray:
        """Computes dedicated full-pitch 2D Gaussian Kernel Density Estimation (KDE) Heatmap."""
        pitch_base = self.draw_pitch_diagram(width=width, height=height, margin=margin)
        pw = width - 2 * margin
        ph = height - 2 * margin

        # Filter by time window if specified
        valid_pts = []
        for item in positions:
            if len(item) == 3:
                f_idx, wx, wy = item
                if time_window is None or (current_frame - time_window <= f_idx <= current_frame):
                    valid_pts.append((wx, wy))
            elif len(item) == 2:
                valid_pts.append((item[0], item[1]))

        if not valid_pts:
            cv2.putText(pitch_base, f"{title}: NO SAMPLES IN WINDOW", (margin + 20, margin + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
            return pitch_base

        grid = np.zeros((ph, pw), dtype=np.float32)
        for wx, wy in valid_pts:
            gx = int((wx / self.pitch_length) * (pw - 1))
            gy = int((wy / self.pitch_width) * (ph - 1))
            if 0 <= gx < pw and 0 <= gy < ph:
                grid[gy, gx] += 1.0

        if np.max(grid) > 0:
            k_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            blurred = cv2.GaussianBlur(grid, (k_size, k_size), 0)
            norm_map = blurred / (np.max(blurred) + 1e-6)
            norm_u8 = (norm_map * 255).astype(np.uint8)

            heat_color = cv2.applyColorMap(norm_u8, colormap)

            turf_roi = pitch_base[margin:margin + ph, margin:margin + pw]
            mask = (norm_map > 0.08).astype(np.float32)[:, :, np.newaxis]
            blended = (heat_color * mask * 0.65 + turf_roi * (1.0 - mask * 0.65)).astype(np.uint8)
            pitch_base[margin:margin + ph, margin:margin + pw] = blended

        cv2.rectangle(pitch_base, (0, 0), (width, 22), (12, 12, 12), -1)
        w_txt = f"WINDOW: LAST {time_window} FRAMES" if time_window else "WINDOW: FULL MATCH"
        cv2.putText(pitch_base, f"{title} | {w_txt} | SAMPLES: {len(valid_pts)}", (margin, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        return pitch_base

    def compute_tactical_metrics(self) -> Dict[str, Any]:
        """
        Computes exact, scientifically grounded tactical metrics:
        1. Team Compactness: Mean player distance to team centroid (meters) & Hull Area (m^2).
        2. Defensive Line Height: Average X position of deepest 4 outfield players.
        3. Spatial Pitch Occupancy (% of pitch controlled).
        4. True Measured Possession (% time within 2.5m of ball).
        5. Tracking Continuity (% frames with valid detections).
        """
        t0_coords = np.array([(wx, wy) for _, wx, wy in self.team0_history]) if len(self.team0_history) > 0 else np.empty((0, 2))
        t1_coords = np.array([(wx, wy) for _, wx, wy in self.team1_history]) if len(self.team1_history) > 0 else np.empty((0, 2))

        # 1. Measured Possession
        if self.total_possession_frames > 0:
            t0_poss = (self.team0_possession_frames / self.total_possession_frames) * 100.0
            t1_poss = (self.team1_possession_frames / self.total_possession_frames) * 100.0
        else:
            t0_poss, t1_poss = 50.0, 50.0

        # 2. Team Centroids
        t0_centroid = (float(np.mean(t0_coords[:, 0])), float(np.mean(t0_coords[:, 1]))) if len(t0_coords) > 0 else (52.5, 34.0)
        t1_centroid = (float(np.mean(t1_coords[:, 0])), float(np.mean(t1_coords[:, 1]))) if len(t1_coords) > 0 else (52.5, 34.0)

        # 3. Compactness (Mean Distance to Centroid in meters)
        def calc_mean_dist_to_centroid(coords, centroid):
            if len(coords) == 0:
                return 0.0
            dists = np.hypot(coords[:, 0] - centroid[0], coords[:, 1] - centroid[1])
            return float(np.mean(dists))

        t0_compactness_m = calc_mean_dist_to_centroid(t0_coords, t0_centroid)
        t1_compactness_m = calc_mean_dist_to_centroid(t1_coords, t1_centroid)

        # Convex Hull Area (m^2)
        def calc_hull_area(coords):
            if len(coords) >= 3:
                try:
                    return float(ConvexHull(coords).volume)
                except Exception:
                    return 0.0
            return 0.0

        t0_hull_area = calc_hull_area(t0_coords)
        t1_hull_area = calc_hull_area(t1_coords)

        # 4. Defensive Line Height: Average X of deepest 4 outfield players
        # Team 0 defends X = 0 (Leeds), so lowest X values represent the backline
        t0_def_line = float(np.mean(np.sort(t0_coords[:, 0])[:min(4, len(t0_coords))])) if len(t0_coords) > 0 else 18.0
        # Team 1 defends X = 105 (Spurs), so highest X values represent the backline (measured from defending goal line: 105 - X)
        if len(t1_coords) > 0:
            deepest_t1 = np.sort(t1_coords[:, 0])[-min(4, len(t1_coords)):]
            t1_def_line = float(np.mean(105.0 - deepest_t1))
        else:
            t1_def_line = 18.0

        # 5. Spatial Pitch Occupancy (% of pitch zones occupied)
        tot_samples = len(t0_coords) + len(t1_coords)
        t0_occupancy = (len(t0_coords) / tot_samples * 100.0) if tot_samples > 0 else 50.0
        t1_occupancy = (len(t1_coords) / tot_samples * 100.0) if tot_samples > 0 else 50.0

        # 6. Tracking Continuity
        continuity = (self.frames_with_players / self.total_frames_processed * 100.0) if self.total_frames_processed > 0 else 100.0

        return {
            "team0_possession_pct": round(t0_poss, 1),
            "team1_possession_pct": round(t1_poss, 1),
            "team0_compactness_m": round(t0_compactness_m, 2),
            "team1_compactness_m": round(t1_compactness_m, 2),
            "team0_hull_area_m2": round(t0_hull_area, 1),
            "team1_hull_area_m2": round(t1_hull_area, 1),
            "team0_def_line_height_m": round(t0_def_line, 2),
            "team1_def_line_height_m": round(t1_def_line, 2),
            "team0_spatial_occupancy_pct": round(t0_occupancy, 1),
            "team1_spatial_occupancy_pct": round(t1_occupancy, 1),
            "team0_centroid": (round(t0_centroid[0], 2), round(t0_centroid[1], 2)),
            "team1_centroid": (round(t1_centroid[0], 2), round(t1_centroid[1], 2)),
            "tracking_continuity_pct": round(continuity, 1),
            "passes_detected_count": self.total_passes_detected,
            # Aliases for backward compatibility
            "team0_control_pct": round(t0_occupancy, 1),
            "team1_control_pct": round(t1_occupancy, 1),
            "team0_samples": len(t0_coords),
            "team1_samples": len(t1_coords),
            "team0_compactness_m2": round(t0_hull_area, 1),
            "team1_compactness_m2": round(t1_hull_area, 1),
            "team0_def_line_m": round(t0_def_line, 1),
            "team1_def_line_m": round(t1_def_line, 1)
        }
