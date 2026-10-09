"""
VAR Broadcast Visualizer and Graphic Overlay Engine.
Renders calibrated pitch offside lines, 3D vertical drop lines, translucent offside shaded zones,
Premier League style VAR decision banners, and 2D tactical radar minimaps.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Any


class VARRenderer:
    def __init__(self):
        # Color definitions (BGR)
        self.COLOR_DEFENDER_LINE = (255, 60, 20)     # Blue / Cyan
        self.COLOR_ATTACKER_LINE = (30, 30, 240)     # Red
        self.COLOR_BALL_LINE = (0, 215, 255)         # Yellow / Gold
        self.COLOR_DROP_LINE = (0, 255, 255)         # Bright Yellow
        self.COLOR_SHADED_ZONE = (15, 30, 15)        # Darkened green pitch
        self.COLOR_TEXT = (255, 255, 255)

    def render_var_overlay(
        self,
        image: np.ndarray,
        offside_result: Any,
        pitch_geom: Optional[Any] = None,
        draw_skeletons: bool = True,
        draw_boxes: bool = False,
        draw_radar: bool = True
    ) -> np.ndarray:
        """
        Produce complete broadcast VAR visualization matching Premier League broadcast graphics.
        """
        output = image.copy()
        h, w = output.shape[:2]

        # 1. Draw Shaded Offside Zone beyond defender line
        if offside_result.offside_line is not None and pitch_geom is not None:
            output = pitch_geom.create_offside_shaded_overlay(
                output,
                offside_result.offside_line,
                goal_direction=offside_result.attack_direction,
                color=self.COLOR_SHADED_ZONE,
                alpha=0.42,
                boundary_world_x=getattr(offside_result, 'offside_boundary_world_x', None)
            )

        # 2. Draw Horizontal Defensive Baseline Line
        if offside_result.offside_line is not None:
            pt1, pt2 = offside_result.offside_line
            # Black border + bright cyan center
            cv2.line(output, pt1, pt2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.line(output, pt1, pt2, self.COLOR_DEFENDER_LINE, 2, cv2.LINE_AA)
            cv2.putText(output, "DEFENSIVE BASELINE", (20, pt1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(output, "DEFENSIVE BASELINE", (20, pt1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 0), 1, cv2.LINE_AA)

        # 3. Draw Horizontal Attacker Line if Offside
        if offside_result.attacker_line is not None:
            pt1, pt2 = offside_result.attacker_line
            cv2.line(output, pt1, pt2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.line(output, pt1, pt2, self.COLOR_ATTACKER_LINE, 2, cv2.LINE_AA)
            cv2.putText(output, f"ATTACKER LINE (+{offside_result.margin_val:.2f}m)", (20, pt1[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(output, f"ATTACKER LINE (+{offside_result.margin_val:.2f}m)", (20, pt1[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (30, 30, 255), 1, cv2.LINE_AA)

        # 4. Draw 3D Vertical Drop Lines from player playable body parts down to pitch
        # For the second-last defender
        if offside_result.second_last_defender is not None:
            self._draw_vertical_drop(output, offside_result.second_last_defender, self.COLOR_DEFENDER_LINE)

        # For offside attacking players
        for att in offside_result.offside_players:
            self._draw_vertical_drop(output, att, self.COLOR_ATTACKER_LINE)

        # 5. Optionally Draw Bounding Boxes and Skeletons
        for p in offside_result.all_players:
            if draw_boxes:
                x1, y1, x2, y2 = p.bbox
                box_color = self.COLOR_DEFENDER_LINE if p.team_id == offside_result.defend_team_id else self.COLOR_ATTACKER_LINE
                cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)

            if draw_skeletons and hasattr(p, 'keypoints') and len(p.keypoints) > 0:
                self._draw_player_skeleton(output, p)

        # 6. Render Premier League Style Top Header ("VAR CHECKING OFFSIDE")
        self._draw_top_var_header(output, offside_result)

        # 7. Render Premier League Style Bottom Decision Banner ("OFFSIDE" / "ONSIDE")
        self._draw_decision_banner(output, offside_result)

        # 8. Optionally draw 2D Tactical Radar Minimap in top-right corner
        if draw_radar:
            self._draw_tactical_radar(output, offside_result)

        return output

    def _draw_vertical_drop(self, image: np.ndarray, player: Any, color: Tuple[int, int, int]):
        """
        Draws the vertical projection line from player's leading playable body part
        down to the pitch ground level (as in Premier League / FIFA SAOT VAR reviews).
        """
        lead_x, lead_y = int(player.leading_point[0]), int(player.leading_point[1])
        ground_x, ground_y = int(player.ground_point[0]), int(player.ground_point[1])

        # If leading part is near ground (e.g. boot), use shoulder/head or bbox top for vertical reference
        top_y = lead_y
        top_x = lead_x
        if ground_y - top_y < 30:
            if hasattr(player, 'bbox'):
                top_y = int(player.bbox[1])
                top_x = lead_x

        # Vertical line projection x-coordinate (at leading playable x)
        proj_x = lead_x
        start_y = min(top_y, ground_y - 25)

        # 1. High-contrast dual-layer vertical line (black shadow + bright neon yellow/cyan)
        cv2.line(image, (proj_x, start_y), (proj_x, ground_y), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(image, (proj_x, start_y), (proj_x, ground_y), (0, 245, 255), 2, cv2.LINE_AA)

        # 2. Upper body anchor reticle
        part_name = getattr(player, 'leading_part_name', 'body').upper()
        cv2.circle(image, (lead_x, lead_y), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (lead_x, lead_y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (lead_x, lead_y), 7, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Ground intersection anchor marker
        cv2.circle(image, (proj_x, ground_y), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (proj_x, ground_y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(image, (proj_x, ground_y), 8, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Small tag indicating vertical projection
        tag_text = f"{part_name} LINE"
        cv2.putText(image, tag_text, (proj_x + 6, start_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, tag_text, (proj_x + 6, start_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    def _draw_player_skeleton(self, image: np.ndarray, player: Any):
        """Draw player skeleton with keypoints."""
        kpts = player.keypoints
        # Skeleton pairs (COCO format)
        pairs = [
            (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        color = (200, 200, 200)

        for p1, p2 in pairs:
            if p1 < len(kpts) and p2 < len(kpts):
                pt1, pt2 = kpts[p1], kpts[p2]
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 1, cv2.LINE_AA)

    def _draw_top_var_header(self, image: np.ndarray, offside_result: Any):
        """Renders Premier League top-left VAR status badge."""
        badge_w, badge_h = 320, 44
        badge = np.zeros((badge_h, badge_w, 3), dtype=np.uint8)
        badge[:] = (40, 10, 50)  # Deep purple Premier League background

        # VAR Logo box
        cv2.rectangle(badge, (4, 4), (54, 40), (255, 255, 255), -1)
        cv2.putText(badge, "VAR", (8, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (40, 10, 50), 2, cv2.LINE_AA)

        # Status text
        status_text = "OFFSIDE REVIEW" if offside_result.decision == "OFFSIDE" else "CHECKING OFFSIDE"
        cv2.putText(badge, status_text, (65, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Blend onto image at top-left (x=25, y=20)
        x_off, y_off = 25, 20
        h_b, w_b = badge.shape[:2]
        if y_off + h_b < image.shape[0] and x_off + w_b < image.shape[1]:
            roi = image[y_off:y_off + h_b, x_off:x_off + w_b]
            blended = cv2.addWeighted(badge, 0.9, roi, 0.1, 0)
            image[y_off:y_off + h_b, x_off:x_off + w_b] = blended
            cv2.rectangle(image, (x_off, y_off), (x_off + w_b, y_off + h_b), (255, 255, 255), 1)

    def _draw_decision_banner(self, image: np.ndarray, offside_result: Any):
        """Renders Premier League bottom broadcast decision card."""
        h, w = image.shape[:2]
        banner_w, banner_h = 440, 64
        x_off = int((w - banner_w) / 2)
        y_off = h - banner_h - 30

        if x_off < 0 or y_off < 0:
            return

        is_offside = (offside_result.decision == "OFFSIDE")
        is_undetermined = (offside_result.decision == "UNDETERMINED")

        if is_undetermined:
            bg_color = (60, 60, 60)  # Neutral dark gray for UNDETERMINED
        elif is_offside:
            bg_color = (20, 20, 210)  # Red for OFFSIDE
        else:
            bg_color = (30, 160, 40)  # Green for ONSIDE

        banner = np.zeros((banner_h, banner_w, 3), dtype=np.uint8)
        banner[:] = (30, 15, 35)  # Dark purple border/backing

        # Main decision block
        block_w = 230 if is_undetermined else 210
        cv2.rectangle(banner, (banner_w - block_w, 0), (banner_w, banner_h), bg_color, -1)

        # Left label: Info / Margin
        if is_undetermined:
            label_title = "CALIBRATION INVALID" if getattr(offside_result, 'calibration_status', '') == "UNCALIBRATED" or getattr(offside_result, 'geometry_status', '') == "INVALID" else "EVIDENCE INVALID"
            margin_str = "NO METRIC DECISION"
        elif is_offside:
            label_title = "OFFSIDE MARGIN"
            margin_str = f"+{offside_result.margin_val:.2f}{offside_result.margin_unit}"
        else:
            label_title = "VAR DECISION"
            margin_str = "LEGAL POSITION"

        cv2.putText(banner, label_title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 200), 1, cv2.LINE_AA)
        cv2.putText(banner, margin_str, (14, 52), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)

        # Right label: Bold decision
        if is_undetermined:
            dec_str = "UNDETERMINED"
            cv2.putText(banner, dec_str, (banner_w - block_w + 12, 42), cv2.FONT_HERSHEY_DUPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            dec_str = "OFFSIDE" if is_offside else "ONSIDE"
            cv2.putText(banner, dec_str, (banner_w - block_w + 22, 44), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

        # Draw banner onto frame
        roi = image[y_off:y_off + banner_h, x_off:x_off + banner_w]
        blended = cv2.addWeighted(banner, 0.95, roi, 0.05, 0)
        image[y_off:y_off + banner_h, x_off:x_off + banner_w] = blended
        cv2.rectangle(image, (x_off, y_off), (x_off + banner_w, y_off + banner_h), (255, 255, 255), 2)

    def _draw_tactical_radar(self, image: np.ndarray, offside_result: Any):
        """Draws a 2D top-down pitch radar in the upper-right corner."""
        h_img, w_img = image.shape[:2]
        radar_w, radar_h = 240, 150
        x_off = w_img - radar_w - 25
        y_off = 20

        radar = np.zeros((radar_h, radar_w, 3), dtype=np.uint8)
        radar[:] = (35, 75, 35)  # Grass green

        # Draw pitch boundaries & markings
        cv2.rectangle(radar, (8, 8), (radar_w - 8, radar_h - 8), (255, 255, 255), 1)
        half_x = radar_w // 2
        cv2.line(radar, (half_x, 8), (half_x, radar_h - 8), (255, 255, 255), 1)
        cv2.circle(radar, (half_x, radar_h // 2), 22, (255, 255, 255), 1)

        # Penalty boxes
        cv2.rectangle(radar, (8, 35), (45, radar_h - 35), (255, 255, 255), 1)
        cv2.rectangle(radar, (radar_w - 45, 35), (radar_w - 8, radar_h - 35), (255, 255, 255), 1)

        # Plot players on radar
        for p in offside_result.all_players:
            px, py = p.ground_point
            norm_x = max(0.0, min(1.0, px / w_img))
            norm_y = max(0.0, min(1.0, py / h_img))
            rx = int(8 + norm_x * (radar_w - 16))
            ry = int(8 + norm_y * (radar_h - 16))

            if p.team_id == offside_result.defend_team_id:
                p_color = (255, 120, 0)
            elif p.offside_decision == "OFFSIDE":
                p_color = (0, 0, 255)
            else:
                p_color = (0, 255, 100)

            cv2.circle(radar, (rx, ry), 4, p_color, -1, cv2.LINE_AA)

        # Plot offside line on radar
        if offside_result.second_last_defender is not None:
            def_x = offside_result.second_last_defender.ground_point[0]
            norm_dx = max(0.0, min(1.0, def_x / w_img))
            rx_line = int(8 + norm_dx * (radar_w - 16))
            cv2.line(radar, (rx_line, 8), (rx_line, radar_h - 8), (0, 255, 255), 1, cv2.LINE_AA)

        # Overlay radar onto image
        roi = image[y_off:y_off + radar_h, x_off:x_off + radar_w]
        blended = cv2.addWeighted(radar, 0.85, roi, 0.15, 0)
        image[y_off:y_off + radar_h, x_off:x_off + radar_w] = blended
        cv2.rectangle(image, (x_off, y_off), (x_off + radar_w, y_off + radar_h), (255, 255, 255), 1)
        cv2.putText(image, "2D PITCH RADAR", (x_off + 10, y_off - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
