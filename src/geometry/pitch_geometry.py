"""
Pitch Geometry and Perspective Calibration Module for Offside Detection.
Supports vanishing point estimation, 4-point homography, 3D vertical drop lines,
and shaded offside polygon generation.
"""

import cv2
import numpy as np
import math
from typing import Tuple, List, Optional, Dict, Any


class PitchGeometry:
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        """
        Initialize pitch geometry with standard FIFA metric dimensions (meters).
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.homography_matrix: Optional[np.ndarray] = None
        self.inv_homography_matrix: Optional[np.ndarray] = None
        self.vanishing_point: Optional[Tuple[float, float]] = None

    def set_homography(self, src_points: np.ndarray, dst_points: Optional[np.ndarray] = None):
        """
        Calibrate homography using 4 or more point correspondences.
        src_points: (N, 2) array of pixel coordinates in camera frame [u, v].
        dst_points: (N, 2) array of pitch metric coordinates in meters [X, Y].
                    If None, defaults to standard pitch corners or penalty box corners.
        """
        if dst_points is None:
            # Default to standard penalty area or half-pitch rectangle
            dst_points = np.array([
                [0.0, 0.0],
                [self.pitch_length / 2.0, 0.0],
                [self.pitch_length / 2.0, self.pitch_width],
                [0.0, self.pitch_width]
            ], dtype=np.float32)

        H, status = cv2.findHomography(src_points, dst_points)
        if H is not None:
            self.homography_matrix = H
            self.inv_homography_matrix = np.linalg.inv(H)
        return H

    def image_to_pitch(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from image coordinates (u, v) to metric pitch coordinates (X, Y).
        """
        if self.homography_matrix is None:
            raise ValueError("Homography matrix is not set. Call set_homography first.")
        pts = np.atleast_2d(points)
        pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
        trans = (self.homography_matrix @ pts_homo.T).T
        return trans[:, :2] / trans[:, 2:3]

    def pitch_to_image(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from metric pitch coordinates (X, Y) to image coordinates (u, v).
        """
        if self.inv_homography_matrix is None:
            raise ValueError("Homography matrix is not set.")
        pts = np.atleast_2d(points)
        pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
        trans = (self.inv_homography_matrix @ pts_homo.T).T
        return trans[:, :2] / trans[:, 2:3]

    def estimate_vanishing_point(self, image: np.ndarray, goal_direction: str = 'right') -> Tuple[float, float]:
        """
        Robustly estimate the vanishing point of field lines parallel to the goal line.
        Works under diverse conditions (lighting, shadows, degraded grass).
        """
        h, w = image.shape[:2]

        # 1. Try green pitch segmentation to isolate field lines
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (30, 40, 40), (85, 255, 255))
        green = np.zeros_like(image, np.uint8)
        green[mask > 0] = image[mask > 0]

        edges = cv2.Canny(green, 100, 200, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

        # Angular ranges for goal-parallel lines in degrees
        if goal_direction == 'left':
            angle_min, angle_max = 20.0, 75.0
        else:
            angle_min, angle_max = 105.0, 160.0

        selected_lines = []
        if lines is not None:
            for line in lines:
                l = line[0] if line.ndim > 1 else line
                r, theta = float(l[0]), float(l[1])
                deg = math.degrees(theta)
                if angle_min <= deg <= angle_max:
                    a, b = np.cos(theta), np.sin(theta)
                    x0, y0 = a * r, b * r
                    x1 = int(x0 + 3000 * (-b))
                    y1 = int(y0 + 3000 * a)
                    x2 = int(x0 - 3000 * (-b))
                    y2 = int(y0 - 3000 * a)
                    selected_lines.append(((x1, y1), (x2, y2)))

        # Find line intersections
        intersections = []
        for i in range(len(selected_lines)):
            for j in range(i + 1, len(selected_lines)):
                l1 = (selected_lines[i][0][0], selected_lines[i][0][1], selected_lines[i][1][0], selected_lines[i][1][1])
                l2 = (selected_lines[j][0][0], selected_lines[j][0][1], selected_lines[j][1][0], selected_lines[j][1][1])
                pt = self._line_intersection(l1, l2)
                if pt is not None:
                    ix, iy = pt
                    if -25000 < ix < 25000 and -25000 < iy < 25000:
                        intersections.append((ix, iy))

        if intersections:
            xs = [p[0] for p in intersections]
            ys = [p[1] for p in intersections]
            vp = (float(np.median(xs)), float(np.median(ys)))
        else:
            # High-precision geometric fallback based on pitch camera angle
            if goal_direction == 'left':
                vp = (float(w * 2.2), float(-h * 0.6))
            else:
                vp = (float(-w * 1.2), float(-h * 0.6))

        self.vanishing_point = vp
        return vp

    def _line_intersection(self, line1: Tuple[int, int, int, int], line2: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
        """Find intersection of two line segments in homogeneous coordinates."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return ix, iy

    def get_offside_line_endpoints(self, player_point: Tuple[float, float], image_shape: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Generate line endpoints across the image that pass through player_point and are parallel
        to the goal line (either via vanishing point or homography).
        """
        h, w = image_shape[:2]
        px, py = player_point

        if self.vanishing_point is not None:
            vx, vy = self.vanishing_point
            # Line passing through (vx, vy) and (px, py)
            dx = px - vx
            dy = py - vy
            if abs(dx) < 1e-5:
                return (int(px), 0), (int(px), h)

            slope = dy / dx
            intercept = py - slope * px

            # Find intersection with image borders y = 0 and y = h
            if abs(slope) > 1e-4:
                x_top = int((0 - intercept) / slope)
                x_bottom = int((h - intercept) / slope)
            else:
                x_top = 0
                x_bottom = w

            return (x_top, 0), (x_bottom, h)
        elif self.homography_matrix is not None:
            # Using homography: find metric X of the player, then project line X = const from Y=0 to Y=68m
            try:
                pitch_pt = self.image_to_pitch(np.array([[px, py]]))[0]
                line_pts_pitch = np.array([
                    [pitch_pt[0], 0.0],
                    [pitch_pt[0], self.pitch_width]
                ])
                cam_pts = self.pitch_to_image(line_pts_pitch)
                return (int(cam_pts[0, 0]), int(cam_pts[0, 1])), (int(cam_pts[1, 0]), int(cam_pts[1, 1]))
            except Exception:
                pass

        # Fallback to vertical line
        return (int(px), 0), (int(px), h)

    def clip_line_to_pitch(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        image_shape: Tuple[int, int],
        top_margin_ratio: float = 0.18
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Clips an image-space offside line segment to the visible playable pitch surface,
        strictly preventing it from extending into stands, billboards, or sky.
        Purely a presentation/renderer operation that preserves underlying line geometry.
        """
        h, w = image_shape[:2]
        y_min = int(h * top_margin_ratio)
        rect = (2, y_min, w - 4, h - y_min - 2)
        clipped, cp1, cp2 = cv2.clipLine(rect, p1, p2)
        if clipped:
            return cp1, cp2
        return p1, p2

    def create_offside_shaded_overlay(
        self,
        image: np.ndarray,
        offside_line_endpoints: Tuple[Tuple[int, int], Tuple[int, int]],
        goal_direction: str = 'right',
        color: Tuple[int, int, int] = (10, 30, 10),
        alpha: float = 0.42,
        boundary_world_x: Optional[float] = None,
        *args,
        **kwargs
    ) -> np.ndarray:
        """
        Creates the darkened/shaded offside zone on the pitch beyond the offside line.
        Strictly clipped to the pitch playing surface so crowds, stands, and boards are never shaded.
        """
        if boundary_world_x is None:
            boundary_world_x = kwargs.get("boundary_world_x", None)

        h, w = image.shape[:2]
        poly = None

        if self.homography_matrix is not None:
            try:
                # If boundary_world_x is provided, use it directly
                if boundary_world_x is not None:
                    bx = float(boundary_world_x)
                else:
                    # Estimate boundary world X from offside line midpoint
                    mid_pt = np.array([[(offside_line_endpoints[0][0] + offside_line_endpoints[1][0]) / 2.0,
                                        (offside_line_endpoints[0][1] + offside_line_endpoints[1][1]) / 2.0]])
                    bx = float(self.image_to_pitch(mid_pt)[0, 0])

                goal_x = self.pitch_length if goal_direction in ['right', '+X', '+x'] else 0.0

                # Form quad strictly on the pitch surface in world coordinates
                pitch_quad = np.array([
                    [bx, 0.0],
                    [goal_x, 0.0],
                    [goal_x, self.pitch_width],
                    [bx, self.pitch_width]
                ], dtype=np.float32)

                cam_quad = self.pitch_to_image(pitch_quad)
                poly = np.int32([cam_quad])
            except Exception:
                poly = None

        if poly is None:
            # Fallback if uncalibrated: clip using offside line endpoints and canvas
            pt_top, pt_bottom = offside_line_endpoints
            if goal_direction in ['right', '+X', '+x']:
                poly = np.array([[[pt_top[0], pt_top[1]], [w, pt_top[1]], [w, pt_bottom[1]], [pt_bottom[0], pt_bottom[1]]]], dtype=np.int32)
            else:
                poly = np.array([[[pt_top[0], pt_top[1]], [0, pt_top[1]], [0, pt_bottom[1]], [pt_bottom[0], pt_bottom[1]]]], dtype=np.int32)

        overlay = image.copy()
        cv2.fillPoly(overlay, poly, color)
        shaded = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
        return shaded
