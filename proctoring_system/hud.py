"""
On-Screen Head-Up Display (HUD) and Visualization Engine.
Renders state-of-the-art surveillance telemetry, color-coded bounding boxes,
top banner status indicators, and facial landmark geometric vectors.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .config import ProctoringConfig
from .detector_yolo import DetectedThreat
from .pose_estimator import HeadPoseResult


class ProctoringHUD:
    """
    Renders high-contrast, informative, and visually striking HUD elements
    onto real-time proctoring video frames.
    """

    def __init__(self, config: ProctoringConfig):
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: np.ndarray,
        pose_res: HeadPoseResult,
        threats: List[DetectedThreat],
        active_violations: List[str],
        accumulating_streaks: Dict[str, int],
        frame_idx: int,
        fps: float,
    ) -> np.ndarray:
        """
        Renders complete HUD onto frame copy.
        """
        canvas = frame.copy()
        h, w, _ = canvas.shape

        # 1. Draw Contraband Bounding Boxes (YOLO)
        for threat in threats:
            self._draw_threat_bbox(canvas, threat)

        # 2. Draw Face Mesh Points & Face Bounding Box
        self._draw_face_geometry(canvas, pose_res)

        # 3. Draw Side Telemetry Dashboard (Glassmorphism overlay)
        self._draw_telemetry_panel(canvas, pose_res, threats, accumulating_streaks, frame_idx, fps)

        # 4. Draw Top Status Banner
        self._draw_top_status_bar(canvas, active_violations, accumulating_streaks, pose_res)

        # 5. Draw Bottom Utility Bar
        self._draw_bottom_bar(canvas, h, w)

        return canvas

    def _draw_threat_bbox(self, canvas: np.ndarray, threat: DetectedThreat):
        """Draws vibrant bounding box with tag banner for detected contraband."""
        x1, y1, x2, y2 = threat.bbox
        color = threat.box_color

        # Outer bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Tech corner accents
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        thickness = 4
        # Top-left
        cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), color, thickness)
        # Top-right
        cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), color, thickness)
        # Bottom-left
        cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), color, thickness)
        # Bottom-right
        cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), color, thickness)

        # Label tag
        label_text = f"CONTRABAND: {threat.label.upper()} ({threat.confidence:.0%})"
        (tw, th), _ = cv2.getTextSize(label_text, self.font, 0.55, 1)
        tag_y1 = max(0, y1 - th - 10)
        tag_y2 = y1
        cv2.rectangle(canvas, (x1, tag_y1), (x1 + tw + 14, tag_y2), color, -1)
        cv2.putText(
            canvas,
            label_text,
            (x1 + 7, tag_y2 - 6),
            self.font,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_face_geometry(self, canvas: np.ndarray, pose_res: HeadPoseResult):
        """Draws face box, nose tip, bilateral cheek markers, and distance vectors."""
        # Draw all detected face bounding boxes
        for fbox in pose_res.all_face_boxes:
            fx1, fy1, fx2, fy2 = fbox
            box_col = self.config.color_face_box if pose_res.face_count == 1 else self.config.color_violation
            cv2.rectangle(canvas, (fx1, fy1), (fx2, fy2), box_col, 1, cv2.LINE_AA)

        if pose_res.face_count == 1 and pose_res.nose_pt and pose_res.left_cheek_pt and pose_res.right_cheek_pt:
            nose = pose_res.nose_pt
            l_cheek = pose_res.left_cheek_pt
            r_cheek = pose_res.right_cheek_pt

            # Distance vectors (cheeks to nose)
            vec_col = (180, 180, 180)
            cv2.line(canvas, l_cheek, nose, vec_col, 1, cv2.LINE_AA)
            cv2.line(canvas, nose, r_cheek, vec_col, 1, cv2.LINE_AA)

            # Bilateral cheek points (landmarks 234 and 454)
            cv2.circle(canvas, l_cheek, 4, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, r_cheek, 4, (0, 220, 255), -1, cv2.LINE_AA)

            # Nose tip point (landmark 1)
            nose_col = (50, 255, 50) if pose_res.orientation == "Forward" else (0, 70, 255)
            cv2.circle(canvas, nose, 5, nose_col, -1, cv2.LINE_AA)
            cv2.drawMarker(canvas, nose, nose_col, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)

    def _draw_telemetry_panel(
        self,
        canvas: np.ndarray,
        pose_res: HeadPoseResult,
        threats: List[DetectedThreat],
        accumulating_streaks: Dict[str, int],
        frame_idx: int,
        fps: float,
    ):
        """Draws live HUD metrics panel with alpha transparency."""
        panel_x = 15
        panel_y = 65
        panel_w = 340
        panel_h = 240

        # Create semi-transparent overlay
        sub_img = canvas[panel_y : panel_y + panel_h, panel_x : panel_x + panel_w]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        # Slate tint
        black_rect[:] = (20, 24, 28)
        blended = cv2.addWeighted(sub_img, 0.35, black_rect, 0.65, 0)
        canvas[panel_y : panel_y + panel_h, panel_x : panel_x + panel_w] = blended

        # Border
        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (80, 90, 100), 1, cv2.LINE_AA)

        # Panel Header
        cv2.putText(canvas, "LIVE PROCTOR TELEMETRY", (panel_x + 12, panel_y + 24), self.font, 0.55, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (panel_x + 12, panel_y + 32), (panel_x + panel_w - 12, panel_y + 32), (60, 70, 80), 1)

        # Telemetry lines
        line_y = panel_y + 54
        spacing = 24

        # FPS & Frame
        cv2.putText(canvas, f"FPS: {fps:.1f}  |  Frame: {frame_idx}", (panel_x + 12, line_y), self.font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        line_y += spacing

        # Head Pose orientation
        pose_col = (50, 230, 100) if pose_res.orientation == "Forward" else (60, 60, 255)
        cv2.putText(canvas, f"Head Pose: {pose_res.orientation}", (panel_x + 12, line_y), self.font, 0.45, pose_col, 1, cv2.LINE_AA)
        line_y += spacing

        # Geometric Gaze Deviation
        dev_str = f"Gaze Dev: {pose_res.deviation_score:+.2f} (Lim: {self.config.gaze_deviation_threshold:.2f})"
        cv2.putText(canvas, dev_str, (panel_x + 12, line_y), self.font, 0.43, (200, 200, 200), 1, cv2.LINE_AA)
        line_y += spacing

        # Candidates / Faces
        face_col = (50, 230, 100) if pose_res.face_count == 1 else (60, 60, 255)
        cv2.putText(canvas, f"Candidates Detected: {pose_res.face_count}", (panel_x + 12, line_y), self.font, 0.45, face_col, 1, cv2.LINE_AA)
        line_y += spacing

        # Contraband Threat count
        threat_count = len(threats)
        threat_col = (50, 230, 100) if threat_count == 0 else (60, 60, 255)
        cv2.putText(canvas, f"Contraband Threats: {threat_count}", (panel_x + 12, line_y), self.font, 0.45, threat_col, 1, cv2.LINE_AA)
        line_y += spacing

        # Temporal streak tracker status
        active_streaks = [f"{k.split('_')[0]}:{v}/{self.config.temporal_streak_threshold}" for k, v in accumulating_streaks.items() if v > 0]
        streak_display = ", ".join(active_streaks) if active_streaks else "None (Stable)"
        streak_col = (255, 180, 50) if active_streaks else (150, 160, 170)
        cv2.putText(canvas, f"Streak: {streak_display}", (panel_x + 12, line_y), self.font, 0.40, streak_col, 1, cv2.LINE_AA)

    def _draw_top_status_bar(
        self,
        canvas: np.ndarray,
        active_violations: List[str],
        accumulating_streaks: Dict[str, int],
        pose_res: HeadPoseResult,
    ):
        """Renders dynamic high-visibility status bar at the top of the screen."""
        _, w, _ = canvas.shape
        bar_h = 50

        if active_violations:
            # Active Violation Confirmed (RED)
            bg_color = self.config.color_violation
            violation_title = " | ".join(active_violations).replace("_", " ")
            status_text = f"ALERT: VIOLATION CONFIRMED -> {violation_title}"
            text_color = (255, 255, 255)
        elif any(count >= (self.config.temporal_streak_threshold // 2) for count in accumulating_streaks.values()):
            # Warning: Streak accumulating (AMBER)
            bg_color = self.config.color_warning
            highest = max(accumulating_streaks.items(), key=lambda x: x[1])
            status_text = f"WARNING: SUSPICIOUS ACTIVITY STREAK ({highest[0]}: {highest[1]}/{self.config.temporal_streak_threshold})"
            text_color = (20, 20, 20)
        else:
            # Candidate Clear & Normal (GREEN)
            bg_color = self.config.color_clear
            status_text = "STATUS: INTEGRITY VERIFIED (CLEAR)"
            text_color = (255, 255, 255)

        # Draw solid or semi-transparent banner
        cv2.rectangle(canvas, (0, 0), (w, bar_h), bg_color, -1)
        # Bottom edge accent
        cv2.line(canvas, (0, bar_h), (w, bar_h), (255, 255, 255), 2)

        # Center the text
        (tw, th), _ = cv2.getTextSize(status_text, self.font, 0.70, 2)
        text_x = max(20, (w - tw) // 2)
        text_y = (bar_h + th) // 2
        cv2.putText(canvas, status_text, (text_x, text_y), self.font, 0.70, text_color, 2, cv2.LINE_AA)

    def _draw_bottom_bar(self, canvas: np.ndarray, h: int, w: int):
        """Draws subtle lower status footer with keybindings and copyright/version."""
        bar_y1 = h - 28
        cv2.rectangle(canvas, (0, bar_y1), (w, h), (15, 18, 22), -1)
        cv2.line(canvas, (0, bar_y1), (w, bar_y1), (45, 55, 65), 1)

        key_hint = "CONTROLS: Press [Q] to Gracefully Exit  |  [R] Reset Streaks  |  Smart Proctor v1.0"
        cv2.putText(canvas, key_hint, (20, h - 9), self.font, 0.40, (180, 190, 200), 1, cv2.LINE_AA)
