"""
Geometric Head Pose Engine and Candidate Verification Module.
Utilizes MediaPipe Face Mesh (468/478 landmarks) running CPU inference.
Computes relative Euclidean distances between the nose tip and bilateral cheeks
to measure gaze deviation, and tracks candidate presence (missing / multiple).
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp

from .config import ProctoringConfig

logger = logging.getLogger(__name__)


@dataclass
class HeadPoseResult:
    face_count: int = 0
    orientation: str = "Unknown"
    deviation_score: float = 0.0
    distance_left: float = 0.0
    distance_right: float = 0.0
    ratio: float = 1.0
    nose_pt: Optional[Tuple[int, int]] = None
    left_cheek_pt: Optional[Tuple[int, int]] = None
    right_cheek_pt: Optional[Tuple[int, int]] = None
    chin_pt: Optional[Tuple[int, int]] = None
    forehead_pt: Optional[Tuple[int, int]] = None
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    all_face_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    violation_type: Optional[str] = None
    confidence: float = 0.0


class GeometricHeadPoseEngine:
    """
    Real-time CPU-based Head Pose and Candidate Tracker using MediaPipe Face Mesh.
    """

    def __init__(self, config: ProctoringConfig):
        self.config = config
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=config.max_num_faces,
            refine_landmarks=config.refine_face_landmarks,
            min_detection_confidence=config.min_face_detection_confidence,
            min_tracking_confidence=config.min_face_tracking_confidence,
        )
        logger.info("MediaPipe Face Mesh engine initialized successfully.")

    def process_frame(self, frame_bgr: np.ndarray) -> HeadPoseResult:
        """
        Processes BGR video frame and computes geometric facial metrics.
        """
        h, w, _ = frame_bgr.shape
        # MediaPipe requires RGB input
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Performance flag
        frame_rgb.flags.writeable = False
        results = self.face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            # Candidate Missing violation
            return HeadPoseResult(
                face_count=0,
                orientation="No Face Detected",
                deviation_score=0.0,
                violation_type="CANDIDATE_MISSING",
                confidence=1.0,
            )

        face_count = len(results.multi_face_landmarks)
        all_boxes = []
        for face_landmarks in results.multi_face_landmarks:
            xs = [int(pt.x * w) for pt in face_landmarks.landmark]
            ys = [int(pt.y * h) for pt in face_landmarks.landmark]
            all_boxes.append((max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys))))

        # Case 1: Multiple Faces Detected
        if face_count > 1:
            return HeadPoseResult(
                face_count=face_count,
                orientation=f"Multiple Faces ({face_count})",
                all_face_boxes=all_boxes,
                violation_type="MULTIPLE_FACES",
                confidence=min(0.99, 0.70 + (face_count * 0.1)),
            )

        # Case 2: Exactly 1 candidate face present - analyze geometric head pose
        primary_landmarks = results.multi_face_landmarks[0].landmark

        def get_coords(idx: int) -> Tuple[int, int]:
            pt = primary_landmarks[idx]
            return int(pt.x * w), int(pt.y * h)

        nose_pt = get_coords(self.config.landmark_nose_tip)
        left_cheek_pt = get_coords(self.config.landmark_left_cheek)
        right_cheek_pt = get_coords(self.config.landmark_right_cheek)
        chin_pt = get_coords(self.config.landmark_chin)
        forehead_pt = get_coords(self.config.landmark_forehead)

        # Compute Euclidean distances
        # Nose tip to bilateral cheek markers
        d_left = math.dist(nose_pt, left_cheek_pt)
        d_right = math.dist(nose_pt, right_cheek_pt)

        # Normalized Gaze Deviation Index:
        # Range is approximately -1.0 to +1.0
        # Positive: nose is farther from left cheek and closer to right cheek (looking toward screen right)
        # Negative: nose is farther from right cheek and closer to left cheek (looking toward screen left)
        denom = d_left + d_right + 1e-6
        deviation_score = (d_right - d_left) / denom
        ratio = d_right / (d_left + 1e-6)

        orientation = "Forward"
        violation_type = None
        confidence = 0.0

        thresh = self.config.gaze_deviation_threshold
        if deviation_score > thresh:
            orientation = "Looking Left"
            violation_type = "GAZE_DEVIATION_LEFT"
            # Normalize confidence based on severity past threshold
            confidence = min(0.99, 0.65 + abs(deviation_score - thresh) * 1.5)
        elif deviation_score < -thresh:
            orientation = "Looking Right"
            violation_type = "GAZE_DEVIATION_RIGHT"
            confidence = min(0.99, 0.65 + abs(deviation_score - (-thresh)) * 1.5)
        else:
            orientation = "Forward"
            violation_type = None
            confidence = 0.0

        return HeadPoseResult(
            face_count=1,
            orientation=orientation,
            deviation_score=deviation_score,
            distance_left=d_left,
            distance_right=d_right,
            ratio=ratio,
            nose_pt=nose_pt,
            left_cheek_pt=left_cheek_pt,
            right_cheek_pt=right_cheek_pt,
            chin_pt=chin_pt,
            forehead_pt=forehead_pt,
            face_bbox=all_boxes[0] if all_boxes else None,
            all_face_boxes=all_boxes,
            violation_type=violation_type,
            confidence=confidence,
        )

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "face_mesh") and self.face_mesh is not None:
            self.face_mesh.close()
