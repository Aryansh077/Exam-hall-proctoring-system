"""
Configuration module for the Smart Exam Hall Proctoring System.
Contains detection parameters, camera settings, thresholds, and logging paths.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Union


@dataclass
class ProctoringConfig:
    # Camera / Ingestion
    camera_source: Union[int, str] = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    reconnect_attempts: int = 5
    reconnect_delay_sec: float = 2.0

    # Ultralytics YOLOv8 settings
    yolo_model_path: str = "yolov8n.pt"
    yolo_device: str = "cpu"  # 'cpu' or 'cuda' or '0'
    
    # COCO Class IDs for contraband
    # 0: person, 67: cell phone, 73: book
    coco_person_id: int = 0
    coco_phone_id: int = 67
    coco_book_id: int = 73

    # Confidence Thresholds
    conf_phone: float = 0.40
    conf_book: float = 0.35
    conf_extra_person: float = 0.50

    # MediaPipe Face Mesh settings
    max_num_faces: int = 4
    refine_face_landmarks: bool = True
    min_face_detection_confidence: float = 0.5
    min_face_tracking_confidence: float = 0.5

    # Face Mesh landmark indices
    # Nose tip: 1, Left cheek: 234, Right cheek: 454, Nose bridge: 6, Chin: 152
    landmark_nose_tip: int = 1
    landmark_left_cheek: int = 234
    landmark_right_cheek: int = 454
    landmark_forehead: int = 10
    landmark_chin: int = 152

    # Geometric Head Pose / Gaze Deviation Thresholds
    # Deviation metric: (d_left - d_right) / (d_left + d_right)
    # When turning right (user perspective), distance to right cheek shrinks -> metric becomes positive.
    # When turning left, distance to left cheek shrinks -> metric becomes negative.
    gaze_deviation_threshold: float = 0.22

    # Temporal Streak Buffer (anti-flicker / false-positive suppression)
    temporal_streak_threshold: int = 15  # Consecutive frames to trigger violation
    violation_cooldown_frames: int = 60   # Wait period before re-logging same persistent violation

    # Storage and Logging
    violations_dir: Path = field(default_factory=lambda: Path("violations"))
    audit_log_csv: Path = field(default_factory=lambda: Path("proctoring_audit_log.csv"))

    # HUD Styling Colors (BGR format)
    color_clear: Tuple[int, int, int] = (46, 204, 113)      # Emerald Green
    color_violation: Tuple[int, int, int] = (60, 60, 235)   # Vivid Red
    color_warning: Tuple[int, int, int] = (39, 140, 245)    # Amber Orange
    color_neutral: Tuple[int, int, int] = (240, 240, 240)  # Near White
    color_dark_bg: Tuple[int, int, int] = (24, 24, 24)      # Dark Slate for HUD panels
    
    # Bounding Box Colors by Violation Type
    color_phone_box: Tuple[int, int, int] = (0, 0, 245)     # Red
    color_book_box: Tuple[int, int, int] = (0, 140, 255)    # Orange
    color_person_box: Tuple[int, int, int] = (200, 50, 180) # Magenta/Purple
    color_face_box: Tuple[int, int, int] = (50, 220, 150)   # Teal/Mint
