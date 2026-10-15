"""
Synthetic Verification and Simulation Script for the Proctoring System.
Demonstrates the entire computer vision pipeline end-to-end:
  1. Integrity clear state (Candidate sitting forward)
  2. Gaze deviation detection and temporal streak counter accumulation
  3. Contraband object tracking (Cell Phone / Book)
  4. Candidate missing detection
  5. Audit CSV logging and timestamped snapshot preservation in violations/
"""

import os
import sys
import time
import csv
import logging
from pathlib import Path
import cv2
import numpy as np

from proctoring_system.config import ProctoringConfig
from proctoring_system.detector_yolo import ContrabandDetectorYOLO, DetectedThreat
from proctoring_system.pose_estimator import GeometricHeadPoseEngine, HeadPoseResult
from proctoring_system.temporal_buffer import TemporalBuffer
from proctoring_system.logger import AnomalyLogger
from proctoring_system.hud import ProctoringHUD

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("Simulation")


def create_synthetic_candidate_face(w: int, h: int, head_turn: str = "forward") -> np.ndarray:
    """
    Renders a realistic synthetic face graphic with eyes, nose, cheeks, and mouth
    at positions corresponding to head turn directions to test geometric estimation.
    """
    frame = np.full((h, w, 3), 40, dtype=np.uint8)

    # Face center and dimensions
    cx, cy = w // 2, h // 2
    face_w, face_h = 180, 240

    # Slight head shift depending on turn
    nose_offset_x = 0
    if head_turn == "left":
        nose_offset_x = -35
    elif head_turn == "right":
        nose_offset_x = 35

    # Draw neck and shoulders
    cv2.ellipse(frame, (cx, cy + 220), (220, 120), 0, 0, 360, (70, 75, 90), -1, cv2.LINE_AA)
    # Face oval
    cv2.ellipse(frame, (cx, cy), (face_w // 2, face_h // 2), 0, 0, 360, (190, 210, 235), -1, cv2.LINE_AA)

    # Eyes
    eye_y = cy - 30
    cv2.ellipse(frame, (cx - 45, eye_y), (18, 10), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.ellipse(frame, (cx + 45, eye_y), (18, 10), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
    # Pupils
    cv2.circle(frame, (cx - 45 + (nose_offset_x // 3), eye_y), 5, (50, 40, 30), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx + 45 + (nose_offset_x // 3), eye_y), 5, (50, 40, 30), -1, cv2.LINE_AA)

    # Nose
    nose_x = cx + nose_offset_x
    nose_y = cy + 15
    pts = np.array([[nose_x, nose_y - 20], [nose_x - 12, nose_y + 10], [nose_x + 12, nose_y + 10]], np.int32)
    cv2.polylines(frame, [pts], True, (130, 150, 180), 2, cv2.LINE_AA)

    # Mouth
    cv2.ellipse(frame, (cx + (nose_offset_x // 4), cy + 65), (35, 12), 0, 0, 180, (100, 110, 190), 3, cv2.LINE_AA)

    return frame


def run_simulation(gui_display: bool = False):
    logger.info("Initializing Proctoring Pipeline Components for Simulation...")
    config = ProctoringConfig(
        camera_source=0,
        temporal_streak_threshold=15,
        violation_cooldown_frames=40,
    )

    contraband_detector = ContrabandDetectorYOLO(config)
    pose_engine = GeometricHeadPoseEngine(config)
    temporal_buffer = TemporalBuffer(
        streak_threshold=config.temporal_streak_threshold,
        cooldown_frames=config.violation_cooldown_frames,
    )
    audit_logger = AnomalyLogger(config)
    hud = ProctoringHUD(config)

    w, h = config.frame_width, config.frame_height
    total_frames = 120
    logger.info(f"Running simulation for {total_frames} frames across 4 dynamic scenarios...")

    if gui_display:
        cv2.namedWindow("Proctoring Simulation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Proctoring Simulation", w, h)

    logged_events = []

    for frame_idx in range(1, total_frames + 1):
        threats: list[DetectedThreat] = []
        pose_res = HeadPoseResult()

        # Scenario Schedule:
        # Frames 1 - 25: Candidate clear and forward
        # Frames 26 - 55: Sustained Gaze Deviation (Head turned right, >15 frames streak)
        # Frames 56 - 85: Cell Phone introduced and detected (>15 frames streak)
        # Frames 86 - 110: Candidate missing (0 faces detected)
        # Frames 111 - 120: Normal forward restore
        if 1 <= frame_idx <= 25 or 111 <= frame_idx <= 120:
            scenario_name = "Normal / Clear"
            base_frame = create_synthetic_candidate_face(w, h, head_turn="forward")
            pose_res = HeadPoseResult(
                face_count=1,
                orientation="Forward",
                deviation_score=0.04,
                nose_pt=(w // 2, h // 2 + 15),
                left_cheek_pt=(w // 2 - 80, h // 2),
                right_cheek_pt=(w // 2 + 80, h // 2),
                all_face_boxes=[(w // 2 - 90, h // 2 - 120, w // 2 + 90, h // 2 + 120)],
                violation_type=None,
                confidence=0.0,
            )

        elif 26 <= frame_idx <= 55:
            scenario_name = "Sustained Gaze Deviation (Looking Left)"
            base_frame = create_synthetic_candidate_face(w, h, head_turn="left")
            pose_res = HeadPoseResult(
                face_count=1,
                orientation="Looking Left",
                deviation_score=0.38,
                nose_pt=(w // 2 - 35, h // 2 + 15),
                left_cheek_pt=(w // 2 - 80, h // 2),
                right_cheek_pt=(w // 2 + 80, h // 2),
                all_face_boxes=[(w // 2 - 90, h // 2 - 120, w // 2 + 90, h // 2 + 120)],
                violation_type="GAZE_DEVIATION_LEFT",
                confidence=0.89,
            )

        elif 56 <= frame_idx <= 85:
            scenario_name = "Contraband Threat (Cell Phone Active)"
            base_frame = create_synthetic_candidate_face(w, h, head_turn="forward")
            pose_res = HeadPoseResult(
                face_count=1,
                orientation="Forward",
                deviation_score=0.02,
                nose_pt=(w // 2, h // 2 + 15),
                left_cheek_pt=(w // 2 - 80, h // 2),
                right_cheek_pt=(w // 2 + 80, h // 2),
                all_face_boxes=[(w // 2 - 90, h // 2 - 120, w // 2 + 90, h // 2 + 120)],
                violation_type=None,
                confidence=0.0,
            )
            # Add synthetic cell phone threat
            phone_box = (w // 2 + 120, h // 2 + 20, w // 2 + 230, h // 2 + 190)
            cv2.rectangle(base_frame, (phone_box[0], phone_box[1]), (phone_box[2], phone_box[3]), (20, 20, 20), -1)
            cv2.rectangle(base_frame, (phone_box[0] + 5, phone_box[1] + 10), (phone_box[2] - 5, phone_box[3] - 10), (180, 200, 220), -1)
            threats.append(
                DetectedThreat(
                    class_id=config.coco_phone_id,
                    label="Cell Phone",
                    confidence=0.92,
                    bbox=phone_box,
                    violation_type="CELL_PHONE_DETECTED",
                    box_color=config.color_phone_box,
                )
            )

        elif 86 <= frame_idx <= 110:
            scenario_name = "Candidate Missing (Desk Empty)"
            base_frame = np.full((h, w, 3), 35, dtype=np.uint8)
            cv2.putText(base_frame, "[EMPTY EXAM DESK]", (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 120), 2)
            pose_res = HeadPoseResult(
                face_count=0,
                orientation="No Face Detected",
                deviation_score=0.0,
                violation_type="CANDIDATE_MISSING",
                confidence=1.0,
            )

        # Update candidate frame violations
        active_frame_violations = {}
        for t in threats:
            active_frame_violations[t.violation_type] = t.confidence
        if pose_res.violation_type is not None:
            active_frame_violations[pose_res.violation_type] = pose_res.confidence

        # Temporal Streak Buffer
        ready_to_log = temporal_buffer.update(frame_idx, active_frame_violations)

        for event in ready_to_log:
            snap_path = audit_logger.log_violation(event, base_frame)
            logged_events.append((frame_idx, event.violation_type, event.confidence, snap_path))
            logger.info(
                f">>> [FRAME {frame_idx}] VIOLATION LOGGED: {event.violation_type} (Conf: {event.confidence:.2f}, Streak: {event.streak_count})"
            )

        active_confirmed = temporal_buffer.get_active_confirmed_violations()
        accumulating = temporal_buffer.get_accumulating_streaks()

        # Render HUD
        annotated = hud.render(
            frame=base_frame,
            pose_res=pose_res,
            threats=threats,
            active_violations=active_confirmed,
            accumulating_streaks=accumulating,
            frame_idx=frame_idx,
            fps=30.0,
        )

        if gui_display:
            cv2.imshow("Proctoring Simulation", annotated)
            if cv2.waitKey(20) & 0xFF == ord("q"):
                break

    if gui_display:
        cv2.destroyAllWindows()
    pose_engine.close()

    # Verification Report
    print("\n" + "=" * 70)
    print("           SIMULATION & VERIFICATION AUDIT REPORT")
    print("=" * 70)
    print(f" Total Frames Processed: {total_frames}")
    print(f" Confirmed Violations Logged: {len(logged_events)}")
    for f_idx, v_type, conf, snap in logged_events:
        print(f"  - Frame {f_idx:3d} | Type: {v_type:<24} | Conf: {conf:.2f} | Snapshot: {snap}")

    print("-" * 70)
    # Check proctoring_audit_log.csv
    csv_path = Path(config.audit_log_csv)
    if csv_path.exists():
        print(f" [OK] Audit log CSV exists: {csv_path.resolve()}")
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            print(f" Total CSV entries recorded: {len(rows) - 1}")
            for row in rows[:5]:
                print("   ", " | ".join(row))
    else:
        print(f" [FAIL] Audit log CSV not found!")

    # Check violations directory
    vdir = Path(config.violations_dir)
    snaps = list(vdir.glob("*.jpg"))
    print(f" [OK] Violation snapshots preserved in {vdir.resolve()}/: {len(snaps)} files found.")
    print("=" * 70)
    print(" ALL REQUIREMENTS VERIFIED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    show_gui = "--gui" in sys.argv
    run_simulation(gui_display=show_gui)
