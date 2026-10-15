"""
Main Proctoring Engine.
Orchestrates Video Ingestion, YOLOv8 Contraband Detection, MediaPipe Geometric Head Pose,
Temporal Streak Counter, Anomaly Logging, and On-Screen HUD.
"""

import time
import logging
from typing import Dict, List, Optional
import cv2
import numpy as np

from .config import ProctoringConfig
from .camera import VideoStreamHandler
from .detector_yolo import ContrabandDetectorYOLO, DetectedThreat
from .pose_estimator import GeometricHeadPoseEngine, HeadPoseResult
from .temporal_buffer import TemporalBuffer, ViolationEvent
from .logger import AnomalyLogger
from .hud import ProctoringHUD

logger = logging.getLogger(__name__)


class ProctoringEngine:
    """
    Central Coordinator managing real-time video pipeline and anomaly auditing.
    """

    def __init__(self, config: Optional[ProctoringConfig] = None):
        self.config = config or ProctoringConfig()

        logger.info("Initializing Smart Exam Hall Proctoring System...")
        self.stream = VideoStreamHandler(
            source=self.config.camera_source,
            target_width=self.config.frame_width,
            target_height=self.config.frame_height,
            fps=self.config.fps,
            reconnect_attempts=self.config.reconnect_attempts,
            reconnect_delay=self.config.reconnect_delay_sec,
        )

        self.contraband_detector = ContrabandDetectorYOLO(self.config)
        self.pose_engine = GeometricHeadPoseEngine(self.config)
        self.temporal_buffer = TemporalBuffer(
            streak_threshold=self.config.temporal_streak_threshold,
            cooldown_frames=self.config.violation_cooldown_frames,
        )
        self.logger = AnomalyLogger(self.config)
        self.hud = ProctoringHUD(self.config)

        self.frame_idx = 0
        self.is_running = False

    def run(self, window_name: str = "Smart Exam Hall Proctoring System - Real-Time Monitor"):
        """
        Executes the real-time proctoring monitoring loop until user presses 'q'.
        """
        if not self.stream.is_opened:
            logger.error("Video stream is not available. Aborting proctoring loop.")
            return

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.config.frame_width, self.config.frame_height)

        self.is_running = True
        self.frame_idx = 0
        fps = 0.0
        fps_calc_time = time.time()
        fps_frame_count = 0

        logger.info("Starting Proctoring Monitoring loop. Press 'q' to gracefully exit.")

        try:
            while self.is_running:
                ret, frame = self.stream.read()
                if not ret or frame is None:
                    logger.warning("No frame received. Waiting for stream...")
                    time.sleep(0.05)
                    continue

                self.frame_idx += 1
                fps_frame_count += 1
                current_time = time.time()
                elapsed = current_time - fps_calc_time
                if elapsed >= 1.0:
                    fps = fps_frame_count / elapsed
                    fps_calc_time = current_time
                    fps_frame_count = 0

                # 1. Contraband Object Detection (YOLOv8)
                threats = self.contraband_detector.detect(frame)

                # 2. Geometric Head Pose & Candidate Tracking (MediaPipe Face Mesh)
                pose_res = self.pose_engine.process_frame(frame)

                # 3. Aggregate Frame Candidate Violations for Temporal Buffer
                active_frame_violations: Dict[str, float] = {}

                # Check contraband threats
                for threat in threats:
                    active_frame_violations[threat.violation_type] = threat.confidence

                # Check pose / face integrity
                if pose_res.violation_type is not None:
                    active_frame_violations[pose_res.violation_type] = pose_res.confidence

                # 4. Temporal Streak Counter Update (False-Positive Suppression)
                ready_to_log: List[ViolationEvent] = self.temporal_buffer.update(
                    current_frame_idx=self.frame_idx,
                    active_candidates=active_frame_violations,
                )

                # 5. Anomaly Logging (Snapshot + CSV Audit Trail)
                for event in ready_to_log:
                    # Log with raw or marked snapshot
                    self.logger.log_violation(event, frame)

                # 6. Retrieve active confirmed violations & accumulating streaks for HUD
                active_confirmed = self.temporal_buffer.get_active_confirmed_violations()
                accumulating = self.temporal_buffer.get_accumulating_streaks()

                # 7. Render Modern HUD
                annotated_frame = self.hud.render(
                    frame=frame,
                    pose_res=pose_res,
                    threats=threats,
                    active_violations=active_confirmed,
                    accumulating_streaks=accumulating,
                    frame_idx=self.frame_idx,
                    fps=fps,
                )

                # 8. Display Frame
                cv2.imshow(window_name, annotated_frame)

                # 9. Handle Keypresses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q") or key == 27:  # 27 is ESC
                    logger.info("Exit key pressed by user.")
                    break
                elif key == ord("r") or key == ord("R"):
                    logger.info("Resetting temporal streak buffers upon user command.")
                    self.temporal_buffer.reset_all()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            self.cleanup()

    def cleanup(self):
        """Releases all capture hardware, inference engines, and OpenCV windows."""
        logger.info("Releasing all resources...")
        self.is_running = False
        try:
            self.stream.release()
        except Exception as e:
            logger.error(f"Error releasing camera stream: {e}")

        try:
            self.pose_engine.close()
        except Exception as e:
            logger.error(f"Error closing pose engine: {e}")

        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Error destroying OpenCV windows: {e}")

        logger.info("Cleanup completed successfully.")
