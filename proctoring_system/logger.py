"""
Anomaly Logging and Audit System.
Saves timestamped photographic evidence of confirmed violations to a violations/ directory
and logs structured violation events to proctoring_audit_log.csv.
"""

import csv
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from .config import ProctoringConfig
from .temporal_buffer import ViolationEvent

logger = logging.getLogger(__name__)


class AnomalyLogger:
    """
    Manages audit logging and violation snapshot preservation for academic integrity compliance.
    """

    def __init__(self, config: ProctoringConfig):
        self.config = config
        self.violations_dir = Path(config.violations_dir)
        self.audit_log_csv = Path(config.audit_log_csv)

        # Ensure violation snapshot directory exists
        self.violations_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV log with header if not present
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        """Creates CSV audit log with standard schema if it does not already exist."""
        if not self.audit_log_csv.exists() or self.audit_log_csv.stat().st_size == 0:
            with open(self.audit_log_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp",
                    "Violation_Type",
                    "Confidence_Score",
                    "Streak_Frames",
                    "Snapshot_Filename",
                ])
            logger.info(f"Initialized audit log: {self.audit_log_csv}")

    def log_violation(
        self,
        event: ViolationEvent,
        frame: np.ndarray,
    ) -> Optional[str]:
        """
        Saves a snapshot of the violating frame and appends an audit row to the CSV.

        Returns:
            The relative path string to the saved snapshot, or None if save failed.
        """
        now_dt = datetime.now()
        timestamp_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        file_ts = now_dt.strftime("%Y%m%d_%H%M%S_%f")[:19]

        # Construct sanitized filename
        clean_vtype = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in event.violation_type)
        snapshot_filename = f"{file_ts}_{clean_vtype}.jpg"
        snapshot_path = self.violations_dir / snapshot_filename

        # Save snapshot image
        try:
            cv2.imwrite(str(snapshot_path), frame)
            logger.info(f"Violation snapshot saved: {snapshot_path}")
        except Exception as e:
            logger.error(f"Failed to save snapshot {snapshot_path}: {e}")
            snapshot_filename = "SAVE_FAILED"

        # Write to CSV log
        try:
            with open(self.audit_log_csv, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp_str,
                    event.violation_type,
                    f"{event.confidence:.4f}",
                    event.streak_count,
                    snapshot_filename,
                ])
            logger.warning(
                f"[AUDIT LOGGED] {event.violation_type} | Conf: {event.confidence:.2f} | Streak: {event.streak_count}"
            )
            return str(snapshot_path)
        except Exception as e:
            logger.error(f"Failed to write to audit log CSV {self.audit_log_csv}: {e}")
            return None
