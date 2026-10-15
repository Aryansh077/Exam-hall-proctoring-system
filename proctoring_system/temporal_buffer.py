"""
False-Positive Temporal Buffer and Streak Counter.
Suppresses transient false alarms (such as rapid blinks, momentary glances, or detector flicker)
by requiring a minimum streak of consecutive frames (default: 15) before triggering a violation alarm.
"""

import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ViolationEvent:
    violation_type: str
    confidence: float
    streak_count: int
    first_detected_at: float
    confirmed_at: float


class TemporalBuffer:
    """
    Tracks frame streaks for each violation type and controls alarm confirmation
    with customizable cooldown periods to avoid duplicate alert log spam.
    """

    def __init__(self, streak_threshold: int = 15, cooldown_frames: int = 60):
        self.streak_threshold = streak_threshold
        self.cooldown_frames = cooldown_frames

        # State tracking per violation type
        self.streaks: Dict[str, int] = {}
        self.first_detected_time: Dict[str, float] = {}
        self.last_logged_frame: Dict[str, int] = {}
        self.max_confidences: Dict[str, float] = {}

    def update(
        self,
        current_frame_idx: int,
        active_candidates: Dict[str, float],
    ) -> List[ViolationEvent]:
        """
        Updates the streak counters given the dictionary of violations detected in the current frame.
        active_candidates: mapping of {violation_type: confidence_score}

        Returns:
            List of confirmed ViolationEvent instances that are ready to be logged
            (i.e. streak threshold met and cooldown period satisfied).
        """
        now = time.time()
        ready_to_log: List[ViolationEvent] = []

        # 1. Update active violations
        for v_type, conf in active_candidates.items():
            if v_type not in self.streaks or self.streaks[v_type] == 0:
                self.streaks[v_type] = 1
                self.first_detected_time[v_type] = now
                self.max_confidences[v_type] = conf
            else:
                self.streaks[v_type] += 1
                self.max_confidences[v_type] = max(self.max_confidences.get(v_type, 0.0), conf)

            # Check if threshold reached
            if self.streaks[v_type] >= self.streak_threshold:
                last_logged = self.last_logged_frame.get(v_type, -999999)
                # Check cooldown to avoid logging 30 times a second
                if (current_frame_idx - last_logged) >= self.cooldown_frames:
                    event = ViolationEvent(
                        violation_type=v_type,
                        confidence=self.max_confidences[v_type],
                        streak_count=self.streaks[v_type],
                        first_detected_at=self.first_detected_time.get(v_type, now),
                        confirmed_at=now,
                    )
                    ready_to_log.append(event)
                    self.last_logged_frame[v_type] = current_frame_idx

        # 2. Reset violations that are not present in this frame
        all_tracked = list(self.streaks.keys())
        for v_type in all_tracked:
            if v_type not in active_candidates:
                self.streaks[v_type] = 0
                self.max_confidences[v_type] = 0.0

        return ready_to_log

    def get_active_confirmed_violations(self) -> List[str]:
        """Returns violation types that currently have streaks >= streak_threshold."""
        return [
            v_type
            for v_type, count in self.streaks.items()
            if count >= self.streak_threshold
        ]

    def get_accumulating_streaks(self) -> Dict[str, int]:
        """Returns violation types that currently have streaks > 0."""
        return {
            v_type: count
            for v_type, count in self.streaks.items()
            if count > 0
        }

    def reset_all(self):
        """Reset all streak counters."""
        self.streaks.clear()
        self.first_detected_time.clear()
        self.last_logged_frame.clear()
        self.max_confidences.clear()
