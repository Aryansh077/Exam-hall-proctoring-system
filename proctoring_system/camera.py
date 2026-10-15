"""
Dynamic Camera Ingestion Module.
Supports built-in webcams, external USB cameras, IP/CCTV cameras via RTSP/HTTP,
and recorded video files with automatic reconnection logic.
"""

import time
import logging
from typing import Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoStreamHandler:
    """
    Robust video capture handler supporting webcams, RTSP/HTTP network streams, and video files.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        target_width: int = 1280,
        target_height: int = 720,
        fps: int = 30,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 2.0,
    ):
        self.source = self._parse_source(source)
        self.target_width = target_width
        self.target_height = target_height
        self.fps = fps
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        self._is_file = isinstance(self.source, str) and not (
            self.source.startswith("rtsp://") or self.source.startswith("http://") or self.source.startswith("https://")
        )

        self._initialize_capture()

    @staticmethod
    def _parse_source(source: Union[int, str]) -> Union[int, str]:
        """Convert string numbers to integer index if user provided '0' or '1'."""
        if isinstance(source, str):
            source_str = source.strip()
            if source_str.isdigit():
                return int(source_str)
            return source_str
        return source

    def _initialize_capture(self) -> bool:
        """Initializes cv2.VideoCapture with optimal backend and properties."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        logger.info(f"Opening video source: {self.source}")
        
        # Choose backend: DSHOW is reliable on Windows for webcams
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.source)
        else:
            # For network stream or video file
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            logger.warning(f"Failed to open video source: {self.source}")
            self.is_opened = False
            return False

        # Attempt to set desired resolution and FPS (applicable to webcams)
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            # Minimize internal buffer lag for live streams
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.is_opened = True
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Stream opened successfully. Resolution: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads next frame with auto-reconnection logic for live network streams
        and loop support for video files.
        """
        if not self.is_opened or self.cap is None:
            if not self._reconnect():
                return False, None

        ret, frame = self.cap.read()

        if not ret or frame is None:
            # If it's a video file, loop it for continuous testing
            if self._is_file:
                logger.info("Video file reached EOF. Looping back to start.")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return True, frame
            
            # If it's a live camera or RTSP stream, attempt reconnection
            logger.warning("Frame read failed. Attempting stream reconnection...")
            if self._reconnect():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return True, frame

            return False, None

        return True, frame

    def _reconnect(self) -> bool:
        """Attempts to reconnect to video source up to reconnect_attempts."""
        for attempt in range(1, self.reconnect_attempts + 1):
            logger.info(f"Reconnection attempt {attempt}/{self.reconnect_attempts}...")
            time.sleep(self.reconnect_delay)
            if self._initialize_capture():
                logger.info("Reconnection successful.")
                return True
        logger.error("All reconnection attempts failed.")
        self.is_opened = False
        return False

    def release(self):
        """Release camera resources cleanly."""
        if self.cap is not None:
            logger.info("Releasing video stream...")
            self.cap.release()
            self.cap = None
        self.is_opened = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
