"""
YOLOv8 Contraband Object Detector.
Leverages pre-trained standard 80-class COCO weights (yolov8n.pt) requiring ZERO custom training.
Tracks:
  - Cell phones (COCO ID 67)
  - Books / notes (COCO ID 73)
  - Extra persons / unauthorized helpers (COCO ID 0)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from .config import ProctoringConfig

logger = logging.getLogger(__name__)


@dataclass
class DetectedThreat:
    class_id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    violation_type: str
    box_color: Tuple[int, int, int]  # BGR


class ContrabandDetectorYOLO:
    """
    Inference wrapper for Ultralytics YOLOv8 nano on COCO dataset.
    """

    COCO_LABELS = {
        0: "person",
        67: "cell phone",
        73: "book",
    }

    def __init__(self, config: ProctoringConfig):
        self.config = config
        logger.info(f"Loading YOLOv8 model: {config.yolo_model_path} onto device: {config.yolo_device}")
        self.model = YOLO(config.yolo_model_path)
        logger.info("YOLOv8 initialized successfully.")

    def detect(self, frame: np.ndarray) -> List[DetectedThreat]:
        """
        Runs object detection on frame and extracts prohibited contraband and extra persons.
        """
        threats: List[DetectedThreat] = []

        # Target class ids in COCO
        target_classes = [
            self.config.coco_person_id,
            self.config.coco_phone_id,
            self.config.coco_book_id,
        ]

        # Inference
        results = self.model.predict(
            source=frame,
            classes=target_classes,
            device=self.config.yolo_device,
            verbose=False,
            conf=min(self.config.conf_phone, self.config.conf_book, self.config.conf_extra_person),
        )

        if not results or len(results) == 0:
            return threats

        res = results[0]
        boxes = res.boxes

        person_boxes = []

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if cls_id == self.config.coco_phone_id:
                if conf >= self.config.conf_phone:
                    threats.append(
                        DetectedThreat(
                            class_id=cls_id,
                            label="Cell Phone",
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            violation_type="CELL_PHONE_DETECTED",
                            box_color=self.config.color_phone_box,
                        )
                    )

            elif cls_id == self.config.coco_book_id:
                if conf >= self.config.conf_book:
                    threats.append(
                        DetectedThreat(
                            class_id=cls_id,
                            label="Book/Notes",
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            violation_type="BOOK_DETECTED",
                            box_color=self.config.color_book_box,
                        )
                    )

            elif cls_id == self.config.coco_person_id:
                if conf >= self.config.conf_extra_person:
                    person_boxes.append((conf, (x1, y1, x2, y2)))

        # Evaluate Person Count:
        # 1 person = Candidate at desk (normal).
        # > 1 person = Unauthorized extra person / proxy / helper present!
        if len(person_boxes) > 1:
            # Sort persons by area or confidence; candidate is typically largest in front of camera
            person_boxes.sort(
                key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
                reverse=True,
            )
            # All additional persons after the primary one are marked as threats
            for extra_conf, extra_bbox in person_boxes[1:]:
                threats.append(
                    DetectedThreat(
                        class_id=self.config.coco_person_id,
                        label="Extra Person",
                        confidence=extra_conf,
                        bbox=extra_bbox,
                        violation_type="EXTRA_PERSON_DETECTED",
                        box_color=self.config.color_person_box,
                    )
                )

        return threats
