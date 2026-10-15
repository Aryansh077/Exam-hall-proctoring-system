"""
CLI Entrypoint for the Smart Exam Hall Proctoring System.
Enables dynamic configuration of camera feeds (webcam, RTSP, HTTP, video),
model weights, detection sensitivities, and streak thresholds.
"""

import argparse
import logging
import sys
from proctoring_system.config import ProctoringConfig
from proctoring_system.proctor import ProctoringEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Exam Hall Proctoring System - Real-Time AI Monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index (e.g. 0, 1), RTSP/HTTP stream URL, or path to video file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics YOLOv8 weights path or name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: 'cpu', 'cuda', '0', etc.",
    )
    parser.add_argument(
        "--streak-threshold",
        type=int,
        default=15,
        help="Consecutive frames required to confirm a violation (false-positive buffer).",
    )
    parser.add_argument(
        "--gaze-threshold",
        type=float,
        default=0.22,
        help="Geometric gaze deviation threshold (nose tip vs bilateral cheeks).",
    )
    parser.add_argument(
        "--phone-conf",
        type=float,
        default=0.40,
        help="Confidence threshold for cell phone detection (COCO class 67).",
    )
    parser.add_argument(
        "--book-conf",
        type=float,
        default=0.35,
        help="Confidence threshold for book/notes detection (COCO class 73).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Target video capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Target video capture height.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG level logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build config
    config = ProctoringConfig(
        camera_source=args.source,
        yolo_model_path=args.model,
        yolo_device=args.device,
        temporal_streak_threshold=args.streak_threshold,
        gaze_deviation_threshold=args.gaze_threshold,
        conf_phone=args.phone_conf,
        conf_book=args.book_conf,
        frame_width=args.width,
        frame_height=args.height,
    )

    print("=" * 70)
    print("   SMART EXAM HALL PROCTORING SYSTEM (AI SURVEILLANCE ENGINE)")
    print("=" * 70)
    print(f" Source Camera       : {config.camera_source}")
    print(f" YOLOv8 Model        : {config.yolo_model_path} ({config.yolo_device})")
    print(f" Temporal Buffer     : {config.temporal_streak_threshold} consecutive frames")
    print(f" Gaze Threshold      : ±{config.gaze_deviation_threshold}")
    print(f" Violations Storage  : {config.violations_dir}/")
    print(f" CSV Audit Trail     : {config.audit_log_csv}")
    print("-" * 70)
    print(" Controls: Press 'q' to gracefully quit, 'r' to reset temporal buffer.")
    print("=" * 70)

    try:
        engine = ProctoringEngine(config)
        engine.run()
    except Exception as e:
        logging.critical(f"Fatal error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
