"""
Main entry point for Lego Cam.
Orchestrates the pipeline and UI loop.
"""

import threading
import queue
import logging
import sys
import sqlite3
import os
import time
import re
import argparse
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

from . import config
from .detection_heuristic import HeuristicDetector, draw_detections
from .detection_yolo import YoloDetector, YoloDetectorError
from .detection_stub import DetectorHolder
from .pipeline import start_capture_thread, start_detection_thread, DetectionBackendState
from .db import init_db, create_session, start_db_worker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def sanitize_filename(text: str) -> str:
    """
    Sanitize a string for use in filenames.

    Args:
        text: String to sanitize

    Returns:
        Sanitized string safe for filenames
    """
    # Replace spaces with underscores and remove special characters
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    return text.strip('_')


def draw_hud(
    frame: np.ndarray,
    session_id: int,
    session_tag: str | None,
    segment_index: int,
    detection_count: int,
    backend_state: DetectionBackendState,
    paused: bool = False,
) -> np.ndarray:
    """
    Draw HUD overlay on the frame.

    Args:
        frame: Input frame to draw on
        session_id: Current session ID
        session_tag: Session tag or None
        segment_index: Current segment index
        detection_count: Number of detections in current frame
        backend_state: Detection backend state (mode and current backend)
        paused: Whether the system is paused

    Returns:
        Frame with HUD drawn
    """
    x, y = config.HUD_POSITION
    line_height = config.HUD_LINE_HEIGHT

    # Line 1: App title with mode and backend
    mode_str = backend_state.mode.value.upper()
    backend_str = backend_state.backend.upper()

    # Show fallback indicator for AUTO mode
    if backend_state.mode == config.DetectionMode.AUTO and backend_state.fallback_count > 0:
        backend_str = f"FALLBACK→{backend_str}"

    # Show broken indicator for SMART mode
    if backend_state.detector_broken:
        backend_str = "BROKEN"

    text = f"Lego Cam v1  [{mode_str} / {backend_str}]"
    if paused:
        text += " [PAUSED]"

    cv2.putText(
        frame,
        text,
        (x, y),
        config.HUD_FONT,
        config.HUD_FONT_SCALE,
        config.HUD_COLOR,
        config.HUD_THICKNESS,
        cv2.LINE_AA,
    )

    # Line 2: Session info
    session_text = f"Session: {session_id}"
    if session_tag:
        session_text += f" ({session_tag})"

    cv2.putText(
        frame,
        session_text,
        (x, y + line_height),
        config.HUD_FONT,
        config.HUD_FONT_SCALE,
        config.HUD_COLOR,
        config.HUD_THICKNESS,
        cv2.LINE_AA,
    )

    # Line 3: Segment info
    segment_text = f"Segment: {segment_index}"

    cv2.putText(
        frame,
        segment_text,
        (x, y + line_height * 2),
        config.HUD_FONT,
        config.HUD_FONT_SCALE,
        config.HUD_COLOR,
        config.HUD_THICKNESS,
        cv2.LINE_AA,
    )

    # Line 4: Detection count
    detection_text = f"Last detections: {detection_count}"

    cv2.putText(
        frame,
        detection_text,
        (x, y + line_height * 3),
        config.HUD_FONT,
        config.HUD_FONT_SCALE,
        config.HUD_COLOR,
        config.HUD_THICKNESS,
        cv2.LINE_AA,
    )

    return frame


def draw_message(frame: np.ndarray, message: str) -> np.ndarray:
    """
    Draw a temporary message overlay on the frame.

    Args:
        frame: Input frame
        message: Message text to display

    Returns:
        Frame with message drawn
    """
    x, y = config.MESSAGE_POSITION

    cv2.putText(
        frame,
        message,
        (x, y),
        config.HUD_FONT,
        config.MESSAGE_FONT_SCALE,
        config.MESSAGE_COLOR,
        config.HUD_THICKNESS,
        cv2.LINE_AA,
    )

    return frame


def draw_quit_confirmation(frame: np.ndarray) -> np.ndarray:
    """
    Draw quit confirmation overlay on the frame.

    Args:
        frame: Input frame

    Returns:
        Frame with quit confirmation drawn
    """
    # Get frame dimensions
    height, width = frame.shape[:2]

    # Draw semi-transparent background rectangle
    overlay = frame.copy()
    rect_x = width // 2 - 200
    rect_y = height // 2 - 40
    rect_w = 400
    rect_h = 80

    cv2.rectangle(
        overlay,
        (rect_x, rect_y),
        (rect_x + rect_w, rect_y + rect_h),
        config.QUIT_CONFIRM_BG_COLOR,
        -1,
    )

    # Blend overlay with original frame
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Draw text
    text1 = "Press Q again to quit"
    text2 = "Any other key to cancel"

    text_x = rect_x + 40
    text_y = rect_y + 35

    cv2.putText(
        frame,
        text1,
        (text_x, text_y),
        config.HUD_FONT,
        config.MESSAGE_FONT_SCALE,
        config.QUIT_CONFIRM_COLOR,
        config.HUD_THICKNESS,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text2,
        (text_x, text_y + 30),
        config.HUD_FONT,
        config.MESSAGE_FONT_SCALE - 0.1,
        config.QUIT_CONFIRM_COLOR,
        1,
        cv2.LINE_AA,
    )

    return frame


def save_snapshot(
    frame: np.ndarray,
    session_id: int,
    session_tag: str | None,
) -> str:
    """
    Save a snapshot of the current frame to disk.

    Args:
        frame: Frame to save (including annotations and HUD)
        session_id: Current session ID
        session_tag: Session tag or None

    Returns:
        Path to saved snapshot
    """
    # Create snapshots directory if it doesn't exist
    snapshots_dir = Path(config.SNAPSHOTS_DIR)
    snapshots_dir.mkdir(exist_ok=True)

    # Build filename
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    session_part = f"session_{session_id}"

    if session_tag:
        tag_part = sanitize_filename(session_tag)
        filename = f"{session_part}_{tag_part}_{timestamp}.png"
    else:
        filename = f"{session_part}_{timestamp}.png"

    filepath = snapshots_dir / filename

    # Save frame
    cv2.imwrite(str(filepath), frame)
    logger.info(f"Snapshot saved: {filepath}")

    return str(filepath)


def ui_loop(
    stop_event: threading.Event,
    ui_queue: queue.Queue,
    db_queue: queue.Queue,
    detector_holder: DetectorHolder,
    session_id: int,
    session_tag: str | None,
    backend_state: DetectionBackendState,
) -> None:
    """
    Main UI loop - MUST run in main thread for OpenCV.

    This loop:
    - Reads (frame, detections) from ui_queue
    - Maintains last frame for display
    - Draws detection boxes and HUD
    - Handles keyboard input (q/p/s/c/r)
    - Shows the frame via cv2.imshow

    Args:
        stop_event: Event to signal shutdown to other threads
        ui_queue: Queue to receive (frame, detections) tuples
        db_queue: Queue to send DB jobs
        detector_holder: DetectorHolder with current detector for calibration
        session_id: Current session ID
        session_tag: Session tag or None
        backend_state: Detection backend state for HUD display
    """
    logger.info("UI loop started")

    # Create window
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    # State
    last_raw_frame = None  # Raw frame without annotations
    last_frame = None  # Frame with annotations
    last_detections = []
    paused = False
    segment_index = 1

    # Message system
    message = None
    message_expire_time = 0

    # Quit confirmation
    quit_pending = False
    quit_expire_time = 0

    try:
        while not stop_event.is_set():
            # Try to get new frame from queue
            if not paused:
                try:
                    frame, detections = ui_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
                    last_raw_frame = frame.copy()  # Store raw frame for calibration
                    last_frame = frame.copy()
                    last_detections = detections

                except queue.Empty:
                    # No new frame, will use last_frame
                    pass

            # If we have a frame to display, draw everything and show it
            if last_frame is not None:
                display_frame = last_frame.copy()

                # Draw detection bounding boxes
                display_frame = draw_detections(display_frame, last_detections)

                # Draw HUD
                display_frame = draw_hud(
                    display_frame,
                    session_id,
                    session_tag,
                    segment_index,
                    len(last_detections),
                    backend_state,
                    paused=paused,
                )

                # Draw temporary message if active
                if message and time.time() < message_expire_time:
                    display_frame = draw_message(display_frame, message)
                elif message and time.time() >= message_expire_time:
                    message = None

                # Draw quit confirmation if pending
                if quit_pending:
                    display_frame = draw_quit_confirmation(display_frame)

                    # Check if quit timeout expired
                    if time.time() >= quit_expire_time:
                        quit_pending = False
                        logger.info("Quit cancelled (timeout)")

                cv2.imshow(config.WINDOW_NAME, display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 255:  # No key pressed
                continue

            # Handle quit with confirmation
            if key == config.KEY_QUIT:
                if quit_pending:
                    # Second 'q' press - actually quit
                    logger.info("Quit confirmed, initiating shutdown")
                    stop_event.set()
                    break
                else:
                    # First 'q' press - enter pending state
                    quit_pending = True
                    quit_expire_time = time.time() + config.QUIT_CONFIRM_TIMEOUT
                    logger.info("Quit pending, press Q again to confirm")

            elif quit_pending:
                # Any other key cancels quit
                quit_pending = False
                logger.info("Quit cancelled")

            elif key == config.KEY_PAUSE:
                paused = not paused
                logger.info(f"Pause toggled: {'PAUSED' if paused else 'RUNNING'}")

            elif key == config.KEY_SNAPSHOT:
                # Save snapshot
                if last_frame is not None:
                    filepath = save_snapshot(display_frame, session_id, session_tag)
                    message = f"Snapshot saved: {Path(filepath).name}"
                    message_expire_time = time.time() + config.MESSAGE_DURATION
                    logger.info(f"Snapshot saved to {filepath}")

            elif key == config.KEY_CALIBRATE:
                # Calibrate detector with current raw frame
                # This uses detector_holder.detector to ensure we calibrate the active detector
                # (important in AUTO mode after fallback from YOLO to heuristic)
                if last_raw_frame is not None:
                    detector_holder.detector.calibrate_from_frame(last_raw_frame)
                    message = "Calibrated"
                    message_expire_time = time.time() + config.MESSAGE_DURATION
                    logger.info("Detector calibrated")

            elif key == config.KEY_RESET_SEGMENT:
                # Increment segment and log marker to DB
                segment_index += 1
                timestamp = time.time()

                # Enqueue segment marker job
                db_queue.put({
                    "type": "segment_marker",
                    "session_id": session_id,
                    "segment_index": segment_index,
                    "timestamp": timestamp,
                })

                message = f"Segment {segment_index}"
                message_expire_time = time.time() + config.MESSAGE_DURATION
                logger.info(f"Segment marker {segment_index} created")

    finally:
        cv2.destroyAllWindows()
        logger.info("UI loop stopped, windows destroyed")


def create_detector(mode: config.DetectionMode):
    """
    Create detector instance based on detection mode.

    Args:
        mode: Detection mode (FAST/SMART/AUTO)

    Returns:
        Tuple of (detector, backend_name)

    Raises:
        SystemExit: If SMART mode requires YOLO but it's unavailable
    """
    if mode == config.DetectionMode.FAST:
        # FAST mode: always use heuristic
        logger.info("FAST mode: Creating heuristic detector")
        detector = HeuristicDetector(
            min_area=config.MIN_CONTOUR_AREA,
            confidence=config.HEURISTIC_CONFIDENCE,
        )
        return detector, "heuristic"

    elif mode == config.DetectionMode.SMART:
        # SMART mode: YOLO only, fail hard if unavailable
        logger.info("SMART mode: Creating YOLO detector (GPU required)")
        try:
            detector = YoloDetector(
                weights_path=config.YOLO_WEIGHTS_PATH,
                device=config.YOLO_DEVICE,
                conf_threshold=config.YOLO_CONF_THRESHOLD,
            )
            logger.info("SMART mode: YOLO detector created successfully")
            return detector, "yolo"
        except YoloDetectorError as e:
            logger.error(f"SMART mode: Failed to create YOLO detector: {e}")
            logger.error("SMART mode requires YOLO. Please use --mode fast or --mode auto instead.")
            sys.exit(1)

    elif mode == config.DetectionMode.AUTO:
        # AUTO mode: Try YOLO, fall back to heuristic if unavailable
        logger.info("AUTO mode: Attempting to create YOLO detector")
        try:
            detector = YoloDetector(
                weights_path=config.YOLO_WEIGHTS_PATH,
                device=config.YOLO_DEVICE,
                conf_threshold=config.YOLO_CONF_THRESHOLD,
            )
            logger.info("AUTO mode: YOLO detector created successfully")
            return detector, "yolo"
        except YoloDetectorError as e:
            logger.warning(f"AUTO mode: Failed to create YOLO detector: {e}")
            logger.warning("AUTO mode: Falling back to heuristic detector")
            detector = HeuristicDetector(
                min_area=config.MIN_CONTOUR_AREA,
                confidence=config.HEURISTIC_CONFIDENCE,
            )
            return detector, "heuristic"

    else:
        raise ValueError(f"Unknown detection mode: {mode}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Lego Cam v1 - Real-time LEGO detection with GPU acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Detection Modes:
  fast   - Heuristic detection only (CPU, always available)
  smart  - YOLO detection only (GPU required, fails if unavailable)
  auto   - YOLO with automatic fallback to heuristic (recommended)

Examples:
  python -m lego_cam --mode fast
  python -m lego_cam --mode smart
  python -m lego_cam --mode auto  (default)
        """
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fast", "smart", "auto"],
        default=config.DEFAULT_DETECTION_MODE.value,
        help=f"Detection mode (default: {config.DEFAULT_DETECTION_MODE.value})"
    )
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for Lego Cam.

    Returns:
        Exit code (0 for success)
    """
    # Parse arguments
    args = parse_args()
    detection_mode = config.DetectionMode(args.mode)

    logger.info("=" * 60)
    logger.info(f"Lego Cam v1 - Mode: {detection_mode.value.upper()}")
    logger.info("=" * 60)

    # Initialize database
    logger.info(f"Initializing database: {config.DB_PATH}")
    init_db(config.DB_PATH)

    # Create detector based on mode
    detector, backend_name = create_detector(detection_mode)

    # Wrap detector in holder to enable sharing between threads
    # This allows AUTO mode fallback and calibration to work on the same detector instance
    detector_holder = DetectorHolder(detector=detector)

    # Create backend state
    backend_state = DetectionBackendState(
        mode=detection_mode,
        backend=backend_name,
    )

    logger.info(f"Detection backend: {backend_name}")

    # Prompt for session tag
    print("\n" + "=" * 60)
    print("Session Setup")
    print("=" * 60)
    session_tag_input = input("Session tag (optional, e.g. 'BOX: green tub'): ").strip()
    session_tag = session_tag_input if session_tag_input else None

    # Create session in database
    conn = sqlite3.connect(config.DB_PATH)
    try:
        session_id = create_session(conn, session_tag)
    finally:
        conn.close()

    logger.info("=" * 60)
    logger.info(f"Session ID: {session_id}")
    if session_tag:
        logger.info(f"Session tag: {session_tag}")
    logger.info(f"Detection mode: {detection_mode.value.upper()}")
    logger.info(f"Detection backend: {backend_name}")
    logger.info(f"Resolution: {config.RESOLUTION}")
    logger.info(f"Target FPS: {config.TARGET_FPS}")
    logger.info("Controls:")
    logger.info("  'q' - Quit (press twice to confirm)")
    logger.info("  'p' - Pause/Unpause")
    logger.info("  's' - Save snapshot")
    logger.info("  'c' - Calibrate background")
    logger.info("  'r' - Start new segment")
    logger.info("=" * 60)

    # Create shared stop event
    stop_event = threading.Event()

    # Create queues
    capture_queue = queue.Queue(maxsize=config.CAPTURE_QUEUE_SIZE)
    ui_queue = queue.Queue(maxsize=config.UI_QUEUE_SIZE)
    db_queue = queue.Queue(maxsize=config.DB_QUEUE_SIZE)

    # Start threads
    logger.info("Starting worker threads...")

    capture_thread = start_capture_thread(stop_event, capture_queue)
    detection_thread = start_detection_thread(
        stop_event, capture_queue, ui_queue, db_queue, detector_holder, session_id, backend_state
    )
    db_thread = start_db_worker(config.DB_PATH, db_queue, stop_event)

    logger.info("All worker threads started")

    # Run UI loop in main thread (required by OpenCV)
    try:
        ui_loop(stop_event, ui_queue, db_queue, detector_holder, session_id, session_tag, backend_state)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        stop_event.set()
    except Exception as e:
        logger.error(f"Error in UI loop: {e}", exc_info=True)
        stop_event.set()

    # Shutdown: end session in DB
    logger.info("Shutting down...")
    logger.info("Ending session in database...")

    # Enqueue end_session job first
    db_queue.put({
        "type": "end_session",
        "session_id": session_id,
    })

    # Then enqueue shutdown sentinel to tell DB worker to exit
    # This guarantees end_session is processed before worker stops
    db_queue.put({
        "type": "shutdown",
    })

    # Wait for threads to finish
    threads = [
        ("Capture", capture_thread),
        ("Detection", detection_thread),
        ("DB", db_thread),
    ]

    for name, thread in threads:
        logger.info(f"Waiting for {name} thread...")
        thread.join(timeout=config.THREAD_JOIN_TIMEOUT)
        if thread.is_alive():
            logger.warning(f"{name} thread did not stop cleanly")
        else:
            logger.info(f"{name} thread stopped")

    logger.info("=" * 60)
    logger.info("Lego Cam shutdown complete")
    logger.info(f"Session {session_id} data saved to {config.DB_PATH}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
