"""
Main entry point for Lego Cam.
Orchestrates the pipeline and UI loop.
"""

import threading
import queue
import logging
import sys
import sqlite3
import cv2
import numpy as np

from . import config
from .detection_heuristic import HeuristicDetector, draw_detections
from .pipeline import start_capture_thread, start_detection_thread
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


def draw_hud(
    frame: np.ndarray,
    session_id: int,
    session_tag: str | None,
    detection_count: int,
    paused: bool = False,
) -> np.ndarray:
    """
    Draw HUD overlay on the frame.

    Args:
        frame: Input frame to draw on
        session_id: Current session ID
        session_tag: Session tag or None
        detection_count: Number of detections in current frame
        paused: Whether the system is paused

    Returns:
        Frame with HUD drawn
    """
    x, y = config.HUD_POSITION
    line_height = config.HUD_LINE_HEIGHT

    # Line 1: App title
    text = config.HUD_TEXT
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

    # Line 3: Detection count
    detection_text = f"Last detections: {detection_count}"

    cv2.putText(
        frame,
        detection_text,
        (x, y + line_height * 2),
        config.HUD_FONT,
        config.HUD_FONT_SCALE,
        config.HUD_COLOR,
        config.HUD_THICKNESS,
        cv2.LINE_AA,
    )

    return frame


def ui_loop(
    stop_event: threading.Event,
    ui_queue: queue.Queue,
    session_id: int,
    session_tag: str | None,
) -> None:
    """
    Main UI loop - MUST run in main thread for OpenCV.

    This loop:
    - Reads (frame, detections) from ui_queue
    - Maintains last frame for display
    - Draws detection boxes and HUD
    - Handles keyboard input (q to quit, p to pause)
    - Shows the frame via cv2.imshow

    Args:
        stop_event: Event to signal shutdown to other threads
        ui_queue: Queue to receive (frame, detections) tuples
        session_id: Current session ID
        session_tag: Session tag or None
    """
    logger.info("UI loop started")

    # Create window
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    # State
    last_frame = None
    last_detections = []
    paused = False

    try:
        while not stop_event.is_set():
            # Try to get new frame from queue
            if not paused:
                try:
                    frame, detections = ui_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
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
                    len(last_detections),
                    paused=paused,
                )

                cv2.imshow(config.WINDOW_NAME, display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == config.KEY_QUIT:
                logger.info("Quit key pressed, initiating shutdown")
                stop_event.set()
                break
            elif key == config.KEY_PAUSE:
                paused = not paused
                logger.info(f"Pause toggled: {'PAUSED' if paused else 'RUNNING'}")

    finally:
        cv2.destroyAllWindows()
        logger.info("UI loop stopped, windows destroyed")


def main() -> int:
    """
    Main entry point for Lego Cam.

    Returns:
        Exit code (0 for success)
    """
    logger.info("=" * 60)
    logger.info("Lego Cam v0 - Heuristic Detection")
    logger.info("=" * 60)

    # Initialize database
    logger.info(f"Initializing database: {config.DB_PATH}")
    init_db(config.DB_PATH)

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
    logger.info(f"Resolution: {config.RESOLUTION}")
    logger.info(f"Target FPS: {config.TARGET_FPS}")
    logger.info("Controls:")
    logger.info("  'q' - Quit")
    logger.info("  'p' - Pause/Unpause")
    logger.info("=" * 60)

    # Create shared stop event
    stop_event = threading.Event()

    # Create queues
    capture_queue = queue.Queue(maxsize=config.CAPTURE_QUEUE_SIZE)
    ui_queue = queue.Queue(maxsize=config.UI_QUEUE_SIZE)
    db_queue = queue.Queue(maxsize=config.DB_QUEUE_SIZE)

    # Create detector (using heuristic detector now)
    detector = HeuristicDetector(
        min_area=config.MIN_CONTOUR_AREA,
        confidence=config.HEURISTIC_CONFIDENCE,
    )

    # Start threads
    logger.info("Starting worker threads...")

    capture_thread = start_capture_thread(stop_event, capture_queue)
    detection_thread = start_detection_thread(
        stop_event, capture_queue, ui_queue, db_queue, detector, session_id
    )
    db_thread = start_db_worker(config.DB_PATH, db_queue, stop_event)

    logger.info("All worker threads started")

    # Run UI loop in main thread (required by OpenCV)
    try:
        ui_loop(stop_event, ui_queue, session_id, session_tag)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        stop_event.set()
    except Exception as e:
        logger.error(f"Error in UI loop: {e}", exc_info=True)
        stop_event.set()

    # Shutdown: end session in DB
    logger.info("Shutting down...")
    logger.info("Ending session in database...")

    # Enqueue end_session job
    db_queue.put({
        "type": "end_session",
        "session_id": session_id,
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
