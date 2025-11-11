"""
Main entry point for Lego Cam.
Orchestrates the pipeline and UI loop.
"""

import threading
import queue
import logging
import sys
import cv2
import numpy as np

from . import config
from .detection_stub import StubDetector
from .pipeline import start_capture_thread, start_detection_thread
from .db_stub import start_db_thread

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def draw_hud(frame: np.ndarray, paused: bool = False) -> np.ndarray:
    """
    Draw HUD overlay on the frame.

    Args:
        frame: Input frame to draw on
        paused: Whether the system is paused

    Returns:
        Frame with HUD drawn
    """
    # Draw main HUD text
    text = config.HUD_TEXT
    if paused:
        text += " [PAUSED]"

    cv2.putText(
        frame,
        text,
        config.HUD_POSITION,
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
) -> None:
    """
    Main UI loop - MUST run in main thread for OpenCV.

    This loop:
    - Reads (frame, detections) from ui_queue
    - Maintains last frame for display
    - Draws HUD and annotations
    - Handles keyboard input (q to quit, p to pause)
    - Shows the frame via cv2.imshow

    Args:
        stop_event: Event to signal shutdown to other threads
        ui_queue: Queue to receive (frame, detections) tuples
    """
    logger.info("UI loop started")

    # Create window
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    # State
    last_frame = None
    paused = False

    try:
        while not stop_event.is_set():
            # Try to get new frame from queue
            if not paused:
                try:
                    frame, detections = ui_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
                    last_frame = frame.copy()

                    # In the future, we'll draw detection boxes here
                    # For now, detections is always empty []

                except queue.Empty:
                    # No new frame, will use last_frame
                    pass

            # If we have a frame to display, draw HUD and show it
            if last_frame is not None:
                display_frame = last_frame.copy()
                display_frame = draw_hud(display_frame, paused=paused)
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
    logger.info("Lego Cam v0 - Foundation Pipeline")
    logger.info("=" * 60)
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

    # Create detector
    detector = StubDetector()

    # Start threads
    logger.info("Starting worker threads...")

    capture_thread = start_capture_thread(stop_event, capture_queue)
    detection_thread = start_detection_thread(
        stop_event, capture_queue, ui_queue, detector
    )
    db_thread = start_db_thread(stop_event, db_queue)

    logger.info("All worker threads started")

    # Run UI loop in main thread (required by OpenCV)
    try:
        ui_loop(stop_event, ui_queue)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        stop_event.set()
    except Exception as e:
        logger.error(f"Error in UI loop: {e}", exc_info=True)
        stop_event.set()

    # Shutdown: wait for threads to finish
    logger.info("Shutting down...")

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
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
