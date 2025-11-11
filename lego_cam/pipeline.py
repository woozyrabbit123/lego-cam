"""
Pipeline threads for Lego Cam.
Handles capture, detection, and the flow of frames through the system.
"""

import threading
import queue
import logging
import time
import cv2
import numpy as np

from . import config
from .detection_stub import Detector

logger = logging.getLogger(__name__)


def capture_thread_fn(
    stop_event: threading.Event,
    capture_queue: queue.Queue,
) -> None:
    """
    Capture thread: reads frames from webcam and pushes to capture_queue.

    This thread:
    - Opens the webcam
    - Sets resolution to 640x480
    - Continuously reads frames
    - Puts frames into capture_queue (overwriting if full)
    - Handles cleanup on shutdown

    Args:
        stop_event: Event to signal thread shutdown
        capture_queue: Queue to send captured frames (maxsize=1)
    """
    logger.info("Capture thread started")

    # Open camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        stop_event.set()
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.RESOLUTION[1])

    # Try to set FPS (not all cameras support this)
    cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

    # Verify actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera opened: {actual_width}x{actual_height}")

    frame_count = 0
    last_log_time = time.time()

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue

            # Ensure frame is correct size (resize if needed)
            if frame.shape[1] != config.RESOLUTION[0] or frame.shape[0] != config.RESOLUTION[1]:
                frame = cv2.resize(frame, config.RESOLUTION)

            # Put frame in queue, non-blocking
            # If queue is full, remove old frame and put new one (maxsize=1 behavior)
            try:
                # Try non-blocking put
                capture_queue.put_nowait(frame)
            except queue.Full:
                # Queue is full, remove old frame and put new one
                try:
                    capture_queue.get_nowait()
                except queue.Empty:
                    pass
                capture_queue.put_nowait(frame)

            frame_count += 1

            # Log FPS periodically
            current_time = time.time()
            if current_time - last_log_time >= 5.0:
                elapsed = current_time - last_log_time
                fps = frame_count / elapsed
                logger.debug(f"Capture FPS: {fps:.1f}")
                frame_count = 0
                last_log_time = current_time

            # Small sleep to avoid spinning too fast
            time.sleep(0.001)

    finally:
        cap.release()
        logger.info("Capture thread stopped, camera released")


def detection_thread_fn(
    stop_event: threading.Event,
    capture_queue: queue.Queue,
    ui_queue: queue.Queue,
    db_queue: queue.Queue,
    detector: Detector,
    session_id: int,
) -> None:
    """
    Detection thread: reads frames from capture_queue, runs detection, pushes to ui_queue and db_queue.

    This thread:
    - Reads frames from capture_queue
    - Runs detector.detect() on each frame
    - Puts (frame, detections) tuple into ui_queue
    - Enqueues frame job to db_queue for logging
    - Handles queue overflow by overwriting old data

    Args:
        stop_event: Event to signal thread shutdown
        capture_queue: Queue to receive captured frames
        ui_queue: Queue to send (frame, detections) tuples
        db_queue: Queue to send DB jobs
        detector: Detector instance to run on frames
        session_id: Current session ID for DB logging
    """
    logger.info(f"Detection thread started (session_id={session_id})")

    detection_count = 0
    last_log_time = time.time()

    try:
        while not stop_event.is_set():
            try:
                # Get frame from capture queue with timeout
                frame = capture_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
            except queue.Empty:
                # No frame available, continue
                continue

            # Record timestamp for this frame
            timestamp = time.time()

            # Run detection
            detections = detector.detect(frame)

            # Prepare result tuple for UI
            result = (frame, detections)

            # Put result in UI queue, non-blocking
            # If queue is full, remove old result and put new one
            try:
                ui_queue.put_nowait(result)
            except queue.Full:
                # Queue is full, remove old result and put new one
                try:
                    ui_queue.get_nowait()
                except queue.Empty:
                    pass
                ui_queue.put_nowait(result)

            # Enqueue frame job to DB (non-blocking, unbounded queue)
            db_job = {
                "type": "frame",
                "session_id": session_id,
                "timestamp": timestamp,
                "detections": detections,
            }
            try:
                db_queue.put_nowait(db_job)
            except queue.Full:
                logger.warning("DB queue full, dropping frame data")

            detection_count += 1

            # Log detection rate periodically
            current_time = time.time()
            if current_time - last_log_time >= 5.0:
                elapsed = current_time - last_log_time
                rate = detection_count / elapsed
                logger.debug(f"Detection rate: {rate:.1f} fps, {len(detections)} detections in last frame")
                detection_count = 0
                last_log_time = current_time

    finally:
        logger.info("Detection thread stopped")


def start_capture_thread(
    stop_event: threading.Event,
    capture_queue: queue.Queue,
) -> threading.Thread:
    """
    Start the capture thread.

    Args:
        stop_event: Event to signal thread shutdown
        capture_queue: Queue to send captured frames

    Returns:
        The started thread object
    """
    thread = threading.Thread(
        target=capture_thread_fn,
        args=(stop_event, capture_queue),
        name="CaptureThread",
        daemon=True,
    )
    thread.start()
    return thread


def start_detection_thread(
    stop_event: threading.Event,
    capture_queue: queue.Queue,
    ui_queue: queue.Queue,
    db_queue: queue.Queue,
    detector: Detector,
    session_id: int,
) -> threading.Thread:
    """
    Start the detection thread.

    Args:
        stop_event: Event to signal thread shutdown
        capture_queue: Queue to receive captured frames
        ui_queue: Queue to send detection results
        db_queue: Queue to send DB jobs
        detector: Detector instance to use
        session_id: Current session ID for DB logging

    Returns:
        The started thread object
    """
    thread = threading.Thread(
        target=detection_thread_fn,
        args=(stop_event, capture_queue, ui_queue, db_queue, detector, session_id),
        name="DetectionThread",
        daemon=True,
    )
    thread.start()
    return thread
