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
from dataclasses import dataclass

from . import config
from .detection_stub import Detector, DetectorHolder

logger = logging.getLogger(__name__)


@dataclass
class DetectionBackendState:
    """
    State tracking for detection backend and adaptive behavior.

    This is shared between the detection thread and UI thread to display
    current mode and backend information in the HUD.
    """
    mode: config.DetectionMode  # Selected detection mode (FAST/SMART/AUTO)
    backend: str  # Current backend in use ("heuristic" or "yolo")
    fallback_count: int = 0  # Number of times we've fallen back to heuristic
    detector_broken: bool = False  # True if detector is permanently broken (SMART mode)

    # Adaptive detection interval
    detection_interval: int = config.INITIAL_DETECTION_INTERVAL  # Current N

    # Scene state
    scene_state: str = "ACTIVE"  # "ACTIVE" or "IDLE"

    # Scan mode
    scan_active: bool = False  # True when scan mode is running


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
    detector_holder: DetectorHolder,
    session_id: int,
    backend_state: DetectionBackendState,
    scan_trigger_queue: queue.Queue,
) -> None:
    """
    Detection thread: reads frames from capture_queue, runs detection, pushes to ui_queue and db_queue.

    This thread:
    - Reads frames from capture_queue
    - Runs detector_holder.detector.detect() on each frame (with adaptive interval N)
    - Puts (frame, detections) tuple into ui_queue
    - Enqueues frame job to db_queue for logging
    - Handles queue overflow by overwriting old data
    - Handles YOLO errors and fallback logic based on detection mode
    - Updates detector_holder.detector on AUTO mode fallback
    - Adapts detection interval N based on detection time
    - Detects idle/static scenes and boosts N
    - Handles scan mode triggers

    Args:
        stop_event: Event to signal thread shutdown
        capture_queue: Queue to receive captured frames
        ui_queue: Queue to send (frame, detections) tuples
        db_queue: Queue to send DB jobs
        detector_holder: DetectorHolder with current detector instance
        session_id: Current session ID for DB logging
        backend_state: Shared state object for mode and backend tracking
        scan_trigger_queue: Queue to receive scan trigger signals
    """
    logger.info(
        f"Detection thread started (session_id={session_id}, "
        f"mode={backend_state.mode.value}, backend={backend_state.backend})"
    )

    detection_count = 0
    last_log_time = time.time()

    # Adaptive detection interval tracking
    frame_counter = 0
    detection_interval = config.INITIAL_DETECTION_INTERVAL
    detection_time_avg = 0.0  # EMA of detection time in ms
    last_adaptation_time = time.time()

    # Idle/static scene detection
    prev_frame_small = None
    static_frame_count = 0

    # Last detections for frame skipping
    last_detections = []

    # Scan mode state
    scan_active = False
    scan_start_time = 0.0
    scan_detections_collected = []
    scan_detection_count = 0

    try:
        while not stop_event.is_set():
            # Check for scan trigger (non-blocking)
            try:
                _scan_trigger = scan_trigger_queue.get_nowait()
                if not scan_active:
                    # Start scan mode
                    scan_active = True
                    scan_start_time = time.time()
                    scan_detections_collected = []
                    scan_detection_count = 0
                    backend_state.scan_active = True
                    logger.info("Scan mode activated")
            except queue.Empty:
                pass

            try:
                # Get frame from capture queue with timeout
                frame = capture_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
            except queue.Empty:
                # No frame available, continue
                continue

            # Record timestamp for this frame
            timestamp = time.time()
            frame_counter += 1

            # Idle/static scene detection
            frame_small = cv2.resize(frame, config.IDLE_DOWNSAMPLE_SIZE)
            frame_small_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

            if prev_frame_small is not None:
                diff = cv2.absdiff(prev_frame_small, frame_small_gray)
                mean_diff = diff.mean()

                if mean_diff < config.IDLE_DIFF_THRESHOLD:
                    static_frame_count += 1
                else:
                    static_frame_count = 0

                # Update scene state
                if static_frame_count >= config.IDLE_FRAMES_REQUIRED:
                    backend_state.scene_state = "IDLE"
                else:
                    backend_state.scene_state = "ACTIVE"

            prev_frame_small = frame_small_gray

            # Determine effective detection interval
            if scan_active:
                # Force detection on every frame during scan
                effective_interval = 1
            elif backend_state.scene_state == "IDLE":
                # Boost interval when idle
                effective_interval = max(detection_interval, config.IDLE_BOOST_INTERVAL)
            else:
                # Use adaptive interval
                effective_interval = detection_interval

            # Update backend state for HUD
            backend_state.detection_interval = effective_interval

            # Decide whether to run detection this frame
            should_detect = (frame_counter % effective_interval == 0)

            # Run detection with error handling
            detections = []

            if should_detect:
                # Skip detection if detector is broken (SMART mode only)
                if backend_state.detector_broken:
                    detections = []
                else:
                    # Measure detection time
                    detect_start = time.time()

                    try:
                        detections = detector_holder.detector.detect(frame)

                        # Update detection time EMA
                        detect_time_ms = (time.time() - detect_start) * 1000.0
                        if detection_time_avg == 0:
                            detection_time_avg = detect_time_ms
                        else:
                            alpha = 1.0 - config.DETECTION_TIME_SMOOTHING
                            detection_time_avg = (alpha * detect_time_ms +
                                                 config.DETECTION_TIME_SMOOTHING * detection_time_avg)

                    except Exception as e:
                        # Handle detection errors based on mode
                        logger.error(f"Detection error: {e}")

                        if backend_state.mode == config.DetectionMode.SMART:
                            # SMART mode: mark detector as broken, stop detecting
                            backend_state.detector_broken = True
                            logger.error(
                                "SMART mode: YOLO detector failed. "
                                "Detection disabled until restart."
                            )
                            detections = []

                        elif backend_state.mode == config.DetectionMode.AUTO:
                            # AUTO mode: fall back to heuristic if using YOLO
                            if backend_state.backend == "yolo":
                                logger.warning(
                                    "AUTO mode: YOLO detector failed. "
                                    "Falling back to heuristic detector."
                                )

                                # Create fallback heuristic detector
                                from .detection_heuristic import HeuristicDetector
                                fallback_detector = HeuristicDetector(
                                    min_area=config.MIN_CONTOUR_AREA,
                                    confidence=config.HEURISTIC_CONFIDENCE,
                                )

                                # Update the shared detector holder
                                # This ensures UI calibration will target the new detector
                                detector_holder.detector = fallback_detector

                                # Update backend state
                                backend_state.backend = "heuristic"
                                backend_state.fallback_count += 1

                                # Try detection again with heuristic
                                try:
                                    detections = detector_holder.detector.detect(frame)
                                except Exception as fallback_error:
                                    logger.error(
                                        f"Fallback heuristic detection also failed: {fallback_error}"
                                    )
                                    detections = []
                            else:
                                # Already using heuristic and it failed
                                logger.error("Heuristic detector failed")
                                detections = []

                        else:
                            # FAST mode: heuristic failed, just skip this frame
                            logger.error("FAST mode: Heuristic detector failed on this frame")
                            detections = []

                # Store last detections
                last_detections = detections
            else:
                # Reuse last detections for skipped frames
                detections = last_detections

            # Scan mode: collect detections
            if scan_active and should_detect:
                scan_detections_collected.extend(detections)
                scan_detection_count += 1

                # Check if scan is complete
                scan_duration = time.time() - scan_start_time
                if (scan_duration >= config.SCAN_WINDOW_SECONDS and
                    scan_detection_count >= config.SCAN_MIN_DETECTIONS):
                    # End scan mode and generate summary
                    scan_active = False
                    backend_state.scan_active = False

                    # Generate scan summary
                    import json
                    from collections import Counter

                    label_counts = Counter(d["label"] for d in scan_detections_collected)
                    color_counts = Counter(d["color"] for d in scan_detections_collected)

                    summary = {
                        "total_detections": len(scan_detections_collected),
                        "detection_cycles": scan_detection_count,
                        "duration_seconds": scan_duration,
                        "labels": dict(label_counts),
                        "colors": dict(color_counts),
                    }

                    summary_json = json.dumps(summary)

                    # Enqueue scan summary to DB
                    db_queue.put({
                        "type": "scan_summary",
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "summary_json": summary_json,
                    })

                    logger.info(f"Scan complete: {len(scan_detections_collected)} detections over {scan_duration:.1f}s")

            # Adapt detection interval periodically
            current_time = time.time()
            if current_time - last_adaptation_time >= config.ADAPTATION_CHECK_SECONDS:
                # Compute frame budget
                frame_budget_ms = 1000.0 / config.TARGET_FPS
                allowed_detection_ms = frame_budget_ms * config.DETECTION_TIME_TARGET_FRACTION

                # Adapt N based on detection time
                if detection_time_avg > allowed_detection_ms * 1.1:
                    if detection_interval < config.MAX_DETECTION_INTERVAL:
                        detection_interval += 1
                        logger.debug(f"Increased detection interval to N={detection_interval}")
                elif detection_time_avg < allowed_detection_ms * 0.5:
                    if detection_interval > config.MIN_DETECTION_INTERVAL:
                        detection_interval -= 1
                        logger.debug(f"Decreased detection interval to N={detection_interval}")

                last_adaptation_time = current_time

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
            if current_time - last_log_time >= 5.0:
                elapsed = current_time - last_log_time
                rate = detection_count / elapsed
                logger.debug(
                    f"Processing rate: {rate:.1f} fps, N={effective_interval}, "
                    f"scene={backend_state.scene_state}, {len(detections)} detections, "
                    f"backend={backend_state.backend}"
                )
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
    detector_holder: DetectorHolder,
    session_id: int,
    backend_state: DetectionBackendState,
    scan_trigger_queue: queue.Queue,
) -> threading.Thread:
    """
    Start the detection thread.

    Args:
        stop_event: Event to signal thread shutdown
        capture_queue: Queue to receive captured frames
        ui_queue: Queue to send detection results
        db_queue: Queue to send DB jobs
        detector_holder: DetectorHolder with current detector instance
        session_id: Current session ID for DB logging
        backend_state: Shared state for mode and backend tracking
        scan_trigger_queue: Queue to receive scan trigger signals

    Returns:
        The started thread object
    """
    thread = threading.Thread(
        target=detection_thread_fn,
        args=(stop_event, capture_queue, ui_queue, db_queue, detector_holder, session_id, backend_state, scan_trigger_queue),
        name="DetectionThread",
        daemon=True,
    )
    thread.start()
    return thread
