"""
Stub database module for Lego Cam.
This will be expanded to handle real SQLite operations and logging later.
"""

import threading
import queue
import logging

logger = logging.getLogger(__name__)


def db_worker_loop(stop_event: threading.Event, db_queue: queue.Queue) -> None:
    """
    Database worker thread loop.
    Currently a stub that just consumes items from the queue.

    In the future, this will:
    - Initialize SQLite connection
    - Create tables for sessions, detections, events
    - Process events from db_queue (insert detections, update sessions, etc.)
    - Handle graceful shutdown and connection cleanup

    Args:
        stop_event: Event to signal thread shutdown
        db_queue: Queue to receive database operations
    """
    logger.info("DB worker thread started (stub mode)")

    while not stop_event.is_set():
        try:
            # Try to get an item from the queue with timeout
            item = db_queue.get(timeout=0.1)

            # Stub: just log that we received something
            logger.debug(f"DB worker received item (stub, discarding): {type(item)}")

            db_queue.task_done()

        except queue.Empty:
            # No items in queue, continue looping
            continue
        except Exception as e:
            logger.error(f"Error in DB worker loop: {e}")

    logger.info("DB worker thread stopped")


def start_db_thread(stop_event: threading.Event, db_queue: queue.Queue) -> threading.Thread:
    """
    Start the database worker thread.

    Args:
        stop_event: Event to signal thread shutdown
        db_queue: Queue to receive database operations

    Returns:
        The started thread object
    """
    thread = threading.Thread(
        target=db_worker_loop,
        args=(stop_event, db_queue),
        name="DBWorker",
        daemon=True,
    )
    thread.start()
    return thread
