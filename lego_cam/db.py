"""
Real database module for Lego Cam.
Handles SQLite operations, schema management, and the DB worker thread.
"""

import sqlite3
import threading
import queue
import logging
import time
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# SQL schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    color TEXT NOT NULL,
    x_min REAL NOT NULL,
    y_min REAL NOT NULL,
    x_max REAL NOT NULL,
    y_max REAL NOT NULL,
    confidence REAL NOT NULL,
    FOREIGN KEY(frame_id) REFERENCES frames(id)
);
"""


def init_db(db_path: str) -> None:
    """
    Initialize the database: create tables and set pragmas.

    Args:
        db_path: Path to SQLite database file
    """
    logger.info(f"Initializing database at {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")

        # Performance pragmas
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")

        # Create tables
        conn.executescript(SCHEMA)
        conn.commit()

        logger.info("Database initialized successfully")
    finally:
        conn.close()


def create_session(conn: sqlite3.Connection, tag: Optional[str] = None) -> int:
    """
    Create a new session in the database.

    Args:
        conn: Database connection
        tag: Optional session tag/description

    Returns:
        The new session ID
    """
    cursor = conn.execute(
        "INSERT INTO sessions (tag, started_at) VALUES (?, datetime('now'))",
        (tag,)
    )
    session_id = cursor.lastrowid
    conn.commit()

    logger.info(f"Created session {session_id} with tag: {tag or '(none)'}")
    return session_id


def end_session(conn: sqlite3.Connection, session_id: int) -> None:
    """
    Mark a session as ended.

    Args:
        conn: Database connection
        session_id: Session to end
    """
    conn.execute(
        "UPDATE sessions SET ended_at = datetime('now') WHERE id = ?",
        (session_id,)
    )
    conn.commit()
    logger.info(f"Ended session {session_id}")


def insert_frame_with_detections(
    conn: sqlite3.Connection,
    session_id: int,
    timestamp: float,
    detections: list[dict],
) -> int:
    """
    Insert a frame and its detections in a single transaction.

    Args:
        conn: Database connection
        session_id: Session this frame belongs to
        timestamp: Frame timestamp (unix time)
        detections: List of detection dicts with keys:
                    label, color, x_min, y_min, x_max, y_max, confidence

    Returns:
        The frame ID
    """
    # Insert frame
    cursor = conn.execute(
        "INSERT INTO frames (session_id, timestamp) VALUES (?, ?)",
        (session_id, timestamp)
    )
    frame_id = cursor.lastrowid

    # Insert detections
    if detections:
        detection_rows = [
            (
                frame_id,
                det["label"],
                det["color"],
                det["x_min"],
                det["y_min"],
                det["x_max"],
                det["y_max"],
                det["confidence"],
            )
            for det in detections
        ]

        conn.executemany(
            """INSERT INTO detections
               (frame_id, label, color, x_min, y_min, x_max, y_max, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            detection_rows
        )

    return frame_id


class DBWorker:
    """
    Database worker that processes jobs from a queue with batching.
    """

    def __init__(
        self,
        db_path: str,
        db_queue: queue.Queue,
        stop_event: threading.Event,
        batch_size: int = 50,
        batch_timeout: float = 5.0,
    ):
        """
        Initialize DB worker.

        Args:
            db_path: Path to SQLite database
            db_queue: Queue to receive jobs from
            stop_event: Event to signal shutdown
            batch_size: Max jobs to batch before committing
            batch_timeout: Max seconds to wait before committing batch
        """
        self.db_path = db_path
        self.db_queue = db_queue
        self.stop_event = stop_event
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.conn: Optional[sqlite3.Connection] = None

    def run(self) -> None:
        """
        Main worker loop: consume jobs, batch them, and commit periodically.
        """
        logger.info("DB worker started")

        try:
            # Open connection
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA journal_mode=WAL")

            pending_jobs = []
            last_commit_time = time.time()

            while not self.stop_event.is_set() or not self.db_queue.empty():
                try:
                    # Try to get a job with timeout
                    job = self.db_queue.get(timeout=0.1)
                    pending_jobs.append(job)
                    self.db_queue.task_done()
                except queue.Empty:
                    pass

                # Check if we should commit
                current_time = time.time()
                should_commit = (
                    len(pending_jobs) >= self.batch_size
                    or (pending_jobs and current_time - last_commit_time >= self.batch_timeout)
                )

                if should_commit:
                    self._process_batch(pending_jobs)
                    pending_jobs.clear()
                    last_commit_time = current_time

            # Process remaining jobs on shutdown
            if pending_jobs:
                logger.info(f"Processing final batch of {len(pending_jobs)} jobs")
                self._process_batch(pending_jobs)

        except Exception as e:
            logger.error(f"Error in DB worker: {e}", exc_info=True)
        finally:
            if self.conn:
                self.conn.close()
            logger.info("DB worker stopped")

    def _process_batch(self, jobs: list[dict]) -> None:
        """
        Process a batch of jobs in a single transaction.

        Args:
            jobs: List of job dicts to process
        """
        if not jobs or not self.conn:
            return

        try:
            # Start transaction
            self.conn.execute("BEGIN")

            for job in jobs:
                job_type = job.get("type")

                if job_type == "frame":
                    # Insert frame with detections
                    insert_frame_with_detections(
                        self.conn,
                        job["session_id"],
                        job["timestamp"],
                        job.get("detections", []),
                    )

                elif job_type == "end_session":
                    # End session
                    end_session(self.conn, job["session_id"])

                else:
                    logger.warning(f"Unknown job type: {job_type}")

            # Commit transaction
            self.conn.commit()
            logger.debug(f"Committed batch of {len(jobs)} jobs")

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            if self.conn:
                self.conn.rollback()


def start_db_worker(
    db_path: str,
    db_queue: queue.Queue,
    stop_event: threading.Event,
) -> threading.Thread:
    """
    Start the database worker thread.

    Args:
        db_path: Path to SQLite database
        db_queue: Queue to receive jobs from
        stop_event: Event to signal shutdown

    Returns:
        The started thread
    """
    worker = DBWorker(db_path, db_queue, stop_event)
    thread = threading.Thread(
        target=worker.run,
        name="DBWorker",
        daemon=True,
    )
    thread.start()
    return thread
