"""
Offline reporting and export utilities for Lego Cam.
Provides database query helpers for analyzing and exporting session data.
"""

import sqlite3
import json
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime


@dataclass
class SessionOverview:
    """Overview statistics for a single session."""
    id: int
    tag: str | None
    started_at: str | None  # ISO string
    ended_at: str | None  # ISO string
    duration_seconds: float | None
    frame_count: int
    detection_count: int
    avg_detections_per_frame: float | None


def get_all_session_overviews(db_path: str) -> list[SessionOverview]:
    """
    Get overview statistics for all sessions.

    Args:
        db_path: Path to SQLite database

    Returns:
        List of SessionOverview objects, sorted by session ID
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()

        # Get all sessions with their stats
        cursor.execute("""
            SELECT
                s.id,
                s.tag,
                s.started_at,
                s.ended_at,
                COUNT(DISTINCT f.id) as frame_count,
                COUNT(d.id) as detection_count
            FROM sessions s
            LEFT JOIN frames f ON f.session_id = s.id
            LEFT JOIN detections d ON d.frame_id = f.id
            GROUP BY s.id
            ORDER BY s.id
        """)

        overviews = []
        for row in cursor.fetchall():
            # Calculate duration if both timestamps exist
            duration_seconds = None
            if row['started_at'] and row['ended_at']:
                try:
                    start = datetime.fromisoformat(row['started_at'])
                    end = datetime.fromisoformat(row['ended_at'])
                    duration_seconds = (end - start).total_seconds()
                except (ValueError, TypeError):
                    pass

            # Calculate average detections per frame
            avg_detections = None
            if row['frame_count'] > 0:
                avg_detections = row['detection_count'] / row['frame_count']

            overview = SessionOverview(
                id=row['id'],
                tag=row['tag'],
                started_at=row['started_at'],
                ended_at=row['ended_at'],
                duration_seconds=duration_seconds,
                frame_count=row['frame_count'],
                detection_count=row['detection_count'],
                avg_detections_per_frame=avg_detections,
            )
            overviews.append(overview)

        return overviews

    finally:
        conn.close()


def get_session_detail(db_path: str, session_id: int) -> dict:
    """
    Get detailed information for a specific session.

    Args:
        db_path: Path to SQLite database
        session_id: Session ID to query

    Returns:
        Dictionary with keys: overview, segments, colors, labels, scans, bookmarks
        Returns None if session doesn't exist

    Raises:
        ValueError: If session doesn't exist
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()

        # Check if session exists
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        session_row = cursor.fetchone()
        if not session_row:
            raise ValueError(f"Session {session_id} not found")

        # Get overview
        overviews = get_all_session_overviews(db_path)
        overview = next((o for o in overviews if o.id == session_id), None)
        if not overview:
            raise ValueError(f"Session {session_id} not found")

        # Get segments
        cursor.execute("""
            SELECT segment_index, timestamp
            FROM segment_markers
            WHERE session_id = ?
            ORDER BY timestamp
        """, (session_id,))
        segment_markers = cursor.fetchall()

        # Build segment boundaries
        segments = []
        if not segment_markers:
            # No explicit segments - treat entire session as segment 1
            cursor.execute("""
                SELECT COUNT(*) as frame_count
                FROM frames
                WHERE session_id = ?
            """, (session_id,))
            frame_count = cursor.fetchone()['frame_count']

            cursor.execute("""
                SELECT COUNT(*) as detection_count
                FROM detections d
                JOIN frames f ON d.frame_id = f.id
                WHERE f.session_id = ?
            """, (session_id,))
            detection_count = cursor.fetchone()['detection_count']

            segments.append({
                "segment_index": 1,
                "start_timestamp": None,
                "frame_count": frame_count,
                "detection_count": detection_count,
            })
        else:
            # Build segments from markers
            segment_boundaries = [(1, 0.0)]  # First segment starts at 0
            for marker in segment_markers:
                segment_boundaries.append((marker['segment_index'], marker['timestamp']))

            # For each segment, count frames and detections
            for i, (seg_idx, seg_start) in enumerate(segment_boundaries):
                # Determine end of this segment
                if i < len(segment_boundaries) - 1:
                    seg_end = segment_boundaries[i + 1][1]
                    time_filter = "f.timestamp >= ? AND f.timestamp < ?"
                    params = (session_id, seg_start, seg_end)
                else:
                    time_filter = "f.timestamp >= ?"
                    params = (session_id, seg_start)

                cursor.execute(f"""
                    SELECT COUNT(*) as frame_count
                    FROM frames f
                    WHERE f.session_id = ? AND {time_filter}
                """, params)
                frame_count = cursor.fetchone()['frame_count']

                cursor.execute(f"""
                    SELECT COUNT(*) as detection_count
                    FROM detections d
                    JOIN frames f ON d.frame_id = f.id
                    WHERE f.session_id = ? AND {time_filter}
                """, params)
                detection_count = cursor.fetchone()['detection_count']

                segments.append({
                    "segment_index": seg_idx,
                    "start_timestamp": seg_start,
                    "frame_count": frame_count,
                    "detection_count": detection_count,
                })

        # Get color counts
        cursor.execute("""
            SELECT d.color, COUNT(*) as count
            FROM detections d
            JOIN frames f ON d.frame_id = f.id
            WHERE f.session_id = ?
            GROUP BY d.color
            ORDER BY count DESC
        """, (session_id,))
        colors = {row['color']: row['count'] for row in cursor.fetchall()}

        # Get label counts
        cursor.execute("""
            SELECT d.label, COUNT(*) as count
            FROM detections d
            JOIN frames f ON d.frame_id = f.id
            WHERE f.session_id = ?
            GROUP BY d.label
            ORDER BY count DESC
        """, (session_id,))
        labels = {row['label']: row['count'] for row in cursor.fetchall()}

        # Get scans
        cursor.execute("""
            SELECT id, timestamp, summary_json
            FROM scans
            WHERE session_id = ?
            ORDER BY timestamp
        """, (session_id,))
        scans = [dict(row) for row in cursor.fetchall()]

        # Get bookmarks
        cursor.execute("""
            SELECT id, frame_timestamp, image_path, note
            FROM bookmarks
            WHERE session_id = ?
            ORDER BY frame_timestamp
        """, (session_id,))
        bookmarks = [dict(row) for row in cursor.fetchall()]

        return {
            "overview": asdict(overview),
            "segments": segments,
            "colors": colors,
            "labels": labels,
            "scans": scans,
            "bookmarks": bookmarks,
        }

    finally:
        conn.close()


def export_session_detections_csv(db_path: str, session_id: int, out_dir: Path) -> Path:
    """
    Export all detections for a session to CSV.

    Args:
        db_path: Path to SQLite database
        session_id: Session ID to export
        out_dir: Output directory (will be created if needed)

    Returns:
        Path to created CSV file

    Raises:
        ValueError: If session doesn't exist
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()

        # Check session exists
        cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
        if not cursor.fetchone():
            raise ValueError(f"Session {session_id} not found")

        # Get segment markers to determine segment_index for each frame
        cursor.execute("""
            SELECT segment_index, timestamp
            FROM segment_markers
            WHERE session_id = ?
            ORDER BY timestamp
        """, (session_id,))
        segment_markers = cursor.fetchall()

        # Build segment boundaries lookup
        segment_boundaries = [(1, 0.0)]  # Default segment 1 at start
        for marker in segment_markers:
            segment_boundaries.append((marker['segment_index'], marker['timestamp']))

        def get_segment_index(frame_timestamp: float) -> int:
            """Determine which segment a frame belongs to."""
            for i in range(len(segment_boundaries) - 1, -1, -1):
                seg_idx, seg_time = segment_boundaries[i]
                if frame_timestamp >= seg_time:
                    return seg_idx
            return 1  # Default to segment 1

        # Query all detections for this session
        cursor.execute("""
            SELECT
                f.session_id,
                f.timestamp as frame_timestamp,
                d.label,
                d.color,
                d.confidence,
                d.x_min,
                d.y_min,
                d.x_max,
                d.y_max
            FROM detections d
            JOIN frames f ON d.frame_id = f.id
            WHERE f.session_id = ?
            ORDER BY f.timestamp, d.id
        """, (session_id,))

        # Write CSV
        csv_path = out_dir / f"session_{session_id}_detections.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'session_id', 'frame_timestamp', 'segment_index',
                'label', 'color', 'confidence',
                'x_min', 'y_min', 'x_max', 'y_max'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in cursor.fetchall():
                segment_index = get_segment_index(row['frame_timestamp'])
                writer.writerow({
                    'session_id': row['session_id'],
                    'frame_timestamp': row['frame_timestamp'],
                    'segment_index': segment_index,
                    'label': row['label'],
                    'color': row['color'],
                    'confidence': row['confidence'],
                    'x_min': row['x_min'],
                    'y_min': row['y_min'],
                    'x_max': row['x_max'],
                    'y_max': row['y_max'],
                })

        return csv_path

    finally:
        conn.close()


def export_session_summary_json(db_path: str, session_id: int, out_dir: Path) -> Path:
    """
    Export session summary as JSON.

    Args:
        db_path: Path to SQLite database
        session_id: Session ID to export
        out_dir: Output directory (will be created if needed)

    Returns:
        Path to created JSON file

    Raises:
        ValueError: If session doesn't exist
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get detailed session info
    detail = get_session_detail(db_path, session_id)

    # Write JSON
    json_path = out_dir / f"session_{session_id}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(detail, f, indent=2)

    return json_path
