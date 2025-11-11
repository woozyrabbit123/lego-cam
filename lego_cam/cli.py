"""
Command-line interface for offline Lego Cam reporting and exports.
Run as: python -m lego_cam.cli <command>
"""

import sys
import argparse
from pathlib import Path
from datetime import timedelta

from . import config
from . import reporting


def format_duration(seconds: float | None) -> str:
    """
    Format duration in seconds as HH:MM:SS.

    Args:
        seconds: Duration in seconds, or None

    Returns:
        Formatted string like "00:05:23" or "OPEN" if None
    """
    if seconds is None:
        return "OPEN"

    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def cmd_list_sessions(args):
    """Handle list-sessions command."""
    try:
        overviews = reporting.get_all_session_overviews(config.DB_PATH)
    except Exception as e:
        print(f"Error reading database: {e}", file=sys.stderr)
        return 1

    if not overviews:
        print("No sessions found in database.")
        return 0

    # Print header
    print(f"{'ID':<4} {'Tag':<25} {'Frames':<8} {'Detections':<12} {'Duration':<10} {'Open?':<6}")
    print("-" * 70)

    # Print each session
    for ov in overviews:
        tag_display = (ov.tag or "")[:24]  # Truncate long tags
        is_open = "yes" if ov.ended_at is None else "no"
        duration_display = format_duration(ov.duration_seconds)

        print(f"{ov.id:<4} {tag_display:<25} {ov.frame_count:<8} "
              f"{ov.detection_count:<12} {duration_display:<10} {is_open:<6}")

    return 0


def cmd_summary(args):
    """Handle summary command."""
    session_id = args.session_id

    try:
        detail = reporting.get_session_detail(config.DB_PATH, session_id)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error reading database: {e}", file=sys.stderr)
        return 1

    # Extract overview
    ov = detail['overview']
    tag_display = ov['tag'] if ov['tag'] else "(no tag)"

    # Print session header
    print(f"\nSession {session_id} – {tag_display}")
    duration_display = format_duration(ov['duration_seconds'])
    avg_display = f"{ov['avg_detections_per_frame']:.2f}" if ov['avg_detections_per_frame'] else "N/A"
    print(f"Duration: {duration_display}  "
          f"(frames: {ov['frame_count']}, detections: {ov['detection_count']}, avg/frame: {avg_display})")

    # Print segments
    segments = detail['segments']
    if segments:
        print(f"\nSegments:")
        for seg in segments:
            print(f"  #{seg['segment_index']:<2}  frames={seg['frame_count']:<6}  "
                  f"detections={seg['detection_count']}")

    # Print colors
    colors = detail['colors']
    if colors:
        print(f"\nColors:")
        for color, count in colors.items():
            print(f"  {color:<12} {count:>6}")

    # Print labels
    labels = detail['labels']
    if labels:
        print(f"\nLabels:")
        for label, count in labels.items():
            print(f"  {label:<12} {count:>6}")

    # Print scans and bookmarks count
    print(f"\nScans: {len(detail['scans'])}")
    print(f"Bookmarks: {len(detail['bookmarks'])}")

    # If scans exist, show brief info
    if detail['scans']:
        print(f"\nScan details:")
        for scan in detail['scans']:
            import json
            summary = json.loads(scan['summary_json'])
            print(f"  Scan {scan['id']}: {summary.get('total_detections', 0)} detections "
                  f"in {summary.get('duration_seconds', 0):.1f}s")

    # If bookmarks exist, show brief info
    if detail['bookmarks']:
        print(f"\nBookmark details:")
        for bm in detail['bookmarks']:
            note_display = f" - {bm['note']}" if bm['note'] else ""
            print(f"  Bookmark {bm['id']}: {Path(bm['image_path']).name}{note_display}")

    print()  # Empty line at end
    return 0


def cmd_export(args):
    """Handle export command."""
    session_id = args.session_id
    out_dir = Path(args.out)
    export_format = args.format

    # Check if session exists first
    try:
        detail = reporting.get_session_detail(config.DB_PATH, session_id)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error reading database: {e}", file=sys.stderr)
        return 1

    # Perform exports based on format
    try:
        if export_format in ["csv", "both"]:
            csv_path = reporting.export_session_detections_csv(config.DB_PATH, session_id, out_dir)
            print(f"Exported CSV to {csv_path}")

        if export_format in ["json", "both"]:
            json_path = reporting.export_session_summary_json(config.DB_PATH, session_id, out_dir)
            print(f"Exported JSON summary to {json_path}")

        return 0

    except Exception as e:
        print(f"Error during export: {e}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="lego-cam-cli",
        description="Offline tools for Lego Cam - analyze and export session data"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # list-sessions command
    subparsers.add_parser(
        "list-sessions",
        help="List all sessions with overview statistics"
    )

    # summary command
    p_summary = subparsers.add_parser(
        "summary",
        help="Show detailed summary for one session"
    )
    p_summary.add_argument(
        "--session-id",
        type=int,
        required=True,
        help="Session ID to summarize"
    )

    # export command
    p_export = subparsers.add_parser(
        "export",
        help="Export detections and summary for one session"
    )
    p_export.add_argument(
        "--session-id",
        type=int,
        required=True,
        help="Session ID to export"
    )
    p_export.add_argument(
        "--out",
        type=str,
        default="exports",
        help="Output directory (default: exports)"
    )
    p_export.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Export format: csv (detections), json (summary), or both (default: both)"
    )

    return parser


def main() -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    # Route to appropriate command handler
    if args.command == "list-sessions":
        return cmd_list_sessions(args)
    elif args.command == "summary":
        return cmd_summary(args)
    elif args.command == "export":
        return cmd_export(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
