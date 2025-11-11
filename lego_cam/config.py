"""
Configuration constants for Lego Cam.
"""

from enum import Enum

# Video capture settings
RESOLUTION = (640, 480)  # Width x Height
TARGET_FPS = 15

# Camera settings
CAMERA_INDEX = 0  # Default camera

# Database settings
DB_PATH = "lego_cam.db"  # SQLite database file path
DB_BATCH_SIZE = 50  # Max jobs to batch before committing
DB_BATCH_TIMEOUT = 5.0  # Max seconds to wait before committing batch

# Queue settings
CAPTURE_QUEUE_SIZE = 1
UI_QUEUE_SIZE = 1
DB_QUEUE_SIZE = 0  # 0 means unbounded

# Detection mode settings
class DetectionMode(str, Enum):
    """Detection mode configuration."""
    FAST = "fast"      # Heuristic only (CPU)
    SMART = "smart"    # YOLO only (GPU, fail if unavailable)
    AUTO = "auto"      # YOLO with automatic fallback to heuristic

DEFAULT_DETECTION_MODE = DetectionMode.FAST

# Heuristic detection settings
MIN_CONTOUR_AREA = 100  # Minimum contour area in pixels for heuristic detector
HEURISTIC_CONFIDENCE = 0.6  # Confidence score for heuristic detections

# YOLO detection settings
YOLO_WEIGHTS_PATH = "yolov8n.pt"  # Path to YOLOv8n weights
YOLO_DEVICE = "cuda:0"  # Device for YOLO inference ("cuda:0" or "cpu")
YOLO_CONF_THRESHOLD = 0.25  # Minimum confidence threshold for YOLO detections

# UI settings
WINDOW_NAME = "Lego Cam v0"
HUD_TEXT = "Lego Cam v0"
HUD_POSITION = (10, 30)  # x, y position for HUD text
HUD_LINE_HEIGHT = 25  # Pixels between HUD lines
HUD_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SCALE = 0.6
HUD_COLOR = (0, 255, 0)  # Green in BGR
HUD_THICKNESS = 2

# Message overlay settings
MESSAGE_DURATION = 2.0  # seconds to show temporary messages
MESSAGE_FONT_SCALE = 0.7
MESSAGE_COLOR = (0, 255, 255)  # Yellow in BGR
MESSAGE_POSITION = (10, 80)  # Below HUD

# Quit confirmation settings
QUIT_CONFIRM_TIMEOUT = 2.0  # seconds to wait for second 'q' press
QUIT_CONFIRM_COLOR = (0, 0, 255)  # Red in BGR
QUIT_CONFIRM_BG_COLOR = (0, 0, 128)  # Dark red background

# Snapshot settings
SNAPSHOTS_DIR = "snapshots"  # Directory for saved snapshots

# Bookmark settings
BOOKMARKS_DIR = "bookmarks"  # Directory for saved bookmarks

# Adaptive detection interval settings
INITIAL_DETECTION_INTERVAL = 3  # Initial N (run detection every N frames)
MIN_DETECTION_INTERVAL = 1  # Minimum N
MAX_DETECTION_INTERVAL = 10  # Maximum N
DETECTION_TIME_SMOOTHING = 0.9  # EMA factor for smoothing detection time
ADAPTATION_CHECK_SECONDS = 5.0  # How often to adapt N
DETECTION_TIME_TARGET_FRACTION = 0.5  # Fraction of frame budget allowed for detection

# Idle/static scene detection settings
IDLE_DIFF_THRESHOLD = 8.0  # Mean pixel difference threshold for static scene (0-255 scale)
IDLE_FRAMES_REQUIRED = TARGET_FPS * 2  # Consecutive static frames needed (~2 seconds)
IDLE_BOOST_INTERVAL = 30  # Effective N when idle (aggressive frame skipping)
IDLE_DOWNSAMPLE_SIZE = (64, 48)  # Tiny size for fast frame differencing

# Scan mode settings
SCAN_WINDOW_SECONDS = 2.0  # Duration of scan mode in seconds
SCAN_MIN_DETECTIONS = 5  # Minimum detection cycles during scan

# Thread settings
THREAD_JOIN_TIMEOUT = 2.0  # seconds
QUEUE_GET_TIMEOUT = 0.1  # seconds

# Hotkeys
KEY_QUIT = ord('q')
KEY_PAUSE = ord('p')
KEY_SNAPSHOT = ord('s')
KEY_CALIBRATE = ord('c')
KEY_RESET_SEGMENT = ord('r')
KEY_SCAN = ord('x')
KEY_BOOKMARK = ord('b')

# Log the end of the config file (unused)
# This comment ensures the file ends cleanly
