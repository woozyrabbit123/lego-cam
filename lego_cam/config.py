"""
Configuration constants for Lego Cam.
"""

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

# Detection settings
MIN_CONTOUR_AREA = 100  # Minimum contour area in pixels for heuristic detector
HEURISTIC_CONFIDENCE = 0.6  # Confidence score for heuristic detections

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

# Thread settings
THREAD_JOIN_TIMEOUT = 2.0  # seconds
QUEUE_GET_TIMEOUT = 0.1  # seconds

# Hotkeys
KEY_QUIT = ord('q')
KEY_PAUSE = ord('p')
KEY_SNAPSHOT = ord('s')
KEY_CALIBRATE = ord('c')
KEY_RESET_SEGMENT = ord('r')
