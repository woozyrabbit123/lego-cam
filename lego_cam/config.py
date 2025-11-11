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

# Thread settings
THREAD_JOIN_TIMEOUT = 2.0  # seconds
QUEUE_GET_TIMEOUT = 0.1  # seconds

# Hotkeys
KEY_QUIT = ord('q')
KEY_PAUSE = ord('p')
