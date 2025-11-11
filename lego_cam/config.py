"""
Configuration constants for Lego Cam.
"""

# Video capture settings
RESOLUTION = (640, 480)  # Width x Height
TARGET_FPS = 15

# Camera settings
CAMERA_INDEX = 0  # Default camera

# Queue settings
CAPTURE_QUEUE_SIZE = 1
UI_QUEUE_SIZE = 1
DB_QUEUE_SIZE = 0  # 0 means unbounded

# UI settings
WINDOW_NAME = "Lego Cam v0"
HUD_TEXT = "Lego Cam v0"
HUD_POSITION = (10, 30)  # x, y position for HUD text
HUD_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SCALE = 0.7
HUD_COLOR = (0, 255, 0)  # Green in BGR
HUD_THICKNESS = 2

# Thread settings
THREAD_JOIN_TIMEOUT = 2.0  # seconds
QUEUE_GET_TIMEOUT = 0.1  # seconds

# Hotkeys
KEY_QUIT = ord('q')
KEY_PAUSE = ord('p')
