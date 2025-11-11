"""
Heuristic color-based detector for Lego Cam.
Uses HSV color filtering and contour detection to find LEGO bricks.
"""

import cv2
import numpy as np
import logging
from typing import Protocol

logger = logging.getLogger(__name__)


# HSV color ranges for common LEGO colors
# Format: (lower_bound, upper_bound) in HSV space
COLOR_RANGES = {
    "red": [
        # Red wraps around in HSV, so we need two ranges
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([170, 100, 100]), np.array([180, 255, 255])),
    ],
    "blue": [
        (np.array([100, 100, 100]), np.array([130, 255, 255])),
    ],
    "yellow": [
        (np.array([20, 100, 100]), np.array([30, 255, 255])),
    ],
    "green": [
        (np.array([40, 100, 100]), np.array([80, 255, 255])),
    ],
    "white": [
        (np.array([0, 0, 200]), np.array([180, 30, 255])),
    ],
}

# Minimum contour area to consider (pixels)
MIN_CONTOUR_AREA = 100

# Default confidence for heuristic detections
HEURISTIC_CONFIDENCE = 0.6


class Detector(Protocol):
    """Protocol/interface for detectors."""

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect objects in the given frame."""
        ...


class HeuristicDetector:
    """
    Heuristic detector using HSV color filtering and contour detection.

    This detector:
    1. Converts frame to HSV color space
    2. Creates masks for each target color
    3. Finds contours in each mask
    4. Filters contours by area
    5. Returns bounding boxes with normalized coordinates
    """

    def __init__(
        self,
        min_area: int = MIN_CONTOUR_AREA,
        confidence: float = HEURISTIC_CONFIDENCE,
    ):
        """
        Initialize heuristic detector.

        Args:
            min_area: Minimum contour area in pixels
            confidence: Confidence score to assign to detections
        """
        self.min_area = min_area
        self.confidence = confidence
        logger.info(f"HeuristicDetector initialized (min_area={min_area})")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect colored objects in the frame using HSV filtering.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of detection dicts with keys:
            - label: "brick" (generic for now)
            - color: color name (e.g., "red", "blue")
            - x_min, y_min, x_max, y_max: normalized coordinates (0..1)
            - confidence: detection confidence
        """
        detections = []

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply Gaussian blur to reduce noise
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        # Process each color
        for color_name, ranges in COLOR_RANGES.items():
            # Create combined mask for all ranges of this color
            combined_mask = None

            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)

                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Apply morphological operations to clean up mask
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                combined_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)

                # Filter by minimum area
                if area < self.min_area:
                    continue

                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Normalize coordinates to 0..1
                x_min = x / width
                y_min = y / height
                x_max = (x + w) / width
                y_max = (y + h) / height

                # Create detection dict
                detection = {
                    "label": "brick",
                    "color": color_name,
                    "x_min": float(x_min),
                    "y_min": float(y_min),
                    "x_max": float(x_max),
                    "y_max": float(y_max),
                    "confidence": self.confidence,
                }

                detections.append(detection)

        return detections


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes for detections on the frame.

    Args:
        frame: Input frame
        detections: List of detection dicts

    Returns:
        Frame with boxes drawn
    """
    height, width = frame.shape[:2]

    # Color map for visualization (BGR)
    color_map = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "white": (255, 255, 255),
    }

    for det in detections:
        # Denormalize coordinates
        x_min = int(det["x_min"] * width)
        y_min = int(det["y_min"] * height)
        x_max = int(det["x_max"] * width)
        y_max = int(det["y_max"] * height)

        # Get color for this detection
        color = color_map.get(det["color"], (0, 255, 0))

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Draw label
        label = f"{det['color']} {det['label']}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        # Draw background for text
        cv2.rectangle(
            frame,
            (x_min, y_min - label_size[1] - 5),
            (x_min + label_size[0], y_min),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return frame
