"""
Stub detector for Lego Cam.
This will be replaced with heuristic and YOLO-based detectors later.
"""

from typing import Protocol
import numpy as np


class Detector(Protocol):
    """Protocol/interface for detectors."""

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect objects in the given frame.

        Args:
            frame: BGR image from OpenCV (numpy array)

        Returns:
            List of detection dictionaries. Each detection should have:
            - 'bbox': (x, y, w, h) bounding box
            - 'confidence': float confidence score
            - 'class': str class name
            - Any other metadata needed
        """
        ...


class StubDetector:
    """
    Stub detector that returns no detections.
    Used for testing the pipeline before implementing real detection logic.
    """

    def __init__(self):
        """Initialize the stub detector."""
        pass

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Stub detection that returns an empty list.

        Args:
            frame: BGR image from OpenCV

        Returns:
            Empty list (no detections)
        """
        # In the future, this will:
        # 1. Run heuristic detection (color-based, etc.)
        # 2. Run YOLO detection
        # 3. Merge and filter results
        # 4. Return list of detection dicts

        return []
