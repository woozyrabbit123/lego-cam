"""
Detector interface for Lego Cam.
Defines the Protocol that all detectors must implement.
"""

from typing import Protocol
from dataclasses import dataclass
import numpy as np


class Detector(Protocol):
    """
    Protocol/interface for all detectors.

    All detector implementations (heuristic, YOLO, etc.) must implement these methods.
    """

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect objects in the given frame.

        Args:
            frame: BGR image from OpenCV (numpy array)

        Returns:
            List of detection dictionaries. Each detection should have:
            - 'label': str object type/class name
            - 'color': str color name (may be "unknown" for some detectors)
            - 'x_min': float normalized x coordinate (0..1)
            - 'y_min': float normalized y coordinate (0..1)
            - 'x_max': float normalized x coordinate (0..1)
            - 'y_max': float normalized y coordinate (0..1)
            - 'confidence': float confidence score (0..1)
        """
        ...

    def calibrate_from_frame(self, frame: np.ndarray) -> None:
        """
        Calibrate the detector using the current frame.

        This may capture a background reference or adjust detector parameters.
        Some detectors (e.g., YOLO) may implement this as a no-op.

        Args:
            frame: BGR image from OpenCV to use for calibration
        """
        ...


@dataclass
class DetectorHolder:
    """
    Mutable holder for a Detector instance.

    This allows the UI thread and detection thread to share a reference
    to the current detector. When AUTO mode falls back from YOLO to heuristic,
    the detection thread can update detector_holder.detector, and the UI
    thread (for calibration) will automatically use the new detector.
    """
    detector: Detector


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
        return []

    def calibrate_from_frame(self, frame: np.ndarray) -> None:
        """No-op calibration for stub detector."""
        pass
