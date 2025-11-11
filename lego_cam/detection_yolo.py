"""
YOLOv8-based detector for Lego Cam.
Uses Ultralytics YOLOv8n model for GPU-accelerated object detection.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class YoloDetectorError(Exception):
    """Exception raised when YOLO detector encounters a fatal error."""
    pass


class YoloDetector:
    """
    YOLO-based detector using YOLOv8n.

    This detector:
    1. Loads YOLOv8n weights via Ultralytics library
    2. Runs inference on GPU (CUDA)
    3. Returns detections in normalized format compatible with heuristic detector
    """

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        device: str = "cuda:0",
        conf_threshold: float = 0.25,
    ):
        """
        Initialize YOLO detector.

        Args:
            weights_path: Path to YOLO weights file
            device: Device to run inference on ("cuda:0" or "cpu")
            conf_threshold: Minimum confidence threshold for detections

        Raises:
            YoloDetectorError: If YOLO model cannot be loaded or GPU unavailable
        """
        self.weights_path = weights_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None

        try:
            # Import ultralytics
            try:
                from ultralytics import YOLO
            except ImportError:
                raise YoloDetectorError(
                    "Ultralytics library not installed. Install with: pip install ultralytics"
                )

            # Load model
            logger.info(f"Loading YOLO model from {weights_path}")
            self.model = YOLO(weights_path)

            # Move to specified device
            if device.startswith("cuda"):
                import torch
                if not torch.cuda.is_available():
                    raise YoloDetectorError(
                        f"CUDA not available, but device '{device}' was specified"
                    )
                logger.info(f"Using GPU: {device}")
            else:
                logger.info(f"Using CPU: {device}")

            # Run a warmup inference
            logger.info("Running warmup inference...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy_frame,
                device=self.device,
                conf=self.conf_threshold,
                verbose=False,
            )
            logger.info(f"YoloDetector initialized successfully")

        except YoloDetectorError:
            raise
        except Exception as e:
            raise YoloDetectorError(f"Failed to initialize YOLO detector: {e}") from e

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect objects using YOLO.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of detection dicts with keys:
            - label: object class name
            - color: "unknown" (YOLO doesn't detect colors)
            - x_min, y_min, x_max, y_max: normalized coordinates (0..1)
            - confidence: detection confidence

        Raises:
            YoloDetectorError: If inference fails
        """
        if self.model is None:
            raise YoloDetectorError("Model not initialized")

        try:
            # Get frame dimensions
            height, width = frame.shape[:2]

            # Run inference
            # YOLO expects RGB, but can handle BGR directly with recent ultralytics
            results = self.model.predict(
                frame,
                device=self.device,
                conf=self.conf_threshold,
                verbose=False,
            )

            detections = []

            # Process results
            if results and len(results) > 0:
                result = results[0]  # First image in batch

                # Extract boxes
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        # Get box coordinates (xyxy format)
                        box = boxes.xyxy[i].cpu().numpy()
                        x_min_px, y_min_px, x_max_px, y_max_px = box

                        # Normalize coordinates to 0..1
                        x_min = float(x_min_px / width)
                        y_min = float(y_min_px / height)
                        x_max = float(x_max_px / width)
                        y_max = float(y_max_px / height)

                        # Get confidence
                        conf = float(boxes.conf[i].cpu().numpy())

                        # Get class name
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        label = result.names[cls_id] if result.names else f"class_{cls_id}"

                        # Create detection dict
                        detection = {
                            "label": label,
                            "color": "unknown",  # YOLO doesn't detect LEGO colors
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max,
                            "confidence": conf,
                        }

                        detections.append(detection)

            return detections

        except Exception as e:
            raise YoloDetectorError(f"YOLO inference failed: {e}") from e

    def calibrate_from_frame(self, frame: np.ndarray) -> None:
        """
        Calibration is a no-op for YOLO detector.

        YOLO doesn't require background calibration like the heuristic detector.

        Args:
            frame: BGR image from OpenCV (ignored)
        """
        logger.debug("Calibration called on YOLO detector (no-op)")
        pass
