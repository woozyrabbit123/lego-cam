"""
Lego Cam - A threaded computer vision pipeline for LEGO detection.

This package provides a foundation for real-time LEGO brick detection
using webcam input, with a clean threaded architecture for future expansion.
"""

__version__ = "0.1.0"
__author__ = "Lego Cam Team"

from .main import main

__all__ = ["main"]
