# Lego Cam

Local-first Lego detection and logging app. Real-time webcam viewer that detects Lego bricks and minifig parts, draws boxes, and logs data to SQLite for analysis. Built in phases (v0–v3) with OpenCV, YOLO, and smart automation.

## Current Status: v0 - Foundation Pipeline

This initial version establishes the core threaded pipeline architecture without actual detection logic. It provides:

- **Threaded Pipeline**: Capture → Detection → UI with proper queue management
- **640x480 Webcam**: Real-time video capture at configured resolution
- **Clean Architecture**: Modular design ready for heuristic and YOLO detectors
- **Basic Controls**: Keyboard hotkeys for quit and pause
- **Stub Components**: DB thread and detector interface ready for expansion

## Features

### Current (v0)
- ✅ Webcam capture at 640x480 resolution
- ✅ Threaded pipeline with queue-based communication
- ✅ Real-time video display with HUD overlay
- ✅ Hotkey controls ('q' to quit, 'p' to pause)
- ✅ Stub detector interface
- ✅ Stub database thread
- ✅ Clean shutdown handling

### Planned (v1+)
- 🔜 Heuristic color-based detection
- 🔜 YOLO-based object detection
- 🔜 SQLite session and detection logging
- 🔜 Rich HUD with FPS, detection counts, session info
- 🔜 Additional hotkeys: S (session), C (clear), R (record), X/B (export)
- 🔜 Bounding box visualization
- 🔜 Detection confidence filtering
- 🔜 Data export functionality

## Architecture

```
┌─────────────┐
│   Camera    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐     capture_queue      ┌──────────────────┐
│ Capture Thread  │────────(size=1)────────▶│ Detection Thread │
└─────────────────┘                         └────────┬─────────┘
                                                     │
                                                     │ ui_queue
                                                     │ (size=1)
                                                     ▼
                                            ┌─────────────────┐
                                            │   UI Thread     │
                                            │  (main thread)  │
                                            └─────────────────┘

┌─────────────────┐     db_queue
│   DB Thread     │◀────(unbounded)────── (future: detection events)
└─────────────────┘
```

## Project Structure

```
lego-cam/
├── pyproject.toml          # Project metadata and dependencies
├── requirements.txt        # Alternative pip requirements
├── README.md              # This file
└── lego_cam/              # Main package
    ├── __init__.py        # Package initialization
    ├── config.py          # Global constants and configuration
    ├── main.py            # Entry point and UI loop
    ├── pipeline.py        # Capture and detection thread functions
    ├── detection_stub.py  # Detector interface (stub)
    └── db_stub.py         # Database thread (stub)
```

## Installation

### Prerequisites
- Python 3.11 or higher
- Webcam connected to your system
- Linux, macOS, or Windows

### Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd lego-cam
   ```

2. **Install dependencies** (choose one method):

   **Option A: Using pip with requirements.txt**
   ```bash
   pip install -r requirements.txt
   ```

   **Option B: Using pip with pyproject.toml**
   ```bash
   pip install .
   ```

   **Option C: Development install**
   ```bash
   pip install -e ".[dev]"
   ```

## Usage

### Running the Application

**Method 1: As a Python module**
```bash
python -m lego_cam
```

**Method 2: Using the installed command** (if installed with `pip install .`)
```bash
lego-cam
```

### Controls

Once the application is running:

- **`q`** - Quit the application and shutdown cleanly
- **`p`** - Pause/unpause the display (freezes UI but threads continue running)

### Expected Output

On startup, you should see:
```
============================================================
Lego Cam v0 - Foundation Pipeline
============================================================
Resolution: (640, 480)
Target FPS: 15
Controls:
  'q' - Quit
  'p' - Pause/Unpause
============================================================
```

The application will:
1. Open your default webcam
2. Display a window titled "Lego Cam v0"
3. Show real-time video with "Lego Cam v0" text overlay
4. Run detection pipeline (currently no-op)
5. Log thread activity to console

### Troubleshooting

**Camera not opening:**
- Check that no other application is using the webcam
- Try setting a different camera index in `lego_cam/config.py` (CAMERA_INDEX)
- Verify camera permissions on your system

**Window not displaying:**
- Ensure you're running in an environment with GUI support
- On Linux, ensure X11 or Wayland is available
- On SSH sessions, X11 forwarding may be needed

**Import errors:**
- Verify all dependencies are installed: `pip list | grep opencv`
- Try reinstalling: `pip install --force-reinstall opencv-python numpy`

## Development

### Code Style
- Uses Python 3.11+ features
- Type hints where helpful (not enforced yet)
- Black formatting (100 char line length)

### Testing
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (when available)
pytest
```

### Logging
Adjust logging level in `lego_cam/main.py`:
```python
logging.basicConfig(level=logging.DEBUG)  # For verbose output
```

## Configuration

Edit `lego_cam/config.py` to customize:

- **RESOLUTION**: Camera resolution (default: 640x480)
- **TARGET_FPS**: Desired frame rate (default: 15)
- **CAMERA_INDEX**: Which camera to use (default: 0)
- **Queue sizes**: Pipeline queue capacities
- **HUD appearance**: Text, color, position
- **Timeouts**: Thread join and queue operation timeouts

## Next Steps (v1 and beyond)

### Phase 1: Heuristic Detection
- Implement color-based LEGO detection
- Add HSV color filtering
- Simple shape detection for bricks
- Draw bounding boxes on detected objects

### Phase 2: YOLO Integration
- Integrate pre-trained or custom YOLO model
- Implement dual detection (heuristic + YOLO)
- Merge and filter detection results
- Confidence thresholding

### Phase 3: Database & Sessions
- SQLite schema for sessions and detections
- Session management (start/stop/resume)
- Detection event logging
- Historical data queries

### Phase 4: Rich Features
- Enhanced HUD with live statistics
- Hotkeys: S (session), C (clear), R (record), X/B (export)
- Data export to CSV/JSON
- Performance metrics and profiling
- Configuration via CLI arguments

## License

MIT License - See LICENSE file for details.

## Contributing

This is currently a personal/educational project. Feedback and suggestions welcome!

## Acknowledgments

Built with:
- OpenCV for video processing
- NumPy for array operations
- Python threading for concurrency
