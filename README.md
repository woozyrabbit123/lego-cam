# Lego Cam

Local-first Lego detection and logging app. Real-time webcam viewer that detects Lego bricks and minifig parts, draws boxes, and logs data to SQLite for analysis. Built with OpenCV, YOLOv8, and smart GPU/CPU fallback.

## Current Status: v1 - GPU-Accelerated Detection

This version adds real detection capabilities with three modes:

- **FAST Mode**: HSV color-based heuristic detection (CPU only, always available)
- **SMART Mode**: YOLOv8n GPU detection (fails gracefully if GPU unavailable)
- **AUTO Mode**: YOLOv8n with automatic fallback to heuristic (recommended)

### Key Features
- ✅ **Multiple Detection Backends**: Choose between heuristic (CPU) and YOLO (GPU)
- ✅ **Automatic Fallback**: AUTO mode seamlessly switches to heuristic if GPU fails
- ✅ **Real-time Detection**: Bounding boxes with color-coded labels
- ✅ **SQLite Logging**: Sessions, frames, detections, and segment markers
- ✅ **Rich HUD**: Shows mode, backend, session, segment, and detection count
- ✅ **Advanced Controls**: Snapshots, calibration, segments, quit confirmation
- ✅ **Threaded Pipeline**: Capture → Detection → UI with proper queue management

## Detection Modes

### FAST Mode (CPU Only)
```bash
python -m lego_cam --mode fast
```
- Uses HSV color filtering and contour detection
- Always available (no GPU required)
- Detects 5 colors: red, blue, yellow, green, white
- Best for: Testing, low-end hardware, CPU-only environments

### SMART Mode (GPU Required)
```bash
python -m lego_cam --mode smart
```
- Uses YOLOv8n with GPU acceleration
- Fails hard if GPU/CUDA unavailable (app won't start)
- Higher accuracy for general object detection
- Best for: Production use with guaranteed GPU availability

### AUTO Mode (Recommended)
```bash
python -m lego_cam --mode auto
```
- Attempts YOLOv8n on GPU first
- Automatically falls back to heuristic if GPU fails
- Runtime fallback if YOLO errors occur
- Best for: Development, mixed environments, maximum reliability

## Features by Version

### v1 (Current)
- ✅ YOLOv8n GPU detection
- ✅ HSV heuristic detection
- ✅ Three detection modes (FAST/SMART/AUTO)
- ✅ Automatic fallback in AUTO mode
- ✅ CLI argument parsing (--mode)
- ✅ SQLite session and detection logging
- ✅ Segment markers for logical grouping
- ✅ Snapshot saving
- ✅ Background calibration
- ✅ Quit confirmation
- ✅ Enhanced HUD with mode/backend display

### Planned (v2+)
- 🔜 YOLO + Heuristic fusion (run both, merge results)
- 🔜 Custom YOLO training for LEGO-specific classes
- 🔜 Per-brick identification and tracking
- 🔜 Data export to CSV/JSON
- 🔜 FPS counter in HUD
- 🔜 Configurable HSV color ranges

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
│   DB Thread     │◀────(unbounded)────── (from detection thread)
└─────────────────┘
```

## Project Structure

```
lego-cam/
├── pyproject.toml            # Project metadata and dependencies
├── requirements.txt          # Alternative pip requirements
├── README.md                # This file
├── lego_cam.db              # SQLite database (created on first run)
└── lego_cam/                # Main package
    ├── __init__.py          # Package initialization
    ├── config.py            # Global constants and configuration
    ├── main.py              # Entry point, UI loop, and CLI
    ├── pipeline.py          # Capture and detection thread functions
    ├── detection_stub.py    # Detector protocol interface
    ├── detection_heuristic.py # HSV color-based detector
    ├── detection_yolo.py    # YOLOv8n GPU detector
    └── db.py                # Database thread and SQLite operations
```

## Installation

### Prerequisites
- Python 3.11 or higher
- Webcam connected to your system
- Linux, macOS, or Windows
- **For SMART/AUTO modes:** NVIDIA GPU with CUDA support (optional for FAST mode)
- **For SMART/AUTO modes:** YOLOv8n weights (auto-downloaded on first run)

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

**Choose a detection mode:**

```bash
# FAST mode (CPU only, always available)
python -m lego_cam --mode fast

# SMART mode (GPU required, fails if unavailable)
python -m lego_cam --mode smart

# AUTO mode (GPU with fallback, recommended)
python -m lego_cam --mode auto

# Default mode (currently FAST)
python -m lego_cam
```

**Using installed command:**
```bash
lego-cam --mode auto
```

### Controls

Once the application is running:

- **`q`** - Quit (press twice to confirm)
- **`p`** - Pause/unpause the display (freezes UI but threads continue running)
- **`s`** - Save snapshot to `snapshots/` folder
- **`c`** - Calibrate background (for heuristic detector)
- **`r`** - Start new segment (creates logical boundary in database)
- **`x`** - Run scan mode (dense detection with summary)
- **`b`** - Bookmark current frame (saves frame + DB entry)

### Expected Output

On startup, you'll see:
```
============================================================
Lego Cam v1 - Mode: FAST
============================================================
Detection backend: heuristic
...
Session Setup
============================================================
Session tag (optional, e.g. 'BOX: green tub'):
```

The HUD displays:
```
Lego Cam v1  [FAST / HEURISTIC]     (or [AUTO / YOLO], etc.)
Session: 1 (my session tag)
Segment: 1  |  N=3  |  Scene: ACTIVE
Last detections: 5
```

The application will:
1. Open your default webcam at 640x480
2. Run detection based on selected mode
3. Draw colored bounding boxes around detected objects
4. Log all frames and detections to SQLite database
5. Display real-time HUD with mode, session, detection interval, and scene state

On exit, a session summary is printed:
```
============================================================
SESSION SUMMARY
============================================================
Session: 1 (my session tag)
Frames: 1234
Detections: 567
Segments: 3
Scans: 2
Bookmarks: 5
Top detected labels:
  person: 234
  brick: 123
  ...
============================================================
```

## Smart Behavior

Lego Cam v1 includes adaptive and intelligent features to optimize performance:

### Adaptive Detection Interval (N)

The system automatically adjusts how often detection runs (every Nth frame) based on detection time:

- **Initial N**: Starts at 3 (runs detection every 3rd frame)
- **Adaptation**: Every 5 seconds, N is adjusted based on measured detection time
- **Range**: N varies between 1 (every frame) and 10 (every 10th frames)
- **Goal**: Keep detection time under 50% of frame budget to maintain responsive UI
- **Benefit**: Heavy detectors (YOLO on CPU) automatically skip more frames; fast detectors run more frequently

The current N value is displayed in the HUD: `N=3`

### Idle/Static Scene Detection

When the scene stops changing, detection interval is boosted aggressively:

- **Detection**: Compares consecutive frames (64x48 downsampled grayscale)
- **Threshold**: Mean pixel difference < 8.0 on 0-255 scale
- **Activation**: After 2 seconds (~30 frames) of static scene
- **Effect**: Detection interval becomes N=30 when idle
- **Recovery**: Returns to adaptive N immediately when motion detected

Scene state is displayed in the HUD: `Scene: ACTIVE` or `Scene: IDLE`

### Scan Mode ('x' hotkey)

Press `x` to run a short dense detection scan:

- **Duration**: 2 seconds of continuous detection (N=1)
- **Collection**: All detections aggregated in-memory
- **Summary**: Counts by label and color, stored in `scans` table
- **Use case**: Quick inventory check or sample capture
- **HUD indicator**: `[SCANNING]` appears during scan

The scan summary is logged to the database with timestamp and JSON statistics.

### Bookmarks ('b' hotkey)

Press `b` to bookmark the current frame:

- **Saves**: Annotated frame to `bookmarks/` directory
- **Database**: Entry in `bookmarks` table with session_id, timestamp, and image path
- **Filename**: `bookmark_session_X_tagY_segZ_TIMESTAMP.png`
- **Use case**: Mark interesting frames for later review

Both `scans` and `bookmarks` are counted in the session summary on exit.

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

**GPU/CUDA issues (SMART/AUTO modes):**
- **SMART mode won't start**: Verify CUDA is installed and accessible: `nvidia-smi`
- **AUTO mode falls back to heuristic**: Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- **YOLOv8n download fails**: Ensure internet connectivity; weights are auto-downloaded on first run
- **Out of memory errors**: Reduce resolution in `config.py` or use FAST mode
- **HUD shows "BROKEN"**: YOLO encountered runtime error in SMART mode; restart app or use AUTO/FAST

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

**Video Capture:**
- **RESOLUTION**: Camera resolution (default: 640x480)
- **TARGET_FPS**: Desired frame rate (default: 15)
- **CAMERA_INDEX**: Which camera to use (default: 0)

**Detection:**
- **DEFAULT_DETECTION_MODE**: Default mode if --mode not specified (default: FAST)
- **YOLO_WEIGHTS_PATH**: Path to YOLO weights (default: yolov8n.pt)
- **YOLO_DEVICE**: GPU device for YOLO (default: cuda:0)
- **YOLO_CONF_THRESHOLD**: Minimum confidence for YOLO detections (default: 0.25)
- **MIN_CONTOUR_AREA**: Minimum area for heuristic detections (default: 100px)
- **HEURISTIC_CONFIDENCE**: Confidence score for heuristic (default: 0.6)

**UI & Database:**
- **DB_PATH**: SQLite database location (default: lego_cam.db)
- **DB_BATCH_SIZE**: Detections per batch (default: 50)
- **HUD appearance**: Text, color, position, font
- **Queue sizes**: Pipeline queue capacities
- **Timeouts**: Thread join and queue operation timeouts

## License

MIT License - See LICENSE file for details.

## Contributing

This is currently a personal/educational project. Feedback and suggestions welcome!

## Acknowledgments

Built with:
- [OpenCV](https://opencv.org/) for video processing
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for GPU-accelerated object detection
- [PyTorch](https://pytorch.org/) for deep learning inference
- [NumPy](https://numpy.org/) for array operations
- Python threading for concurrency
