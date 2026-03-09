# RTSP Human Capture

A multi-stream RTSP person-detection tool built on **YOLOv4 / YOLOv3** (with an automatic HOG fallback) and **OpenCV**. When a person enters a camera frame the tool saves an annotated JPEG snapshot or starts recording an MP4 clip. Multiple cameras run in parallel threads and can be watched in a single composited grid window.

## Features

- **YOLOv4 / YOLOv3 detection** with automatic fallback to OpenCV HOG when model files are absent
- **CUDA GPU acceleration** — automatically detected and enabled; falls back to CPU
- **Single or multiple RTSP streams** processed concurrently via threads
- **Two save modes** — `image` (annotated JPEG snapshot on person entry) or `video` (MP4 clip of the entire presence window)
- **Live display** — dedicated window for a single stream; resizable grid window for multiple streams
- **INI config file** (`config.cfg`) for all paths and detection settings; individual values can be overridden from the CLI
- **Automatic reconnect** — each stream retries up to 5 times on read failure before giving up

## Project Structure

```text
rtsp-human-capture/
├── main.py                 # CLI entry point
├── config.py               # Config loader (AppConfig dataclass)
├── config.cfg              # Default configuration file
├── person_detector.py      # YOLOv4 / YOLOv3 / HOG inference (thread-safe)
├── display_manager.py      # Grid window composition and display thread
├── stream_processor.py     # Per-stream loop, save logic, reconnect
├── multi_stream_manager.py # Thread orchestration for multiple streams
├── pyproject.toml
├── model/                  # Place model files here (see below)
│   ├── yolov4.weights
│   ├── yolov4.cfg
│   └── coco.names
└── deps/
    └── opencv_contrib_python-*.whl
```

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- *(Optional)* NVIDIA GPU with CUDA for hardware-accelerated inference
- *(Optional)* Get the `opencv-contrib-python` CUDA wheels from <https://github.com/cudawarped/opencv-python-cuda-wheels/releases/latest> and place the appropriate `.whl` file in the `deps/` directory, then install with:

## Installation

```bash
git clone https://github.com/itsyourap/rtsp-human-capture
cd rtsp-human-capture
uv sync
```

If `uv sync` fails due to missing OpenCV CUDA wheels, you can install the CPU-only version as a fallback:

```bash
uv pip install opencv-python
```

## Model Files

Download the model files and place them in the `model/` directory (path is configurable via `config.cfg`):

| File | Download |
| --- | --- |
| `yolov4.weights` | [AlexeyAB/darknet releases](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) |
| `yolov4.cfg` | [darknet/cfg/yolov4.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg) |
| `coco.names` | [darknet/data/coco.names](https://github.com/AlexeyAB/darknet/blob/master/data/coco.names) |

YOLOv3 files are also supported and tried as a fallback. If neither is found, the built-in OpenCV HOG detector is used automatically.

## Configuration

Edit `config.cfg` to set default paths and detection parameters:

```ini
[paths]
# Directory containing model files
model_dir = model

# Root directory for saved outputs
# Multi-stream runs create sub-folders: <output_dir>/stream_<id>/
output_dir = output

[detection]
confidence_threshold = 0.5   # 0.0 – 1.0
person_area_threshold = 1000  # minimum bounding-box area in pixels
frame_skip = 15               # analyse every Nth frame (15 ≈ 2 fps on 30 fps stream)
```

All `[detection]` values can be overridden per-run with CLI flags.

## Usage

### Single RTSP stream

```bash
uv run main.py --rtsp "rtsp://camera1.local/stream" --save image
uv run main.py --rtsp "rtsp://camera1.local/stream" --save video --no-display
```

### Multiple RTSP streams

```bash
# Supply URLs directly
uv run main.py --rtsp-list "rtsp://cam1.local" "rtsp://cam2.local" --save video --display

# Or load from a file (one URL per line, # for comments)
uv run main.py --rtsp-file streams.txt --save image --display
```

### Test with a local image

```bash
uv run main.py --test-image photo.jpg --save image
```

### Override config values at runtime

```bash
uv run main.py --rtsp-file streams.txt --save video \
  --config custom.cfg \
  --confidence 0.6 \
  --frame-skip 10 \
  --area-threshold 2000
```

### All CLI options

| Flag | Description |
| --- | --- |
| `--config PATH` | Config file to load (default: `config.cfg`) |
| `--rtsp URL` | Single RTSP stream URL |
| `--rtsp-list URL …` | One or more RTSP stream URLs |
| `--rtsp-file PATH` | Text file with one RTSP URL per line |
| `--test-image PATH` | Run detection on a local image file |
| `--save image\|video` | **Required.** Save annotated snapshots or MP4 clips |
| `--display` | Show live grid window (multiple streams) |
| `--no-display` | Suppress the live window (single stream) |
| `--confidence FLOAT` | Detection confidence threshold (overrides config) |
| `--area-threshold INT` | Minimum bounding-box area in pixels (overrides config) |
| `--frame-skip INT` | Analyse every Nth frame (overrides config) |

## Output

Saved files are written under `output_dir` (default `output/`):

```text
output/
├── stream_1/
│   ├── person_entry_1_20260309_143022_1741528222.jpg   # image mode
│   └── person_clip_1_20260309_143022_1741528222.mp4    # video mode
└── stream_2/
    └── ...
```

For single-stream mode files are saved directly inside `output_dir`.
