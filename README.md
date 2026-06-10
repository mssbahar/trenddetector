# Hand Gesture & Movement Detection

Real-time webcam hand tracking with static gesture detection and custom movement sequence recognition. Triggers sound and visual effects via OpenCV, MediaPipe, and Pygame.

## Prerequisites

- Python 3.11+
- Webcam
- Optional: sound/video/image assets (see [Asset files](#asset-files))

## Setup

```bash
cd Trend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Download the hand tracking model (required once):

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "models\hand_landmarker.task"
```

## Run

```bash
python main.py
```

## Controls

| Key | Action |
|-----|--------|
| `d` | Toggle debug mode (shows direction history and normalized coords) |
| `r` | Reset movement buffer |
| `q` / `Esc` | Quit |

## Features

- **Static gestures**: Open Palm, Fist, Peace Sign, Thumbs Up
- **Movement patterns**: Direction sequences (e.g. left→right→left→center) and coordinate paths
- **Effects**: PNG image overlays and MP4 video overlays
- **Cooldown**: 2.5 s minimum between repeated triggers

## Project Structure

```
Trend/
├── main.py                 # Application entry point
├── camera.py               # Webcam capture + FPS
├── hand_tracker.py         # MediaPipe hand tracking
├── gesture_detector.py     # Static gesture rules
├── movement_tracker.py     # Normalized trajectory buffer
├── pattern_recognizer.py   # Pattern matching engine
├── event_system.py         # Cooldown + event dispatch
├── sound_manager.py        # Pygame audio
├── effect_manager.py       # Image/video overlays
├── config.py               # All settings and pattern bindings
├── models/hand_data.py     # Data classes
└── assets/
    ├── sounds/
    ├── videos/
    └── images/
```

## Adding Custom Movement Patterns

Edit `CUSTOM_PATTERNS` in [`config.py`](config.py):

```python
{
    "id": "my_trend",
    "type": "direction",           # or "coordinate"
    "pattern": ["up", "down", "up", "center"],
    "display_name": "My Trend",
    "sound": str(SOUNDS_DIR / "trend1.wav"),
    "effect": str(IMAGES_DIR / "sparkle.png"),
    "effect_type": "image",        # "image", "video", or "none"
}
```

No core code changes required.

## Asset Files

Place files in `assets/` (app runs without them — warnings only):

| File | Purpose |
|------|---------|
| `assets/sounds/palm.wav` | Open Palm |
| `assets/sounds/fist.wav` | Fist |
| `assets/sounds/peace.wav` | Peace Sign |
| `assets/sounds/thumbs_up.wav` | Thumbs Up |
| `assets/sounds/trend1.wav` | Trend patterns |
| `assets/sounds/trend2.wav` | Circle Swipe |
| `assets/images/sparkle.png` | Trend Wave overlay |
| `assets/images/star.png` | Open Palm overlay |
| `assets/videos/burst.mp4` | Circle Swipe video |

## Troubleshooting

- **Webcam not found**: Change `WEBCAM_INDEX` in `config.py` (try 0, 1, 2).
- **Low FPS**: Reduce `FRAME_WIDTH` / `FRAME_HEIGHT` in `config.py`.
- **No audio**: Ensure pygame mixer initializes; check that `.wav` files exist.
- **Gestures not detected**: Improve lighting; hold hand fully in frame.

## Future Extensions

Designed for later integration of:
- Full-body MediaPipe Pose tracking
- ML-based trend classification
- LSTM/Transformer sequence learning
- TikTok-style auto trend detection

See [`prd.md`](prd.md) for full requirements.
