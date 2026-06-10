# Trend Detector

Real-time webcam hand tracking with static gestures, custom movement patterns, and TikTok-style combo detection. Built with OpenCV, MediaPipe Tasks API, and Pygame.

**Repository:** [github.com/mssbahar/trenddetector](https://github.com/mssbahar/trenddetector)

## What It Does

| Mode | Trigger | Sound / Video |
|------|---------|---------------|
| **TikTok 1** | Hold nose pinch (thumb + index on face) | `kicau.mp3` + cat & dance GIFs |
| **TikTok 2** | Hold sideways hand sweep | `thaidance.mp3` + Power Rangers video (both sides) |
| **Static gestures** | Open Palm, Fist, Peace Sign, Thumbs Up | One-shot sound + optional overlay |
| **Movement patterns** | Direction or coordinate sequences | One-shot sound + overlay (3 s) |

TikTok combos use **hold-to-play**: sound and overlays run only while the gesture is detected and stop immediately when you release. Gestures and movement patterns use a one-shot trigger with a 2.5 s cooldown.

## Prerequisites

- Python 3.11+ (tested on 3.14 with `pygame-ce`)
- Webcam
- Windows, macOS, or Linux

## Setup

```bash
git clone https://github.com/mssbahar/trenddetector.git
cd trenddetector
python -m venv .venv
```

**Windows**

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Download the MediaPipe hand model (required once):

```powershell
# Windows PowerShell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "models\hand_landmarker.task"
```

```bash
# macOS / Linux
curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -o models/hand_landmarker.task
```

## Run

```bash
python main.py
```

## Controls

| Key | Action |
|-----|--------|
| `d` | Toggle debug mode (direction history, normalized coords) |
| `r` | Reset movement buffer, gestures, combos, and active effects |
| `q` / `Esc` | Quit |

## TikTok Combos

Configured in [`config.py`](config.py) under `TIKTOK_COMBOS`.

### TikTok 1 — Nose Pinch

1. Pinch thumb and index finger together.
2. Move the pinch up to your nose / upper face area.
3. **Hold** the pinch — sound loops and GIF overlays appear on left and right.
4. Release — everything stops.

### TikTok 2 — Hand Sweep

1. Keep one hand in frame (no nose pinch).
2. Sweep the hand **sideways** (left or right) in a clear motion.
3. **Hold** the sweep — music loops and Power Rangers video plays on both sides.
4. Stop moving — everything stops.

TikTok 1 takes priority if both gestures could match.

## Other Features

- **Static gestures**: Rule-based detection with multi-frame confirmation
- **Movement patterns**: Direction sequences (e.g. left→right→left→center) and coordinate paths
- **Landmark smoothing**: EMA filtering reduces jitter (`LANDMARK_SMOOTHING_ALPHA` in config)
- **Multi-overlay effects**: GIF, MP4, and PNG overlays with alpha blending (Pillow required for GIFs)
- **HUD**: FPS, gesture, pattern progress, combo status, active effect name

## Project Structure

```
trenddetector/
├── main.py                 # Application entry point
├── config.py               # Settings, gesture bindings, TikTok combos, patterns
├── camera.py               # Webcam capture + FPS
├── hand_tracker.py         # MediaPipe Hand Landmarker (Tasks API)
├── landmark_smoother.py    # EMA smoothing for landmarks
├── hand_pose.py            # Pinch / pose helpers for combo detection
├── gesture_detector.py     # Static gesture rules
├── movement_tracker.py     # Normalized trajectory buffer
├── pattern_recognizer.py   # Pattern matching engine
├── combo_detector.py       # TikTok 1 & 2 hold-to-play detection
├── event_system.py         # Cooldown + hold sync + event dispatch
├── sound_manager.py        # Pygame audio (loop + one-shot)
├── effect_manager.py       # GIF / video / image overlays
├── analyze_example.py      # Utility to analyze reference TikTok videos
├── models/
│   ├── hand_data.py        # Data classes
│   └── hand_landmarker.task  # Download separately (see Setup)
└── assets/
    ├── sounds/             # kicau.mp3, thaidance.mp3, …
    ├── videos/             # GIFs, MP4 overlays
    ├── images/             # PNG overlays for gestures/patterns
    └── example/            # Reference TikTok clips + frame samples
```

## Configuration

All bindings live in [`config.py`](config.py):

- `GESTURE_BINDINGS` — static gesture → sound + effect
- `TIKTOK_COMBOS` — combo id → sound + overlay list (`side`: `"left"` or `"right"`)
- `CUSTOM_PATTERNS` — movement pattern definitions

### Adding a Custom Movement Pattern

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

No core code changes required — restart the app after editing config.

### Adding a TikTok Combo Overlay

```python
"tiktok1": {
    "display_name": "TikTok 1 — Nose Pinch",
    "sound": str(SOUNDS_DIR / "kicau.mp3"),
    "overlays": [
        {"path": str(VIDEOS_DIR / "Cat GIF.gif"), "side": "right"},
        {"path": str(VIDEOS_DIR / "Dance Animation GIF.gif"), "side": "left"},
    ],
},
```

Adjust overlay size with `EFFECT_SIDE_SCALE` and `EFFECT_SIDE_MARGIN` in config.

## Bundled Assets

These files are included in the repo:

| File | Used by |
|------|---------|
| `assets/sounds/kicau.mp3` | TikTok 1 |
| `assets/sounds/thaidance.mp3` | TikTok 2 |
| `assets/videos/Cat GIF.gif` | TikTok 1 (right) |
| `assets/videos/Dance Animation GIF.gif` | TikTok 1 (left) |
| `assets/videos/power rangers.mp4` | TikTok 2 (left + right) |
| `assets/example/Tiktok1.mp4`, `Tiktok2.mp4` | Reference / analysis only |

Optional gesture and pattern assets (`palm.wav`, `sparkle.png`, etc.) can be added under `assets/` — the app logs warnings and continues if files are missing.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Webcam not found | Change `WEBCAM_INDEX` in `config.py` (try 0, 1, 2) |
| Low FPS | Lower `FRAME_WIDTH` / `FRAME_HEIGHT` in `config.py` |
| No audio | Check mixer init in logs; confirm `.mp3` / `.wav` paths exist |
| GIF overlays missing | Install Pillow: `pip install Pillow` |
| `hand_landmarker.task` missing | Run the download command in [Setup](#setup) |
| TikTok 1 not triggering | Pinch thumb+index and move hand to nose; improve lighting |
| TikTok 2 not triggering | Sweep one hand clearly sideways; avoid nose pinch at same time |
| Pygame install fails on Python 3.14+ | Use `pygame-ce` (already in `requirements.txt`) |

## Dependencies

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
pygame-ce>=2.5.0
Pillow>=10.0.0
```

## Future Extensions

See [`prd.md`](prd.md) for full requirements. Planned next steps:

- Full-body MediaPipe Pose tracking
- ML-based trend classification
- LSTM / Transformer sequence learning
