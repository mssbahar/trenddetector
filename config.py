"""Central configuration for the hand gesture and movement detection system."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
ASSETS_DIR = PROJECT_ROOT / "assets"
SOUNDS_DIR = ASSETS_DIR / "sounds"
VIDEOS_DIR = ASSETS_DIR / "videos"
IMAGES_DIR = ASSETS_DIR / "images"
HAND_LANDMARKER_MODEL = PROJECT_ROOT / "models" / "hand_landmarker.task"

# ── Camera ─────────────────────────────────────────────────────────────────
WEBCAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_SMOOTHING = 30

# ── MediaPipe Hands ────────────────────────────────────────────────────────
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# ── Landmark Smoothing ─────────────────────────────────────────────────────
LANDMARK_SMOOTHING_ENABLED = True
LANDMARK_SMOOTHING_ALPHA = 0.35  # lower = smoother/slower, higher = snappier

# ── Movement Tracking ────────────────────────────────────────────────────────
MOVEMENT_BUFFER_SIZE = 60
MOVEMENT_TRACK_POINT = "index_tip"  # "wrist" or "index_tip"
DIRECTION_THRESHOLD = 0.08
DIRECTION_MIN_SEGMENT = 3
NORMALIZED_DEAD_ZONE = 0.02

# ── Pattern Recognition ────────────────────────────────────────────────────
DIRECTION_TOLERANCE = 1
COORDINATE_TOLERANCE = 0.15
COORDINATE_RESAMPLE_POINTS = 8
PATTERN_MIN_FRAMES = 10

# ── Event System ───────────────────────────────────────────────────────────
COOLDOWN_SECONDS = 2.5
GESTURE_CONFIRM_FRAMES = 3
COMBO_CONFIRM_FRAMES = 3
VERTICAL_HAND_SEPARATION = 0.06
TIKTOK1_WAVE_MIN_DELTA = 0.035
TIKTOK1_WAVE_MIN_SEGMENT = 2
TIKTOK1_PINCH_THRESHOLD = 0.075
TIKTOK2_SWEEP_MIN_SPAN = 0.08
TIKTOK2_SWEEP_MIN_DELTA = 0.025
TIKTOK2_SWEEP_MIN_SEGMENT = 3

# ── Effects ────────────────────────────────────────────────────────────────
EFFECT_DURATION_SECONDS = 3.0
EFFECT_OVERLAY_SCALE = 0.4
EFFECT_SIDE_SCALE = 0.22
EFFECT_POSITION = (20, 20)
EFFECT_SIDE_MARGIN = 12

# ── UI ─────────────────────────────────────────────────────────────────────
HUD_FONT_SCALE = 0.6
HUD_COLOR = (0, 255, 128)
HUD_BG_COLOR = (0, 0, 0)
DEBUG_COLOR = (255, 200, 0)

# ── Gesture Bindings ───────────────────────────────────────────────────────
GESTURE_BINDINGS = {
    "open_palm": {
        "display_name": "Open Palm",
        "sound": str(SOUNDS_DIR / "palm.wav"),
        "effect": str(IMAGES_DIR / "star.png"),
        "effect_type": "image",
    },
    "fist": {
        "display_name": "Fist",
        "sound": str(SOUNDS_DIR / "fist.wav"),
        "effect": "",
        "effect_type": "none",
    },
    "peace_sign": {
        "display_name": "Peace Sign",
        "sound": str(SOUNDS_DIR / "peace.wav"),
        "effect": "",
        "effect_type": "none",
    },
    "thumbs_up": {
        "display_name": "Thumbs Up",
        "sound": str(SOUNDS_DIR / "thumbs_up.wav"),
        "effect": "",
        "effect_type": "none",
    },
}

# ── TikTok full-sequence combos ─────────────────────────────────────────────
TIKTOK_COMBOS = {
    "tiktok1": {
        "display_name": "TikTok 1 — Nose Pinch",
        "sound": str(SOUNDS_DIR / "kicau.mp3"),
        "overlays": [
            {"path": str(VIDEOS_DIR / "Cat GIF.gif"), "side": "right"},
            {"path": str(VIDEOS_DIR / "Dance Animation GIF.gif"), "side": "left"},
        ],
    },
    "tiktok2": {
        "display_name": "TikTok 2 — Hand Sweep",
        "sound": str(SOUNDS_DIR / "thaidance.mp3"),
        "overlays": [
            {"path": str(VIDEOS_DIR / "power rangers.mp4"), "side": "left", "mute": True},
            {"path": str(VIDEOS_DIR / "power rangers.mp4"), "side": "right", "mute": True},
        ],
    },
}

# ── Custom Movement Patterns ───────────────────────────────────────────────
CUSTOM_PATTERNS = [
    {
        "id": "trend_wave",
        "type": "direction",
        "pattern": ["left", "right", "left", "center"],
        "display_name": "Trend Wave",
        "sound": str(SOUNDS_DIR / "trend1.wav"),
        "effect": str(IMAGES_DIR / "sparkle.png"),
        "effect_type": "image",
    },
    {
        "id": "circle_swipe",
        "type": "coordinate",
        "pattern": [(0.0, -0.3), (0.3, 0.0), (0.0, 0.3), (-0.3, 0.0), (0.0, 0.0)],
        "display_name": "Circle Swipe",
        "sound": str(SOUNDS_DIR / "trend2.wav"),
        "effect": str(VIDEOS_DIR / "burst.mp4"),
        "effect_type": "video",
    },
    {
        "id": "vertical_bounce",
        "type": "direction",
        "pattern": ["up", "down", "up", "center"],
        "display_name": "Vertical Bounce",
        "sound": str(SOUNDS_DIR / "trend1.wav"),
        "effect": "",
        "effect_type": "none",
    },
]
