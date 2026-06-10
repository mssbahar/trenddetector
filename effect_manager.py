"""Video, GIF, and image overlay effects on the camera feed."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class MediaKind(str, Enum):
    NONE = "none"
    IMAGE = "image"
    GIF = "gif"
    VIDEO = "video"


@dataclass
class _OverlaySlot:
    path: str
    side: str
    kind: MediaKind
    scale: float
    gif_frames: List[np.ndarray] = field(default_factory=list)
    gif_index: int = 0
    gif_last_advance: float = 0.0
    gif_frame_delay: float = 0.08
    video_cap: Optional[cv2.VideoCapture] = None
    current: Optional[np.ndarray] = None


class EffectManager:
    def __init__(
        self,
        duration: float = config.EFFECT_DURATION_SECONDS,
        side_scale: float = config.EFFECT_SIDE_SCALE,
        side_margin: int = config.EFFECT_SIDE_MARGIN,
    ) -> None:
        self._duration = duration
        self._side_scale = side_scale
        self._side_margin = side_margin
        self._active_name: str = ""
        self._active_until: float = 0.0
        self._hold_mode: bool = False
        self._slots: List[_OverlaySlot] = []

    def start_overlays(self, overlays: List[dict], display_name: str) -> None:
        """Start overlays and keep playing until stop() is called."""
        if self._hold_mode and self._active_name == display_name and self._slots:
            return
        self.stop()
        self._hold_mode = True
        self._load_slots(overlays, display_name)

    def stop(self) -> None:
        self._hold_mode = False
        self._active_until = 0.0
        self._clear()

    def trigger_overlays(self, overlays: List[dict], display_name: str) -> None:
        """One-shot overlays with timed duration (gestures/patterns)."""
        self.stop()
        self._hold_mode = False
        if not self._load_slots(overlays, display_name):
            return
        self._active_until = time.perf_counter() + self._duration

    def _load_slots(self, overlays: List[dict], display_name: str) -> bool:
        if not overlays:
            return False

        loaded: List[_OverlaySlot] = []
        for item in overlays:
            slot = self._load_overlay(item)
            if slot is not None:
                loaded.append(slot)

        if not loaded:
            return False

        self._slots = loaded
        self._active_name = display_name
        logger.info("Effect started: %s (%d overlays)", display_name, len(loaded))
        return True

    def _load_overlay(self, item: dict) -> Optional[_OverlaySlot]:
        path = item.get("path", "")
        side = item.get("side", "left")
        scale = float(item.get("scale", self._side_scale))
        file_path = Path(path)

        if not file_path.exists():
            logger.warning("Effect file not found: %s", path)
            return None

        ext = file_path.suffix.lower()
        slot = _OverlaySlot(path=path, side=side, kind=MediaKind.NONE, scale=scale)

        if ext == ".gif":
            frames = self._load_gif_frames(file_path)
            if not frames:
                return None
            slot.kind = MediaKind.GIF
            slot.gif_frames = frames
            slot.current = frames[0]
            slot.gif_frame_delay = max(0.05, self._gif_frame_delay(file_path))
            return slot

        if ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                logger.warning("Could not open video: %s", path)
                return None
            slot.kind = MediaKind.VIDEO
            slot.video_cap = cap
            return slot

        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("Could not load image: %s", path)
            return None
        slot.kind = MediaKind.IMAGE
        slot.current = img
        return slot

    @staticmethod
    def _load_gif_frames(file_path: Path) -> List[np.ndarray]:
        try:
            from PIL import Image, ImageSequence
        except ImportError:
            logger.warning("Pillow required for GIF overlays.")
            return []

        try:
            im = Image.open(file_path)
            frames: List[np.ndarray] = []
            for frame in ImageSequence.Iterator(im):
                rgba = np.array(frame.convert("RGBA"))
                frames.append(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            return frames
        except Exception as exc:
            logger.warning("Failed to load GIF '%s': %s", file_path, exc)
            return []

    @staticmethod
    def _gif_frame_delay(file_path: Path) -> float:
        try:
            from PIL import Image
            im = Image.open(file_path)
            duration_ms = im.info.get("duration", 80)
            return max(0.04, duration_ms / 1000.0)
        except Exception:
            return 0.08

    def update(self) -> None:
        if not self.is_active():
            return

        now = time.perf_counter()
        for slot in self._slots:
            if slot.kind == MediaKind.GIF and slot.gif_frames:
                if now - slot.gif_last_advance >= slot.gif_frame_delay:
                    slot.gif_index = (slot.gif_index + 1) % len(slot.gif_frames)
                    slot.current = slot.gif_frames[slot.gif_index]
                    slot.gif_last_advance = now
            elif slot.kind == MediaKind.VIDEO and slot.video_cap is not None:
                ok, frame = slot.video_cap.read()
                if not ok:
                    slot.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = slot.video_cap.read()
                if ok:
                    slot.current = frame

    def render(self, frame: np.ndarray) -> np.ndarray:
        if not self.is_active():
            return frame

        for slot in self._slots:
            if slot.current is not None:
                frame = self._blend_overlay(frame, slot.current, slot.side, slot.scale)
        return frame

    def trigger(self, path: str, effect_type: str, display_name: str) -> None:
        """Legacy single-overlay trigger."""
        if not path or effect_type == "none":
            return
        self.trigger_overlays([{"path": path, "side": "left"}], display_name)

    def is_active(self) -> bool:
        if self._hold_mode:
            return bool(self._slots)
        if time.perf_counter() > self._active_until:
            if self._slots:
                self._clear()
            return False
        return bool(self._slots)

    def get_active_name(self) -> str:
        if self.is_active():
            return self._active_name
        return ""

    def _side_position(
        self, side: str, fw: int, fh: int, tw: int, th: int
    ) -> Tuple[int, int]:
        y = max((fh - th) // 2, self._side_margin)
        if side == "right":
            x = max(fw - tw - self._side_margin, 0)
        else:
            x = self._side_margin
        return x, y

    def _blend_overlay(
        self, frame: np.ndarray, overlay: np.ndarray, side: str, scale: float
    ) -> np.ndarray:
        fh, fw = frame.shape[:2]
        target_w = max(int(fw * scale), 1)
        aspect = overlay.shape[0] / overlay.shape[1]
        target_h = max(int(target_w * aspect), 1)
        resized = cv2.resize(overlay, (target_w, target_h))

        x, y = self._side_position(side, fw, fh, target_w, target_h)

        if resized.shape[2] == 4:
            bgr = resized[:, :, :3]
            alpha = resized[:, :, 3] / 255.0
            roi = frame[y : y + target_h, x : x + target_w]
            for c in range(3):
                roi[:, :, c] = (
                    alpha * bgr[:, :, c] + (1 - alpha) * roi[:, :, c]
                ).astype(np.uint8)
            frame[y : y + target_h, x : x + target_w] = roi
        else:
            frame[y : y + target_h, x : x + target_w] = resized

        return frame

    def _clear(self) -> None:
        for slot in self._slots:
            if slot.video_cap is not None:
                slot.video_cap.release()
        self._slots = []
        self._active_name = ""

    def release(self) -> None:
        self._clear()
