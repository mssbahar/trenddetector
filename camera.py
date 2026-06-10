"""Webcam capture with FPS tracking."""

from __future__ import annotations

import logging
import time
from collections import deque

import cv2

import config

logger = logging.getLogger(__name__)


class CameraCapture:
    def __init__(
        self,
        index: int = config.WEBCAM_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None
        self._frame_times: deque[float] = deque(maxlen=config.FPS_SMOOTHING)
        self._fps = 0.0

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {self.index}. "
                "Check that a camera is connected and not in use."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> tuple[bool, object, float]:
        if self._cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")

        ok, frame = self._cap.read()
        now = time.perf_counter()
        self._frame_times.append(now)

        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self._fps = (len(self._frame_times) - 1) / elapsed

        return ok, frame, self._fps

    @property
    def fps(self) -> float:
        return self._fps

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("Camera released.")
