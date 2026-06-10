"""MediaPipe hand tracking and landmark visualization."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config
from landmark_smoother import LandmarkSmoother
from models.hand_data import HandFrameData, Landmark

logger = logging.getLogger(__name__)

FINGERTIP_INDICES = {4, 8, 12, 16, 20}
INDEX_TIP = 8
MIDDLE_MCP = 9


class _DrawLandmark:
    """Minimal landmark for drawing_utils (needs x, y, z, visibility, presence)."""

    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.visibility = None
        self.presence = None


class HandTracker:
    def __init__(self) -> None:
        model_path = str(config.HAND_LANDMARKER_MODEL)
        if not config.HAND_LANDMARKER_MODEL.exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}. "
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0
        self._last_draw_landmarks: List[List[_DrawLandmark]] = []
        self._smoother = LandmarkSmoother(alpha=config.LANDMARK_SMOOTHING_ALPHA)

    def process(self, frame: np.ndarray) -> List[HandFrameData]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
        timestamp = time.perf_counter()
        hands: List[HandFrameData] = []
        self._last_draw_landmarks = []

        if not result.hand_landmarks:
            self._smoother.remove_stale(set())
            return hands

        active_labels: set[str] = set()

        for hand_landmarks, handedness_list in zip(
            result.hand_landmarks,
            result.handedness or [[]] * len(result.hand_landmarks),
        ):
            label = handedness_list[0].category_name if handedness_list else "Unknown"
            active_labels.add(label)

            landmarks = [
                Landmark(x=lm.x, y=lm.y, z=lm.z, index=i)
                for i, lm in enumerate(hand_landmarks)
            ]
            if config.LANDMARK_SMOOTHING_ENABLED:
                landmarks = self._smoother.smooth(label, landmarks)

            self._last_draw_landmarks.append(
                [_DrawLandmark(lm.x, lm.y, lm.z) for lm in landmarks]
            )

            wrist = (landmarks[0].x, landmarks[0].y, landmarks[0].z)
            fingertips = {
                i: (landmarks[i].x, landmarks[i].y, landmarks[i].z)
                for i in FINGERTIP_INDICES
            }
            hands.append(
                HandFrameData(
                    handedness=label,
                    landmarks=landmarks,
                    wrist=wrist,
                    fingertips=fingertips,
                    timestamp=timestamp,
                )
            )

        self._smoother.remove_stale(active_labels)
        return hands

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        for hand_landmarks in self._last_draw_landmarks:
            vision.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
            )
        return frame

    def reset_smoothing(self) -> None:
        self._smoother.reset()

    @staticmethod
    def get_hand_scale(hand: HandFrameData) -> float:
        wx, wy, _ = hand.wrist
        mx = hand.landmarks[MIDDLE_MCP].x
        my = hand.landmarks[MIDDLE_MCP].y
        scale = np.sqrt((mx - wx) ** 2 + (my - wy) ** 2)
        return max(scale, 1e-6)

    @staticmethod
    def get_track_point(hand: HandFrameData, point_name: str) -> tuple[float, float, float]:
        if point_name == "wrist":
            return hand.wrist
        if point_name == "index_tip":
            return hand.fingertips[INDEX_TIP]
        return hand.wrist

    def close(self) -> None:
        self._landmarker.close()
