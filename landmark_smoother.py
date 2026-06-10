"""Exponential moving average smoothing for hand landmarks."""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np

from models.hand_data import Landmark


class LandmarkSmoother:
    """Per-hand EMA filter to reduce landmark jitter frame-to-frame."""

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self._state: Dict[str, np.ndarray] = {}

    def smooth(self, handedness: str, landmarks: List[Landmark]) -> List[Landmark]:
        raw = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)

        if handedness not in self._state:
            self._state[handedness] = raw.copy()
        else:
            prev = self._state[handedness]
            self._state[handedness] = self.alpha * raw + (1.0 - self.alpha) * prev

        return self._to_landmarks(self._state[handedness])

    def remove_stale(self, active_labels: Set[str]) -> None:
        for label in list(self._state):
            if label not in active_labels:
                del self._state[label]

    def reset(self) -> None:
        self._state.clear()

    @staticmethod
    def _to_landmarks(arr: np.ndarray) -> List[Landmark]:
        return [
            Landmark(x=float(p[0]), y=float(p[1]), z=float(p[2]), index=i)
            for i, p in enumerate(arr)
        ]
