"""Track normalized hand movement over time for pattern analysis."""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

import config
from hand_tracker import HandTracker
from models.hand_data import HandFrameData, MovementPoint

logger = logging.getLogger(__name__)


class MovementTracker:
    def __init__(
        self,
        buffer_size: int = config.MOVEMENT_BUFFER_SIZE,
        track_point: str = config.MOVEMENT_TRACK_POINT,
        direction_threshold: float = config.DIRECTION_THRESHOLD,
        min_segment: int = config.DIRECTION_MIN_SEGMENT,
    ) -> None:
        self.buffer_size = buffer_size
        self.track_point = track_point
        self.direction_threshold = direction_threshold
        self.min_segment = min_segment
        self._buffers: Dict[int, Deque[MovementPoint]] = {}
        self._direction_history: Dict[int, Deque[str]] = {}

    def update(self, hands: List[HandFrameData]) -> None:
        active_indices = set()

        for hand_idx, hand in enumerate(hands):
            active_indices.add(hand_idx)
            norm_x, norm_y = self._normalize_point(hand)
            buf = self._buffers.setdefault(hand_idx, deque(maxlen=self.buffer_size))
            buf.append(
                MovementPoint(
                    x=norm_x,
                    y=norm_y,
                    timestamp=hand.timestamp,
                )
            )
            self._update_directions(hand_idx)

        stale = [k for k in self._buffers if k not in active_indices]
        for k in stale:
            del self._buffers[k]
            self._direction_history.pop(k, None)

    def _normalize_point(self, hand: HandFrameData) -> Tuple[float, float]:
        px, py, pz = HandTracker.get_track_point(hand, self.track_point)
        wx, wy, _ = hand.wrist
        scale = HandTracker.get_hand_scale(hand)
        return (px - wx) / scale, (py - wy) / scale

    def _update_directions(self, hand_idx: int) -> None:
        buf = self._buffers.get(hand_idx)
        if not buf or len(buf) < 2:
            return

        hist = self._direction_history.setdefault(
            hand_idx, deque(maxlen=self.buffer_size)
        )
        prev = buf[-2]
        curr = buf[-1]
        dx = curr.x - prev.x
        dy = curr.y - prev.y

        if abs(dx) < config.NORMALIZED_DEAD_ZONE and abs(dy) < config.NORMALIZED_DEAD_ZONE:
            direction = "center"
        elif abs(dx) >= abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"

        curr.direction = direction
        hist.append(direction)

    def get_trajectory(self, hand_idx: int = 0) -> List[Tuple[float, float]]:
        buf = self._buffers.get(hand_idx)
        if not buf:
            return []
        return [(p.x, p.y) for p in buf]

    def get_recent_directions(self, hand_idx: int = 0) -> List[str]:
        """Collapse raw direction history into stable segments."""
        hist = self._direction_history.get(hand_idx)
        if not hist:
            return []

        collapsed: List[str] = []
        current_dir = None
        count = 0

        for d in hist:
            if d == current_dir:
                count += 1
            else:
                if current_dir is not None and count >= self.min_segment:
                    if collapsed and collapsed[-1] == current_dir:
                        pass
                    else:
                        collapsed.append(current_dir)
                elif current_dir is not None and d == "center" and count < self.min_segment:
                    pass
                current_dir = d
                count = 1

        if current_dir is not None and count >= self.min_segment:
            if not collapsed or collapsed[-1] != current_dir:
                collapsed.append(current_dir)

        return collapsed

    def get_progress_ratio(self, hand_idx: int = 0) -> float:
        buf = self._buffers.get(hand_idx)
        if not buf:
            return 0.0
        return len(buf) / self.buffer_size

    def get_latest_point(self, hand_idx: int = 0) -> Optional[MovementPoint]:
        buf = self._buffers.get(hand_idx)
        if not buf:
            return None
        return buf[-1]

    def reset(self, hand_idx: Optional[int] = None) -> None:
        if hand_idx is not None:
            self._buffers.pop(hand_idx, None)
            self._direction_history.pop(hand_idx, None)
        else:
            self._buffers.clear()
            self._direction_history.clear()

    @property
    def active_hand_count(self) -> int:
        return len(self._buffers)
