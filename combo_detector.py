"""Simplified TikTok combo detection — hold-to-play model."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import config
from hand_pose import is_pinch, is_tiktok1_pinch
from models.hand_data import HandFrameData
from movement_tracker import MovementTracker

logger = logging.getLogger(__name__)


@dataclass
class ComboStatus:
    combo_id: str
    display_name: str
    progress: float
    steps: str


@dataclass
class _HandSweepTrack:
    wrist_x: Deque[float] = field(default_factory=lambda: deque(maxlen=45))
    dirs: List[str] = field(default_factory=list)


@dataclass
class _TikTok2State:
    hands: Dict[str, _HandSweepTrack] = field(default_factory=dict)


class ComboDetector:
    def __init__(self) -> None:
        self._tiktok2 = _TikTok2State()
        self._status: Optional[ComboStatus] = None
        self._active_combo: Optional[str] = None

    def update(
        self, hands: List[HandFrameData], movement_tracker: MovementTracker
    ) -> Optional[str]:
        """
        Return active combo id while gesture is held, None when idle.
        tiktok1: nose pinch detected now
        tiktok2: hand actively sweeping sideways (no nose pinch)
        """
        self._active_combo = None
        self._status = None

        if not hands:
            self._tiktok2 = _TikTok2State()
            return None

        if any(is_tiktok1_pinch(h) for h in hands):
            binding = config.TIKTOK_COMBOS["tiktok1"]
            self._active_combo = "tiktok1"
            self._status = ComboStatus(
                "tiktok1",
                binding["display_name"],
                1.0,
                "ACTIVE — pinching nose",
            )
            self._tiktok2 = _TikTok2State()
            return "tiktok1"

        sweeping, t2_status = self._eval_tiktok2_active(hands)
        if sweeping:
            self._active_combo = "tiktok2"
            self._status = t2_status
            return "tiktok2"

        t1_hint = self._tiktok1_hint(hands)
        hand_near_face = any(
            h.wrist[1] < 0.62
            or is_pinch(h.landmarks, config.TIKTOK1_PINCH_THRESHOLD)
            for h in hands
        )
        if hand_near_face:
            binding = config.TIKTOK_COMBOS["tiktok1"]
            self._status = ComboStatus("tiktok1", binding["display_name"], 0.0, t1_hint)
        elif t2_status:
            self._status = t2_status
        else:
            self._status = ComboStatus("none", "idle", 0.0, "no gesture")

        return None

    def get_status(self) -> Optional[ComboStatus]:
        return self._status

    def get_active_combo(self) -> Optional[str]:
        return self._active_combo

    def _eval_tiktok2_active(
        self, hands: List[HandFrameData]
    ) -> tuple[bool, Optional[ComboStatus]]:
        binding = config.TIKTOK_COMBOS["tiktok2"]
        state = self._tiktok2
        active_labels = {h.handedness for h in hands}

        for label in list(state.hands):
            if label not in active_labels:
                del state.hands[label]

        best_steps = "sweep one hand sideways..."
        best_progress = 0.0
        any_sweeping = False

        for hand in hands:
            if is_tiktok1_pinch(hand):
                continue

            label = hand.handedness
            track = state.hands.setdefault(label, _HandSweepTrack())
            self._update_sweep_track(hand, track)

            sweeping_now = self._is_hand_sweeping_now(track)
            span_x = abs(track.wrist_x[-1] - track.wrist_x[0]) if len(track.wrist_x) > 1 else 0.0
            dirs_text = "->".join(track.dirs[-3:]) if track.dirs else "move sideways"

            if sweeping_now:
                any_sweeping = True
                progress = 1.0
                steps = f"ACTIVE — sweeping ({label})"
            else:
                progress = min(span_x / config.TIKTOK2_SWEEP_MIN_SPAN, 0.9)
                steps = f"sweep {label}: {dirs_text}"

            if progress > best_progress:
                best_progress = progress
                best_steps = steps

        status = ComboStatus("tiktok2", binding["display_name"], best_progress, best_steps)
        return any_sweeping, status

    @staticmethod
    def _is_hand_sweeping_now(track: _HandSweepTrack) -> bool:
        if len(track.wrist_x) < config.TIKTOK2_SWEEP_MIN_SEGMENT + 1:
            return False
        recent = list(track.wrist_x)[-(config.TIKTOK2_SWEEP_MIN_SEGMENT + 1) :]
        dx = recent[-1] - recent[0]
        if abs(dx) < config.TIKTOK2_SWEEP_MIN_DELTA * config.TIKTOK2_SWEEP_MIN_SEGMENT:
            return False
        span_x = abs(track.wrist_x[-1] - track.wrist_x[0])
        return span_x >= config.TIKTOK2_SWEEP_MIN_SPAN * 0.4

    def _update_sweep_track(self, hand: HandFrameData, track: _HandSweepTrack) -> None:
        wx = hand.wrist[0]
        threshold = config.TIKTOK2_SWEEP_MIN_DELTA
        min_seg = config.TIKTOK2_SWEEP_MIN_SEGMENT

        if len(track.wrist_x) >= min_seg:
            dx = wx - track.wrist_x[-min_seg]
            if abs(dx) >= threshold * min_seg:
                direction = "right" if dx > 0 else "left"
                if not track.dirs or track.dirs[-1] != direction:
                    track.dirs.append(direction)

        track.wrist_x.append(wx)

    def _tiktok1_hint(self, hands: List[HandFrameData]) -> str:
        for hand in hands:
            lm = hand.landmarks
            pinch_y = (lm[4].y + lm[8].y) / 2
            if is_pinch(lm, config.TIKTOK1_PINCH_THRESHOLD) and pinch_y >= 0.58:
                return "move hand up to face"
            if pinch_y < 0.58 and not is_pinch(lm, config.TIKTOK1_PINCH_THRESHOLD):
                return "pinch thumb+index on nose"
        return "pinch your nose"

    def reset(self) -> None:
        self._tiktok2 = _TikTok2State()
        self._status = None
        self._active_combo = None
