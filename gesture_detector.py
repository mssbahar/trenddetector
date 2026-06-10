"""Rule-based static gesture detection from hand landmarks."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

import config
from models.hand_data import GestureResult, HandFrameData, Landmark

logger = logging.getLogger(__name__)

# Landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


class GestureDetector:
    def __init__(self, confirm_frames: int = config.GESTURE_CONFIRM_FRAMES) -> None:
        self._confirm_frames = confirm_frames
        self._history: Dict[int, List[str]] = {}

    def detect(self, hands: List[HandFrameData]) -> Optional[GestureResult]:
        if not hands:
            self._history.clear()
            return None

        best: Optional[GestureResult] = None
        for hand_idx, hand in enumerate(hands):
            gesture_name, confidence = self._classify_hand(hand)
            if gesture_name is None:
                self._history.pop(hand_idx, None)
                continue

            history = self._history.setdefault(hand_idx, [])
            history.append(gesture_name)
            if len(history) > self._confirm_frames:
                history.pop(0)

            if len(history) == self._confirm_frames and len(set(history)) == 1:
                result = GestureResult(
                    name=gesture_name,
                    confidence=confidence,
                    hand_index=hand_idx,
                )
                if best is None or confidence > best.confidence:
                    best = result

        return best

    def get_current_gesture(self, hands: List[HandFrameData]) -> str:
        if not hands:
            return "none"
        name, _ = self._classify_hand(hands[0])
        return name or "none"

    def _classify_hand(self, hand: HandFrameData) -> tuple[Optional[str], float]:
        lm = hand.landmarks

        thumb_ext = self._is_thumb_extended(lm)
        index_ext = self._is_finger_extended(lm, INDEX_MCP, INDEX_PIP, INDEX_TIP)
        middle_ext = self._is_finger_extended(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP)
        ring_ext = self._is_finger_extended(lm, RING_MCP, RING_PIP, RING_TIP)
        pinky_ext = self._is_finger_extended(lm, PINKY_MCP, PINKY_PIP, PINKY_TIP)

        scores: Dict[str, float] = {}

        # Open palm: all fingers extended
        palm_score = sum([thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]) / 5
        scores["open_palm"] = palm_score if palm_score >= 0.8 else 0.0

        # Fist: no fingers extended
        fist_score = 1.0 - sum([index_ext, middle_ext, ring_ext, pinky_ext]) / 4
        scores["fist"] = fist_score if fist_score >= 0.75 else 0.0

        # Peace sign: index + middle extended, ring + pinky curled
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            scores["peace_sign"] = 0.9
        else:
            scores["peace_sign"] = 0.0

        # Thumbs up: thumb extended up, others curled
        if thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            if self._is_thumb_pointing_up(lm):
                scores["thumbs_up"] = 0.9
            else:
                scores["thumbs_up"] = 0.0
        else:
            scores["thumbs_up"] = 0.0

        best_name = max(scores, key=scores.get)
        best_score = scores[best_name]
        if best_score < 0.7:
            return None, 0.0
        return best_name, best_score

    @staticmethod
    def _dist(a: Landmark, b: Landmark) -> float:
        return float(np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2))

    def _is_finger_extended(
        self, lm: list, mcp_idx: int, pip_idx: int, tip_idx: int
    ) -> bool:
        wrist = lm[WRIST]
        mcp = lm[mcp_idx]
        pip = lm[pip_idx]
        tip = lm[tip_idx]
        tip_to_wrist = self._dist(tip, wrist)
        pip_to_wrist = self._dist(pip, wrist)
        tip_to_mcp = self._dist(tip, mcp)
        pip_to_mcp = self._dist(pip, mcp)
        return tip_to_wrist > pip_to_wrist and tip_to_mcp > pip_to_mcp * 0.85

    def _is_thumb_extended(self, lm: list) -> bool:
        thumb_tip = lm[THUMB_TIP]
        thumb_ip = lm[THUMB_IP]
        thumb_mcp = lm[THUMB_MCP]
        index_mcp = lm[INDEX_MCP]
        tip_to_ip = self._dist(thumb_tip, thumb_ip)
        ip_to_mcp = self._dist(thumb_ip, thumb_mcp)
        tip_to_index = self._dist(thumb_tip, index_mcp)
        return tip_to_ip > ip_to_mcp * 0.6 and tip_to_index > 0.06

    @staticmethod
    def _is_thumb_pointing_up(lm: list) -> bool:
        return lm[THUMB_TIP].y < lm[THUMB_MCP].y

    def reset(self) -> None:
        self._history.clear()
