"""Reusable hand pose checks for TikTok combo detection."""

from __future__ import annotations

import numpy as np

import config
from models.hand_data import HandFrameData, Landmark

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


def _dist(a: Landmark, b: Landmark) -> float:
    return float(np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2))


def is_finger_extended(lm: list, mcp_idx: int, pip_idx: int, tip_idx: int) -> bool:
    wrist = lm[WRIST]
    tip_to_wrist = _dist(lm[tip_idx], wrist)
    pip_to_wrist = _dist(lm[pip_idx], wrist)
    tip_to_mcp = _dist(lm[tip_idx], lm[mcp_idx])
    pip_to_mcp = _dist(lm[pip_idx], lm[mcp_idx])
    return tip_to_wrist > pip_to_wrist and tip_to_mcp > pip_to_mcp * 0.85


def is_pinch(lm: list, threshold: float = 0.06) -> bool:
    return _dist(lm[THUMB_TIP], lm[INDEX_TIP]) < threshold


def is_tiktok1_pinch(hand: HandFrameData, threshold: float | None = None) -> bool:
    """
    Lenient nose pinch for TikTok 1.
    Thumb+index close while hand is in the upper / face area.
    """
    if threshold is None:
        threshold = config.TIKTOK1_PINCH_THRESHOLD
    lm = hand.landmarks
    if not is_pinch(lm, threshold):
        return False

    pinch_y = (lm[THUMB_TIP].y + lm[INDEX_TIP].y) / 2
    pinch_x = (lm[THUMB_TIP].x + lm[INDEX_TIP].x) / 2

    if pinch_y >= 0.58:
        return False
    if not (0.12 < pinch_x < 0.88):
        return False
    if hand.wrist[1] > 0.72:
        return False

    return True


def is_nose_pinch(hand: HandFrameData, threshold: float | None = None) -> bool:
    """Alias for TikTok 1 pinch detection."""
    return is_tiktok1_pinch(hand, threshold)


def is_wave_hand_pose(hand: HandFrameData) -> bool:
    """
    Open hand facing the camera (TikTok 1 wave hand).
    Uses depth/spread checks — not standard finger-extension, which fails
    when fingertips point toward the camera.
    """
    lm = hand.landmarks
    if is_pinch(lm, 0.08):
        return False
    if is_nose_pinch(hand):
        return False

    # Fingertips closer to camera than their knuckles = fingers pointing at you
    pointing_at_camera = 0
    for tip, mcp in ((INDEX_TIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_MCP),
                     (RING_TIP, RING_MCP), (PINKY_TIP, PINKY_MCP)):
        if lm[tip].z < lm[mcp].z + 0.03:
            pointing_at_camera += 1
    if pointing_at_camera < 3:
        return False

    tips = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
    xs = [lm[i].x for i in tips]
    ys = [lm[i].y for i in tips]
    spread = max(max(xs) - min(xs), max(ys) - min(ys))
    return spread >= 0.04


def is_palm_facing_camera(lm: list) -> bool:
    """Strict palm-to-camera check (legacy). Prefer is_wave_hand_pose for TikTok 1."""
    return is_wave_hand_pose(
        HandFrameData(
            handedness="Unknown",
            landmarks=lm,
            wrist=(lm[WRIST].x, lm[WRIST].y, lm[WRIST].z),
            fingertips={},
            timestamp=0.0,
        )
    )


def is_palm_out(lm: list) -> bool:
    """Open palm facing outward/toward camera — used for TikTok 2."""
    extended = sum(
        [
            is_finger_extended(lm, INDEX_MCP, INDEX_PIP, INDEX_TIP),
            is_finger_extended(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
            is_finger_extended(lm, RING_MCP, RING_PIP, RING_TIP),
            is_finger_extended(lm, PINKY_MCP, PINKY_PIP, PINKY_TIP),
        ]
    )
    if extended < 3:
        return False
    if is_pinch(lm, 0.05):
        return False
    return True


def get_hand_by_label(hands: list[HandFrameData], label: str) -> HandFrameData | None:
    for hand in hands:
        if hand.handedness == label:
            return hand
    return None


def vertical_relation(hands: list[HandFrameData]) -> str | None:
    """Return 'right_up' if right hand higher than left, 'left_up' if opposite."""
    right = get_hand_by_label(hands, "Right")
    left = get_hand_by_label(hands, "Left")
    if not right or not left:
        return None
    diff = right.wrist[1] - left.wrist[1]
    sep = config.VERTICAL_HAND_SEPARATION
    if diff < -sep:
        return "right_up"
    if diff > sep:
        return "left_up"
    return None
