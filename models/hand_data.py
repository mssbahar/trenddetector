"""Structured data types for hand tracking and movement analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    index: int


@dataclass
class HandFrameData:
    handedness: str
    landmarks: List[Landmark]
    wrist: Tuple[float, float, float]
    fingertips: Dict[int, Tuple[float, float, float]]
    timestamp: float


@dataclass
class MovementPoint:
    x: float
    y: float
    timestamp: float
    direction: str = ""


@dataclass
class GestureResult:
    name: str
    confidence: float
    hand_index: int


@dataclass
class PatternMatch:
    pattern_id: str
    display_name: str
    progress: float
    matched_steps: int
    total_steps: int


@dataclass
class TriggerEvent:
    trigger_id: str
    trigger_type: str
    display_name: str
    sound_path: str = ""
    effect_path: str = ""
    effect_type: str = "none"
    overlays: List[dict] | None = None
