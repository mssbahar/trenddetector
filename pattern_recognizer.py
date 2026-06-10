"""Match hand movement trajectories against registered patterns."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from models.hand_data import PatternMatch, TriggerEvent
from movement_tracker import MovementTracker

logger = logging.getLogger(__name__)


class PatternRecognizer:
    def __init__(self) -> None:
        self._patterns: List[Dict[str, Any]] = []
        self._last_progress: Optional[PatternMatch] = None

    def register_from_config(self, patterns: List[Dict[str, Any]]) -> None:
        self._patterns = list(patterns)
        logger.info("Registered %d movement patterns.", len(self._patterns))

    def update(self, movement_tracker: MovementTracker) -> Optional[TriggerEvent]:
        self._last_progress = None
        best_match: Optional[TriggerEvent] = None

        for hand_idx in range(movement_tracker.active_hand_count):
            trajectory = movement_tracker.get_trajectory(hand_idx)
            if len(trajectory) < config.PATTERN_MIN_FRAMES:
                continue

            for pattern_def in self._patterns:
                match = self._evaluate_pattern(
                    pattern_def, movement_tracker, hand_idx, trajectory
                )
                if match is None:
                    continue

                progress = PatternMatch(
                    pattern_id=pattern_def["id"],
                    display_name=pattern_def["display_name"],
                    progress=match["progress"],
                    matched_steps=match["matched_steps"],
                    total_steps=match["total_steps"],
                )
                if self._last_progress is None or progress.progress > self._last_progress.progress:
                    self._last_progress = progress

                if match["matched"]:
                    best_match = TriggerEvent(
                        trigger_id=pattern_def["id"],
                        trigger_type="movement",
                        display_name=pattern_def["display_name"],
                        sound_path=pattern_def.get("sound", ""),
                        effect_path=pattern_def.get("effect", ""),
                        effect_type=pattern_def.get("effect_type", "none"),
                    )
                    return best_match

        return best_match

    def get_progress(self) -> Optional[PatternMatch]:
        return self._last_progress

    def _evaluate_pattern(
        self,
        pattern_def: Dict[str, Any],
        movement_tracker: MovementTracker,
        hand_idx: int,
        trajectory: List[Tuple[float, float]],
    ) -> Optional[Dict[str, Any]]:
        pattern_type = pattern_def.get("type", "direction")
        if pattern_type == "direction":
            return self._match_direction(pattern_def, movement_tracker, hand_idx)
        if pattern_type == "coordinate":
            return self._match_coordinate(pattern_def, trajectory)
        return None

    def _match_direction(
        self,
        pattern_def: Dict[str, Any],
        movement_tracker: MovementTracker,
        hand_idx: int,
    ) -> Optional[Dict[str, Any]]:
        expected: List[str] = pattern_def["pattern"]
        observed = movement_tracker.get_recent_directions(hand_idx)
        total = len(expected)

        if not observed:
            return {
                "matched": False,
                "progress": 0.0,
                "matched_steps": 0,
                "total_steps": total,
            }

        matched_steps = self._longest_direction_subsequence(
            observed, expected, config.DIRECTION_TOLERANCE
        )
        progress = matched_steps / total if total else 0.0
        matched = matched_steps >= total

        if matched:
            tail = observed[-total:]
            matched = self._directions_match(tail, expected, config.DIRECTION_TOLERANCE)

        return {
            "matched": matched,
            "progress": progress,
            "matched_steps": matched_steps,
            "total_steps": total,
        }

    def _match_coordinate(
        self,
        pattern_def: Dict[str, Any],
        trajectory: List[Tuple[float, float]],
    ) -> Optional[Dict[str, Any]]:
        expected: List[Tuple[float, float]] = [
            (float(p[0]), float(p[1])) for p in pattern_def["pattern"]
        ]
        resampled = self._resample_trajectory(
            trajectory, config.COORDINATE_RESAMPLE_POINTS
        )
        expected_resampled = self._resample_trajectory(
            expected, config.COORDINATE_RESAMPLE_POINTS
        )

        if len(resampled) < 2:
            return {
                "matched": False,
                "progress": 0.0,
                "matched_steps": 0,
                "total_steps": len(expected_resampled),
            }

        distances = [
            np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
            for a, b in zip(resampled, expected_resampled)
        ]
        avg_dist = float(np.mean(distances))
        tolerance = config.COORDINATE_TOLERANCE
        progress = max(0.0, 1.0 - avg_dist / tolerance)
        matched = avg_dist <= tolerance

        matched_steps = sum(1 for d in distances if d <= tolerance)

        return {
            "matched": matched,
            "progress": min(1.0, progress),
            "matched_steps": matched_steps,
            "total_steps": len(expected_resampled),
        }

    @staticmethod
    def _resample_trajectory(
        points: List[Tuple[float, float]], n: int
    ) -> List[Tuple[float, float]]:
        if len(points) < 2:
            return points

        arr = np.array(points, dtype=np.float64)
        diffs = np.diff(arr, axis=0)
        seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_length = cumulative[-1]

        if total_length < 1e-8:
            return [tuple(arr[0])] * n

        targets = np.linspace(0, total_length, n)
        resampled = []
        for t in targets:
            idx = np.searchsorted(cumulative, t, side="right") - 1
            idx = min(max(idx, 0), len(arr) - 2)
            seg_len = seg_lengths[idx]
            if seg_len < 1e-8:
                resampled.append(tuple(arr[idx]))
            else:
                alpha = (t - cumulative[idx]) / seg_len
                point = arr[idx] + alpha * (arr[idx + 1] - arr[idx])
                resampled.append((float(point[0]), float(point[1])))
        return resampled

    @staticmethod
    def _directions_match(
        observed: List[str], expected: List[str], tolerance: int
    ) -> bool:
        if len(observed) < len(expected):
            return False
        return (
            PatternRecognizer._longest_direction_subsequence(
                observed, expected, tolerance
            )
            >= len(expected)
        )

    @staticmethod
    def _longest_direction_subsequence(
        observed: List[str], expected: List[str], tolerance: int
    ) -> int:
        """Count how many expected directions match in order within tolerance."""
        if not expected:
            return 0

        matched = 0
        obs_idx = 0
        errors = 0

        for exp_dir in expected:
            found = False
            while obs_idx < len(observed):
                obs_dir = observed[obs_idx]
                obs_idx += 1
                if obs_dir == "center" and exp_dir != "center":
                    continue
                if obs_dir == exp_dir:
                    matched += 1
                    found = True
                    errors = 0
                    break
                errors += 1
                if errors > tolerance:
                    break
            if not found:
                break

        return matched
