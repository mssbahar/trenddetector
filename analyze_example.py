"""Analyze hand movements in example TikTok videos."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
EXAMPLE_DIR = PROJECT_ROOT / "assets" / "example"

INDEX_TIP = 8
MIDDLE_MCP = 9
DIRECTION_THRESHOLD = 0.06
MIN_SEGMENT = 2


def create_landmarker() -> vision.HandLandmarker:
    base = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def normalize_point(landmarks, track_index: int = INDEX_TIP) -> tuple[float, float]:
    wx = landmarks[0].x
    wy = landmarks[0].y
    mx = landmarks[MIDDLE_MCP].x
    my = landmarks[MIDDLE_MCP].y
    scale = max(np.sqrt((mx - wx) ** 2 + (my - wy) ** 2), 1e-6)
    px = landmarks[track_index].x
    py = landmarks[track_index].y
    return (px - wx) / scale, (py - wy) / scale


def direction_from_delta(dx: float, dy: float) -> str:
    if abs(dx) < 0.02 and abs(dy) < 0.02:
        return "center"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def collapse_directions(raw: list[str], min_segment: int = MIN_SEGMENT) -> list[str]:
    if not raw:
        return []
    collapsed: list[str] = []
    current = raw[0]
    count = 1
    for d in raw[1:]:
        if d == current:
            count += 1
        else:
            if count >= min_segment and (not collapsed or collapsed[-1] != current):
                collapsed.append(current)
            current = d
            count = 1
    if count >= min_segment and (not collapsed or collapsed[-1] != current):
        collapsed.append(current)
    return collapsed


def resample(points: list[tuple[float, float]], n: int = 8) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
    arr = np.array(points, dtype=np.float64)
    diffs = np.diff(arr, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    if total < 1e-8:
        return [tuple(arr[0])] * n
    targets = np.linspace(0, total, n)
    out = []
    for t in targets:
        idx = min(max(int(np.searchsorted(cumulative, t, side="right") - 1), 0), len(arr) - 2)
        seg_len = seg_lengths[idx]
        alpha = 0.0 if seg_len < 1e-8 else (t - cumulative[idx]) / seg_len
        p = arr[idx] + alpha * (arr[idx + 1] - arr[idx])
        out.append((round(float(p[0]), 2), round(float(p[1]), 2)))
    return out


def analyze_video(video_path: Path, landmarker: vision.HandLandmarker) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0

    trajectories: dict[int, list[tuple[float, float]]] = {}
    raw_dirs: dict[int, list[str]] = {}
    hand_frames = 0
    timestamp_ms = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += int(1000 / fps)

        if result.hand_landmarks:
            hand_frames += 1
            for i, hand_lm in enumerate(result.hand_landmarks):
                norm = normalize_point(hand_lm)
                trajectories.setdefault(i, []).append(norm)
                dirs = raw_dirs.setdefault(i, [])
                if len(trajectories[i]) >= 2:
                    prev = trajectories[i][-2]
                    dx = norm[0] - prev[0]
                    dy = norm[1] - prev[1]
                    if abs(dx) > DIRECTION_THRESHOLD or abs(dy) > DIRECTION_THRESHOLD:
                        dirs.append(direction_from_delta(dx, dy))

    cap.release()

    hands_out = []
    for hand_idx, traj in trajectories.items():
        collapsed = collapse_directions(raw_dirs.get(hand_idx, []))
        # Split into movement bursts separated by pauses
        bursts = []
        current_burst: list[str] = []
        for d in collapsed:
            if d == "center":
                if current_burst:
                    bursts.append(current_burst)
                    current_burst = []
            else:
                current_burst.append(d)
        if current_burst:
            bursts.append(current_burst)

        main_burst = max(bursts, key=len) if bursts else collapsed
        if main_burst and main_burst[-1] != "center":
            suggested_pattern = main_burst + ["center"]
        else:
            suggested_pattern = main_burst or collapsed

        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        hands_out.append({
            "hand_index": hand_idx,
            "frames_tracked": len(traj),
            "direction_sequence": collapsed,
            "movement_bursts": bursts,
            "suggested_direction_pattern": suggested_pattern,
            "suggested_coordinate_pattern": resample(traj, 8),
            "range_x": [round(min(xs), 2), round(max(xs), 2)] if xs else [0, 0],
            "range_y": [round(min(ys), 2), round(max(ys), 2)] if ys else [0, 0],
        })

    return {
        "file": video_path.name,
        "duration_sec": round(duration, 2),
        "fps": round(fps, 1),
        "total_frames": total_frames,
        "frames_with_hands": hand_frames,
        "hands": hands_out,
    }


def main() -> None:
    videos = sorted(EXAMPLE_DIR.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files in {EXAMPLE_DIR}")
        sys.exit(1)

    landmarker = create_landmarker()
    results = []
    try:
        for video in videos:
            print(f"Analyzing {video.name}...", flush=True)
            results.append(analyze_video(video, landmarker))
            # Reset landmarker between videos (timestamps must stay monotonic)
            landmarker.close()
            landmarker = create_landmarker()
    finally:
        landmarker.close()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
