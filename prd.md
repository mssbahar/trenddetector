# Product Requirements Document: Hand Gesture & Movement Detection System

## 1. Problem Statement

Content creators and interactive installations need a real-time system that detects hand gestures and custom movement sequences from a webcam, then triggers synchronized sound and visual effects. This prototype serves as the foundation for future "trend dance" detection and TikTok-style auto trend classification.

## 2. Target Users

- Content creators prototyping interactive video effects
- Developers building gesture-controlled installations
- Engineers preparing a pipeline for ML-based dance trend classification

## 3. Goals

| Goal | Metric |
|------|--------|
| Real-time hand tracking | ≥15 FPS on standard webcam |
| Static gesture recognition | 4 gestures with <500 ms latency |
| Custom movement sequences | Direction and coordinate patterns with tolerance |
| Effect triggering | Sound + overlay with 2–3 s cooldown |
| Extensibility | New patterns added via config only |

## 4. Functional Requirements

### 4.1 Webcam System
- OpenCV webcam capture with real-time feed
- FPS display (rolling average)
- MediaPipe hand landmark visualization (21 points per hand)

### 4.2 Hand Tracking
- MediaPipe Hands, 1–2 hands
- Extract landmark x, y, z; wrist reference; fingertip positions
- Structured per-frame hand data for movement analysis

### 4.3 Static Gesture Detection
- Open Palm, Fist, Peace Sign, Thumbs Up
- Rule-based landmark detection
- Multi-frame confirmation to reduce false positives

### 4.4 Custom Movement Tracking
- Rolling buffer (30–100 frames) of normalized wrist or index-tip trajectory
- Scale/position normalization (wrist reference, hand-scale division)
- Direction patterns: `["left", "right", "left", "center"]`
- Coordinate patterns: `[(x1,y1), (x2,y2), ...]`
- Tolerance threshold, speed variation handling, post-trigger cooldown

### 4.5 Event System
- Trigger on gesture OR movement match
- Play sound, show video/image overlay, display trigger name
- Minimum 2–3 second cooldown per trigger

### 4.6 Sound Manager
- Pygame-based audio playback
- Gesture/movement → sound file mapping
- Graceful handling of missing files

### 4.7 Effect Manager
- Video overlay (2–3 s) or PNG image overlay on camera feed
- Alpha blending support

### 4.8 UI / HUD
- FPS, current gesture, movement pattern progress, active effect name
- Debug mode toggle (ON/OFF)

## 5. Non-Functional Requirements

- **Modularity**: Clean separation between detection, tracking, recognition, and effects
- **Performance**: Target <100 ms per frame processing
- **Reliability**: App runs without asset files (warnings only)
- **Python**: 3.11+
- **Platform**: Windows primary; cross-platform where possible

## 6. Success Criteria

1. Webcam opens and displays landmarks for 1–2 hands
2. All 4 static gestures detected reliably in normal lighting
3. At least one direction pattern (e.g. left→right→left→center) triggers correctly
4. Cooldown prevents repeated triggers within 2–3 seconds
5. Missing sound files log warnings without crashing
6. Debug mode shows normalized coordinates and direction history

## 7. Out of Scope (v1)

- Full-body pose tracking
- ML-based trend classification
- LSTM/Transformer sequence learning
- TikTok auto trend detection
- Multi-camera support
- Recording/export of sessions

## 8. Asset Contract

### Sounds (`assets/sounds/`)
| File | Trigger |
|------|---------|
| `palm.wav` | Open Palm gesture |
| `fist.wav` | Fist gesture |
| `peace.wav` | Peace Sign gesture |
| `thumbs_up.wav` | Thumbs Up gesture |
| `trend1.wav` | Trend Wave pattern |
| `trend2.wav` | Circle Swipe pattern |

### Images (`assets/images/`)
| File | Trigger |
|------|---------|
| `sparkle.png` | Trend Wave effect |
| `star.png` | Open Palm effect |

### Videos (`assets/videos/`)
| File | Trigger |
|------|---------|
| `burst.mp4` | Circle Swipe effect |

App runs without these files; warnings are logged and effects are skipped.

## 9. Future Extension Points

| Feature | Extension Point |
|---------|-----------------|
| Full-body pose | Replace/augment `hand_tracker.py` with MediaPipe Pose |
| ML classification | Feed normalized trajectory to model; emit via `event_system` |
| Sequence learning | Persist trajectory buffer for LSTM/Transformer training |
| Auto trend detection | Batch offline module on saved trajectories |

## 10. Controls

| Key | Action |
|-----|--------|
| `d` | Toggle debug mode |
| `r` | Reset movement buffer |
| `q` / `Esc` | Quit |

## 11. Architecture

```
main.py → camera → hand_tracker → gesture_detector ─┐
                                 → movement_tracker → pattern_recognizer ─┤
                                                                          → event_system → sound_manager
                                                                                        → effect_manager
config.py (central bindings)
models/hand_data.py (dataclasses)
```
