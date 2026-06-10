"""Hand gesture and movement detection application entry point."""

from __future__ import annotations

import logging
import sys

import cv2

import config
from camera import CameraCapture
from combo_detector import ComboDetector
from effect_manager import EffectManager
from event_system import EventSystem
from gesture_detector import GestureDetector
from hand_tracker import HandTracker
from movement_tracker import MovementTracker
from pattern_recognizer import PatternRecognizer
from sound_manager import SoundManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

WINDOW_NAME = "Hand Gesture & Movement Detection"


def draw_hud(
    frame,
    fps: float,
    gesture: str,
    pattern_progress: str,
    combo_progress: str,
    active_effect: str,
    debug_mode: bool,
    movement_tracker: MovementTracker,
    hands_count: int,
) -> None:
    lines = [
        f"FPS: {fps:.1f}",
        f"Gesture: {gesture}",
        f"Pattern: {pattern_progress}",
        f"Combo: {combo_progress}",
        f"Effect: {active_effect or 'none'}",
        f"Debug: {'ON' if debug_mode else 'OFF'}  |  [d] toggle  [r] reset  [q] quit",
    ]

    y = 30
    for line in lines:
        cv2.rectangle(frame, (8, y - 22), (8 + len(line) * 12, y + 6), config.HUD_BG_COLOR, -1)
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.HUD_FONT_SCALE,
            config.HUD_COLOR,
            1,
            cv2.LINE_AA,
        )
        y += 30

    if debug_mode:
        debug_lines = [f"Hands: {hands_count}"]
        for hand_idx in range(min(hands_count, 2)):
            directions = movement_tracker.get_recent_directions(hand_idx)
            traj = movement_tracker.get_trajectory(hand_idx)
            debug_lines.append(
                f"H{hand_idx} dirs: {' -> '.join(directions[-6:]) if directions else 'none'}"
            )
            if traj:
                debug_lines.append(f"H{hand_idx} pos: ({traj[-1][0]:.2f}, {traj[-1][1]:.2f})")

        for line in debug_lines:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                config.DEBUG_COLOR,
                1,
                cv2.LINE_AA,
            )
            y += 22


def main() -> int:
    camera = CameraCapture()
    hand_tracker = HandTracker()
    gesture_detector = GestureDetector()
    movement_tracker = MovementTracker()
    pattern_recognizer = PatternRecognizer()
    combo_detector = ComboDetector()
    sound_manager = SoundManager()
    effect_manager = EffectManager()
    event_system = EventSystem(sound_manager, effect_manager)

    pattern_recognizer.register_from_config(config.CUSTOM_PATTERNS)
    sound_manager.init()

    debug_mode = False

    try:
        camera.open()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    logger.info("Application started. Press 'q' to quit.")

    try:
        while True:
            ok, frame, fps = camera.read()
            if not ok:
                logger.warning("Failed to read frame from camera.")
                break

            hands = hand_tracker.process(frame)
            movement_tracker.update(hands)

            gesture_result = gesture_detector.detect(hands)
            current_gesture = gesture_detector.get_current_gesture(hands)

            combo_detector.update(hands, movement_tracker)
            event_system.sync_active_combo(combo_detector.get_active_combo())

            if gesture_result:
                event_system.emit_gesture(gesture_result)

            movement_event = pattern_recognizer.update(movement_tracker)
            if movement_event and event_system.emit_movement(movement_event):
                movement_tracker.reset()

            effect_manager.update()
            frame = effect_manager.render(frame)
            hand_tracker.draw_landmarks(frame)

            progress = pattern_recognizer.get_progress()
            if progress:
                pattern_text = (
                    f"{progress.display_name}: "
                    f"{progress.matched_steps}/{progress.total_steps} "
                    f"({progress.progress * 100:.0f}%)"
                )
            else:
                pattern_text = "none"

            combo = combo_detector.get_status()
            if combo:
                combo_text = f"{combo.display_name}: {combo.steps} ({combo.progress * 100:.0f}%)"
            else:
                combo_text = "none"

            draw_hud(
                frame,
                fps,
                current_gesture,
                pattern_text,
                combo_text,
                effect_manager.get_active_name() or event_system.get_active_display() or "none",
                debug_mode,
                movement_tracker,
                len(hands),
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug_mode = not debug_mode
            if key == ord("r"):
                movement_tracker.reset()
                gesture_detector.reset()
                combo_detector.reset()
                event_system.sync_active_combo(None)
                hand_tracker.reset_smoothing()
                logger.info("Movement buffer reset.")

    finally:
        camera.release()
        hand_tracker.close()
        effect_manager.release()
        sound_manager.quit()
        cv2.destroyAllWindows()
        logger.info("Application shut down cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
