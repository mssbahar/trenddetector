"""Event dispatch with cooldown management."""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Optional

import config
from effect_manager import EffectManager
from models.hand_data import GestureResult, TriggerEvent
from sound_manager import SoundManager

logger = logging.getLogger(__name__)


class EventSystem:
    def __init__(
        self,
        sound_manager: SoundManager,
        effect_manager: EffectManager,
        cooldown: float = config.COOLDOWN_SECONDS,
    ) -> None:
        self._sound = sound_manager
        self._effect = effect_manager
        self._cooldown = cooldown
        self._last_trigger: Dict[str, float] = {}
        self._active_display: str = ""
        self._active_until: float = 0.0
        self._hold_combo_id: Optional[str] = None
        self._on_trigger: Optional[Callable[[TriggerEvent], None]] = None

    def on_trigger(self, callback: Callable[[TriggerEvent], None]) -> None:
        self._on_trigger = callback

    def emit_gesture(self, gesture: GestureResult) -> bool:
        from config import GESTURE_BINDINGS

        binding = GESTURE_BINDINGS.get(gesture.name)
        if not binding:
            return False

        event = TriggerEvent(
            trigger_id=gesture.name,
            trigger_type="gesture",
            display_name=binding["display_name"],
            sound_path=binding.get("sound", ""),
            effect_path=binding.get("effect", ""),
            effect_type=binding.get("effect_type", "none"),
        )
        return self._emit(event)

    def emit_movement(self, event: TriggerEvent) -> bool:
        return self._emit(event)

    def emit_combo(self, event: TriggerEvent) -> bool:
        return self._emit(event)

    def sync_active_combo(self, combo_id: Optional[str]) -> None:
        """Hold-to-play: start sound/video while detected, stop when idle."""
        if combo_id == self._hold_combo_id:
            return

        self._stop_hold()

        if not combo_id:
            self._active_display = ""
            return

        binding = config.TIKTOK_COMBOS.get(combo_id)
        if not binding:
            return

        self._hold_combo_id = combo_id
        self._active_display = binding["display_name"]

        sound = binding.get("sound", "")
        if sound:
            self._sound.play_loop(sound)

        overlays = binding.get("overlays")
        if overlays:
            self._effect.start_overlays(overlays, binding["display_name"])

        logger.info("Combo active: %s", binding["display_name"])

    def _stop_hold(self) -> None:
        if self._hold_combo_id is None:
            return
        self._sound.stop()
        self._effect.stop()
        self._hold_combo_id = None

    def _emit(self, event: TriggerEvent) -> bool:
        now = time.perf_counter()
        last = self._last_trigger.get(event.trigger_id, 0.0)
        if now - last < self._cooldown:
            logger.debug(
                "Cooldown active for '%s' (%.1fs remaining).",
                event.trigger_id,
                self._cooldown - (now - last),
            )
            return False

        self._last_trigger[event.trigger_id] = now
        self._active_display = event.display_name
        self._active_until = now + self._cooldown

        if event.sound_path:
            self._sound.play(event.sound_path)

        if event.overlays:
            self._effect.trigger_overlays(event.overlays, event.display_name)
        elif event.effect_path and event.effect_type != "none":
            self._effect.trigger(
                event.effect_path,
                event.effect_type,
                event.display_name,
            )

        logger.info("Triggered: %s (%s)", event.display_name, event.trigger_type)

        if self._on_trigger:
            self._on_trigger(event)

        return True

    def get_active_display(self) -> str:
        if self._hold_combo_id:
            return self._active_display
        if time.perf_counter() > self._active_until:
            return ""
        return self._active_display

    def reset_cooldowns(self) -> None:
        self._last_trigger.clear()
        self.sync_active_combo(None)
