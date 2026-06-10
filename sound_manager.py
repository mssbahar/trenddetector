"""Pygame-based sound playback with safe missing-file handling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Set

import pygame

logger = logging.getLogger(__name__)


class SoundManager:
    def __init__(self, volume: float = 0.8) -> None:
        self._volume = volume
        self._cache: Dict[str, pygame.mixer.Sound] = {}
        self._missing: Set[str] = set()
        self._initialized = False
        self._loop_path: str | None = None

    def init(self) -> None:
        if self._initialized:
            return
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._initialized = True
            logger.info("Pygame mixer initialized.")
        except pygame.error as exc:
            logger.warning("Could not initialize audio: %s", exc)

    def play_loop(self, path: str) -> None:
        """Loop sound while gesture is held."""
        if not self._initialized:
            self.init()
        if not self._initialized:
            return

        file_path = Path(path)
        if not file_path.exists():
            if path not in self._missing:
                logger.warning("Sound file not found: %s", path)
                self._missing.add(path)
            return

        try:
            if file_path.suffix.lower() == ".mp3":
                if self._loop_path != path:
                    pygame.mixer.music.load(str(file_path))
                    self._loop_path = path
                pygame.mixer.music.set_volume(self._volume)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)
                return

            if path not in self._cache:
                self._cache[path] = pygame.mixer.Sound(str(file_path))
            sound = self._cache[path]
            sound.set_volume(self._volume)
            if self._loop_path != path:
                pygame.mixer.stop()
                sound.play(-1)
                self._loop_path = path
        except pygame.error as exc:
            logger.warning("Failed to loop sound '%s': %s", path, exc)

    def stop(self) -> None:
        self._loop_path = None
        if self._initialized:
            pygame.mixer.music.stop()
            pygame.mixer.stop()

    def play(self, path: str) -> None:
        if not self._initialized:
            self.init()
        if not self._initialized:
            return

        file_path = Path(path)
        if not file_path.exists():
            if path not in self._missing:
                logger.warning("Sound file not found: %s", path)
                self._missing.add(path)
            return

        try:
            if file_path.suffix.lower() == ".mp3":
                pygame.mixer.music.load(str(file_path))
                pygame.mixer.music.set_volume(self._volume)
                pygame.mixer.music.play()
                return

            if path not in self._cache:
                self._cache[path] = pygame.mixer.Sound(str(file_path))
            sound = self._cache[path]
            sound.set_volume(self._volume)
            sound.play()
        except pygame.error as exc:
            logger.warning("Failed to play sound '%s': %s", path, exc)

    def stop_all(self) -> None:
        self.stop()

    def quit(self) -> None:
        self.stop()
        if self._initialized:
            pygame.mixer.quit()
            self._initialized = False
