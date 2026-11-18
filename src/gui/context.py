"""Shared state passed to every page.

Pages never talk to each other; they call the context, which owns the
`Application`, the resolved configuration, and the palette. When something
changes, the context emits a signal and the pages that care reload themselves.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import QObject, Signal

from config import AppConfig
from src.application import Application
from src.database import DatabaseError, FaceDatabase
from src.similarity import GalleryIndex

from .theme import Palette, palette_for_style


class AppContext(QObject):
    """The single bridge between the pages and the backend."""

    config_changed = Signal()
    theme_changed = Signal()
    gallery_changed = Signal()
    status_message = Signal(str)

    def __init__(self, application: Application, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._application = application
        self._config = application.config
        self._palette = palette_for_style(self._config.ui.style)

    @property
    def application(self) -> Application:
        return self._application

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def palette(self) -> Palette:
        return self._palette

    @property
    def tier(self) -> str:
        return str(self._application.tier)

    def hardware_summary(self) -> str:
        return self._application.hardware.summary

    def database(self) -> FaceDatabase:
        return self._application.database

    def gallery(self) -> GalleryIndex:
        return self._application.create_gallery()

    def invalidate_gallery(self) -> None:
        self._application.invalidate_gallery()
        self.gallery_changed.emit()

    def save_settings(self, overrides: Mapping[str, Any]) -> None:
        """Persist overrides and adopt the re-validated configuration."""
        previous_style = self._config.ui.style
        self._config = self._application.save_settings(overrides)
        self._after_config_change(previous_style)

    def reset_settings(self) -> None:
        previous_style = self._config.ui.style
        self._config = self._application.reset_settings()
        self._after_config_change(previous_style)
        self.status_message.emit("Settings reset to the shipped defaults.")

    def _after_config_change(self, previous_style: str) -> None:
        self._palette = palette_for_style(self._config.ui.style)
        self.invalidate_gallery()
        self.config_changed.emit()
        if self._config.ui.style != previous_style:
            self.theme_changed.emit()

    def identity_count(self) -> int:
        try:
            return self.database().count_employees()
        except DatabaseError:
            return 0

    def sample_count(self) -> int:
        try:
            return sum(record.sample_count for record in self.database().list_employees())
        except DatabaseError:
            return 0

    def model_signature(self) -> str:
        detector = self._application.detector_if_loaded
        signature = getattr(detector, "model_signature", "") if detector is not None else ""
        return str(signature or "not loaded")
