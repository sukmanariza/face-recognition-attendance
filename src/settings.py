"""User settings persisted to `settings.json`.

`config.yaml` stays the shipped baseline (with its explanatory comments);
everything the Settings screen can change is stored as a sparse override file and
merged on top of it. Precedence, highest first:

1. `settings.json` (explicit user choice)
2. the active performance tier (see `src/performance.py`)
3. `config.yaml` (defaults)

Merging goes through `config.build_config`, so an edited file cannot introduce a
value the normal loader would reject.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config import AppConfig, ConfigError, build_config

from .hardware import Hardware, detect_hardware
from .performance import apply_profile

logger = logging.getLogger(__name__)

SETTINGS_VERSION = 1


class SettingsError(RuntimeError):
    """Settings file could not be read or written."""


def merge_mappings(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `overrides` into a copy of `base`, returning a new dict."""
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_mappings(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


class SettingsStore:
    """Read and write sparse configuration overrides."""

    def __init__(self, path: Path, hardware: Hardware | None = None) -> None:
        self._path = Path(path)
        self._hardware = hardware

    @property
    def path(self) -> Path:
        return self._path

    @property
    def hardware(self) -> Hardware:
        if self._hardware is None:
            self._hardware = detect_hardware()
        return self._hardware

    def load(self) -> dict[str, Any]:
        """Return the stored overrides; a broken file is ignored with a warning."""
        if not self._path.is_file():
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable settings file %s: %s", self._path, exc)
            return {}
        if not isinstance(payload, dict):
            logger.warning("Ignoring settings file with unexpected shape: %s", self._path)
            return {}
        payload.pop("version", None)
        return payload

    def save(self, overrides: Mapping[str, Any]) -> None:
        """Replace the override file atomically."""
        payload = {"version": SETTINGS_VERSION, **merge_mappings({}, overrides)}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # The handle is closed by the ``with`` below and then renamed into
            # place; that is the atomic-write pattern, so it is not managed here.
            handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
                "w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=".settings-",
                suffix=".tmp",
                delete=False,
            )
            with handle as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(handle.name, self._path)
        except OSError as exc:
            raise SettingsError(f"Could not write settings to {self._path}: {exc}") from exc

    def reset(self) -> None:
        """Delete the override file, returning to the shipped defaults."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            raise SettingsError(f"Could not delete {self._path}: {exc}") from exc

    def effective_config(self, baseline: AppConfig) -> AppConfig:
        """Apply the performance tier, then the user overrides, then validate."""
        tuned, _tier = apply_profile(baseline, self.hardware)
        merged = merge_mappings(tuned.to_mapping(), self.load())
        try:
            return build_config(merged, baseline.source_path)
        except ConfigError as exc:
            logger.warning("Settings overrides were rejected (%s); using defaults.", exc)
            return tuned
