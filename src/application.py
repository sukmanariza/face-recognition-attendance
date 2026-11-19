"""Composition root: builds and owns every long-lived object.

The GUI, the CLI, and the benchmark all need the same objects wired the same way.
Building them here keeps that wiring in one place, and keeps the heavy resources
(the ONNX model and the SQLite connection) created lazily so opening the window
does not wait for a 300 MB model load.

Configuration is resolved once, here: `settings.json` on top of the hardware
performance tier on top of `config.yaml`. `update_config` re-applies a new
configuration and disposes only the objects that it invalidates.
"""

from __future__ import annotations

import logging
import platform
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from config import AppConfig, load_config

from . import execution_providers
from .attendance import AttendanceService
from .camera import Camera, probe_cameras
from .crypto import FieldCipher, passphrase_from_environment
from .database import FaceDatabase
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognitionSession
from .face_registration import FaceRegistrar, FaceRegistrationSession
from .hardware import Hardware, detect_hardware
from .liveness import LivenessDetector, build_liveness_detector
from .performance import Tier, apply_profile, describe_profile
from .quality import FaceQualityAssessor
from .settings import SettingsStore
from .similarity import GalleryIndex

logger = logging.getLogger(__name__)


class Application:
    """Owns the configuration, database, model, and the sessions built on them."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        hardware: Hardware | None = None,
        use_settings: bool = True,
        passphrase: str | None = None,
    ) -> None:
        baseline = config if config is not None else load_config()
        self._hardware = hardware or detect_hardware()
        self._settings = SettingsStore(baseline.settings_path, self._hardware)
        tuned, self._tier = apply_profile(baseline, self._hardware)
        if use_settings:
            self._config = self._settings.effective_config(tuned)
        else:
            self._config = tuned
        self._baseline = baseline

        self._database: FaceDatabase | None = None
        self._detector: FaceDetector | None = None
        self._assessor: FaceQualityAssessor | None = None
        self._assessor_config: object | None = None
        self._liveness: LivenessDetector | None = None
        self._gallery_cache: tuple[float, GalleryIndex] | None = None
        self._attendance: AttendanceService | None = None
        self._cipher: FieldCipher | None = None
        self._passphrase = passphrase or passphrase_from_environment()

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def baseline(self) -> AppConfig:
        """Shipped `config.yaml` values, before tiers and user overrides."""
        return self._baseline

    @property
    def settings(self) -> SettingsStore:
        return self._settings

    @property
    def hardware(self) -> Hardware:
        return self._hardware

    @property
    def tier(self) -> Tier:
        return self._tier

    def update_config(self, config: AppConfig) -> None:
        """Adopt a new configuration, disposing only what it invalidates."""
        previous = self._config
        self._config = config

        model_changed = config.model != previous.model or config.model_dir != previous.model_dir
        if model_changed and self._detector is not None:
            self._detector.close()
            self._detector = None
        if config.database_path != previous.database_path:
            if self._database is not None:
                self._database.close()
                self._database = None
            self._gallery_cache = None
        elif config.database != previous.database:
            # Key path or encryption flag changed: reopen so the cipher matches.
            if self._database is not None:
                self._database.close()
                self._database = None
            self._cipher = None
            self._gallery_cache = None
        if config.attendance != previous.attendance:
            self._attendance = AttendanceService(self.database, config.attendance)
        if config.quality != previous.quality and self._assessor is not None:
            self._assessor = FaceQualityAssessor(config.quality)
            self._assessor_config = config.quality
        if config.liveness != previous.liveness:
            self._liveness = None

    def save_settings(self, overrides: Mapping[str, Any]) -> AppConfig:
        """Persist overrides, then apply the re-validated configuration."""
        current = self._settings.load()
        merged: dict[str, Any] = {**current}
        for section, values in overrides.items():
            if isinstance(values, Mapping) and isinstance(merged.get(section), Mapping):
                merged[section] = {**merged[section], **values}
            else:
                merged[section] = values
        self._settings.save(merged)
        self._reload()
        return self._config

    def reset_settings(self) -> AppConfig:
        """Drop every override and fall back to the performance tier."""
        self._settings.reset()
        self._reload()
        return self._config

    def _reload(self) -> None:
        tuned, self._tier = apply_profile(self._baseline, self._hardware)
        self.update_config(self._settings.effective_config(tuned))

    @property
    def database(self) -> FaceDatabase:
        if self._database is None:
            self._database = FaceDatabase(
                self._config.database_path,
                self._config.database.busy_timeout_ms,
                self.cipher,
            )
        return self._database

    @property
    def cipher(self) -> FieldCipher | None:
        """The database cipher, or None when encryption is turned off."""
        if not self._config.database.encrypt:
            return None
        if self._cipher is None:
            self._cipher = FieldCipher.from_key_file(
                self._config.key_path, self._passphrase
            )
        return self._cipher

    @property
    def attendance(self) -> AttendanceService:
        if self._attendance is None:
            self._attendance = AttendanceService(self.database, self._config.attendance)
        return self._attendance

    @property
    def detector(self) -> FaceDetector:
        if self._detector is None:
            self._detector = FaceDetector(self._config.model, self._config.model_dir)
        if not self._detector.is_loaded:
            self._detector.load()
        return self._detector

    @property
    def detector_if_loaded(self) -> FaceDetector | None:
        """The detector only if it already exists, so callers can avoid loading it."""
        return self._detector

    @property
    def assessor(self) -> FaceQualityAssessor:
        if self._assessor is None or self._assessor_config != self._config.quality:
            self._assessor = FaceQualityAssessor(self._config.quality)
            self._assessor_config = self._config.quality
        return self._assessor

    @property
    def liveness(self) -> LivenessDetector:
        if self._liveness is None:
            self._liveness = build_liveness_detector(self._config.liveness)
        return self._liveness

    def create_gallery(self) -> GalleryIndex:
        """Gallery from the database; cached until the database changes."""
        stamp = self._database_gallery_stamp()
        if self._gallery_cache is not None and self._gallery_cache[0] == stamp:
            return self._gallery_cache[1]
        gallery = GalleryIndex(self.database.load_gallery())
        self._gallery_cache = (stamp, gallery)
        return gallery

    def _database_gallery_stamp(self) -> float:
        path = self._config.database_path
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def invalidate_gallery(self) -> None:
        self._gallery_cache = None

    def create_recognition_session(self) -> FaceRecognitionSession:
        return FaceRecognitionSession(
            self.detector,
            self.create_gallery(),
            self._config.recognition,
            self.liveness,
            enforce_presentation=self._config.liveness.enforcing,
            attendance=self.attendance,
        )

    def create_registrar(self) -> FaceRegistrar:
        return FaceRegistrar(self.database, self._config.registration)

    def create_registration_session(self) -> FaceRegistrationSession:
        return FaceRegistrationSession(
            self.detector,
            self.assessor,
            self._config.registration,
            self.liveness,
            self.create_gallery(),
        )

    def create_camera(self) -> Camera:
        return Camera(self._config.camera)

    @staticmethod
    def probe_cameras(max_index: int = 4) -> list[int]:
        return probe_cameras(max_index=max_index)

    def describe_compute(self) -> str:
        """Which backend the models will run on, and what else is installed."""
        requested = self._config.model.providers
        if self._detector is not None and self._detector.is_loaded:
            return f"{execution_providers.describe(self._detector.providers)} (loaded)"
        chosen, available = execution_providers.plan(requested)
        if not chosen:
            return "CPU only (onnxruntime reports no provider)"
        installed = [name for name in available if name != "AzureExecutionProvider"]
        return (
            f"{execution_providers.describe(chosen)} available in this onnxruntime build "
            f"({', '.join(sorted(installed))})"
        )

    def describe_environment(self) -> Sequence[str]:
        """Human-readable summary for the status view and the CLI."""
        lines = [
            f"Python            {platform.python_version()} ({platform.system()} "
            f"{platform.release()}, {platform.machine()})",
            f"Executable        {sys.executable}",
            f"Hardware          {self._hardware.summary}",
            f"Performance       {describe_profile(self._tier, self._hardware)}",
            f"Compute           {self.describe_compute()}",
            f"Configuration     {self._config.source_path}",
            f"Settings          {self._settings.path}"
            + ("" if self._settings.path.is_file() else " (not created yet)"),
            f"Database          {self._config.database_path}",
        ]
        if self._detector is not None and self._detector.is_loaded:
            lines.append(f"Model             {self._detector.describe()}")
            lines.append(f"Signature         {self._detector.model_signature}")
        else:
            lines.append(f"Model             {self._config.model.name} (not loaded yet)")
        liveness = self._config.liveness
        detail = liveness.mode
        if liveness.enabled:
            detail += f" ({self.liveness.name}, sensitivity {liveness.sensitivity:.2f})"
        lines.append(f"Presentation      {detail}")
        storage = "encrypted (AES-GCM)" if self._config.database.encrypt else "plain text"
        lines.append(f"Storage           {storage}")
        lines.append(f"Attendance        {self.attendance.describe()}")
        return lines

    def close(self) -> None:
        """Release the model and the database. Safe to call more than once."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        if self._database is not None:
            self._database.close()
            self._database = None
        self._gallery_cache = None
        self._liveness = None

    def __enter__(self) -> Application:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


