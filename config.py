"""Configuration loading, validation, and serialisation.

`config.yaml` is the shipped baseline. User-adjustable values written from the
Settings screen live in `settings.json` and are merged on top of the baseline at
start-up (see `src/settings.py`). This module owns the single validation path
used by both sources, so a bad value is rejected naming the exact key.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any

import yaml

from src import execution_providers

DEFAULT_CONFIG_FILENAME = "config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / DEFAULT_CONFIG_FILENAME

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigError(RuntimeError):
    """Configuration file is missing, unreadable, or invalid."""


def _reject_unknown(data: Mapping[str, Any], allowed: Sequence[str], path: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        prefix = f"{path}." if path else ""
        raise ConfigError(
            f"Unknown configuration key(s): {', '.join(prefix + str(k) for k in unknown)}. "
            f"Valid keys: {', '.join(sorted(allowed))}."
        )


def _describe(value: Any) -> str:
    return f"{value!r} ({type(value).__name__})"


def _get_int(
    data: Mapping[str, Any],
    key: str,
    default: int,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}.{key} must be an integer, got {_describe(value)}.")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path}.{key} must be at least {minimum}, got {value}.")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{path}.{key} must be at most {maximum}, got {value}.")
    return value


def _get_float(
    data: Mapping[str, Any],
    key: str,
    default: float,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}.{key} must be a number, got {_describe(value)}.")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ConfigError(f"{path}.{key} must be a finite number, got {value}.")
    if minimum is not None:
        if exclusive_minimum and result <= minimum:
            raise ConfigError(f"{path}.{key} must be greater than {minimum}, got {value}.")
        if not exclusive_minimum and result < minimum:
            raise ConfigError(f"{path}.{key} must be at least {minimum}, got {value}.")
    if maximum is not None and result > maximum:
        raise ConfigError(f"{path}.{key} must be at most {maximum}, got {value}.")
    return result


def _get_bool(data: Mapping[str, Any], key: str, default: bool, path: str) -> bool:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{path}.{key} must be true or false, got {_describe(value)}.")
    return value


def _get_str(
    data: Mapping[str, Any],
    key: str,
    default: str,
    path: str,
    *,
    choices: tuple[str, ...] | None = None,
) -> str:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, str):
        raise ConfigError(f"{path}.{key} must be text, got {_describe(value)}.")
    result = value.strip()
    if not result:
        raise ConfigError(f"{path}.{key} must not be empty.")
    if choices is not None and result.lower() not in choices:
        raise ConfigError(f"{path}.{key} must be one of {', '.join(choices)}; got {value!r}.")
    return result.lower() if choices is not None else result


def _get_str_list(
    data: Mapping[str, Any], key: str, default: Sequence[str], path: str
) -> tuple[str, ...]:
    if key not in data:
        return tuple(default)
    value = data[key]
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigError(f"{path}.{key} must be a list of strings, got {_describe(value)}.")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{path}.{key}[{index}] must be a non-empty string.")
        items.append(item.strip())
    return tuple(items)


def _get_int_pair(
    data: Mapping[str, Any], key: str, default: tuple[int, int], path: str, *, minimum: int
) -> tuple[int, int]:
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 2:
        raise ConfigError(
            f"{path}.{key} must be a pair of integers, e.g. [640, 480]; got {_describe(value)}."
        )
    first, second = value
    for item in (first, second):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ConfigError(f"{path}.{key} only accepts integers, got {_describe(item)}.")
        if item < minimum:
            raise ConfigError(f"{path}.{key} values must be at least {minimum}, got {item}.")
    return int(first), int(second)


def _parse_clock_time(value: str, path: str) -> time:
    """Parse ``HH:MM`` (24 hour) into a `time`."""
    parts = value.split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ConfigError(f"{path} must look like 08:00, got {value!r}.")
    hours, minutes = int(parts[0]), int(parts[1])
    if not (0 <= hours <= 23 and 0 <= minutes <= 59):
        raise ConfigError(f"{path} must be a valid time of day, got {value!r}.")
    return time(hour=hours, minute=minutes)


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    requested_fps: int = 60
    backend: str = "auto"
    warmup_frames: int = 3
    read_timeout_seconds: float = 3.0
    max_consecutive_failures: int = 30

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CameraConfig:
        path = "camera"
        return cls(
            index=_get_int(data, "index", 0, path, minimum=0, maximum=16),
            width=_get_int(data, "width", 640, path, minimum=160, maximum=7680),
            height=_get_int(data, "height", 480, path, minimum=120, maximum=4320),
            requested_fps=_get_int(data, "requested_fps", 60, path, minimum=1, maximum=240),
            backend=_get_str(
                data, "backend", "auto", path, choices=("auto", "dshow", "msmf", "v4l2", "any")
            ),
            warmup_frames=_get_int(data, "warmup_frames", 3, path, minimum=0, maximum=120),
            read_timeout_seconds=_get_float(
                data, "read_timeout_seconds", 3.0, path, minimum=0.1, maximum=60.0
            ),
            max_consecutive_failures=_get_int(
                data, "max_consecutive_failures", 30, path, minimum=1, maximum=10000
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    name: str = "buffalo_s"
    root: str = "models"
    providers: tuple[str, ...] = ("auto",)
    det_size: tuple[int, int] = (320, 320)
    det_thresh: float = 0.5
    warmup: bool = True
    graph_optimization: str = "all"
    intra_op_threads: int = 2
    inter_op_threads: int = 1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ModelConfig:
        path = "model"
        providers = _get_str_list(data, "providers", ("auto",), path)
        for index, name in enumerate(providers):
            if name not in execution_providers.known_names():
                raise ConfigError(
                    f"{path}.providers[{index}] is {name!r}, which is not a known "
                    f"execution provider. Use one of: "
                    f"{', '.join(execution_providers.known_names())}."
                )
        return cls(
            name=_get_str(data, "name", "buffalo_s", path),
            root=_get_str(data, "root", "models", path),
            providers=providers,
            det_size=_get_int_pair(data, "det_size", (320, 320), path, minimum=64),
            det_thresh=_get_float(data, "det_thresh", 0.5, path, minimum=0.01, maximum=0.99),
            warmup=_get_bool(data, "warmup", True, path),
            graph_optimization=_get_str(
                data,
                "graph_optimization",
                "all",
                path,
                choices=("all", "extended", "basic", "disabled"),
            ),
            intra_op_threads=_get_int(data, "intra_op_threads", 2, path, minimum=0, maximum=64),
            inter_op_threads=_get_int(data, "inter_op_threads", 1, path, minimum=0, maximum=64),
        )

    def to_mapping(self) -> dict[str, Any]:
        mapping = dataclasses.asdict(self)
        mapping["providers"] = list(self.providers)
        mapping["det_size"] = list(self.det_size)
        return mapping


@dataclass(frozen=True)
class RecognitionConfig:
    similarity_threshold: float = 0.45
    ambiguity_margin: float = 0.0
    detection_interval: int = 2
    embedding_interval_seconds: float = 0.5
    confirmation_hits: int = 3
    smoothing_alpha: float = 0.65
    track_max_age_seconds: float = 1.5
    max_faces: int = 5
    iou_threshold: float = 0.3

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RecognitionConfig:
        path = "recognition"
        return cls(
            similarity_threshold=_get_float(
                data, "similarity_threshold", 0.45, path, minimum=-1.0, maximum=1.0
            ),
            ambiguity_margin=_get_float(
                data, "ambiguity_margin", 0.0, path, minimum=0.0, maximum=1.0
            ),
            detection_interval=_get_int(data, "detection_interval", 2, path, minimum=1, maximum=60),
            embedding_interval_seconds=_get_float(
                data, "embedding_interval_seconds", 0.5, path, minimum=0.0, maximum=60.0
            ),
            confirmation_hits=_get_int(data, "confirmation_hits", 3, path, minimum=1, maximum=120),
            smoothing_alpha=_get_float(
                data,
                "smoothing_alpha",
                0.65,
                path,
                minimum=0.0,
                maximum=1.0,
                exclusive_minimum=True,
            ),
            track_max_age_seconds=_get_float(
                data, "track_max_age_seconds", 1.5, path, minimum=0.1, maximum=60.0
            ),
            max_faces=_get_int(data, "max_faces", 5, path, minimum=1, maximum=64),
            iou_threshold=_get_float(data, "iou_threshold", 0.3, path, minimum=0.01, maximum=1.0),
        )

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RegistrationConfig:
    samples: int = 5
    capture_interval_seconds: float = 0.4
    min_sample_consistency: float = 0.50
    duplicate_similarity_threshold: float = 0.60
    frontal_yaw_tolerance: float = 0.10
    turned_yaw: float = 0.18
    tilt_roll_min_deg: float = 8.0
    tilt_roll_max_deg: float = 20.0
    stage_max_roll_deg: float = 14.0
    closer_scale_ratio: float = 1.20

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RegistrationConfig:
        path = "registration"
        return cls(
            samples=_get_int(data, "samples", 5, path, minimum=1, maximum=5),
            capture_interval_seconds=_get_float(
                data, "capture_interval_seconds", 0.4, path, minimum=0.0, maximum=30.0
            ),
            min_sample_consistency=_get_float(
                data, "min_sample_consistency", 0.50, path, minimum=-1.0, maximum=1.0
            ),
            duplicate_similarity_threshold=_get_float(
                data, "duplicate_similarity_threshold", 0.60, path, minimum=-1.0, maximum=1.0
            ),
            frontal_yaw_tolerance=_get_float(
                data, "frontal_yaw_tolerance", 0.10, path, minimum=0.02, maximum=0.5
            ),
            turned_yaw=_get_float(data, "turned_yaw", 0.18, path, minimum=0.05, maximum=0.8),
            tilt_roll_min_deg=_get_float(
                data, "tilt_roll_min_deg", 8.0, path, minimum=0.0, maximum=45.0
            ),
            tilt_roll_max_deg=_get_float(
                data, "tilt_roll_max_deg", 20.0, path, minimum=1.0, maximum=60.0
            ),
            stage_max_roll_deg=_get_float(
                data, "stage_max_roll_deg", 14.0, path, minimum=1.0, maximum=45.0
            ),
            closer_scale_ratio=_get_float(
                data, "closer_scale_ratio", 1.20, path, minimum=1.0, maximum=3.0
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class QualityConfig:
    min_face_size_px: int = 120
    min_det_score: float = 0.65
    blur_threshold: float = 60.0
    brightness_min: float = 60.0
    brightness_max: float = 200.0
    contrast_min: float = 20.0
    max_yaw_ratio: float = 0.45
    max_roll_deg: float = 25.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> QualityConfig:
        path = "quality"
        config = cls(
            min_face_size_px=_get_int(
                data, "min_face_size_px", 120, path, minimum=16, maximum=2000
            ),
            min_det_score=_get_float(data, "min_det_score", 0.65, path, minimum=0.0, maximum=1.0),
            blur_threshold=_get_float(
                data, "blur_threshold", 60.0, path, minimum=0.0, maximum=10000.0
            ),
            brightness_min=_get_float(
                data, "brightness_min", 60.0, path, minimum=0.0, maximum=255.0
            ),
            brightness_max=_get_float(
                data, "brightness_max", 200.0, path, minimum=0.0, maximum=255.0
            ),
            contrast_min=_get_float(data, "contrast_min", 20.0, path, minimum=0.0, maximum=255.0),
            max_yaw_ratio=_get_float(data, "max_yaw_ratio", 0.45, path, minimum=0.01, maximum=1.0),
            max_roll_deg=_get_float(data, "max_roll_deg", 25.0, path, minimum=1.0, maximum=90.0),
        )
        if config.brightness_min >= config.brightness_max:
            raise ConfigError(
                f"{path}.brightness_min ({config.brightness_min}) must be lower than "
                f"{path}.brightness_max ({config.brightness_max})."
            )
        return config

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class DatabaseConfig:
    path: str = "database/face_database.db"
    busy_timeout_ms: int = 5000
    encrypt: bool = True
    key_file: str = "database/face_key.bin"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DatabaseConfig:
        path = "database"
        return cls(
            path=_get_str(data, "path", "database/face_database.db", path),
            busy_timeout_ms=_get_int(
                data, "busy_timeout_ms", 5000, path, minimum=100, maximum=120000
            ),
            encrypt=_get_bool(data, "encrypt", True, path),
            key_file=_get_str(data, "key_file", "database/face_key.bin", path),
        )

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class AttendanceConfig:
    """Daily presence rules."""

    enabled: bool = True
    work_start: str = "08:00"
    late_after_minutes: int = 15
    refresh_seconds: int = 60
    require_live_face: bool = True
    export_directory: str = "attendance"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AttendanceConfig:
        path = "attendance"
        work_start = _get_str(data, "work_start", "08:00", path)
        _parse_clock_time(work_start, f"{path}.work_start")
        return cls(
            enabled=_get_bool(data, "enabled", True, path),
            work_start=work_start,
            late_after_minutes=_get_int(
                data, "late_after_minutes", 15, path, minimum=0, maximum=720
            ),
            refresh_seconds=_get_int(data, "refresh_seconds", 60, path, minimum=5, maximum=3600),
            require_live_face=_get_bool(data, "require_live_face", True, path),
            export_directory=_get_str(data, "export_directory", "attendance", path),
        )

    def start_time(self) -> time:
        return _parse_clock_time(self.work_start, "attendance.work_start")

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class UiConfig:
    window_title: str = "Face ID"
    display_max_width: int = 720
    show_fps: bool = True
    show_similarity: bool = True
    style: str = "dark"
    animations: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> UiConfig:
        path = "ui"
        return cls(
            window_title=_get_str(data, "window_title", "Face ID", path),
            display_max_width=_get_int(
                data, "display_max_width", 720, path, minimum=240, maximum=3840
            ),
            show_fps=_get_bool(data, "show_fps", True, path),
            show_similarity=_get_bool(data, "show_similarity", True, path),
            style=_get_str(data, "style", "dark", path, choices=("dark", "light")),
            animations=_get_bool(data, "animations", True, path),
        )

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class LivenessConfig:
    """Presentation check: real 3D face versus a photo or a screen."""

    mode: str = "advisory"
    min_observations: int = 8
    sensitivity: float = 0.5

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> LivenessConfig:
        path = "liveness"
        return cls(
            mode=_get_str(
                data, "mode", "advisory", path, choices=("off", "advisory", "enforce")
            ),
            min_observations=_get_int(
                data, "min_observations", 8, path, minimum=2, maximum=60
            ),
            sensitivity=_get_float(data, "sensitivity", 0.5, path, minimum=0.0, maximum=1.0),
        )

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def enforcing(self) -> bool:
        return self.mode == "enforce"

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> LoggingConfig:
        level = _get_str(
            data, "level", "INFO", "logging", choices=tuple(x.lower() for x in _LOG_LEVELS)
        ).upper()
        return cls(level=level)

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class BenchmarkConfig:
    duration_seconds: float = 60.0
    warmup_seconds: float = 5.0
    sample_interval_seconds: float = 1.0
    target_display_fps: float = 25.0
    output_directory: str = "benchmarks"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BenchmarkConfig:
        path = "benchmark"
        return cls(
            duration_seconds=_get_float(
                data, "duration_seconds", 60.0, path, minimum=5.0, maximum=3600.0
            ),
            warmup_seconds=_get_float(
                data, "warmup_seconds", 5.0, path, minimum=0.0, maximum=600.0
            ),
            sample_interval_seconds=_get_float(
                data, "sample_interval_seconds", 1.0, path, minimum=0.2, maximum=30.0
            ),
            target_display_fps=_get_float(
                data, "target_display_fps", 25.0, path, minimum=1.0, maximum=240.0
            ),
            output_directory=_get_str(data, "output_directory", "benchmarks", path),
        )

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class PerformanceConfig:
    """Hardware tier selection; resolved by `src/performance.py`."""

    profile: str = "auto"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PerformanceConfig:
        return cls(
            profile=_get_str(
                data,
                "profile",
                "auto",
                "performance",
                choices=("auto", "minimal", "low", "balanced", "high"),
            )
        )

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTION_TYPES: dict[str, type] = {
    "camera": CameraConfig,
    "model": ModelConfig,
    "recognition": RecognitionConfig,
    "registration": RegistrationConfig,
    "quality": QualityConfig,
    "database": DatabaseConfig,
    "attendance": AttendanceConfig,
    "ui": UiConfig,
    "liveness": LivenessConfig,
    "logging": LoggingConfig,
    "benchmark": BenchmarkConfig,
    "performance": PerformanceConfig,
}

SECTION_NAMES: tuple[str, ...] = tuple(_SECTION_TYPES)

SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    name: tuple(f.name for f in dataclasses.fields(section_type))
    for name, section_type in _SECTION_TYPES.items()
}


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    model: ModelConfig
    recognition: RecognitionConfig
    registration: RegistrationConfig
    quality: QualityConfig
    database: DatabaseConfig
    attendance: AttendanceConfig
    ui: UiConfig
    liveness: LivenessConfig
    logging: LoggingConfig
    benchmark: BenchmarkConfig
    performance: PerformanceConfig
    source_path: Path = DEFAULT_CONFIG_PATH

    @property
    def base_dir(self) -> Path:
        """Directory that relative paths in the configuration resolve against."""
        return self.source_path.parent

    def resolve(self, value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (self.base_dir / candidate).resolve()

    @property
    def model_dir(self) -> Path:
        return self.resolve(Path(self.model.root) / self.model.name)

    @property
    def database_path(self) -> Path:
        return self.resolve(self.database.path)

    @property
    def key_path(self) -> Path:
        return self.resolve(self.database.key_file)

    @property
    def attendance_dir(self) -> Path:
        return self.resolve(self.attendance.export_directory)

    @property
    def benchmark_dir(self) -> Path:
        return self.resolve(self.benchmark.output_directory)

    @property
    def settings_path(self) -> Path:
        return self.base_dir / "settings.json"

    @property
    def log_level(self) -> str:
        return self.logging.level

    def to_mapping(self) -> dict[str, Any]:
        """Serialise every section so it can be re-validated after a merge."""
        return {name: getattr(self, name).to_mapping() for name in SECTION_NAMES}

    def replace(self, **changes: Any) -> AppConfig:
        return dataclasses.replace(self, **changes)


def build_config(root: Mapping[str, Any], source_path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Validate a complete mapping (baseline already merged with overrides)."""
    _reject_unknown(root, SECTION_NAMES, "")
    sections: dict[str, Any] = {}
    for name, section_type in _SECTION_TYPES.items():
        data = root.get(name) or {}
        if not isinstance(data, Mapping):
            raise ConfigError(f"{name} must be a mapping of keys to values, got {_describe(data)}.")
        _reject_unknown(data, SECTION_FIELDS[name], name)
        sections[name] = section_type.from_mapping(data)
    return AppConfig(**sections, source_path=source_path)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load `config.yaml`. User overrides are applied by `src/settings.py`."""
    config_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(
            f"Configuration file not found: {config_path}. "
            "Run `python install.py` or copy config.yaml into the project folder."
        )
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml could not be parsed: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"config.yaml could not be read: {exc}") from exc
    return build_config(raw or {}, config_path)
