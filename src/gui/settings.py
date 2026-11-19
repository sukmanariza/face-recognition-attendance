"""Settings page: every value a user can reasonably change, in one place.

Values are stored as a sparse override file (`settings.json`) merged on top of
`config.yaml` and the hardware performance tier, so a reset really does return to
the shipped defaults. Anything that can only take effect when the model or camera
is rebuilt is marked in the UI instead of pretending it applies immediately.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QLocale, Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.similarity import to_percentage

from . import widgets
from .context import AppContext


class _DecimalField(QWidget):
    """Decimal spin box with a label and an optional hint."""

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        step: float = 0.1,
        suffix: str = "",
        hint: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._spin = QDoubleSpinBox()
        # The UI is English, so decimals use a dot even on a comma-locale system.
        self._spin.setLocale(QLocale.c())
        self._spin.setRange(minimum, maximum)
        self._spin.setSingleStep(step)
        self._spin.setDecimals(2)
        self._spin.setValue(value)
        if suffix:
            self._spin.setSuffix(suffix)
        self._spin.setMaximumWidth(140)
        self._spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(3)
        row = QHBoxLayout()
        row.setSpacing(12)
        text = QLabel(label)
        text.setMinimumWidth(180)
        text.setWordWrap(True)
        row.addWidget(text, 0)
        row.addWidget(self._spin, 1)
        row.addStretch(0)
        outer.addLayout(row)
        if hint:
            note = QLabel(hint)
            note.setObjectName("Hint")
            note.setWordWrap(True)
            note.setContentsMargins(192, 0, 0, 0)
            outer.addWidget(note)

    def value(self) -> float:
        return float(self._spin.value())

    def set_value(self, value: float) -> None:
        self._spin.setValue(value)


class SettingsPage(QWidget):
    """Editable configuration with save and reset."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._build_ui()
        self.load_values(context.config)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(14)

        root.addWidget(
            widgets.PageHeader(
                "Settings",
                "Changes are stored in settings.json and apply on the next start. "
                "Fields marked with a dot need a restart because the model or the camera "
                "is rebuilt.",
            )
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self._content = QVBoxLayout(content)
        self._content.setContentsMargins(0, 0, 14, 0)
        self._content.setSpacing(12)

        self._content.addWidget(self._build_recognition_card())
        self._content.addWidget(self._build_registration_card())
        self._content.addWidget(self._build_quality_card())
        self._content.addWidget(self._build_presentation_card())
        self._content.addWidget(self._build_attendance_card())
        self._content.addWidget(self._build_camera_card())
        self._content.addWidget(self._build_performance_card())
        self._content.addWidget(self._build_interface_card())
        self._content.addWidget(self._build_data_card())
        self._content.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        footer = QHBoxLayout()
        footer.setSpacing(10)
        self._save_button = widgets.primary_button("Save settings")
        self._save_button.clicked.connect(self.save)
        self._reset_button = widgets.secondary_button("Reset to defaults")
        self._reset_button.clicked.connect(self.reset)
        self._revert_button = widgets.secondary_button("Discard changes")
        self._revert_button.clicked.connect(lambda: self.load_values(self._context.config))
        footer.addWidget(self._save_button)
        footer.addWidget(self._revert_button)
        footer.addStretch(1)
        footer.addWidget(self._reset_button)
        footer_holder = QWidget()
        footer_holder.setLayout(footer)
        root.addWidget(footer_holder)

        self._banner = widgets.Banner()
        root.addWidget(self._banner)

    def _build_recognition_card(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("Recognition"))

        self._threshold = widgets.SliderField(
            "Similarity threshold (cosine)",
            0,
            95,
            45,
            lambda value: f"{value / 100:.2f}",
            hint="Raw cosine score a match needs. Measured on this model, people who "
            "are different score under 0.20 and the same person above 0.48, so keep this "
            "between those two numbers. Run `python main.py calibrate` to measure yours.",
        )
        card.add(self._threshold)

        self._ambiguity = widgets.SliderField(
            "Ambiguity margin",
            0,
            30,
            0,
            lambda value: f"{value}%",
            hint="Reject a match when the runner-up is within this margin of the winner. "
            "0 disables the check.",
        )
        card.add(self._ambiguity)

        self._confirmation = widgets.NumericField(
            "Confirmation hits",
            1,
            15,
            3,
            hint="Consecutive agreeing results before a name is shown.",
        )
        card.add(self._confirmation)

        self._detection_interval = widgets.NumericField(
            "Detection interval",
            1,
            10,
            2,
            suffix=" frames",
            hint="Detect on every Nth frame and reuse the boxes in between. Higher is faster.",
        )
        card.add(self._detection_interval)

        self._embedding_interval = _DecimalField(
            "Embedding interval",
            0.0,
            5.0,
            0.5,
            step=0.1,
            suffix=" s",
            hint="Minimum time between embedding the same face. Higher saves CPU.",
        )
        card.add(self._embedding_interval)

        self._max_faces = widgets.NumericField(
            "Maximum faces", 1, 10, 5, hint="Faces processed per frame."
        )
        card.add(self._max_faces)
        return card

    def _build_registration_card(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("Enrollment"))
        self._samples = widgets.NumericField(
            "Samples per identity",
            1,
            5,
            5,
            hint="One staged pose per sample: frontal, turn left, turn right, tilt, closer.",
        )
        card.add(self._samples)

        self._consistency = widgets.SliderField(
            "Sample consistency",
            0,
            100,
            50,
            lambda value: f"{value}%",
            hint="How closely a new sample must match the samples already captured.",
        )
        card.add(self._consistency)

        self._duplicate = widgets.SliderField(
            "Duplicate face guard",
            0,
            100,
            60,
            lambda value: f"{value}%",
            hint="Refuse to register a face that already belongs to another identity.",
        )
        card.add(self._duplicate)

        self._capture_interval = _DecimalField(
            "Capture interval",
            0.0,
            5.0,
            0.4,
            step=0.1,
            suffix=" s",
            hint="Pause between two accepted samples so the pose can change.",
        )
        card.add(self._capture_interval)
        return card

    def _build_quality_card(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("Image quality gate"))
        self._min_face = widgets.NumericField(
            "Minimum face size",
            40,
            800,
            120,
            suffix=" px",
            hint="Smaller faces produce less reliable embeddings.",
        )
        card.add(self._min_face)

        self._min_det_score = widgets.SliderField(
            "Minimum detection score",
            30,
            99,
            65,
            lambda value: f"{value}%",
            hint="Lower it when the detector misses your face; raise it in cluttered scenes.",
        )
        card.add(self._min_det_score)

        self._blur = widgets.NumericField(
            "Sharpness threshold",
            10,
            400,
            60,
            hint="Laplacian variance. Raise for a sharp camera, lower in dim rooms.",
        )
        card.add(self._blur)

        self._brightness_min = widgets.NumericField("Minimum brightness", 0, 255, 60)
        card.add(self._brightness_min)

        self._brightness_max = widgets.NumericField("Maximum brightness", 1, 255, 200)
        card.add(self._brightness_max)

        self._max_yaw = widgets.SliderField(
            "Maximum head turn",
            10,
            100,
            45,
            lambda value: f"{value / 100:.2f}",
            hint="Upper bound on the yaw ratio for an accepted sample.",
        )
        card.add(self._max_yaw)
        return card

    def _build_presentation_card(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("Presentation check"))
        note = QLabel(
            "Tells a real face apart from a photo or a screen by measuring whether the face "
            "has depth when the head moves. A photo or screen is refused during enrollment "
            "in every mode except off."
        )
        note.setObjectName("Hint")
        note.setWordWrap(True)
        card.add(note)

        self._liveness_mode = widgets.ChoiceField(
            "Mode",
            (
                ("off", "Off (measure nothing)"),
                ("advisory", "Advisory (label only)"),
                ("enforce", "Enforce (refuse a detected photo)"),
            ),
            "advisory",
            hint="Advisory keeps working with a weak camera; enforce blocks the match instead.",
        )
        card.add(self._liveness_mode)

        self._liveness_observations = widgets.NumericField(
            "Frames before a verdict",
            2,
            60,
            8,
            hint="The check needs movement over time, so it stays undecided until then.",
        )
        card.add(self._liveness_observations)

        self._liveness_sensitivity = widgets.SliderField(
            "Sensitivity",
            0,
            100,
            50,
            lambda value: f"{value}%",
            hint="Higher reacts to smaller head movements. Lower waits for a clearer turn.",
        )
        card.add(self._liveness_sensitivity)
        return card

    def _build_attendance_card(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("Attendance"))

        self._attendance_enabled = widgets.ToggleField(
            "Record attendance from recognition",
            True,
            hint="A confirmed match marks the person present once per day.",
        )
        card.add(self._attendance_enabled)

        self._work_start = widgets.TimeField(
            "Work starts at",
            "08:00",
            hint="A first sighting later than this plus the grace period is marked late.",
        )
        card.add(self._work_start)

        self._late_minutes = widgets.NumericField(
            "Grace period", 0, 240, 15, suffix=" min"
        )
        card.add(self._late_minutes)

        self._refresh_seconds = widgets.NumericField(
            "Repeat sighting interval",
            5,
            3600,
            60,
            suffix=" s",
            hint="How often a still-visible face updates the last-seen time.",
        )
        card.add(self._refresh_seconds)

        self._require_live = widgets.ToggleField(
            "Refuse a face flagged as a photo",
            True,
            hint="Only applies while the presentation check is on.",
        )
        card.add(self._require_live)
        return card

    def _build_camera_card(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("Camera"))
        note = QLabel("·  Changed camera values are applied on the next start.")
        note.setObjectName("Hint")
        card.add(note)

        self._camera_index = widgets.NumericField("Camera index", 0, 16, 0)
        card.add(self._camera_index)

        self._camera_width = widgets.NumericField("Width", 160, 1920, 640, suffix=" px")
        card.add(self._camera_width)

        self._camera_height = widgets.NumericField("Height", 120, 1080, 480, suffix=" px")
        card.add(self._camera_height)

        self._camera_backend = widgets.ChoiceField(
            "Backend",
            (
                ("auto", "Auto"),
                ("dshow", "DirectShow (Windows)"),
                ("msmf", "Media Foundation (Windows)"),
                ("v4l2", "V4L2 (Linux)"),
                ("any", "Any"),
            ),
            "auto",
        )
        card.add(self._camera_backend)

        self._camera_fps = widgets.NumericField("Requested FPS", 1, 120, 30)
        card.add(self._camera_fps)
        return card

    def _build_performance_card(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("Performance"))
        note = QLabel("·  Detector size and thread count are applied on the next start.")
        note.setObjectName("Hint")
        card.add(note)

        self._profile = widgets.ChoiceField(
            "Hardware profile",
            (
                ("auto", "Auto detect"),
                ("minimal", "Minimal (1 GB board)"),
                ("low", "Low power (single-board, 2 to 4 cores)"),
                ("balanced", "Balanced (4-core laptop)"),
                ("high", "High (6+ cores)"),
            ),
            "auto",
            hint="Auto reads the CPU core count and the installed RAM.",
        )
        card.add(self._profile)

        self._det_size = widgets.ChoiceField(
            "Detector input size",
            (
                ("320", "320 x 320 (fastest)"),
                ("480", "480 x 480 (small or distant faces)"),
                ("640", "640 x 640 (slowest)"),
            ),
            "320",
        )
        card.add(self._det_size)

        self._intra_threads = widgets.NumericField(
            "Inference threads",
            0,
            16,
            2,
            hint="0 lets ONNX Runtime decide. Keep it at or below the physical core count.",
        )
        card.add(self._intra_threads)
        return card

    def _build_interface_card(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("Interface"))
        self._style = widgets.ChoiceField(
            "Theme", (("dark", "Dark grey"), ("light", "Light")), "dark"
        )
        card.add(self._style)

        self._animations = widgets.ToggleField(
            "Animated transitions", True, hint="Turn off on very slow machines."
        )
        card.add(self._animations)

        self._show_fps = widgets.ToggleField("Show FPS", True)
        card.add(self._show_fps)

        self._show_similarity = widgets.ToggleField(
            "Show similarity percentage", True, hint="The number is a similarity score, not a "
            "probability."
        )
        card.add(self._show_similarity)

        self._display_width = widgets.NumericField(
            "Display width", 240, 1920, 720, suffix=" px"
        )
        card.add(self._display_width)
        return card

    def _build_data_card(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("Data"))
        self._database_label = QLabel("n/a")
        self._database_label.setObjectName("MonoValue")
        self._database_label.setWordWrap(True)
        self._database_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        card.add(self._database_label)

        self._settings_label = QLabel("n/a")
        self._settings_label.setObjectName("MonoValue")
        self._settings_label.setWordWrap(True)
        self._settings_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        card.add(self._settings_label)
        return card

    def load_values(self, config: Any) -> None:
        self._threshold.set_value(to_percentage(config.recognition.similarity_threshold))
        self._ambiguity.set_value(round(config.recognition.ambiguity_margin * 100))
        self._confirmation.set_value(config.recognition.confirmation_hits)
        self._detection_interval.set_value(config.recognition.detection_interval)
        self._embedding_interval.set_value(config.recognition.embedding_interval_seconds)
        self._max_faces.set_value(config.recognition.max_faces)

        self._samples.set_value(config.registration.samples)
        self._consistency.set_value(round(config.registration.min_sample_consistency * 100))
        self._duplicate.set_value(round(config.registration.duplicate_similarity_threshold * 100))
        self._capture_interval.set_value(config.registration.capture_interval_seconds)

        self._min_face.set_value(config.quality.min_face_size_px)
        self._min_det_score.set_value(round(config.quality.min_det_score * 100))
        self._blur.set_value(round(config.quality.blur_threshold))
        self._brightness_min.set_value(round(config.quality.brightness_min))
        self._brightness_max.set_value(round(config.quality.brightness_max))
        self._max_yaw.set_value(round(config.quality.max_yaw_ratio * 100))

        self._liveness_mode.set_value(config.liveness.mode)
        self._liveness_observations.set_value(config.liveness.min_observations)
        self._liveness_sensitivity.set_value(round(config.liveness.sensitivity * 100))

        self._attendance_enabled.set_checked(config.attendance.enabled)
        self._work_start.set_value(config.attendance.work_start)
        self._late_minutes.set_value(config.attendance.late_after_minutes)
        self._refresh_seconds.set_value(config.attendance.refresh_seconds)
        self._require_live.set_checked(config.attendance.require_live_face)

        self._camera_index.set_value(config.camera.index)
        self._camera_width.set_value(config.camera.width)
        self._camera_height.set_value(config.camera.height)
        self._camera_backend.set_value(config.camera.backend)
        self._camera_fps.set_value(config.camera.requested_fps)

        self._profile.set_value(config.performance.profile)
        self._det_size.set_value(str(config.model.det_size[0]))
        self._intra_threads.set_value(config.model.intra_op_threads)

        self._style.set_value(config.ui.style)
        self._animations.set_checked(config.ui.animations)
        self._show_fps.set_checked(config.ui.show_fps)
        self._show_similarity.set_checked(config.ui.show_similarity)
        self._display_width.set_value(config.ui.display_max_width)

        self._database_label.setText(f"Database   {config.database_path}")
        self._settings_label.setText(f"Settings   {config.settings_path}")
        self._banner.clear_message()

    def collect(self) -> dict[str, Any]:
        """Build the override mapping from the current form values."""
        det_size = int(self._det_size.value())
        return {
            "recognition": {
                "similarity_threshold": self._threshold.value() / 100.0,
                "ambiguity_margin": self._ambiguity.value() / 100.0,
                "confirmation_hits": self._confirmation.value(),
                "detection_interval": self._detection_interval.value(),
                "embedding_interval_seconds": self._embedding_interval.value(),
                "max_faces": self._max_faces.value(),
            },
            "registration": {
                "samples": self._samples.value(),
                "min_sample_consistency": self._consistency.value() / 100.0,
                "duplicate_similarity_threshold": self._duplicate.value() / 100.0,
                "capture_interval_seconds": self._capture_interval.value(),
            },
            "quality": {
                "min_face_size_px": self._min_face.value(),
                "min_det_score": self._min_det_score.value() / 100.0,
                "blur_threshold": float(self._blur.value()),
                "brightness_min": float(self._brightness_min.value()),
                "brightness_max": float(self._brightness_max.value()),
                "max_yaw_ratio": self._max_yaw.value() / 100.0,
            },
            "liveness": {
                "mode": self._liveness_mode.value(),
                "min_observations": self._liveness_observations.value(),
                "sensitivity": self._liveness_sensitivity.value() / 100.0,
            },
            "attendance": {
                "enabled": self._attendance_enabled.is_checked(),
                "work_start": self._work_start.value(),
                "late_after_minutes": self._late_minutes.value(),
                "refresh_seconds": self._refresh_seconds.value(),
                "require_live_face": self._require_live.is_checked(),
            },
            "camera": {
                "index": self._camera_index.value(),
                "width": self._camera_width.value(),
                "height": self._camera_height.value(),
                "backend": self._camera_backend.value(),
                "requested_fps": self._camera_fps.value(),
            },
            "model": {
                "det_size": [det_size, det_size],
                "intra_op_threads": self._intra_threads.value(),
            },
            "performance": {"profile": self._profile.value()},
            "ui": {
                "style": self._style.value(),
                "animations": self._animations.is_checked(),
                "show_fps": self._show_fps.is_checked(),
                "show_similarity": self._show_similarity.is_checked(),
                "display_max_width": self._display_width.value(),
            },
        }

    def _restart_fields(self) -> tuple[tuple[str, Any, Any], ...]:
        config = self._context.config
        overrides = self.collect()
        return (
            ("camera index", config.camera.index, overrides["camera"]["index"]),
            ("camera resolution", (config.camera.width, config.camera.height),
             (overrides["camera"]["width"], overrides["camera"]["height"])),
            ("camera backend", config.camera.backend, overrides["camera"]["backend"]),
            ("detector size", list(config.model.det_size), overrides["model"]["det_size"]),
            ("inference threads", config.model.intra_op_threads,
             overrides["model"]["intra_op_threads"]),
        )

    def save(self) -> None:
        overrides = self.collect()
        changed_at_restart = [
            name
            for name, before, after in self._restart_fields()
            if before != after
        ]
        try:
            self._context.save_settings(overrides)
        except Exception as exc:  # noqa: BLE001 - surfaced in the banner
            self._banner.show_message(f"Could not save settings: {exc}", "error")
            return

        message = "Settings saved."
        if changed_at_restart:
            message += " Restart the app for: " + ", ".join(changed_at_restart) + "."
        self._banner.show_message(message, "success")
        self.load_values(self._context.config)

    def reset(self) -> None:
        try:
            self._context.reset_settings()
        except Exception as exc:  # noqa: BLE001 - surfaced in the banner
            self._banner.show_message(f"Could not reset settings: {exc}", "error")
            return
        self.load_values(self._context.config)
        self._banner.show_message("Settings reset to the shipped defaults.", "success")
