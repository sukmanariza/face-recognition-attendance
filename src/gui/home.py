"""Overview page: what the app knows right now, and the two things to do next."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from src.database import DatabaseError
from src.similarity import to_percentage

from . import widgets
from .context import AppContext


class HomePage(QWidget):
    """Landing page with live counters and shortcuts."""

    navigate = Signal(str)

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)

        layout.addWidget(
            widgets.PageHeader(
                "Overview",
                "Everything runs on this machine: no network calls, no cloud service, "
                "and the face data never leaves the database file.",
            )
        )

        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        self._identities = widgets.StatTile("Registered identities", "0")
        self._present = widgets.StatTile("Present today", "0")
        self._camera = widgets.StatTile("Camera", "n/a")
        self._performance = widgets.StatTile("Performance", context.tier.title())
        tiles.addWidget(self._identities)
        tiles.addWidget(self._present)
        tiles.addWidget(self._camera)
        tiles.addWidget(self._performance)
        layout.addLayout(tiles)

        layout.addWidget(self._build_actions())
        layout.addWidget(self._build_environment())
        layout.addStretch(1)

        context.gallery_changed.connect(self.refresh)
        context.config_changed.connect(self.refresh)
        self.refresh()

    def _build_actions(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("Start here"))
        card.add(
            self._muted(
                "Recording a face takes about ten seconds and needs five guided poses. "
                "Recognition needs at least one identity in the database."
            )
        )

        row = QHBoxLayout()
        row.setSpacing(10)
        record = widgets.primary_button("Record a new face")
        record.clicked.connect(lambda: self.navigate.emit("enrollment"))
        recognize = widgets.secondary_button("Open recognition")
        recognize.clicked.connect(lambda: self.navigate.emit("recognition"))
        manage = widgets.secondary_button("Manage identities")
        manage.clicked.connect(lambda: self.navigate.emit("users"))
        attendance = widgets.secondary_button("Today's attendance")
        attendance.clicked.connect(lambda: self.navigate.emit("attendance"))
        row.addWidget(record)
        row.addWidget(recognize)
        row.addWidget(manage)
        row.addWidget(attendance)
        row.addStretch(1)

        holder = QWidget()
        holder.setLayout(row)
        card.body().addWidget(holder)
        return card

    def _build_environment(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("Environment"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(7)
        grid.setColumnStretch(1, 1)

        self._environment_labels: dict[str, QLabel] = {}
        rows = (
            "Hardware",
            "Performance profile",
            "Recognition threshold",
            "Storage",
            "Database",
            "Configuration",
        )
        for index, name in enumerate(rows):
            key = QLabel(name)
            key.setObjectName("Muted")
            value = QLabel("n/a")
            value.setObjectName("MonoValue")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(key, index, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(value, index, 1)
            self._environment_labels[name] = value

        holder = QWidget()
        holder.setLayout(grid)
        card.body().addWidget(holder)
        return card

    def _muted(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("Muted")
        label.setWordWrap(True)
        return label

    def refresh(self) -> None:
        config = self._context.config

        try:
            database = self._context.database()
            identities = database.count_employees()
            self._identities.set_value(
                str(identities),
                "Ready for recognition" if identities else "Record a face to begin",
            )
        except DatabaseError as exc:
            self._identities.set_value("?", f"Database unavailable: {exc}")

        if config.attendance.enabled:
            present = self._context.application.attendance.present_count()
            self._present.set_value(
                str(present),
                f"Start {config.attendance.work_start}, late after "
                f"{config.attendance.late_after_minutes} min",
            )
        else:
            self._present.set_value("off", "Attendance is disabled in Settings")

        camera = config.camera
        self._camera.set_value(
            f"{camera.width}x{camera.height}",
            f"Camera index {camera.index}, requested {camera.requested_fps} FPS",
        )
        self._performance.set_value(
            self._context.tier.title(),
            f"Tuned for {config.model.det_size[0]}x{config.model.det_size[1]} detection",
        )

        self._environment_labels["Hardware"].setText(self._context.hardware_summary())
        self._environment_labels["Performance profile"].setText(
            f"{self._context.tier} (from performance.profile = {config.performance.profile})"
        )
        self._environment_labels["Recognition threshold"].setText(
            f"{to_percentage(config.recognition.similarity_threshold)}% similarity"
        )
        self._environment_labels["Database"].setText(str(config.database_path))
        self._environment_labels["Storage"].setText(
            "encrypted (AES-GCM)" if config.database.encrypt else "plain text"
        )
        self._environment_labels["Configuration"].setText(str(config.source_path))
