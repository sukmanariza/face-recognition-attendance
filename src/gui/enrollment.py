"""Staged face enrollment.

The capture is a fixed sequence of poses (frontal, turn left, turn right, tilt,
closer). The next sample is accepted only when the frame matches the current pose
and passes the quality checks, so five samples means five genuinely different
angles instead of five near-identical frames, which is what keeps recognition
stable. The backend does the deciding; this page only displays the state.

Model loading and camera warm-up run on a worker thread, because both block for
seconds and would otherwise freeze the window.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.database import DatabaseError, DuplicateEmployeeError, validate_full_name
from src.enrollment_plan import build_plan
from src.face_registration import (
    FaceRegistrationSession,
    RegistrationError,
    SampleStatus,
)
from src.guidance import checklist_hints
from src.pipeline import CameraSource, DisplayPump, RecognitionPipeline
from src.presentation import PresentationVerdict

from . import widgets
from .context import AppContext


class EnrollmentPage(QWidget):
    """Guided capture plus the form that stores the result."""

    enrollment_saved = Signal()

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._session: FaceRegistrationSession | None = None
        self._pipeline: RecognitionPipeline | None = None
        self._pump: DisplayPump | None = None
        self._worker: widgets.Worker | None = None
        self._stage_labels: list[QLabel] = []
        self._last_prompt = ""
        self._update_id: str | None = None
        self._ready_to_save = False
        self._capturing = False

        self._timer = QTimer(self)
        self._timer.setInterval(8)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self._update_buttons()
        context.config_changed.connect(self._rebuild_plan)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(16)

        root.addWidget(
            widgets.PageHeader(
                "Record a new face",
                "Sit facing the camera in even light. The capture follows five poses and "
                "rejects anything blurry, dark, or turned the wrong way.",
            )
        )

        body = QHBoxLayout()
        body.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(10)
        self._video = widgets.VideoView("Press Prepare to start the camera")
        left.addWidget(self._video, 1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        left.addWidget(self._progress)

        self._prompt = QLabel("Waiting to start.")
        self._prompt.setObjectName("SectionTitle")
        self._prompt.setWordWrap(True)
        left.addWidget(self._prompt)

        self._presentation = QLabel("Presentation: not checked")
        self._presentation.setObjectName("Hint")
        self._presentation.setWordWrap(True)
        left.addWidget(self._presentation)

        left_holder = QWidget()
        left_holder.setLayout(left)
        body.addWidget(left_holder, 3)

        right = QVBoxLayout()
        right.setSpacing(12)
        right.addWidget(self._build_form())
        right.addWidget(self._build_stages())
        right.addWidget(self._build_checklist())
        right.addStretch(1)
        right_holder = QWidget()
        right_holder.setLayout(right)

        panel = QScrollArea()
        panel.setWidgetResizable(True)
        panel.setWidget(right_holder)
        panel.setMinimumWidth(360)
        panel.setMaximumWidth(460)
        panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body.addWidget(panel, 2)

        root.addLayout(body, 1)

        self._banner = widgets.Banner()
        root.addWidget(self._banner)

    def _build_form(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("Identity"))

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(8)

        self._id_field = QLineEdit()
        self._id_field.setReadOnly(True)
        self._name_field = QLineEdit()
        self._name_field.setPlaceholderText("Full name")
        self._name_field.setMaxLength(64)
        self._name_field.textChanged.connect(self._on_name_changed)
        form.addRow("ID", self._id_field)
        form.addRow("Name", self._name_field)

        holder = QWidget()
        holder.setLayout(form)
        card.body().addWidget(holder)

        self._name_hint = QLabel("A name is required: it is what attendance shows.")
        self._name_hint.setObjectName("Hint")
        self._name_hint.setWordWrap(True)
        card.body().addWidget(self._name_hint)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        self._prepare_button = widgets.primary_button("Prepare camera")
        self._prepare_button.clicked.connect(self.prepare)
        self._start_button = widgets.primary_button("Start capture")
        self._start_button.setEnabled(False)
        self._start_button.clicked.connect(self.start_capture)
        self._cancel_button = widgets.secondary_button("Cancel")
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self.cancel_capture)
        self._save_button = widgets.primary_button("Save identity")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self.save)

        first_row = QHBoxLayout()
        first_row.setSpacing(10)
        first_row.addWidget(self._prepare_button, 1)
        first_row_holder = QWidget()
        first_row_holder.setLayout(first_row)
        card.body().addWidget(first_row_holder)

        second_row = QHBoxLayout()
        second_row.setSpacing(10)
        second_row.addWidget(self._start_button, 1)
        second_row.addWidget(self._cancel_button, 1)
        second_row_holder = QWidget()
        second_row_holder.setLayout(second_row)
        card.body().addWidget(second_row_holder)

        card.body().addWidget(self._save_button)
        return card

    def _build_stages(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("Poses"))
        hint = QLabel("Follow the order. Each pose is captured once, from the current frames.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        card.add(hint)
        self._stage_holder = QWidget()
        self._stage_layout = QVBoxLayout(self._stage_holder)
        self._stage_layout.setContentsMargins(0, 0, 0, 0)
        self._stage_layout.setSpacing(4)
        card.body().addWidget(self._stage_holder)
        self._rebuild_plan()
        return card

    def _build_checklist(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("Before you start"))
        for line in checklist_hints(self._context.config.quality):
            label = QLabel(f"·  {line}")
            label.setObjectName("Hint")
            label.setWordWrap(True)
            card.add(label)
        return card

    def _rebuild_plan(self) -> None:
        while self._stage_layout.count():
            item = self._stage_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        self._stage_labels.clear()

        plan = build_plan(self._context.config.registration)
        for index, stage in enumerate(plan, start=1):
            label = QLabel(f"○  {index}. {stage.title}")
            label.setObjectName("Muted")
            label.setWordWrap(True)
            self._stage_layout.addWidget(label)
            self._stage_labels.append(label)
        self._reset_stage_marks()

    def _reset_stage_marks(self) -> None:
        plan = build_plan(self._context.config.registration)
        for index, (stage, label) in enumerate(zip(plan, self._stage_labels, strict=False)):
            label.setText(f"○  {index + 1}. {stage.title}")
            label.setObjectName("Muted")
            label.style().unpolish(label)
            label.style().polish(label)

    def _mark_stage(self, collected: int) -> None:
        plan = build_plan(self._context.config.registration)
        for index, (stage, label) in enumerate(zip(plan, self._stage_labels, strict=False)):
            if index < collected:
                label.setText(f"✓  {index + 1}. {stage.title}")
                label.setObjectName("BadgeAccent")
            elif index == collected:
                label.setText(f"→  {index + 1}. {stage.title}")
                label.setObjectName("SectionTitle")
            else:
                label.setText(f"○  {index + 1}. {stage.title}")
                label.setObjectName("Muted")
            label.style().unpolish(label)
            label.style().polish(label)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._update_id is None:
            self._refresh_identity_fields()
        self._update_buttons()

    def _name_problem(self) -> str:
        """Empty when the typed name is acceptable, otherwise the reason."""
        raw = self._name_field.text()
        try:
            validate_full_name(raw)
        except DatabaseError as exc:
            if not raw.strip():
                return "A name is required before capture: it is what attendance shows."
            return str(exc)
        return ""

    def _on_name_changed(self) -> None:
        self._update_buttons()

    def _update_buttons(self) -> None:
        """One place decides what is clickable, so no path can skip the name check.

        A name is required before the capture starts, because the samples are only
        useful when they can be saved under a real person. Leaving the page with a
        finished capture asks for confirmation instead of throwing the samples away.
        """
        problem = self._name_problem()
        self._name_hint.setText(problem or "Shown on the attendance list.")
        self._name_hint.setObjectName("ToneDanger" if problem else "Hint")
        self._name_hint.style().unpolish(self._name_hint)
        self._name_hint.style().polish(self._name_hint)

        camera_ready = self._session is not None
        self._prepare_button.setEnabled(not camera_ready and not self._capturing)
        self._start_button.setEnabled(camera_ready and not problem and not self._capturing)
        self._cancel_button.setEnabled(self._capturing)
        self._save_button.setEnabled(self._ready_to_save and not problem)
        self._name_field.setReadOnly(self._capturing)

    def has_unsaved_capture(self) -> bool:
        """True when a completed capture is sitting in memory, not yet in the database."""
        return self._ready_to_save and self._session is not None

    def start_update(self, employee_id: str, full_name: str) -> None:
        """Switch this page into re-recording mode for an existing identity."""
        self._update_id = employee_id
        self._id_field.setText(employee_id)
        self._name_field.setText(full_name)
        self._name_field.setReadOnly(False)
        self._save_button.setText("Replace samples")
        self._banner.show_message(
            f"Re-recording {employee_id} ({full_name}). The old samples are replaced when "
            "you save.",
            "info",
        )
        self._update_buttons()

    def start_new(self) -> None:
        """Leave re-recording mode and go back to enrolling a new person."""
        self._update_id = None
        self._name_field.clear()
        self._name_field.setReadOnly(False)
        self._save_button.setText("Save identity")
        self._refresh_identity_fields()
        self._update_buttons()

    def _refresh_identity_fields(self) -> None:
        try:
            next_id = self._context.database().next_employee_id()
        except DatabaseError:
            next_id = ""
        self._id_field.setText(next_id)
        self._reset_stage_marks()

    def prepare(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._banner.show_message("Loading the model and opening the camera…", "info")
        self._prepare_button.setEnabled(False)

        application = self._context.application

        def action() -> object:
            application.detector.load()
            session = application.create_registration_session()
            camera = application.create_camera()
            camera.open()
            pipeline = RecognitionPipeline(application.config, CameraSource(camera))
            pipeline.begin_registration(session)
            pipeline.start()
            return session, pipeline

        self._worker = widgets.Worker(action, self)
        self._worker.succeeded.connect(self._on_prepared)
        self._worker.failed.connect(self._on_prepare_failed)
        self._worker.start()

    def _on_prepared(self, payload: object) -> None:
        session, pipeline = payload  # type: ignore[misc]
        self._session = session
        self._pipeline = pipeline
        self._pump = DisplayPump(pipeline)
        self._timer.start()
        self._update_buttons()
        self._banner.show_message("Camera ready. Enter the name, then start the capture.", "success")

    def _on_prepare_failed(self, message: str) -> None:
        self._session = None
        self._update_buttons()
        self._banner.show_message(f"Could not start the camera: {message}", "error")

    def start_capture(self) -> None:
        if self._session is None:
            return
        problem = self._name_problem()
        if problem:
            self._name_hint.setText(problem)
            self._name_field.setFocus()
            self._banner.show_message(problem, "warning")
            return
        self._capturing = True
        self._ready_to_save = False
        self._session.reset()
        self._reset_stage_marks()
        self._progress.setValue(0)
        self._update_buttons()
        self._banner.show_message("Hold still and follow the poses shown on the left.", "info")

    def cancel_capture(self) -> None:
        if self._session is not None:
            self._session.freeze()
        self._stop_stream()
        self._session = None
        self._capturing = False
        self._ready_to_save = False
        self._progress.setValue(0)
        self._reset_stage_marks()
        self._update_buttons()
        self._banner.show_message("Capture cancelled. Nothing was saved.", "warning")

    def _stop_stream(self) -> None:
        self._timer.stop()
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None
        self._pump = None
        self._video.clear_frame("Press Prepare to start the camera")

    def shutdown(self) -> None:
        """Release the camera and the worker; called when the window closes."""
        self._stop_stream()
        self._session = None
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)
        self._worker = None

    def _tick(self) -> None:
        if self._pump is None:
            return
        output = self._pump.poll()
        if output is None:
            return
        attempt = output.attempt
        if attempt is None:
            self._video.set_frame(output.frame.image)
            return

        caption = f"{attempt.collected}/{attempt.required}  {attempt.status.value}"
        self._video.set_frame(output.frame.image, caption=caption)
        self._progress.setValue(int(attempt.progress * 100))
        self._mark_stage(attempt.collected)

        if attempt.prompt != self._last_prompt:
            self._prompt.setText(attempt.prompt)
            self._last_prompt = attempt.prompt

        self._show_presentation(attempt.presentation)

        if attempt.status is SampleStatus.REJECTED and attempt.message:
            self._banner.show_message(attempt.message, "warning")
        elif attempt.status is SampleStatus.CAPTURED:
            self._banner.show_message(
                f"{attempt.collected} of {attempt.required} samples captured.", "success"
            )
        elif attempt.status is SampleStatus.COMPLETE:
            self._on_capture_complete()

    def _show_presentation(self, verdict: PresentationVerdict | None) -> None:
        if not self._context.config.liveness.enabled:
            self._presentation.setText("Presentation check: off in Settings")
            self._presentation.setObjectName("Hint")
        elif verdict is None:
            self._presentation.setText("Presentation: not checked")
            self._presentation.setObjectName("Hint")
        else:
            self._presentation.setText(f"Presentation: {verdict.label} ({verdict.detail})")
            self._presentation.setObjectName(
                "ToneDanger" if verdict.is_suspected else "Hint"
            )
        self._presentation.style().unpolish(self._presentation)
        self._presentation.style().polish(self._presentation)

    def _on_capture_complete(self) -> None:
        if self._session is None:
            return
        self._session.freeze()
        self._stop_stream()
        self._capturing = False
        self._ready_to_save = True
        self._progress.setValue(100)
        self._mark_stage(self._session.collected)
        self._prompt.setText("All samples captured. Review the name, then save.")
        self._update_buttons()
        problem = self._name_problem()
        if problem:
            # The samples are already captured, so ask for the missing name instead
            # of dropping them.
            self._name_field.setFocus()
            self._banner.show_message(
                f"{problem} Type it to save the captured samples.", "warning"
            )
            return
        self._banner.show_message(
            "Capture complete. Saving writes the embeddings to the database.", "success"
        )

    def save(self) -> None:
        if self._session is None:
            return
        problem = self._name_problem()
        if problem:
            self._name_field.setFocus()
            self._banner.show_message(problem, "error")
            return
        employee_id = self._id_field.text().strip()
        full_name = self._name_field.text().strip()
        registrar = self._context.application.create_registrar()
        try:
            samples = self._session.build_samples()
            if self._update_id is not None:
                result = registrar.save_update(
                    employee_id, samples, self._session.min_consistency(), full_name=full_name
                )
            else:
                result = registrar.save_new(
                    employee_id, full_name, samples, self._session.min_consistency()
                )
        except (RegistrationError, DuplicateEmployeeError, DatabaseError) as exc:
            self._banner.show_message(str(exc), "error")
            return

        self._context.invalidate_gallery()
        self._banner.show_message(result.describe(), "success")
        self.enrollment_saved.emit()

        self._session = None
        self._update_id = None
        self._capturing = False
        self._ready_to_save = False
        self._save_button.setText("Save identity")
        self._name_field.clear()
        self._progress.setValue(0)
        self._reset_stage_marks()
        self._prompt.setText("Saved. Record another face or open recognition.")
        self._refresh_identity_fields()
        self._update_buttons()
