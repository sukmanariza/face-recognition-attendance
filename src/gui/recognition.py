"""Live recognition page.

Shows the newest frame, the best match for each face as a percentage, and what the
pipeline costs while it runs. Similarity is displayed as a percentage because it is
easier to read at a glance; the README is explicit that it measures distance in
embedding space and is not a probability of being the same person.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from src.attendance import AttendanceDecision
from src.face_recognizer import FaceState, FrameRecognition
from src.pipeline import CameraSource, DisplayPump, Mode, RecognitionPipeline
from src.presentation import PresentationVerdict, SpoofLevel

from . import widgets
from .context import AppContext


class RecognitionPage(QWidget):
    """Camera preview plus the live match readout."""

    navigate = Signal(str)

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._pipeline: RecognitionPipeline | None = None
        self._pump: DisplayPump | None = None
        self._worker: widgets.Worker | None = None
        self._last_face_key: tuple[str, str] | None = None

        self._timer = QTimer(self)
        # 8 ms is a poll, not a cap: the page paints only when a new result exists,
        # so the display follows the camera instead of being limited to 30 FPS.
        self._timer.setInterval(8)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self._refresh_threshold()
        context.config_changed.connect(self._refresh_threshold)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(16)

        self._header = widgets.PageHeader(
            "Face recognition",
            "Names appear after the same identity wins several consecutive checks, "
            "which stops a single doubtful frame from flipping the label.",
        )
        root.addWidget(self._header)

        body = QHBoxLayout()
        body.setSpacing(16)

        self._video = widgets.VideoView("Press Start to open the camera")
        body.addWidget(self._video, 3)

        panel = QVBoxLayout()
        panel.setSpacing(12)
        panel.addWidget(self._build_match_card())
        panel.addWidget(self._build_stats_card())
        panel.addStretch(1)
        panel_holder = QWidget()
        panel_holder.setLayout(panel)
        panel_holder.setMinimumWidth(320)
        body.addWidget(panel_holder, 2)

        root.addLayout(body, 1)
        self._banner = widgets.Banner()
        root.addWidget(self._banner)

    def _build_match_card(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("Best match"))

        heading_row = QHBoxLayout()
        heading_row.setSpacing(8)
        self._name = QLabel("No face")
        self._name.setObjectName("FaceName")
        self._name.setWordWrap(True)
        self._name.setMinimumHeight(26)
        heading_row.addWidget(self._name, 1)
        self._state_badge = widgets.Badge("idle")
        self._state_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading_row.addWidget(self._state_badge, 0, Qt.AlignmentFlag.AlignTop)
        holder = QWidget()
        holder.setLayout(heading_row)
        card.add(holder)

        self._identity = QLabel("waiting for a face")
        self._identity.setObjectName("Hint")
        card.add(self._identity)

        self._similarity = QProgressBar()
        self._similarity.setRange(0, 100)
        self._similarity.setValue(0)
        self._similarity.setTextVisible(False)
        self._similarity.setFixedHeight(8)
        card.add(self._similarity)

        self._similarity_text = QLabel("Similarity not measured yet")
        self._similarity_text.setObjectName("Muted")
        card.add(self._similarity_text)

        presentation_row = QHBoxLayout()
        presentation_row.setSpacing(10)
        presentation_key = QLabel("Presentation")
        presentation_key.setObjectName("Muted")
        self._presentation = QLabel("Not checked")
        self._presentation.setObjectName("ToneMuted")
        self._presentation.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        presentation_row.addWidget(presentation_key, 1)
        presentation_row.addWidget(self._presentation, 0)
        presentation_holder = QWidget()
        presentation_holder.setLayout(presentation_row)
        card.add(presentation_holder)

        self._presentation_detail = QLabel("")
        self._presentation_detail.setObjectName("Hint")
        self._presentation_detail.setWordWrap(True)
        card.add(self._presentation_detail)

        attendance_row = QHBoxLayout()
        attendance_row.setSpacing(10)
        attendance_key = QLabel("Attendance")
        attendance_key.setObjectName("Muted")
        self._attendance = QLabel("not recorded")
        self._attendance.setObjectName("ToneMuted")
        self._attendance.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        attendance_row.addWidget(attendance_key, 1)
        attendance_row.addWidget(self._attendance, 0)
        attendance_holder = QWidget()
        attendance_holder.setLayout(attendance_row)
        card.add(attendance_holder)

        self._reason = QLabel("")
        self._reason.setObjectName("Hint")
        self._reason.setWordWrap(True)
        card.add(self._reason)

        self._threshold_text = QLabel("")
        self._threshold_text.setObjectName("Hint")
        self._threshold_text.setWordWrap(True)
        card.add(self._threshold_text)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        self._start_button = widgets.primary_button("Start")
        self._start_button.clicked.connect(self.start)
        self._stop_button = widgets.secondary_button("Stop")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self.stop)
        buttons.addWidget(self._start_button)
        buttons.addWidget(self._stop_button)
        buttons.addStretch(1)
        buttons_holder = QWidget()
        buttons_holder.setLayout(buttons)
        card.body().addWidget(buttons_holder)
        return card

    def _build_stats_card(self) -> QWidget:
        card = widgets.Card()
        card.add(widgets.SectionTitle("This session"))
        self._stats_labels: dict[str, QLabel] = {}
        for name in ("Capture", "Inference", "Frame time", "Faces", "Dropped frames"):
            row = QHBoxLayout()
            row.setSpacing(10)
            key = QLabel(name)
            key.setObjectName("Muted")
            value = QLabel("n/a")
            value.setObjectName("MonoValue")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(key, 1)
            row.addWidget(value, 0)
            holder = QWidget()
            holder.setLayout(row)
            card.add(holder)
            self._stats_labels[name] = value
        return card

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_threshold()

    def _refresh_threshold(self) -> None:
        config = self._context.config
        threshold = config.recognition.similarity_threshold
        hits = config.recognition.confirmation_hits
        text = (
            f"Threshold {round(threshold * 100)}% similarity, confirmed over {hits} "
            "consecutive checks."
        )
        if config.recognition.ambiguity_margin > 0:
            text += (
                f" A runner-up within {config.recognition.ambiguity_margin:.2f} of the winner "
                "is treated as ambiguous."
            )
        self._threshold_text.setText(text)

    def start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if not self._context.identity_count():
            self._banner.show_message(
                "No identities are registered yet. Record a face first.", "warning"
            )
            return

        self._start_button.setEnabled(False)
        self._banner.show_message("Preparing the model and the camera…", "info")
        application = self._context.application

        def action() -> object:
            application.detector.load()
            session = application.create_recognition_session()
            camera = application.create_camera()
            camera.open()
            pipeline = RecognitionPipeline(application.config, CameraSource(camera))
            pipeline.set_recognition_session(session)
            pipeline.set_mode(Mode.RECOGNIZE)
            pipeline.start()
            return pipeline

        self._worker = widgets.Worker(action, self)
        self._worker.succeeded.connect(self._on_started)
        self._worker.failed.connect(self._on_start_failed)
        self._worker.start()

    def _on_started(self, payload: object) -> None:
        if not isinstance(payload, RecognitionPipeline):
            self._on_start_failed("The pipeline could not be created.")
            return
        self._pipeline = payload
        self._pump = DisplayPump(payload)
        self._timer.start()
        self._stop_button.setEnabled(True)
        self._banner.clear_message()

    def _on_start_failed(self, message: str) -> None:
        self._start_button.setEnabled(True)
        self._banner.show_message(f"Could not start: {message}", "error")

    def stop(self) -> None:
        self._timer.stop()
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None
        self._pump = None
        self._video.clear_frame("Press Start to open the camera")
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._reset_readout()

    def shutdown(self) -> None:
        self.stop()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)
        self._worker = None

    def _tick(self) -> None:
        if self._pump is None or self._pipeline is None:
            return
        output = self._pump.poll()
        if output is None:
            return
        self._pipeline.note_displayed()

        recognition = output.recognition
        if recognition is None:
            self._video.set_frame(output.frame.image)
            return

        caption = self._caption_for(recognition)
        self._video.set_frame(output.frame.image, caption=caption)
        self._update_match(recognition)
        self._update_stats(recognition)

    def _caption_for(self, recognition: FrameRecognition) -> str:
        parts = [f"{recognition.face_count} face(s)", f"{recognition.inference_ms:.0f} ms"]
        if self._context.config.ui.show_fps:
            parts.append(f"{recognition.fps:.1f} FPS")
        return "   ".join(parts)

    def _update_match(self, recognition: FrameRecognition) -> None:
        if not recognition.faces:
            self._reset_readout()
            return

        shown = recognition.matched or recognition.faces
        face = shown[0]
        key = (face.display_name, face.display_id)
        if key != self._last_face_key:
            self._name.setText(face.display_name)
            self._identity.setText(face.display_id if face.accepted else "not confirmed")
            self._last_face_key = key

        if face.state is FaceState.MATCHED:
            self._state_badge.setText("matched")
            self._state_badge.setObjectName("BadgeAccent")
        elif face.state is FaceState.VERIFYING:
            self._state_badge.setText("confirming")
            self._state_badge.setObjectName("Badge")
        else:
            self._state_badge.setText("no match")
            self._state_badge.setObjectName("Badge")
        self._state_badge.style().unpolish(self._state_badge)
        self._state_badge.style().polish(self._state_badge)

        percentage = face.percentage
        self._similarity.setValue(percentage)
        if self._context.config.ui.show_similarity:
            self._similarity_text.setText(f"Similarity {percentage}%  ({face.similarity:.3f})")
        else:
            self._similarity_text.setText("Similarity hidden in Settings")
        self._show_presentation(face.presentation, multiple=recognition.face_count > 1)
        self._show_attendance(face.attendance)
        self._reason.setText(face.reason)
        self._reason.setVisible(bool(face.reason))

    def _show_presentation(self, verdict: PresentationVerdict | None, multiple: bool) -> None:
        config = self._context.config.liveness
        if not config.enabled:
            self._set_presentation("Off in Settings", "ToneMuted", "")
            return
        if multiple:
            self._set_presentation(
                "Not checked",
                "ToneMuted",
                "The check needs one face at a time, because it compares that face with itself.",
            )
            return
        if verdict is None:
            self._set_presentation("Not checked", "ToneMuted", "")
            return
        tone = {
            SpoofLevel.GENUINE: "ToneSuccess",
            SpoofLevel.SUSPECTED: "ToneDanger",
        }.get(verdict.level, "ToneMuted")
        self._set_presentation(verdict.label, tone, verdict.detail)

    def _set_presentation(self, text: str, tone: str, detail: str) -> None:
        self._presentation.setText(text)
        self._presentation.setObjectName(tone)
        self._presentation.style().unpolish(self._presentation)
        self._presentation.style().polish(self._presentation)
        self._presentation_detail.setText(detail)
        self._presentation_detail.setVisible(bool(detail))

    def _show_attendance(self, decision: AttendanceDecision | None) -> None:
        if not self._context.config.attendance.enabled:
            self._set_attendance("off in Settings", "ToneMuted", "")
            return
        mark = decision.mark if decision is not None else None
        if mark is None:
            reason = decision.reason if decision is not None else "Confirm a match to record it."
            self._set_attendance("not recorded", "ToneMuted", reason)
            return
        entry = mark.entry
        tone = "ToneSuccess" if entry.status == "present" else "ToneWarning"
        self._set_attendance(
            f"{entry.status} {entry.first_seen[11:19]}",
            tone,
            f"{entry.sightings} sighting(s) today, last at {entry.last_seen[11:19]}.",
        )

    def _set_attendance(self, text: str, tone: str, detail: str) -> None:
        self._attendance.setText(text)
        self._attendance.setObjectName(tone)
        self._attendance.style().unpolish(self._attendance)
        self._attendance.style().polish(self._attendance)
        self._attendance.setToolTip(detail)

    def _reset_readout(self) -> None:
        self._last_face_key = None
        self._name.setText("No face")
        self._identity.setText("waiting for a face")
        self._state_badge.setText("idle")
        self._similarity.setValue(0)
        self._similarity_text.setText("Similarity not measured yet")
        self._set_presentation("Not checked", "ToneMuted", "")
        self._set_attendance("not recorded", "ToneMuted", "")
        self._reason.setText("")
        self._reason.setVisible(False)
        for label in self._stats_labels.values():
            label.setText("n/a")

    def _update_stats(self, recognition: FrameRecognition) -> None:
        pipeline = self._pipeline
        if pipeline is None:
            return
        stats = pipeline.stats()
        self._stats_labels["Capture"].setText(f"{stats.capture_fps:.1f} FPS")
        self._stats_labels["Inference"].setText(f"{stats.inference_fps:.1f} FPS")
        self._stats_labels["Frame time"].setText(f"{recognition.inference_ms:.0f} ms")
        self._stats_labels["Faces"].setText(str(recognition.face_count))
        self._stats_labels["Dropped frames"].setText(str(stats.dropped_frames))
