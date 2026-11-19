"""Reusable widgets shared by every page.

Each widget owns one thing and exposes a small API, so pages describe layout
instead of repeating stylesheet details.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ClassVar

import numpy as np
from PySide6.QtCore import QLocale, QRect, QSize, Qt, QThread, QTime, Signal
from PySide6.QtGui import QFont, QImage, QPainter, QPaintEvent, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from .theme import HEADING_FONT_SIZE_PT, MONO_FAMILY


def to_qimage_shared(image: np.ndarray) -> tuple[QImage, np.ndarray]:
    """Wrap a frame without copying it; the caller must keep the buffer alive.

    Copying a 640x480 frame costs about 1 MB per paint, which is real work at 30 to
    60 frames per second on a small board. The camera hands out a fresh array per
    frame, so holding a reference in the widget is enough to keep it valid.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot convert an empty frame to a QImage.")
    contiguous = np.ascontiguousarray(image)
    height, width = contiguous.shape[:2]
    if contiguous.ndim == 2:
        fmt = QImage.Format.Format_Grayscale8
        stride = width
    else:
        channels = contiguous.shape[2]
        fmt = QImage.Format.Format_BGR888 if channels == 3 else QImage.Format.Format_RGBA8888
        stride = width * channels
    return QImage(contiguous.data, width, height, stride, fmt), contiguous


class Worker(QThread):
    """Run one blocking call off the UI thread and report the result.

    Loading the model and opening the camera both take seconds; doing that on the
    GUI thread freezes the window and makes the app look hung.
    """

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, action: Callable[[], object], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._action = action

    def run(self) -> None:
        try:
            self.succeeded.emit(self._action())
        except Exception as exc:  # noqa: BLE001 - the message is shown in the UI
            self.failed.emit(str(exc))


class Card(QFrame):
    """Bordered container with a vertical layout."""

    def __init__(self, spacing: int = 10, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 14, 16, 14)
        self._layout.setSpacing(spacing)

    def body(self) -> QVBoxLayout:
        return self._layout

    def add(self, widget: QWidget) -> QWidget:
        self._layout.addWidget(widget)
        return widget

class PageHeader(QWidget):
    """Page title with an optional one-line explanation."""

    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._title = QLabel(title)
        self._title.setObjectName("PageTitle")
        layout.addWidget(self._title)
        self._subtitle = QLabel(subtitle)
        self._subtitle.setObjectName("PageSubtitle")
        self._subtitle.setWordWrap(True)
        self._subtitle.setVisible(bool(subtitle))
        layout.addWidget(self._subtitle)


class StatTile(Card):
    """A single number with its label and a short caption."""

    def __init__(
        self, label: str, value: str = "n/a", caption: str = "", parent: QWidget | None = None
    ) -> None:
        super().__init__(spacing=2, parent=parent)
        self.setObjectName("Tile")
        self._label = QLabel(label)
        self._label.setObjectName("TileLabel")
        self._value = QLabel(value)
        self._value.setObjectName("TileValue")
        self._caption = QLabel(caption)
        self._caption.setObjectName("Hint")
        self._caption.setWordWrap(True)
        self._caption.setVisible(bool(caption))
        self.body().addWidget(self._label)
        self.body().addWidget(self._value)
        self.body().addWidget(self._caption)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, value: str, caption: str | None = None) -> None:
        self._value.setText(value)
        if caption is not None:
            self._caption.setText(caption)
            self._caption.setVisible(bool(caption))


class Badge(QLabel):
    """Small status chip."""

    def __init__(self, text: str = "", accent: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("BadgeAccent" if accent else "Badge")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(max(20, self.fontMetrics().height() + 8))


def scroll_area(
    inner: QWidget,
    *,
    min_width: int = 0,
    max_width: int | None = None,
    horizontal: bool = False,
) -> QScrollArea:
    """Wrap a panel so its content stays reachable when the window is short.

    A page whose minimum height is larger than the screen gets its children squeezed
    until labels and buttons are clipped, which is exactly what a scroll area avoids:
    the panel keeps its natural height and the scrollbar appears instead.
    """
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(inner)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    area.setHorizontalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAsNeeded if horizontal else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    area.viewport().setAutoFillBackground(False)
    inner.setAutoFillBackground(False)
    if min_width:
        area.setMinimumWidth(min_width)
    if max_width is not None:
        area.setMaximumWidth(max_width)
    return area


class Banner(QLabel):
    """Inline message with a severity, used for errors and confirmations.

    Colours live in the stylesheet (`QLabel#BannerError` and friends), so the light
    and dark themes each get a readable pairing instead of one hard-coded set.
    """

    LEVELS: ClassVar[dict[str, str]] = {
        "info": "BannerInfo",
        "success": "BannerSuccess",
        "warning": "BannerWarning",
        "error": "BannerError",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("BannerInfo")
        self.setWordWrap(True)
        self.setVisible(False)

    def show_message(self, text: str, level: str = "info") -> None:
        name = self.LEVELS.get(level, "BannerInfo")
        # The camera pages call this on every result, so the repolish below would
        # otherwise run many times per second for the same message.
        if text == self.text() and name == self.objectName() and self.isVisible():
            return
        self.setObjectName(name)
        self.style().unpolish(self)
        self.style().polish(self)
        self.setText(text)
        self.setVisible(bool(text))

    def clear_message(self) -> None:
        self.setText("")
        self.setVisible(False)


class SliderField(QWidget):
    """Slider with a live formatted value and a change callback."""

    def __init__(
        self,
        label: str,
        minimum: int,
        maximum: int,
        value: int,
        formatter: Callable[[int], str],
        hint: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._formatter = formatter
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(12)
        self._label = QLabel(label)
        self._label.setMinimumWidth(180)
        header.addWidget(self._label)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(minimum, maximum)
        self._slider.setValue(value)
        self._slider.setSingleStep(1)
        header.addWidget(self._slider, 1)
        self._value = QLabel(formatter(value))
        self._value.setObjectName("MonoValue")
        self._value.setMinimumWidth(72)
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self._value)
        outer.addLayout(header)

        if hint:
            note = QLabel(hint)
            note.setObjectName("Hint")
            note.setWordWrap(True)
            note.setContentsMargins(192, 0, 0, 0)
            outer.addWidget(note)

        self._slider.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self, value: int) -> None:
        self._value.setText(self._formatter(value))

    def value(self) -> int:
        return self._slider.value()

    def set_value(self, value: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(value)
        self._value.setText(self._formatter(value))
        self._slider.blockSignals(False)


class NumericField(QWidget):
    """Integer spin box with unit suffix."""

    def __init__(
        self,
        label: str,
        minimum: int,
        maximum: int,
        value: int,
        suffix: str = "",
        hint: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._spin = QSpinBox()
        self._spin.setRange(minimum, maximum)
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

    def value(self) -> int:
        return self._spin.value()

    def set_value(self, value: int) -> None:
        self._spin.setValue(value)


class TimeField(QWidget):
    """Time of day in ``HH:MM``, 24 hour clock."""

    def __init__(
        self, label: str, value: str = "08:00", hint: str = "", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._edit = QTimeEdit()
        self._edit.setDisplayFormat("HH:mm")
        self._edit.setLocale(QLocale.c())
        self._edit.setTime(QTime.fromString(value, "HH:mm"))
        self._edit.setMaximumWidth(140)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(3)
        row = QHBoxLayout()
        row.setSpacing(12)
        text = QLabel(label)
        text.setMinimumWidth(180)
        text.setWordWrap(True)
        row.addWidget(text, 0)
        row.addWidget(self._edit, 0)
        row.addStretch(1)
        outer.addLayout(row)
        if hint:
            note = QLabel(hint)
            note.setObjectName("Hint")
            note.setWordWrap(True)
            note.setContentsMargins(192, 0, 0, 0)
            outer.addWidget(note)

    def value(self) -> str:
        return self._edit.time().toString("HH:mm")

    def set_value(self, value: str) -> None:
        parsed = QTime.fromString(value, "HH:mm")
        if parsed.isValid():
            self._edit.setTime(parsed)


class ChoiceField(QWidget):
    """Combo box bound to string values."""

    def __init__(
        self,
        label: str,
        options: Sequence[tuple[str, str]],
        current: str,
        hint: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._combo = QComboBox()
        for value, text in options:
            self._combo.addItem(text, value)
        self._combo.setMaximumWidth(220)
        index = self._combo.findData(current)
        if index >= 0:
            self._combo.setCurrentIndex(index)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(3)
        row = QHBoxLayout()
        row.setSpacing(12)
        text = QLabel(label)
        text.setMinimumWidth(180)
        text.setWordWrap(True)
        row.addWidget(text, 0)
        row.addWidget(self._combo, 1)
        row.addStretch(0)
        outer.addLayout(row)
        if hint:
            note = QLabel(hint)
            note.setObjectName("Hint")
            note.setWordWrap(True)
            note.setContentsMargins(192, 0, 0, 0)
            outer.addWidget(note)

    def value(self) -> str:
        return str(self._combo.currentData())

    def set_value(self, value: str) -> None:
        index = self._combo.findData(value)
        if index >= 0:
            self._combo.setCurrentIndex(index)


class ToggleField(QWidget):
    """Checkbox with a caption."""

    def __init__(
        self, label: str, checked: bool, hint: str = "", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        self._check = QCheckBox(label)
        self._check.setChecked(checked)
        outer.addWidget(self._check)
        if hint:
            note = QLabel(hint)
            note.setObjectName("Hint")
            note.setWordWrap(True)
            note.setContentsMargins(24, 0, 0, 0)
            outer.addWidget(note)

    def is_checked(self) -> bool:
        return self._check.isChecked()

    def set_checked(self, checked: bool) -> None:
        self._check.setChecked(checked)


def primary_button(text: str, parent: QWidget | None = None) -> QPushButton:
    button = QPushButton(text, parent)
    button.setObjectName("Primary")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    _fit_text(button, text)
    return button


def secondary_button(text: str, parent: QWidget | None = None) -> QPushButton:
    button = QPushButton(text, parent)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    _fit_text(button, text)
    return button


def danger_button(text: str, parent: QWidget | None = None) -> QPushButton:
    button = QPushButton(text, parent)
    button.setObjectName("Danger")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    _fit_text(button, text)
    return button


def _fit_text(button: QPushButton, text: str) -> None:
    """Reserve enough room for the label.

    With a stylesheet applied, a QPushButton reports a hint smaller than its text
    plus padding, and a tight layout then clips the label. The height matters as
    much as the width: without a minimum, a squeezed layout renders the button as a
    coloured bar with no text at all.
    """
    metrics = button.fontMetrics()
    button.setMinimumWidth(metrics.horizontalAdvance(text) + 40)
    button.setMinimumHeight(max(30, metrics.height() + 16))


class SectionTitle(QLabel):
    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("SectionTitle")
        font = self.font()
        font.setPointSize(HEADING_FONT_SIZE_PT)
        font.setWeight(QFont.Weight.DemiBold)
        self.setFont(font)


class VideoView(QFrame):
    """Paints the newest camera frame, letterboxed, with an optional caption."""

    def __init__(self, placeholder: str = "Camera is off", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("VideoFrame")
        self.setMinimumSize(QSize(320, 240))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image: QImage | None = None
        self._buffer: np.ndarray | None = None
        self._placeholder = placeholder
        self._caption = ""

    def set_frame(self, image: np.ndarray, caption: str = "") -> None:
        self._image, self._buffer = to_qimage_shared(image)
        self._caption = caption
        self.update()

    def clear_frame(self, placeholder: str | None = None) -> None:
        self._image = None
        self._buffer = None
        self._caption = ""
        if placeholder is not None:
            self._placeholder = placeholder
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        # Let the stylesheet paint the themed background and the 1 px border first.
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        area = self.contentsRect().adjusted(1, 1, -1, -1)

        if self._image is None:
            painter.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, self._placeholder)
            painter.end()
            return

        target = self._scaled_rect(area, self._image)
        painter.drawImage(target, self._image)
        if self._caption:
            painter.setPen(self.palette().color(QPalette.ColorRole.Text))
            font = painter.font()
            font.setPointSize(11)
            font.setFamily(MONO_FAMILY)
            painter.setFont(font)
            painter.drawText(
                target.adjusted(12, 10, -12, -10),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                self._caption,
            )
        painter.end()

    @staticmethod
    def _scaled_rect(area: QRect, image: QImage) -> QRect:
        scaled = image.size().scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRect(
            area.x() + (area.width() - scaled.width()) // 2,
            area.y() + (area.height() - scaled.height()) // 2,
            scaled.width(),
            scaled.height(),
        )
