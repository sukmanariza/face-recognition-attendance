"""OpenCV overlays for the headless preview (`python main.py recognize`).

The Qt interface draws with QPainter and does not use this module. This exists for
machines without a GUI (a Raspberry Pi over SSH, a headless board) and for
screenshots, so the same information is available without a display server.

Colours are BGR because OpenCV expects them in that order.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import UiConfig

from .face_recognizer import FaceState, FrameRecognition
from .face_registration import SampleAttempt, SampleStatus

_FONT = cv2.FONT_HERSHEY_DUPLEX
_FONT_PLAIN = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_THICKNESS = 1


@dataclass(frozen=True)
class Palette:
    """BGR colours for boxes, labels, and text."""

    background: tuple[int, int, int]
    text: tuple[int, int, int]
    matched: tuple[int, int, int]
    verifying: tuple[int, int, int]
    unknown: tuple[int, int, int]
    accent: tuple[int, int, int]
    panel: tuple[int, int, int]


# Mirrors src/gui/theme.py so the CLI preview and the window look like one app.
_DARK = Palette(
    background=(20, 20, 20),
    text=(230, 234, 237),
    matched=(122, 179, 95),
    verifying=(74, 180, 224),
    unknown=(92, 112, 232),
    accent=(24, 122, 196),
    panel=(34, 36, 38),
)

_LIGHT = Palette(
    background=(243, 242, 245),
    text=(40, 40, 40),
    matched=(61, 102, 38),
    verifying=(9, 86, 127),
    unknown=(34, 47, 143),
    accent=(12, 82, 138),
    panel=(230, 228, 228),
)


def palette_for(ui_config: UiConfig) -> Palette:
    return _LIGHT if ui_config.style == "light" else _DARK


def resize_for_display(image: np.ndarray, max_width: int) -> np.ndarray:
    """Scale down to fit ``max_width``, keeping the aspect ratio."""
    height, width = image.shape[:2]
    if max_width <= 0 or width <= max_width:
        return image
    scale = max_width / float(width)
    return cv2.resize(
        image, (max_width, max(1, round(height * scale))), interpolation=cv2.INTER_AREA
    )


def _label_box(
    image: np.ndarray,
    origin: tuple[int, int],
    text: str,
    colour: tuple[int, int, int],
    palette: Palette,
    scale: float = 0.5,
) -> None:
    (text_width, text_height), baseline = cv2.getTextSize(text, _FONT, scale, _TEXT_THICKNESS)
    x, y = origin
    top = max(0, y - text_height - baseline - 6)
    cv2.rectangle(
        image, (x, top), (x + text_width + 10, y), colour, thickness=-1, lineType=cv2.LINE_AA
    )
    cv2.putText(
        image,
        text,
        (x + 5, y - baseline - 3),
        _FONT,
        scale,
        palette.background,
        _TEXT_THICKNESS,
        cv2.LINE_AA,
    )


def draw_detection_box(
    image: np.ndarray,
    bbox: np.ndarray,
    palette: Palette,
    label: str = "",
    level: str = "unknown",
    thickness: int = 2,
) -> None:
    """Draw one box with a corner-accent style and an optional label."""
    x1, y1, x2, y2 = (round(float(value)) for value in bbox[:4])
    colour = {
        "matched": palette.matched,
        "verifying": palette.verifying,
    }.get(level, palette.unknown)

    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness, lineType=cv2.LINE_AA)
    # Corner ticks make the box read as a frame instead of a static rectangle.
    tick = max(8, min(24, (x2 - x1) // 5, (y2 - y1) // 5))
    for start, end in (
        ((x1, y1), (x1 + tick, y1)),
        ((x1, y1), (x1, y1 + tick)),
        ((x2, y1), (x2 - tick, y1)),
        ((x2, y1), (x2, y1 + tick)),
        ((x1, y2), (x1 + tick, y2)),
        ((x1, y2), (x1, y2 - tick)),
        ((x2, y2), (x2 - tick, y2)),
        ((x2, y2), (x2, y2 - tick)),
    ):
        cv2.line(image, start, end, colour, thickness + 1, cv2.LINE_AA)

    if label:
        _label_box(image, (x1, max(y1 - 4, 18)), label, colour, palette)


def draw_face(
    image: np.ndarray,
    bbox: np.ndarray,
    palette: Palette,
    label: str = "",
    level: str = "unknown",
) -> None:
    """Convenience alias used by the CLI preview."""
    draw_detection_box(image, bbox, palette, label=label, level=level)


def draw_hud(
    image: np.ndarray, palette: Palette, lines: list[str], origin: tuple[int, int] = (12, 26)
) -> None:
    """Draw a translucent panel with status lines in the top-left corner."""
    if not lines:
        return
    x, y = origin
    scale = 0.5
    widths, heights = zip(
        *[cv2.getTextSize(line, _FONT_PLAIN, scale, 1)[0] for line in lines], strict=True
    )
    panel_width = max(widths) + 20
    line_height = max(heights) + 10
    panel_height = line_height * len(lines) + 10

    overlay = image.copy()
    cv2.rectangle(overlay, (x - 8, y - 20), (x - 8 + panel_width, y - 20 + panel_height),
                  palette.panel, thickness=-1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, dst=image)

    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x, y + index * line_height),
            _FONT_PLAIN,
            scale,
            palette.text,
            1,
            cv2.LINE_AA,
        )


def draw_banner(image: np.ndarray, palette: Palette, text: str, colour: tuple[int, int, int] | None = None) -> None:
    """Centred message used for errors and prompts."""
    height, width = image.shape[:2]
    scale = 0.7
    (text_width, text_height), baseline = cv2.getTextSize(text, _FONT, scale, 2)
    x = max(0, (width - text_width) // 2)
    y = height // 2
    cv2.rectangle(
        image,
        (x - 16, y - text_height - baseline - 12),
        (x + text_width + 16, y + baseline + 8),
        colour or palette.panel,
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        _FONT,
        scale,
        palette.text,
        2,
        cv2.LINE_AA,
    )


def draw_progress(
    image: np.ndarray,
    palette: Palette,
    fraction: float,
    label: str,
    margin: int = 16,
) -> None:
    """Bottom progress bar used by the enrollment preview."""
    height, width = image.shape[:2]
    bar_height = 10
    y = height - margin - bar_height
    x1, x2 = margin, width - margin
    cv2.rectangle(image, (x1, y), (x2, y + bar_height), palette.panel, thickness=-1)
    filled = int((x2 - x1) * max(0.0, min(1.0, fraction)))
    if filled > 0:
        cv2.rectangle(image, (x1, y), (x1 + filled, y + bar_height), palette.accent, thickness=-1)
    if label:
        cv2.putText(
            image,
            label,
            (x1, y - 10),
            _FONT_PLAIN,
            0.5,
            palette.text,
            1,
            cv2.LINE_AA,
        )


def render_recognition(
    frame: np.ndarray, recognition: FrameRecognition, ui_config: UiConfig
) -> np.ndarray:
    """Frame with boxes, names and percentages for the CLI preview."""
    palette = palette_for(ui_config)
    image = frame.copy()
    for face in recognition.faces:
        if face.accepted:
            level = "matched"
        elif face.state is FaceState.VERIFYING:
            level = "verifying"
        else:
            level = "unknown"
        label = face.label if ui_config.show_similarity else face.display_name
        draw_face(image, face.bbox, palette, label=label, level=level)

    lines = [
        f"{recognition.face_count} face(s)  |  threshold {recognition.threshold:.2f} cosine",
    ]
    if ui_config.show_fps:
        lines.append(f"{recognition.fps:.1f} FPS  |  {recognition.inference_ms:.0f} ms")
    presentation = next(
        (face.presentation for face in recognition.faces if face.presentation is not None), None
    )
    if presentation is not None:
        lines.append(f"presentation: {presentation.label}")
    draw_hud(image, palette, lines)
    return resize_for_display(image, ui_config.display_max_width)


def render_registration(
    frame: np.ndarray, attempt: SampleAttempt, ui_config: UiConfig
) -> np.ndarray:
    """Frame with the current stage instruction and capture progress."""
    palette = palette_for(ui_config)
    image = frame.copy()
    level = "matched" if attempt.status is SampleStatus.CAPTURED else "unknown"
    for face in attempt.faces:
        draw_face(image, face.bbox, palette, label=attempt.status.value.upper(), level=level)

    draw_hud(
        image,
        palette,
        [
            f"Stage {min(attempt.collected + 1, attempt.required)}/{attempt.required}: "
            f"{attempt.stage_title}",
            attempt.prompt,
            attempt.message,
        ],
    )
    draw_progress(
        image,
        palette,
        attempt.progress,
        f"{attempt.collected}/{attempt.required} samples captured",
    )
    return resize_for_display(image, ui_config.display_max_width)


def render_idle(frame: np.ndarray, ui_config: UiConfig) -> np.ndarray:
    palette = palette_for(ui_config)
    image = frame.copy()
    draw_banner(image, palette, "Idle")
    return resize_for_display(image, ui_config.display_max_width)
