"""Preview painting shared by the camera pages.

The video shows the newest captured frame, never the newest inference result: a slow
detection must not make the picture stutter. The caption is whatever the last result
produced, so text and picture stay independent of each other.
"""

from __future__ import annotations

from src.pipeline import RecognitionPipeline

from .widgets import VideoView


class PreviewPainter:
    """Paints each captured frame once into a video widget, with the latest caption."""

    def __init__(self, video: VideoView) -> None:
        self._video = video
        self._caption = ""
        self._sequence = -1

    def set_caption(self, caption: str) -> None:
        """Text drawn over the picture; it only changes when a new result arrives."""
        self._caption = caption

    def reset(self) -> None:
        """Forget the last frame and caption, so the next frame is painted again."""
        self._sequence = -1
        self._caption = ""

    def paint(self, pipeline: RecognitionPipeline | None) -> bool:
        """Show the latest frame once; returns True when something was painted."""
        if pipeline is None:
            return False
        frame = pipeline.latest_frame()
        if frame is None or frame.sequence == self._sequence:
            return False
        self._sequence = frame.sequence
        self._video.set_frame(frame.image, caption=self._caption)
        pipeline.note_displayed()
        return True
