"""OpenCV preview overlays used by the headless CLI."""

from __future__ import annotations

import unittest

import numpy as np

from config import UiConfig
from src.face_recognizer import FaceState, FrameRecognition, RecognizedFace
from src.face_registration import SampleAttempt, SampleStatus
from src.ui import (
    draw_banner,
    draw_detection_box,
    draw_hud,
    draw_progress,
    palette_for,
    render_idle,
    render_recognition,
    render_registration,
    resize_for_display,
)
from tests.support import make_face

UI = UiConfig()


def frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.full((height, width, 3), 120, dtype=np.uint8)


def recognized(state: FaceState = FaceState.MATCHED) -> RecognizedFace:
    return RecognizedFace(
        face=make_face((60.0, 60.0, 260.0, 260.0)),
        track_id=1,
        state=state,
        similarity=0.62,
        similarity_smoothed=0.6,
        hits=3,
        employee_id="001",
        full_name="Alice",
        reason="matched",
    )


class PaletteTests(unittest.TestCase):
    def test_dark_is_the_default(self) -> None:
        self.assertIs(palette_for(UiConfig()), palette_for(UiConfig(style="nonsense")))

    def test_light_differs_from_dark(self) -> None:
        self.assertNotEqual(palette_for(UiConfig(style="light")).background, palette_for(UiConfig()).background)

    def test_no_teal_or_purple_channels_in_the_dark_accent(self) -> None:
        palette = palette_for(UiConfig())
        blue, green, red = palette.accent
        # The accent must stay warm: red above green, green above blue.
        self.assertGreater(red, green)
        self.assertGreater(green, blue)


class ResizeTests(unittest.TestCase):
    def test_large_frames_are_scaled_down(self) -> None:
        scaled = resize_for_display(frame(1000, 500), max_width=500)
        self.assertEqual(scaled.shape[1], 500)
        self.assertEqual(scaled.shape[0], 250)

    def test_small_frames_are_untouched(self) -> None:
        original = frame(320, 240)
        self.assertIs(resize_for_display(original, max_width=720), original)

    def test_zero_width_means_no_limit(self) -> None:
        original = frame(320, 240)
        self.assertIs(resize_for_display(original, max_width=0), original)


class DrawingTests(unittest.TestCase):
    def test_detection_box_changes_pixels(self) -> None:
        image = frame()
        before = image.copy()
        draw_detection_box(
            image, np.array([100.0, 100.0, 300.0, 300.0]), palette_for(UiConfig()), "Alice", "matched"
        )
        self.assertFalse(np.array_equal(image, before))

    def test_hud_draws_inside_the_frame(self) -> None:
        image = frame()
        draw_hud(image, palette_for(UiConfig()), ["line one", "line two"])
        self.assertEqual(image.shape[:2], (480, 640))

    def test_banner_is_centred(self) -> None:
        image = frame()
        draw_banner(image, palette_for(UiConfig()), "No camera")
        self.assertTrue(np.any(image != 120))

    def test_progress_bar_respects_the_fraction(self) -> None:
        empty = frame()
        full = frame()
        draw_progress(empty, palette_for(UiConfig()), 0.0, "")
        draw_progress(full, palette_for(UiConfig()), 1.0, "")
        self.assertFalse(np.array_equal(empty, full))


class RenderTests(unittest.TestCase):
    def test_recognition_render_returns_a_frame(self) -> None:
        image = render_recognition(frame(), FrameRecognition(faces=(recognized(),)), UI)
        self.assertEqual(image.ndim, 3)

    def test_recognition_render_handles_no_faces(self) -> None:
        image = render_recognition(frame(), FrameRecognition(), UI)
        self.assertEqual(image.ndim, 3)

    def test_recognition_render_handles_verifying_faces(self) -> None:
        image = render_recognition(
            frame(), FrameRecognition(faces=(recognized(FaceState.VERIFYING),)), UI
        )
        self.assertEqual(image.ndim, 3)

    def test_registration_render_returns_a_frame(self) -> None:
        attempt = SampleAttempt(
            status=SampleStatus.CAPTURED,
            message="Sample saved.",
            collected=2,
            required=5,
            prompt="Turn left",
            faces=(make_face(),),
            stage_title="Turn left",
        )
        image = render_registration(frame(), attempt, UI)
        self.assertEqual(image.ndim, 3)

    def test_idle_render_returns_a_frame(self) -> None:
        self.assertEqual(render_idle(frame(), UI).ndim, 3)

    def test_display_width_is_applied(self) -> None:
        wide = frame(1400, 700)
        image = render_idle(wide, UiConfig(display_max_width=700))
        self.assertEqual(image.shape[1], 700)


if __name__ == "__main__":
    unittest.main()
