"""GUI-level checks: theme invariants, page contents, and the settings round trip.

These run on the offscreen Qt platform, so they need no display but do need
PySide6 installed. They assert behaviour rather than pixels: no forbidden hues, no
duplicated "model" row, and a settings save that really reaches the store.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QDate, QRect
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from config import load_config
from src.camera import Frame

FORBIDDEN_HEX = {
    "#22d3ee",  # cyan
    "#06b6d4",  # teal
    "#14b8a6",  # teal-green
    "#8b5cf6",  # violet
    "#a855f7",  # purple
    "#6366f1",  # indigo
    "#d946ef",  # fuchsia
}


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    cleaned = value.lstrip("#")
    return int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16)


def _relative_luminance(value: str) -> float:
    channels = [component / 255 for component in _hex_to_rgb(value)]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground: str, background: str) -> float:
    """WCAG 2.x contrast ratio between two hex colours."""
    first = _relative_luminance(foreground)
    second = _relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


class ThemeTests(unittest.TestCase):
    def test_dark_background_is_a_neutral_grey(self) -> None:
        from src.gui.theme import palette_for_style

        red, green, blue = _hex_to_rgb(palette_for_style("dark").background)
        self.assertLessEqual(abs(red - green), 6)
        self.assertLessEqual(abs(green - blue), 6)

    def test_accent_is_warm_in_both_themes(self) -> None:
        from src.gui.theme import palette_for_style

        for style in ("dark", "light"):
            red, green, blue = _hex_to_rgb(palette_for_style(style).accent)
            self.assertGreater(red, green, f"{style} accent is not warm")
            self.assertGreater(green, blue, f"{style} accent is not warm")

    def test_no_forbidden_hue_appears_in_either_palette(self) -> None:
        from src.gui.theme import palette_for_style, stylesheet

        for style in ("dark", "light"):
            palette = palette_for_style(style)
            text = stylesheet(palette).lower()
            for forbidden in FORBIDDEN_HEX:
                self.assertNotIn(forbidden, text, f"{forbidden} is used in the {style} theme")

    def test_stylesheet_has_no_unsubstituted_placeholders(self) -> None:
        from src.gui.theme import palette_for_style, stylesheet

        text = stylesheet(palette_for_style("dark"))
        self.assertNotIn("$", text)

    def test_stylesheet_defines_the_primary_button(self) -> None:
        from src.gui.theme import palette_for_style, stylesheet

        self.assertIn("QPushButton#Primary", stylesheet(palette_for_style("dark")))

    def test_every_text_pair_meets_wcag_aa(self) -> None:
        from src.gui.theme import DARK, LIGHT

        pairs = (
            ("text", ("background", "surface", "surface_alt", "surface_sunken")),
            ("text_muted", ("background", "surface", "surface_alt", "surface_sunken")),
            ("accent", ("accent_soft",)),
            ("success", ("surface", "surface_alt", "surface_sunken")),
            ("warning", ("surface", "surface_alt", "surface_sunken")),
            ("danger", ("surface", "surface_alt", "surface_sunken")),
            ("text_inverse", ("accent",)),
        )
        for palette in (DARK, LIGHT):
            for foreground, backgrounds in pairs:
                for background in backgrounds:
                    ratio = _contrast(
                        getattr(palette, foreground), getattr(palette, background)
                    )
                    self.assertGreaterEqual(
                        ratio,
                        4.5,
                        f"{palette.name}: {foreground} on {background} is {ratio:.2f}:1",
                    )

    def test_stylesheet_keeps_a_visible_focus_indicator(self) -> None:
        from src.gui.theme import palette_for_style, stylesheet

        text = stylesheet(palette_for_style("dark"))
        self.assertIn("QPushButton:focus", text)
        self.assertIn("QSlider:focus::handle:horizontal", text)
        self.assertIn("QCheckBox:focus::indicator", text)

    def test_stylesheet_never_removes_the_focus_outline(self) -> None:
        from src.gui.theme import palette_for_style, stylesheet

        for style in ("dark", "light"):
            self.assertNotIn("outline: none", stylesheet(palette_for_style(style)))


class PageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        existing = QApplication.instance()
        cls.app = existing if isinstance(existing, QApplication) else QApplication([])
        cls._directory = tempfile.TemporaryDirectory()
        base = load_config()
        settings_path = Path(cls._directory.name) / "config.yaml"
        cls.config = base.replace(source_path=settings_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def setUp(self) -> None:
        from src.application import Application
        from src.gui import MainWindow

        self.application = Application(self.config)
        self.window = MainWindow(self.application)
        self.addCleanup(self.window.shutdown)

    def _texts(self, page: QWidget) -> list[str]:
        return [label.text() for label in page.findChildren(QLabel)]

    def test_every_page_can_be_shown(self) -> None:
        for key in ("home", "recognition", "enrollment", "users", "settings"):
            self.window.navigate(key)
            self.app.processEvents()
        self.assertEqual(self.window.windowTitle(), self.config.ui.window_title)

    def test_home_has_no_sample_total_tile(self) -> None:
        self.window.navigate("home")
        self.app.processEvents()
        page = self.window.home_page
        text = " ".join(self._texts(page)).lower()
        self.assertNotIn("total sample", text)
        self.assertNotIn("face samples", text)

    def test_recognition_page_has_no_model_row(self) -> None:
        self.window.navigate("recognition")
        self.app.processEvents()
        page = self.window.recognition_page
        labels = [text.strip().lower() for text in self._texts(page)]
        self.assertNotIn("model", labels)

    def test_recognition_page_shows_a_percentage_threshold(self) -> None:
        self.window.navigate("recognition")
        self.app.processEvents()
        page = self.window.recognition_page
        text = " ".join(self._texts(page))
        self.assertIn("45%", text)

    def test_the_idle_readout_is_not_repainted_on_every_frame(self) -> None:
        """The page sees 10 to 30 results per second; identical text must be free."""
        self.window.navigate("recognition")
        page = self.window.recognition_page
        page._reset_readout()

        spy = mock.Mock()
        page._name.setText = spy
        page._reset_readout()

        spy.assert_not_called()

    def test_the_badge_is_only_restyled_when_it_changes(self) -> None:
        self.window.navigate("recognition")
        page = self.window.recognition_page
        page._set_badge("matched", "BadgeAccent")

        spy = mock.Mock()
        page._state_badge.setText = spy
        page._set_badge("matched", "BadgeAccent")
        spy.assert_not_called()

        page._set_badge("no match", "Badge")
        spy.assert_called_once_with("no match")

    def test_enrollment_stage_marks_are_written_once(self) -> None:
        self.window.navigate("enrollment")
        page = self.window.enrollment_page
        page._mark_stage(2)

        spy = mock.Mock()
        page._stage_labels[0].setText = spy
        page._mark_stage(2)

        spy.assert_not_called()

    def test_settings_are_written_to_disk(self) -> None:
        self.window.navigate("settings")
        self.app.processEvents()
        page = self.window.settings_page
        page._threshold.set_value(60)
        page.save()

        stored = Path(self._directory.name) / "settings.json"
        self.assertTrue(stored.is_file())
        payload = json.loads(stored.read_text(encoding="utf-8"))
        self.assertAlmostEqual(payload["recognition"]["similarity_threshold"], 0.6)
        self.assertAlmostEqual(self.window._context.config.recognition.similarity_threshold, 0.6)

    def test_presentation_settings_round_trip(self) -> None:
        page = self.window.settings_page
        page._liveness_mode.set_value("enforce")
        page._liveness_sensitivity.set_value(80)
        page.save()

        payload = json.loads(
            (Path(self._directory.name) / "settings.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["liveness"]["mode"], "enforce")
        self.assertAlmostEqual(payload["liveness"]["sensitivity"], 0.8)
        self.assertTrue(self.window._context.config.liveness.enforcing)

    def test_theme_toggle_reaches_the_whole_application(self) -> None:
        from src.gui.theme import palette_for_style

        page = self.window.settings_page
        page._style.set_value("light")
        page.save()
        self.app.processEvents()
        self.assertEqual(self.window._context.config.ui.style, "light")
        self.assertIn(palette_for_style("light").background, self.app.styleSheet())

        page.reset()
        self.app.processEvents()
        self.assertEqual(self.window._context.config.ui.style, "dark")
        self.assertIn(palette_for_style("dark").background, self.app.styleSheet())

    def test_settings_reset_removes_the_file(self) -> None:
        page = self.window.settings_page
        page._threshold.set_value(70)
        page.save()
        page.reset()
        stored = Path(self._directory.name) / "settings.json"
        self.assertFalse(stored.exists())
        self.assertAlmostEqual(
            self.window._context.config.recognition.similarity_threshold, 0.45
        )

    def test_settings_page_collects_every_section(self) -> None:
        page = self.window.settings_page
        overrides = page.collect()
        for section in (
            "recognition",
            "registration",
            "quality",
            "liveness",
            "attendance",
            "camera",
            "model",
            "ui",
        ):
            self.assertIn(section, overrides)

    def test_attendance_settings_round_trip(self) -> None:
        page = self.window.settings_page
        page._attendance_enabled.set_checked(False)
        page._late_minutes.set_value(30)
        page._work_start.set_value("09:15")
        page.save()

        payload = json.loads(
            (Path(self._directory.name) / "settings.json").read_text(encoding="utf-8")
        )
        self.assertFalse(payload["attendance"]["enabled"])
        self.assertEqual(payload["attendance"]["late_after_minutes"], 30)
        self.assertEqual(payload["attendance"]["work_start"], "09:15")
        self.assertFalse(self.window._context.config.attendance.enabled)

    def test_attendance_page_lists_a_day(self) -> None:
        from datetime import datetime

        import numpy as np

        from src.database import FaceSample

        database = self.window._context.database()
        database.add_employee("001", "Ana", [FaceSample(embedding=np.ones(4, dtype=np.float32))])
        today = self.window._context.application.attendance.today()
        database.mark_present(
            "001", today, datetime.now().astimezone().isoformat(timespec="seconds")
        )
        page = self.window.attendance_page
        page.refresh()
        self.assertEqual(page._table.rowCount(), 1)
        self.assertIn("1 present", page._summary.text())

    def test_attendance_page_empty_state(self) -> None:
        page = self.window.attendance_page
        page._date.setDate(QDate(2001, 1, 1))
        page.refresh()
        self.assertEqual(page._table.rowCount(), 0)
        self.assertIn("Nobody recorded", page._summary.text())

    def test_enrollment_page_lists_the_five_poses(self) -> None:
        page = self.window.enrollment_page
        text = " ".join(self._texts(page))
        for pose in ("Front", "Turn left", "Turn right", "Tilt", "Closer"):
            self.assertIn(pose, text)

    def test_rerecord_opens_the_enrollment_page_in_update_mode(self) -> None:
        enrollment = self.window.enrollment_page
        page = self.window.users_page
        page.record_requested.emit("003", "Ana")

        self.assertIs(self.window._stack.currentWidget(), enrollment)
        self.assertEqual(enrollment._id_field.text(), "003")
        self.assertEqual(enrollment._name_field.text(), "Ana")
        self.assertEqual(enrollment._save_button.text(), "Replace samples")

    def test_navigating_to_enrollment_leaves_update_mode(self) -> None:
        enrollment = self.window.enrollment_page
        self.window._start_rerecord("003", "Ana")
        self.window.navigate("enrollment")
        self.assertIsNone(enrollment._update_id)
        self.assertEqual(enrollment._save_button.text(), "Save identity")

    def test_users_page_lists_the_registered_identity(self) -> None:
        page = self.window.users_page
        page.refresh()
        table = page._table
        for row in range(table.rowCount()):
            item = table.item(row, 1)
            assert item is not None
            self.assertTrue(item.text())


class _FakeSession:
    """Stands in for FaceRegistrationSession: only the calls the page makes."""

    def __init__(self) -> None:
        self.resets = 0
        self.freezes = 0

    def reset(self) -> None:
        self.resets += 1

    def freeze(self) -> None:
        self.freezes += 1


class EnrollmentNameTests(unittest.TestCase):
    """A face can only be captured and saved under a real name."""

    @classmethod
    def setUpClass(cls) -> None:
        existing = QApplication.instance()
        cls.app = existing if isinstance(existing, QApplication) else QApplication([])
        cls._directory = tempfile.TemporaryDirectory()
        cls.config = load_config().replace(
            source_path=Path(cls._directory.name) / "config.yaml"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def setUp(self) -> None:
        from src.application import Application
        from src.gui import MainWindow

        self.application = Application(self.config, use_settings=False)
        self.window = MainWindow(self.application)
        self.addCleanup(self.window.shutdown)
        self.page = self.window.enrollment_page
        self.window.navigate("enrollment")
        # Pretend the camera is ready, which is what enables the capture button.
        self.session = _FakeSession()
        self.page._session = self.session  # type: ignore[assignment]
        self.page._update_buttons()

    def test_capture_is_blocked_without_a_name(self) -> None:
        self.assertFalse(self.page._start_button.isEnabled())
        self.assertIn("name", self.page._name_hint.text().lower())

    def test_a_whitespace_only_name_does_not_count(self) -> None:
        self.page._name_field.setText("   ")
        self.assertFalse(self.page._start_button.isEnabled())

    def test_typing_a_name_enables_the_capture(self) -> None:
        self.page._name_field.setText("Ana")
        self.assertTrue(self.page._start_button.isEnabled())

    def test_clearing_the_name_disables_it_again(self) -> None:
        self.page._name_field.setText("Ana")
        self.page._name_field.clear()
        self.assertFalse(self.page._start_button.isEnabled())

    def test_start_capture_refuses_an_empty_name(self) -> None:
        self.page.start_capture()
        self.assertEqual(self.session.resets, 0)
        self.assertFalse(self.page._capturing)

    def test_start_capture_runs_once_the_name_is_typed(self) -> None:
        self.page._name_field.setText("Ana")
        self.page.start_capture()
        self.assertEqual(self.session.resets, 1)
        self.assertTrue(self.page._capturing)

    def test_save_is_blocked_until_captured_and_named(self) -> None:
        self.page._name_field.setText("Ana")
        self.assertFalse(self.page._save_button.isEnabled())
        self.page._ready_to_save = True
        self.page._update_buttons()
        self.assertTrue(self.page._save_button.isEnabled())

    def test_re_record_prefills_the_name_and_allows_capture(self) -> None:
        self.window._start_rerecord("003", "Ana Maria")
        self.assertTrue(self.page._start_button.isEnabled())

    def test_leaving_without_a_capture_does_not_ask(self) -> None:
        self.assertTrue(self.window._confirm_leave_enrollment())
        self.assertEqual(self.window._current_key, "enrollment")

    def test_leaving_with_an_unsaved_capture_asks_first(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        from src.gui import app as gui_app

        self.page._ready_to_save = True
        with mock.patch.object(
            gui_app.QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel
        ):
            self.window.navigate("home")
        self.assertEqual(self.window._current_key, "enrollment")
        self.assertIs(self.window._stack.currentWidget(), self.page)

        with mock.patch.object(
            gui_app.QMessageBox, "question", return_value=QMessageBox.StandardButton.Discard
        ):
            self.window.navigate("home")
        self.assertEqual(self.window._current_key, "home")
        self.assertFalse(self.page.has_unsaved_capture())


class ResponsiveLayoutTests(unittest.TestCase):
    """The window has to fit small screens and reflow instead of clipping."""

    @classmethod
    def setUpClass(cls) -> None:
        existing = QApplication.instance()
        cls.app = existing if isinstance(existing, QApplication) else QApplication([])
        cls._directory = tempfile.TemporaryDirectory()
        base = load_config()
        cls.config = base.replace(source_path=Path(cls._directory.name) / "config.yaml")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def setUp(self) -> None:
        from src.application import Application
        from src.gui import MainWindow

        self.application = Application(self.config)
        self.window = MainWindow(self.application)
        # Shown on purpose: Qt only runs a layout pass for visible widgets, and
        # these tests are about what the layout does when the window is resized.
        self.window.show()
        self.addCleanup(self.window.hide)
        self.addCleanup(self.window.shutdown)

    def _available(self) -> QRect:
        screen = self.app.primaryScreen()
        self.assertIsNotNone(screen)
        return screen.availableGeometry()

    def test_requested_sizes_fit_the_screen(self) -> None:
        from src.gui.app import _initial_size, _minimum_size

        available = self._available()
        width, height = _initial_size(self.window)
        minimum_width, minimum_height = _minimum_size()

        self.assertLessEqual(width, available.width())
        self.assertLessEqual(height, available.height())
        self.assertLessEqual(minimum_width, width)
        self.assertLessEqual(minimum_height, height)
        self.assertLessEqual(minimum_width, available.width())
        self.assertLessEqual(minimum_height, available.height())
        self.assertGreaterEqual(minimum_width, 320)
        self.assertGreaterEqual(minimum_height, 240)

    def test_every_page_fits_a_short_window(self) -> None:
        for key in ("home", "recognition", "enrollment", "attendance", "users", "settings"):
            with self.subTest(page=key):
                self.window.navigate(key)
                self.app.processEvents()
                page = self.window._pages[key]
                self.assertLessEqual(page.minimumSizeHint().height(), 560)

    def test_recognition_page_stacks_on_a_narrow_window(self) -> None:
        from PySide6.QtWidgets import QBoxLayout

        self.window.navigate("recognition")
        page = self.window.recognition_page

        self.window.resize(780, 560)
        self.app.processEvents()
        self.assertEqual(page._body.direction(), QBoxLayout.Direction.TopToBottom)
        self.assertLessEqual(page._video.maximumHeight(), 400)

        self.window.resize(1300, 720)
        self.app.processEvents()
        self.assertEqual(page._body.direction(), QBoxLayout.Direction.LeftToRight)

    def test_enrollment_page_stacks_on_a_narrow_window(self) -> None:
        from PySide6.QtWidgets import QBoxLayout

        self.window.navigate("enrollment")
        page = self.window.enrollment_page

        self.window.resize(780, 560)
        self.app.processEvents()
        self.assertEqual(page._body.direction(), QBoxLayout.Direction.TopToBottom)

        self.window.resize(1000, 720)
        self.app.processEvents()
        self.assertEqual(page._body.direction(), QBoxLayout.Direction.LeftToRight)

    def test_toolbars_stay_on_one_row_at_normal_sizes(self) -> None:
        from PySide6.QtWidgets import QBoxLayout

        self.window.navigate("attendance")
        self.window.resize(1000, 620)
        self.app.processEvents()
        self.assertEqual(
            self.window.attendance_page._toolbar.direction(),
            QBoxLayout.Direction.LeftToRight,
        )
        self.window.resize(780, 560)
        self.app.processEvents()
        self.assertEqual(
            self.window.attendance_page._toolbar.direction(),
            QBoxLayout.Direction.TopToBottom,
        )

        self.window.navigate("users")
        self.window.resize(1000, 620)
        self.app.processEvents()
        self.assertEqual(
            self.window.users_page._actions.direction(), QBoxLayout.Direction.LeftToRight
        )

    def test_home_tiles_reduce_their_column_count(self) -> None:
        self.window.navigate("home")
        page = self.window.home_page

        for width, columns in ((1500, 4), (1000, 2), (780, 1)):
            with self.subTest(width=width):
                self.window.resize(width, 680)
                self.app.processEvents()
                # Real geometry, not columnCount(): QGridLayout keeps the widest
                # grid it has ever used until it is rebuilt.
                tiles = [tile for tile in page._tiles if tile.isVisible()]
                top = tiles[0].y()
                self.assertEqual(len([tile for tile in tiles if tile.y() == top]), columns)

    def test_buttons_keep_a_readable_height(self) -> None:
        self.window.navigate("recognition")
        page = self.window.recognition_page
        self.window.resize(780, 560)
        self.app.processEvents()

        for button in (page._start_button, page._stop_button):
            self.assertGreaterEqual(button.minimumHeight(), 28)
            self.assertGreaterEqual(button.height(), 28)


class PreviewPainterTests(unittest.TestCase):
    """The preview follows the camera, so each frame is painted exactly once."""

    @classmethod
    def setUpClass(cls) -> None:
        existing = QApplication.instance()
        cls.app = existing if isinstance(existing, QApplication) else QApplication([])

    def setUp(self) -> None:
        from src.gui.preview import PreviewPainter
        from src.gui.widgets import VideoView

        self.video = VideoView("off")
        self.painter = PreviewPainter(self.video)
        self.pipeline = mock.Mock()
        self.pipeline.latest_frame.return_value = Frame(
            np.zeros((8, 8, 3), dtype=np.uint8), 4
        )

    def test_a_frame_is_painted_once(self) -> None:
        self.assertTrue(self.painter.paint(self.pipeline))
        self.assertFalse(self.painter.paint(self.pipeline))
        self.assertEqual(self.pipeline.note_displayed.call_count, 1)

    def test_a_new_frame_is_painted_again(self) -> None:
        self.painter.paint(self.pipeline)
        self.pipeline.latest_frame.return_value = Frame(
            np.zeros((8, 8, 3), dtype=np.uint8), 5
        )
        self.assertTrue(self.painter.paint(self.pipeline))

    def test_the_caption_of_the_last_result_is_drawn(self) -> None:
        self.painter.set_caption("2 of 5 samples captured")
        self.painter.paint(self.pipeline)
        self.assertEqual(self.video._caption, "2 of 5 samples captured")

    def test_nothing_is_painted_without_a_pipeline(self) -> None:
        self.assertFalse(self.painter.paint(None))

    def test_reset_forgets_the_last_frame(self) -> None:
        self.painter.paint(self.pipeline)
        self.painter.reset()
        self.assertTrue(self.painter.paint(self.pipeline))


if __name__ == "__main__":
    unittest.main()


