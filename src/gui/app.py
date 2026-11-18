"""Main window: sidebar navigation plus one stacked page per task."""

from __future__ import annotations

import sys
from typing import cast

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.application import Application

from .attendance import AttendancePage
from .context import AppContext
from .enrollment import EnrollmentPage
from .home import HomePage
from .recognition import RecognitionPage
from .settings import SettingsPage
from .theme import BASE_FONT_SIZE_PT, apply_theme, palette_for_style
from .users import UsersPage

PAGES = (
    ("home", "Overview"),
    ("recognition", "Recognition"),
    ("enrollment", "Record a face"),
    ("attendance", "Attendance"),
    ("users", "Identities"),
    ("settings", "Settings"),
)
APP_VERSION = "1.0"


class MainWindow(QWidget):
    """Application shell."""

    def __init__(self, application: Application) -> None:
        super().__init__()
        self._context = AppContext(application, self)
        self._context.theme_changed.connect(self._apply_theme)
        self._context.status_message.connect(self._show_status)

        config = self._context.config
        self.setObjectName("Root")
        self.setWindowTitle(config.ui.window_title)
        self.resize(1180, 760)
        self.setMinimumSize(940, 620)

        # Applied before the pages exist: the stylesheet changes font sizes, and a
        # widget laid out with the old metrics keeps a clipped, too-small rect.
        self._apply_theme()
        self._build_ui()
        self.navigate("home")

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())

        self._stack = QStackedWidget()
        self._pages: dict[str, QWidget] = {
            "home": HomePage(self._context),
            "recognition": RecognitionPage(self._context),
            "enrollment": EnrollmentPage(self._context),
            "attendance": AttendancePage(self._context),
            "users": UsersPage(self._context),
            "settings": SettingsPage(self._context),
        }
        for page in self._pages.values():
            self._stack.addWidget(page)
        root.addWidget(self._stack, 1)

        home = self._pages["home"]
        if isinstance(home, HomePage):
            home.navigate.connect(self.navigate)
        enrollment = self._pages["enrollment"]
        if isinstance(enrollment, EnrollmentPage):
            enrollment.enrollment_saved.connect(lambda: self._show_status("Identity saved."))
        users = self._pages["users"]
        if isinstance(users, UsersPage):
            users.record_requested.connect(self._start_rerecord)

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(208)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 18, 14, 16)
        layout.setSpacing(6)

        name = QLabel("Face ID")
        name.setObjectName("AppName")
        layout.addWidget(name)
        subtitle = QLabel("Local recognition")
        subtitle.setObjectName("Hint")
        layout.addWidget(subtitle)
        layout.addSpacing(14)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_buttons: dict[str, QPushButton] = {}
        self._current_key = "home"
        for key, label in PAGES:
            button = QPushButton(label)
            button.setObjectName("NavItem")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, target=key: self.navigate(target))
            self._nav_group.addButton(button)
            layout.addWidget(button)
            self._nav_buttons[key] = button

        layout.addStretch(1)

        hardware = QLabel(self._hardware_lines())
        hardware.setObjectName("Hint")
        hardware.setWordWrap(True)
        layout.addWidget(hardware)

        tier = QLabel(f"Profile: {self._context.tier}")
        tier.setObjectName("Hint")
        layout.addWidget(tier)

        version = QLabel(f"Version {APP_VERSION}")
        version.setObjectName("Hint")
        layout.addWidget(version)
        return sidebar

    def _hardware_lines(self) -> str:
        info = self._context.application.hardware
        ram_gb = info.total_ram_mb // 1024 if info.total_ram_mb else 0
        ram = f" · {ram_gb} GB RAM" if ram_gb else ""
        return f"{info.cpu_cores} cores{ram}\n{info.platform_tag} · {info.machine}"

    def navigate(self, key: str) -> None:
        page = self._pages.get(key)
        if page is None:
            return
        if key != self._current_key and not self._confirm_leave_enrollment():
            # The click already moved the sidebar highlight; put it back.
            previous = self._nav_buttons.get(self._current_key)
            if previous is not None:
                previous.setChecked(True)
            return
        if key == "enrollment":
            # Opening the page from the sidebar or a shortcut means "new person".
            self.enrollment_page.start_new()
        self._stack.setCurrentWidget(page)
        self._current_key = key
        button = self._nav_buttons.get(key)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        self._fade_in(page)

    def _confirm_leave_enrollment(self) -> bool:
        """Return True when it is safe to leave the enrollment page.

        A completed capture lives only in memory until Save writes it, so leaving
        the page has to be a deliberate choice rather than a lost face.
        """
        if self._current_key != "enrollment":
            return True
        page = self.enrollment_page
        if not page.has_unsaved_capture():
            return True
        answer = QMessageBox.question(
            self,
            "Discard the captured samples?",
            "The five poses are captured but not saved. Leaving this page discards "
            "them and nothing is written to the database.",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Discard:
            return False
        page.cancel_capture()
        return True

    def _start_rerecord(self, employee_id: str, full_name: str) -> None:
        """Open the enrollment page ready to replace one identity's samples."""
        self.navigate("enrollment")
        self.enrollment_page.start_update(employee_id, full_name)

    # Typed accessors: pages are stored generically, but callers (and tests) want
    # the concrete class without a runtime check.
    @property
    def home_page(self) -> HomePage:
        return cast(HomePage, self._pages["home"])

    @property
    def recognition_page(self) -> RecognitionPage:
        return cast(RecognitionPage, self._pages["recognition"])

    @property
    def enrollment_page(self) -> EnrollmentPage:
        return cast(EnrollmentPage, self._pages["enrollment"])

    @property
    def attendance_page(self) -> AttendancePage:
        return cast(AttendancePage, self._pages["attendance"])

    @property
    def users_page(self) -> UsersPage:
        return cast(UsersPage, self._pages["users"])

    @property
    def settings_page(self) -> SettingsPage:
        return cast(SettingsPage, self._pages["settings"])

    def _fade_in(self, page: QWidget) -> None:
        if not self._context.config.ui.animations:
            return
        effect = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(140)
        animation.setStartValue(0.35)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(lambda: page.setGraphicsEffect(None))  # type: ignore[arg-type]
        animation.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)  # type: ignore[arg-type]

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if isinstance(app, QApplication):
            apply_theme(app, palette_for_style(self._context.config.ui.style))

    def _show_status(self, message: str) -> None:
        if message:
            self.setWindowTitle(f"{self._context.config.ui.window_title}: {message}")
        else:
            self.setWindowTitle(self._context.config.ui.window_title)

    def shutdown(self) -> None:
        for page in self._pages.values():
            closer = getattr(page, "shutdown", None)
            if callable(closer):
                closer()
        self._context.application.close()

    def closeEvent(self, event) -> None:
        if not self._confirm_leave_enrollment():
            event.ignore()
            return
        self.shutdown()
        super().closeEvent(event)


def run_app(application: Application) -> int:
    """Create the Qt application, show the window, and run the event loop."""
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication(sys.argv[:1])
    app.setApplicationName("Face ID")
    app.setApplicationVersion(APP_VERSION)
    app.setStyle("Fusion")
    font = app.font()
    font.setPointSize(BASE_FONT_SIZE_PT)
    app.setFont(font)

    window = MainWindow(application)
    window.show()
    try:
        return app.exec()
    finally:
        application.close()
