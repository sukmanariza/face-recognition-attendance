"""Colour palette and Qt stylesheet.

One neutral grey ramp plus a single amber accent. Deliberately no gradients, no
glow, and no second hue: the interface stays readable in a dim room and does not
compete with the camera image, which is the only element that should draw the eye.

Every colour lives in `Palette`; widgets never hard-code a hex value, so switching
between the dark and light themes needs no widget changes.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

FONT_FAMILY_FALLBACKS = ("Segoe UI", "Inter", "Noto Sans", "DejaVu Sans", "Sans Serif")
BASE_FONT_SIZE_PT = 10
TITLE_FONT_SIZE_PT = 19
HEADING_FONT_SIZE_PT = 13
MONO_FAMILY = "Consolas"


@dataclass(frozen=True)
class Palette:
    """Every colour the interface uses, in one place."""

    name: str
    background: str
    surface: str
    surface_alt: str
    surface_sunken: str
    border: str
    border_strong: str
    text: str
    text_muted: str
    text_inverse: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_soft: str
    success: str
    warning: str
    danger: str
    overlay: str

    def color(self, attribute: str) -> QColor:
        return QColor(getattr(self, attribute))


DARK = Palette(
    name="dark",
    background="#141414",
    surface="#1C1B1A",
    surface_alt="#242322",
    surface_sunken="#0F0F0E",
    border="#302E2C",
    border_strong="#403D3A",
    text="#EDEAE6",
    text_muted="#A09A93",
    text_inverse="#14100A",
    accent="#C47A18",
    accent_hover="#D98A22",
    accent_pressed="#A96713",
    accent_soft="#241A0C",
    success="#5FB37A",
    warning="#E0B44A",
    danger="#E8705C",
    overlay="#0A0A09",
)

LIGHT = Palette(
    name="light",
    background="#F4F2EF",
    surface="#FFFFFF",
    surface_alt="#F7F5F2",
    surface_sunken="#EAE6E0",
    border="#DED9D2",
    border_strong="#C4BDB4",
    text="#1C1A18",
    text_muted="#6B655E",
    text_inverse="#FFFFFF",    accent="#8A520C",
    accent_hover="#7A470A",
    accent_pressed="#663C08",
    accent_soft="#F6E9D4",
    success="#26663D",
    warning="#7F5609",
    danger="#8F2F22",
    overlay="#FFFFFF",
)

_TEMPLATE = string.Template(
    """
/* Only the root fills with the window colour. Child widgets stay transparent,
   otherwise every label and layout holder would paint a darker rectangle over
   the card it sits on. */
QWidget {
    color: $text;
    font-size: ${font_size}pt;
    background: transparent;
}
QWidget#Root, QMainWindow, QDialog {
    background-color: $background;
}
QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget {
    background: transparent;
}

QToolTip {
    background-color: $surface_alt;
    color: $text;
    border: 1px solid $border_strong;
    padding: 4px 6px;
}
/* structure */
QFrame#Sidebar {
    background-color: $surface_sunken;
    border-right: 1px solid $border;
}
QFrame#Card, QFrame#Tile {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: 8px;
}
QFrame#TileAccent {
    background-color: $surface;
    border: 1px solid $border_strong;
    border-radius: 8px;
}
QFrame#Divider {
    background-color: $border;
    max-height: 1px;
    border: none;
}
QFrame#VideoFrame {
    background-color: $surface;
    border: 1px solid $border_strong;
    border-radius: 8px;
}

/* text */
QLabel#AppName {
    font-size: ${title_size}pt;
    font-weight: 600;
    color: $text;
}
QLabel#PageTitle {
    font-size: ${title_size}pt;
    font-weight: 600;
    color: $text;
}
QLabel#FaceName {
    font-size: ${heading_size}pt;
    font-weight: 600;
    color: $text;
}
QLabel#PageSubtitle, QLabel#Muted, QLabel#Hint {
    color: $text_muted;
}
QLabel#SectionTitle {
    font-size: ${heading_size}pt;
    font-weight: 600;
    color: $text;
}
QLabel#TileValue {
    font-size: ${title_size}pt;
    font-weight: 600;
    color: $text;
}
QLabel#TileLabel {
    color: $text_muted;
}
/* Scroll areas are page furniture, not surfaces: keep them invisible so a panel
   does not look like a second card. */
QScrollArea {
    background: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: $border_strong;
    border-radius: 5px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: $text_muted;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
}

QLabel#MonoValue {
    font-family: "$mono";
    color: $text;
}
/* Tone classes for a single value whose meaning is good, warning, or bad. */
QLabel#ToneSuccess {
    color: $success;
    font-weight: 600;
}
QLabel#ToneWarning {
    color: $warning;
    font-weight: 600;
}
QLabel#ToneDanger {
    color: $danger;
    font-weight: 600;
}
QLabel#ToneMuted {
    color: $text_muted;
}
QLabel#Badge {
    background-color: $surface_alt;
    border: 1px solid $border;
    border-radius: 8px;
    padding: 2px 8px;
    color: $text_muted;
}
QLabel#BadgeAccent {
    background-color: $accent_soft;
    border: 1px solid $accent;
    border-radius: 8px;
    padding: 2px 8px;
    color: $accent;
}

/* Banner tones: the object name carries the severity, so each theme pairs text
   and background with enough contrast. */
QLabel#BannerInfo, QLabel#BannerSuccess, QLabel#BannerWarning, QLabel#BannerError {
    background-color: $surface;
    border: 1px solid $border_strong;
    border-radius: 6px;
    padding: 8px 10px;
}
QLabel#BannerInfo {
    color: $text_muted;
}
QLabel#BannerSuccess {
    color: $success;
    border-color: $success;
}
QLabel#BannerWarning {
    color: $warning;
    border-color: $warning;
}
QLabel#BannerError {
    color: $danger;
    border-color: $danger;
}

/* buttons */
QPushButton {
    background-color: $surface_alt;
    color: $text;
    border: 1px solid $border_strong;
    border-radius: 6px;
    padding: 7px 14px;
}
QPushButton:hover {
    background-color: $surface;
    border-color: $accent;
}
QPushButton:pressed {
    background-color: $surface_sunken;
}
QPushButton:disabled {
    color: $text_muted;
    border-color: $border;
    background-color: $surface;
}
QPushButton#Primary {
    background-color: $accent;
    color: $text_inverse;
    border: 1px solid $accent;
    font-weight: 600;
}
QPushButton#Primary:hover {
    background-color: $accent_hover;
    border-color: $accent_hover;
}
QPushButton#Primary:pressed {
    background-color: $accent_pressed;
}
QPushButton#Primary:disabled {
    background-color: $surface_alt;
    color: $text_muted;
    border-color: $border;
}
/* Keyboard focus: a 2 px accent ring, inset by 1 px so the layout does not shift. */
QPushButton:focus {
    border: 2px solid $accent;
    padding: 6px 13px;
}
QPushButton#Primary:focus {
    border: 2px solid $text;
    padding: 6px 13px;
}
QPushButton#Danger {
    color: $danger;
    border-color: $danger;
    background-color: transparent;
}
QPushButton#Danger:hover {
    background-color: $surface_alt;
}
QPushButton#NavItem {
    background-color: transparent;
    border: none;
    border-radius: 6px;
    padding: 9px 12px;
    text-align: left;
    color: $text_muted;
}
QPushButton#NavItem:hover {
    color: $text;
    background-color: $surface;
}
QPushButton#NavItem:checked {
    color: $text;
    background-color: $surface;
    border-left: 3px solid $accent;
    padding-left: 9px;
}

/* inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: $surface_sunken;
    border: 1px solid $border_strong;
    border-radius: 6px;
    padding: 6px 8px;
    color: $text;
    selection-background-color: $accent;
    selection-color: $text_inverse;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: $accent;
}
QComboBox::drop-down {
    border: none;
    width: 18px;
}
/* Qt draws no arrow once a spin button is styled, and QSS cannot draw one without
   an image, so the buttons are hidden and the field is typed or driven by keys. */
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    width: 0;
    border: none;
}
QComboBox QAbstractItemView {
    background-color: $surface;
    border: 1px solid $border_strong;
    selection-background-color: $accent_soft;
    selection-color: $text;
}
QComboBox QAbstractItemView:focus {
    border-color: $accent;
}

QSlider::groove:horizontal {
    height: 6px;
    background: $surface_sunken;
    border: 1px solid $border;
    border-radius: 3px;
    margin: 0 7px;
}
QSlider::sub-page:horizontal {
    background: $accent;
    border: none;
    border-radius: 3px;
}
QSlider::add-page:horizontal {
    background: $surface_sunken;
    border: none;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: $text;
    border: 2px solid $accent;
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: $accent;
}
/* Qt recomputes a sub-control's rect from the matching rule, so every state that
   restyles the handle has to repeat its size and margin or it gets stretched. */
QSlider:focus::handle:horizontal {
    background: $accent;
    border: 2px solid $text;
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 8px;
}

QCheckBox {
    spacing: 8px;
    color: $text;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid $border_strong;
    background-color: $surface_sunken;
}
QCheckBox::indicator:checked {
    background-color: $accent;
    border-color: $accent;
}
QCheckBox::indicator:hover {
    border-color: $accent;
}
QCheckBox:focus::indicator {
    border: 2px solid $accent;
    width: 16px;
    height: 16px;
    border-radius: 4px;
    background-color: $surface_sunken;
}

/* lists and tables */
QListWidget, QTableWidget, QTableView {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: 8px;
    alternate-background-color: $surface_alt;
}
QListWidget:focus, QTableWidget:focus, QTableView:focus {
    border-color: $accent;
}
QListWidget::item {
    padding: 8px 10px;
    border-bottom: 1px solid $border;
}
QListWidget::item:selected, QTableWidget::item:selected {
    background-color: $accent_soft;
    color: $text;
}
QListWidget::item:hover {
    background-color: $surface_alt;
}
QHeaderView::section {
    background-color: $surface_alt;
    color: $text_muted;
    border: none;
    border-bottom: 1px solid $border;
    padding: 7px 8px;
    font-weight: 600;
}

/* progress and scrollbars */
QProgressBar {
    background-color: $surface_alt;
    border: 1px solid $border;
    border-radius: 3px;
    height: 8px;
    text-align: center;
    color: $text_muted;
}
QProgressBar::chunk {
    background-color: $accent;
    border-radius: 2px;
}

QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: $border_strong;
    border-radius: 5px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover {
    background: $text_muted;
}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {
    background: none;
    height: 0;
    width: 0;
}

QScrollArea {
    border: none;
    background: transparent;
}

QStatusBar {
    background-color: $surface_sunken;
    color: $text_muted;
    border-top: 1px solid $border;
}
QStatusBar::item {
    border: none;
}
"""
)


def palette_for_style(style: str) -> Palette:
    return LIGHT if str(style).lower() == "light" else DARK


def stylesheet(palette: Palette) -> str:
    """Render the QSS with the palette substituted in."""
    return _TEMPLATE.substitute(
        font_size=BASE_FONT_SIZE_PT,
        title_size=TITLE_FONT_SIZE_PT,
        heading_size=HEADING_FONT_SIZE_PT,
        mono=MONO_FAMILY,
        **{
            field: str(getattr(palette, field))
            for field in palette.__dataclass_fields__
            if field != "name"
        },
    )


def _base_font() -> QFont:
    font = QFont()
    font.setFamilies(list(FONT_FAMILY_FALLBACKS))
    font.setPointSize(BASE_FONT_SIZE_PT)
    return font


def apply_theme(app: QApplication, palette: Palette) -> None:
    """Apply the palette to the application, including native dialogs."""
    app.setStyle("Fusion")
    app.setFont(_base_font())

    qt_palette = QPalette()
    qt_palette.setColor(QPalette.ColorRole.Window, QColor(palette.background))
    qt_palette.setColor(QPalette.ColorRole.WindowText, QColor(palette.text))
    qt_palette.setColor(QPalette.ColorRole.Base, QColor(palette.surface_sunken))
    qt_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(palette.surface_alt))
    qt_palette.setColor(QPalette.ColorRole.Text, QColor(palette.text))
    qt_palette.setColor(QPalette.ColorRole.Button, QColor(palette.surface_alt))
    qt_palette.setColor(QPalette.ColorRole.ButtonText, QColor(palette.text))
    qt_palette.setColor(QPalette.ColorRole.Highlight, QColor(palette.accent))
    qt_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(palette.text_inverse))
    qt_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(palette.surface_alt))
    qt_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(palette.text))
    qt_palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(palette.text_muted))
    app.setPalette(qt_palette)
    app.setStyleSheet(stylesheet(palette))
