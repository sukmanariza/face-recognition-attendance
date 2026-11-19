"""Attendance page: who is present, and the corrections an operator may need."""

from __future__ import annotations

from datetime import date

from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QBoxLayout,
    QDateEdit,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.database import DatabaseError

from . import widgets
from .context import AppContext

COLUMNS = ("ID", "Name", "First seen", "Last seen", "Status", "Sightings")

# Below this width the toolbar goes on two rows instead of one.
_STACK_BELOW_WIDTH = 640


class AttendancePage(QWidget):
    """Daily presence list with export and delete."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._build_ui()
        context.gallery_changed.connect(self.refresh)
        self.refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(14)

        root.addWidget(
            widgets.PageHeader(
                "Attendance",
                "A confirmed match marks the person present once per day. Further "
                "sightings update the last-seen time instead of adding a row.",
            )
        )

        self._toolbar = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self._toolbar.setSpacing(10)
        toolbar = self._toolbar
        toolbar.addWidget(QLabel("Date"))
        self._date = QDateEdit()
        self._date.setCalendarPopup(True)
        self._date.setDisplayFormat("yyyy-MM-dd")
        self._date.setDate(QDate.currentDate())
        self._date.dateChanged.connect(self.refresh)
        toolbar.addWidget(self._date)

        today_button = widgets.secondary_button("Today")
        today_button.clicked.connect(self._go_today)
        toolbar.addWidget(today_button)

        refresh_button = widgets.secondary_button("Refresh")
        refresh_button.clicked.connect(self.refresh)
        toolbar.addWidget(refresh_button)
        toolbar.addStretch(1)

        self._export_button = widgets.secondary_button("Export month")
        self._export_button.setToolTip("Write the selected month as a CSV file")
        self._export_button.clicked.connect(self.export_month)
        toolbar.addWidget(self._export_button)
        self._delete_button = widgets.danger_button("Delete row")
        self._delete_button.setToolTip("Remove the selected attendance row")
        self._delete_button.setEnabled(False)
        self._delete_button.clicked.connect(self.delete_selected)
        toolbar.addWidget(self._delete_button)

        toolbar_holder = QWidget()
        toolbar_holder.setLayout(toolbar)
        root.addWidget(toolbar_holder)

        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(list(COLUMNS))
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setColumnWidth(0, 80)
        self._table.setColumnWidth(1, 200)
        self._table.setColumnWidth(2, 90)
        self._table.setColumnWidth(3, 90)
        self._table.setColumnWidth(4, 80)
        self._table.itemSelectionChanged.connect(self._sync_buttons)
        root.addWidget(self._table, 1)

        self._summary = QLabel("")
        self._summary.setObjectName("Hint")
        root.addWidget(self._summary)

        self._banner = widgets.Banner()
        root.addWidget(self._banner)

        self._sync_buttons()
        self._apply_layout()

    def _apply_layout(self) -> None:
        """Put the toolbar on two rows when the window is narrow."""
        narrow = self.width() < _STACK_BELOW_WIDTH
        direction = (
            QBoxLayout.Direction.TopToBottom if narrow else QBoxLayout.Direction.LeftToRight
        )
        if self._toolbar.direction() != direction:
            self._toolbar.setDirection(direction)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_layout()

    def _go_today(self) -> None:
        self._date.setDate(QDate.currentDate())
        self.refresh()

    def _selected(self) -> int | None:
        row = self._table.currentRow()
        if row < 0:
            return None
        item = self._table.item(row, 0)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return int(value) if value is not None else None

    def _sync_buttons(self) -> None:
        self._delete_button.setEnabled(self._selected() is not None)

    def selected_date(self) -> str:
        value: QDate = self._date.date()
        return value.toString("yyyy-MM-dd")

    def refresh(self) -> None:
        work_date = self.selected_date()
        try:
            entries = self._context.application.attendance.day(work_date)
        except (DatabaseError, RuntimeError) as exc:
            self._banner.show_message(f"Could not read attendance: {exc}", "error")
            entries = []

        self._table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            values = (
                entry.employee_id,
                entry.full_name,
                entry.first_seen[11:19],
                entry.last_seen[11:19],
                entry.status,
                str(entry.sightings),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, entry.attendance_id)
                self._table.setItem(row, column, item)

        late = sum(1 for entry in entries if entry.status == "late")
        if entries:
            self._summary.setText(
                f"{len(entries)} present on {work_date}, {late} of them late."
            )
        else:
            self._summary.setText(
                f"Nobody recorded on {work_date}. Recognition marks attendance "
                "automatically while the camera is running."
            )
        self._sync_buttons()

    def export_month(self) -> None:
        today = date.fromisoformat(self.selected_date())
        first = today.replace(day=1)
        target = self._context.config.attendance_dir / f"attendance-{first:%Y-%m}.csv"
        try:
            path = self._context.application.attendance.export_csv(target, first, today)
        except (DatabaseError, OSError) as exc:
            self._banner.show_message(f"Export failed: {exc}", "error")
            return
        self._banner.show_message(f"Exported {path.name} to {path.parent}", "success")

    def delete_selected(self) -> None:
        attendance_id = self._selected()
        if attendance_id is None:
            return
        confirmed = QMessageBox.question(
            self,
            "Delete attendance",
            "Delete this attendance row? The person will be marked present again if "
            "the camera sees them today.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return
        try:
            deleted = self._context.application.attendance.delete(attendance_id)
        except DatabaseError as exc:
            self._banner.show_message(f"Could not delete: {exc}", "error")
            return
        if deleted:
            self._banner.show_message("Attendance row deleted.", "success")
        else:
            self._banner.show_message("That row no longer exists.", "warning")
        self.refresh()
