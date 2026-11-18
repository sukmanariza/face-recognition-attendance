"""Identity management: what is stored, rename, and delete."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.database import DatabaseError, EmployeeNotFoundError

from . import widgets
from .context import AppContext


class UsersPage(QWidget):
    """Table of registered identities with the safe maintenance actions."""

    record_requested = Signal(str, str)

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._build_ui()
        context.gallery_changed.connect(self.refresh)
        self.refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(16)

        root.addWidget(
            widgets.PageHeader(
                "Identities",
                "Deactivating keeps the person's attendance history but stops new matches. "
                "Deleting removes the embeddings for good, and the ID is never reused.",
            )
        )

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["ID", "Name", "State", "Samples", "Registered"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setColumnWidth(0, 90)
        self._table.setColumnWidth(1, 240)
        self._table.setColumnWidth(2, 90)
        self._table.setColumnWidth(3, 90)
        self._table.itemSelectionChanged.connect(self._sync_buttons)
        root.addWidget(self._table, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        self._refresh_button = widgets.secondary_button("Refresh")
        self._refresh_button.clicked.connect(self.refresh)
        self._rename_button = widgets.secondary_button("Rename")
        self._rename_button.clicked.connect(self.rename_selected)
        self._record_button = widgets.secondary_button("Re-record face")
        self._record_button.clicked.connect(self.record_selected)
        self._state_button = widgets.secondary_button("Deactivate")
        self._state_button.clicked.connect(self.toggle_active)
        self._delete_button = widgets.danger_button("Delete")
        self._delete_button.clicked.connect(self.delete_selected)
        buttons.addWidget(self._refresh_button)
        buttons.addStretch(1)
        buttons.addWidget(self._rename_button)
        buttons.addWidget(self._record_button)
        buttons.addWidget(self._state_button)
        buttons.addWidget(self._delete_button)
        holder = QWidget()
        holder.setLayout(buttons)
        root.addWidget(holder)

        self._banner = widgets.Banner()
        root.addWidget(self._banner)

        self._count = QLabel("")
        self._count.setObjectName("Hint")
        root.addWidget(self._count)

        self._sync_buttons()

    def refresh(self) -> None:
        try:
            records = self._context.database().list_employees()
        except DatabaseError as exc:
            self._banner.show_message(f"Database problem: {exc}", "error")
            return

        self._table.setRowCount(len(records))
        for row, record in enumerate(records):
            values = (
                record.employee_id,
                record.full_name,
                "active" if record.active else "inactive",
                str(record.sample_count),
                record.created_at.replace("T", " ").replace("+00:00", " UTC"),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (0, 2, 3):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                    )
                self._table.setItem(row, column, item)

        active = sum(1 for record in records if record.active)
        total_samples = sum(record.sample_count for record in records)
        noun = "identity" if len(records) == 1 else "identities"
        self._count.setText(
            f"{len(records)} {noun} ({active} active), {total_samples} embeddings stored."
            if records
            else "No identities yet. Use 'Record a new face' on the Overview page."
        )
        self._sync_buttons()

    def _selected_id(self) -> str | None:
        row = self._table.currentRow()
        if row < 0:
            return None
        item = self._table.item(row, 0)
        return item.text() if item is not None else None

    def _selected_name(self) -> str:
        row = self._table.currentRow()
        if row < 0:
            return ""
        item = self._table.item(row, 1)
        return item.text() if item is not None else ""

    def _sync_buttons(self) -> None:
        has_selection = self._selected_id() is not None
        self._rename_button.setEnabled(has_selection)
        self._record_button.setEnabled(has_selection)
        self._state_button.setEnabled(has_selection)
        self._state_button.setText("Deactivate" if self._selected_active() else "Reactivate")
        self._delete_button.setEnabled(has_selection)

    def record_selected(self) -> None:
        """Ask the window to open the enrollment page in re-record mode."""
        employee_id = self._selected_id()
        if employee_id is None:
            return
        self.record_requested.emit(employee_id, self._selected_name())

    def rename_selected(self) -> None:
        employee_id = self._selected_id()
        if employee_id is None:
            return
        new_name, accepted = QInputDialog.getText(
            self, "Rename identity", "Full name:", text=self._selected_name()
        )
        if not accepted or not new_name.strip():
            return
        try:
            record = self._context.database().rename_employee(employee_id, new_name)
        except (DatabaseError, EmployeeNotFoundError) as exc:
            self._banner.show_message(str(exc), "error")
            return
        self._banner.show_message(f"Renamed {record.employee_id} to {record.full_name}.", "success")
        self.refresh()

    def _selected_active(self) -> bool:
        row = self._table.currentRow()
        if row < 0:
            return True
        item = self._table.item(row, 2)
        return item is None or item.text() != "inactive"

    def toggle_active(self) -> None:
        employee_id = self._selected_id()
        if employee_id is None:
            return
        target = not self._selected_active()
        try:
            record = self._context.database().set_employee_active(employee_id, target)
        except (DatabaseError, EmployeeNotFoundError) as exc:
            self._banner.show_message(str(exc), "error")
            return
        self._context.invalidate_gallery()
        state = "reactivated" if record.active else "deactivated"
        self._banner.show_message(
            f"{record.employee_id} ({record.full_name}) {state}.",
            "success" if record.active else "warning",
        )
        self.refresh()

    def delete_selected(self) -> None:
        employee_id = self._selected_id()
        if employee_id is None:
            return
        confirmed = QMessageBox.question(
            self,
            "Delete identity",
            f"Delete {employee_id} ({self._selected_name()}) and all of its embeddings?\n"
            "Its attendance history is deleted too. Use Deactivate instead if you only "
            "want to stop new matches.\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return
        try:
            deleted = self._context.database().delete_employee(employee_id)
        except DatabaseError as exc:
            self._banner.show_message(str(exc), "error")
            return
        self._context.invalidate_gallery()
        if deleted:
            self._banner.show_message(f"Deleted {employee_id}.", "success")
        else:
            self._banner.show_message(f"{employee_id} was not found.", "warning")
        self.refresh()
