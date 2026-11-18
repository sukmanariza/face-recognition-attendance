"""Attendance: once per day, late detection, and the operator corrections."""

from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from config import AttendanceConfig
from src.attendance import AttendanceService
from src.database import EmployeeNotFoundError, FaceDatabase, FaceSample
from src.presentation import PresentationVerdict, SpoofLevel
from tests.support import unit_vector

MONDAY_MORNING = datetime(2026, 11, 16, 7, 50).astimezone()
MONDAY_LATE = datetime(2026, 11, 16, 8, 40).astimezone()
MONDAY_MIDDAY = datetime(2026, 11, 16, 12, 0).astimezone()


def sample(seed: int = 1) -> FaceSample:
    return FaceSample(embedding=unit_vector(seed, 8), det_score=0.9)


def verdict(level: SpoofLevel) -> PresentationVerdict:
    return PresentationVerdict(level=level, confidence=1.0, detail="test", observations=8)


class FakeClock:
    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment

    def set(self, moment: datetime) -> None:
        self.moment = moment


class AttendanceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.database = FaceDatabase(self.root / "faces.db")
        self.addCleanup(self.database.close)
        self.database.add_employee("001", "Ana", [sample(1)])
        self.database.add_employee("002", "Bob", [sample(2)])
        self.clock = FakeClock(MONDAY_MORNING)
        self.service = AttendanceService(
            self.database, AttendanceConfig(), self.clock
        )

    def first_entry(self):
        entries = self.service.day("2026-11-16")
        self.assertTrue(entries)
        return entries[0]

    def test_a_first_sighting_creates_one_row(self) -> None:
        decision = self.service.record("001")
        self.assertTrue(decision.recorded)
        mark = decision.mark
        self.assertIsNotNone(mark)
        assert mark is not None
        self.assertTrue(mark.created)
        self.assertEqual(mark.entry.status, "present")
        self.assertEqual(len(self.service.day("2026-11-16")), 1)

    def test_the_same_person_is_never_duplicated(self) -> None:
        self.service.record("001")
        self.clock.set(MONDAY_MIDDAY)
        second = self.service.record("001")
        self.assertTrue(second.recorded)
        self.assertFalse(second.mark.created)  # type: ignore[union-attr]
        self.assertEqual(len(self.service.day("2026-11-16")), 1)
        self.assertEqual(self.first_entry().sightings, 2)

    def test_repeat_sightings_are_throttled(self) -> None:
        self.service.record("001")
        writes_before = self.first_entry().sightings
        self.clock.set(MONDAY_MORNING.replace(second=30))
        self.service.record("001")
        self.assertEqual(self.first_entry().sightings, writes_before)

    def test_late_arrival_is_marked(self) -> None:
        self.clock.set(MONDAY_LATE)
        self.service.record("002")
        self.assertEqual(self.first_entry().status, "late")

    def test_grace_period_is_honoured(self) -> None:
        service = AttendanceService(
            self.database, AttendanceConfig(late_after_minutes=60), self.clock
        )
        self.clock.set(MONDAY_LATE)
        service.record("001")
        self.assertEqual(self.first_entry().status, "present")

    def test_a_photo_is_not_recorded(self) -> None:
        decision = self.service.record("001", verdict(SpoofLevel.SUSPECTED))
        self.assertFalse(decision.recorded)
        self.assertIn("photo", decision.reason.lower())
        self.assertEqual(self.service.day("2026-11-16"), [])

    def test_a_photo_is_recorded_when_the_rule_is_off(self) -> None:
        service = AttendanceService(
            self.database, AttendanceConfig(require_live_face=False), self.clock
        )
        self.assertTrue(service.record("001", verdict(SpoofLevel.SUSPECTED)).recorded)

    def test_disabled_attendance_records_nothing(self) -> None:
        service = AttendanceService(
            self.database, AttendanceConfig(enabled=False), self.clock
        )
        self.assertFalse(service.record("001").recorded)
        self.assertEqual(self.service.day("2026-11-16"), [])

    def test_unknown_identity_is_reported_not_raised(self) -> None:
        decision = self.service.record("999")
        self.assertFalse(decision.recorded)
        self.assertIn("could not record", decision.reason.lower())

    def test_present_count(self) -> None:
        self.service.record("001")
        self.assertEqual(self.service.present_count(), 1)

    def test_operator_can_delete_a_row(self) -> None:
        self.service.record("001")
        entry = self.first_entry()
        self.assertTrue(self.service.delete(entry.attendance_id))
        self.assertEqual(self.service.day("2026-11-16"), [])

    def test_operator_can_delete_by_identity(self) -> None:
        self.service.record("001")
        self.assertTrue(self.service.delete_for("001", "2026-11-16"))
        self.assertEqual(self.service.day("2026-11-16"), [])

    def test_export_writes_one_row_per_record(self) -> None:
        self.service.record("001")
        self.clock.set(MONDAY_LATE)
        self.service.record("002")
        target = self.root / "export" / "attendance.csv"
        path = self.service.export_csv(
            target,
            datetime(2026, 11, 16).date(),
            datetime(2026, 11, 16).date(),
        )
        rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
        self.assertEqual(rows[0][0], "work_date")
        self.assertEqual(len(rows), 3)
        self.assertEqual({row[2] for row in rows[1:]}, {"Ana", "Bob"})

    def test_describe_mentions_the_policy(self) -> None:
        text = self.service.describe()
        self.assertIn("08:00", text)
        self.assertIn("late after 15", text)


class DatabaseAttendanceTests(unittest.TestCase):
    """Storage-level guarantees, without the service in the way."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.database = FaceDatabase(Path(self._directory.name) / "faces.db")
        self.addCleanup(self.database.close)
        self.database.add_employee("001", "Ana", [sample(1)])
        self.database.add_employee("002", "Bob", [sample(2)])

    def test_unique_constraint_holds_at_the_storage_layer(self) -> None:
        self.database.mark_present("001", "2026-11-16", MONDAY_MORNING.isoformat())
        self.database.mark_present("001", "2026-11-16", MONDAY_LATE.isoformat())
        rows = self.database.attendance_for("2026-11-16")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].sightings, 2)

    def test_entries_are_ordered_by_first_seen(self) -> None:
        self.database.mark_present("002", "2026-11-16", MONDAY_LATE.isoformat())
        self.database.mark_present("001", "2026-11-16", MONDAY_MORNING.isoformat())
        rows = self.database.attendance_for("2026-11-16")
        self.assertEqual([row.employee_id for row in rows], ["001", "002"])

    def test_range_and_dates(self) -> None:
        self.database.mark_present("001", "2026-11-16", MONDAY_MORNING.isoformat())
        self.database.mark_present("001", "2026-11-17", MONDAY_MORNING.isoformat())
        self.assertEqual(len(self.database.attendance_range("2026-11-16", "2026-11-17")), 2)
        self.assertEqual(self.database.attendance_dates(), ["2026-11-17", "2026-11-16"])

    def test_deleting_an_identity_removes_its_attendance(self) -> None:
        self.database.mark_present("001", "2026-11-16", MONDAY_MORNING.isoformat())
        self.database.delete_employee("001")
        self.assertEqual(self.database.attendance_for("2026-11-16"), [])

    def test_marking_an_unknown_identity_raises(self) -> None:
        with self.assertRaises(EmployeeNotFoundError):
            self.database.mark_present("999", "2026-11-16", MONDAY_MORNING.isoformat())

    def test_audit_trail_records_the_change(self) -> None:
        self.database.mark_present("001", "2026-11-16", MONDAY_MORNING.isoformat())
        actions = [action for _at, action, _detail in self.database.audit_entries()]
        self.assertIn("attendance.mark", actions)


if __name__ == "__main__":
    unittest.main()
