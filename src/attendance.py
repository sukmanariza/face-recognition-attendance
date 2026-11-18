"""Daily attendance: turn confirmed recognitions into presence records.

The rules are deliberately boring, because attendance is the part people argue
about:

* a person is recorded once per calendar day, guaranteed by a unique constraint in
  the database, not by application logic;
* further sightings the same day update ``last_seen`` and a counter, throttled by
  ``refresh_seconds`` so a face in front of the camera does not rewrite the row on
  every frame;
* the status is ``late`` when the first sighting is after the start time plus the
  grace period, otherwise ``present``;
* a face flagged by the presentation check is not recorded while
  ``require_live_face`` is on, which is what stops a photo on a phone from clocking
  somebody in.

The clock is the machine's clock. This is a local system: if the workstation clock is
wrong, the attendance is wrong, and the README says so instead of pretending
otherwise.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from config import AttendanceConfig

from .database import AttendanceEntry, AttendanceMark, FaceDatabase
from .presentation import PresentationVerdict

logger = logging.getLogger(__name__)

STATUS_PRESENT = "present"
STATUS_LATE = "late"
CSV_COLUMNS = (
    "work_date",
    "employee_id",
    "full_name",
    "first_seen",
    "last_seen",
    "status",
    "sightings",
    "source",
)


@dataclass(frozen=True)
class AttendanceDecision:
    """What the service decided about one sighting."""

    mark: AttendanceMark | None
    reason: str

    @property
    def recorded(self) -> bool:
        return self.mark is not None


def local_now() -> datetime:
    return datetime.now(UTC).astimezone()


class AttendanceService:
    """Record presence once per day and answer attendance questions."""

    def __init__(
        self,
        database: FaceDatabase,
        config: AttendanceConfig,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._database = database
        self._config = config
        self._clock = clock or local_now
        self._last: dict[str, AttendanceMark] = {}
        self._last_write: dict[str, datetime] = {}

    @property
    def config(self) -> AttendanceConfig:
        return self._config

    def today(self) -> str:
        return self._clock().date().isoformat()

    def status_for(self, moment: datetime) -> str:
        start = self._config.start_time()
        # Keep the moment's timezone: comparing an aware datetime with a naive one
        # raises, and the office clock is local time.
        beginning = moment.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
        late_after = beginning + timedelta(minutes=self._config.late_after_minutes)
        return STATUS_LATE if moment > late_after else STATUS_PRESENT

    def record(
        self,
        employee_id: str,
        presentation: PresentationVerdict | None = None,
        when: datetime | None = None,
    ) -> AttendanceDecision:
        """Mark one sighting, or explain why it was not recorded."""
        if not self._config.enabled:
            return AttendanceDecision(None, "Attendance is off in Settings.")
        if (
            self._config.require_live_face
            and presentation is not None
            and presentation.is_suspected
        ):
            return AttendanceDecision(
                None, "Not recorded: the face came from a photo or a screen."
            )

        moment = when or self._clock()
        work_date = moment.date().isoformat()
        key = f"{employee_id}:{work_date}"

        previous = self._last.get(key)
        last_write = self._last_write.get(key)
        if previous is not None and last_write is not None:
            elapsed = (moment - last_write).total_seconds()
            if elapsed < self._config.refresh_seconds:
                return AttendanceDecision(previous, "Already recorded today.")

        try:
            mark = self._database.mark_present(
                employee_id,
                work_date,
                moment.isoformat(timespec="seconds"),
                status=self.status_for(moment),
            )
        except Exception as exc:  # noqa: BLE001 - a bad write must not stop the stream
            logger.warning("Attendance write failed for %s: %s", employee_id, exc)
            return AttendanceDecision(None, f"Could not record attendance: {exc}")

        self._last[key] = mark
        self._last_write[key] = moment
        if mark.created:
            logger.info(
                "Attendance: %s %s (%s)", mark.entry.employee_id, work_date, mark.entry.status
            )
        return AttendanceDecision(mark, "Recorded." if mark.created else "Updated.")

    def day(self, work_date: str | None = None) -> list[AttendanceEntry]:
        return self._database.attendance_for(work_date or self.today())

    def present_count(self, work_date: str | None = None) -> int:
        try:
            return len(self.day(work_date))
        except Exception:  # noqa: BLE001 - the home tile must never raise
            return 0

    def delete(self, attendance_id: int) -> bool:
        return self._database.delete_attendance(attendance_id)

    def delete_for(self, employee_id: str, work_date: str | None = None) -> bool:
        return self._database.delete_attendance_for(employee_id, work_date or self.today())

    def export_csv(
        self, path: Path, start: date | None = None, end: date | None = None
    ) -> Path:
        """Write one row per attendance record to a UTF-8 CSV."""
        today = self._clock().date()
        first = start or today
        last = end or today
        entries = self._database.attendance_range(first.isoformat(), last.isoformat())
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            for entry in entries:
                writer.writerow(
                    [
                        entry.work_date,
                        entry.employee_id,
                        entry.full_name,
                        entry.first_seen,
                        entry.last_seen,
                        entry.status,
                        entry.sightings,
                        entry.source,
                    ]
                )
        return target

    def describe(self) -> str:
        start: time = self._config.start_time()
        return (
            f"{'enabled' if self._config.enabled else 'off'}, start {start.strftime('%H:%M')}, "
            f"late after {self._config.late_after_minutes} min, "
            f"{'live face required' if self._config.require_live_face else 'any match'}"
        )
