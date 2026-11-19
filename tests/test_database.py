"""Database storage: identities, embeddings, ID handling, and safety checks."""

from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np

from config import load_config
from src import attendance as attendance_module
from src import database as database_module
from src.crypto import FieldCipher
from src.database import (
    DatabaseError,
    DuplicateEmployeeError,
    EmployeeNotFoundError,
    FaceDatabase,
    FaceSample,
    validate_employee_id,
    validate_full_name,
)
from tests.support import unit_vector


def sample(seed: int = 1, signature: str = "stub:test", dim: int = 8) -> FaceSample:
    return FaceSample(
        embedding=unit_vector(seed, dim),
        det_score=0.9,
        quality_score=0.7,
        yaw_ratio=0.1,
        roll_deg=2.0,
        model_signature=signature,
    )


class DatabaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.path = Path(self._directory.name) / "faces.db"
        self.database = FaceDatabase(self.path)
        self.addCleanup(self.database.close)


class ValidationTests(unittest.TestCase):
    def test_valid_ids_are_trimmed(self) -> None:
        self.assertEqual(validate_employee_id("  001 "), "001")

    def test_invalid_ids_are_rejected(self) -> None:
        for value in ("", "../x", "a b", "-lead", "x" * 40):
            with self.assertRaises(DatabaseError):
                validate_employee_id(value)

    def test_names_are_normalised(self) -> None:
        self.assertEqual(validate_full_name("  Ana   Maria "), "Ana Maria")

    def test_empty_name_is_rejected(self) -> None:
        with self.assertRaises(DatabaseError):
            validate_full_name("   ")

    def test_long_name_is_rejected(self) -> None:
        with self.assertRaises(DatabaseError):
            validate_full_name("x" * 80)

    def test_sample_rejects_a_zero_embedding(self) -> None:
        with self.assertRaises(DatabaseError):
            FaceSample(embedding=np.zeros(4, dtype=np.float32))

    def test_sample_rejects_non_finite_values(self) -> None:
        with self.assertRaises(DatabaseError):
            FaceSample(embedding=np.array([1.0, np.inf], dtype=np.float32))


class EmployeeTests(DatabaseTestCase):
    def test_add_and_read_back(self) -> None:
        record = self.database.add_employee("001", "Ana", [sample(1), sample(2)])
        self.assertEqual(record.employee_id, "001")
        self.assertEqual(record.full_name, "Ana")
        self.assertEqual(record.sample_count, 2)
        self.assertEqual(self.database.count_employees(), 1)

    def test_duplicate_id_is_rejected(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1)])
        with self.assertRaises(DuplicateEmployeeError):
            self.database.add_employee("001", "Other", [sample(2)])

    def test_list_is_sorted_by_id(self) -> None:
        self.database.add_employee("002", "B", [sample(1)])
        self.database.add_employee("001", "A", [sample(2)])
        self.assertEqual([r.employee_id for r in self.database.list_employees()], ["001", "002"])

    def test_get_unknown_returns_none(self) -> None:
        self.assertIsNone(self.database.get_employee("999"))

    def test_rename(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1)])
        record = self.database.rename_employee("001", "Ana Maria")
        self.assertEqual(record.full_name, "Ana Maria")

    def test_rename_unknown_raises(self) -> None:
        with self.assertRaises(EmployeeNotFoundError):
            self.database.rename_employee("404", "Nobody")

    def test_replace_samples(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1)])
        record = self.database.replace_samples("001", [sample(2), sample(3)])
        self.assertEqual(record.sample_count, 2)

    def test_replace_removes_the_old_embeddings(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1), sample(2)])
        self.database.replace_samples("001", [sample(3)])
        gallery = self.database.load_gallery()
        self.assertEqual(gallery[0].embeddings.shape[0], 1)

    def test_delete(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1)])
        self.assertTrue(self.database.delete_employee("001"))
        self.assertEqual(self.database.count_employees(), 0)

    def test_delete_unknown_returns_false(self) -> None:
        self.assertFalse(self.database.delete_employee("404"))

    def test_delete_last_identity_clears_model_metadata(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1)])
        self.assertEqual(self.database.model_signature, "stub:test")
        self.database.delete_employee("001")
        self.assertIsNone(self.database.model_signature)


class IdGenerationTests(DatabaseTestCase):
    def test_first_id_is_padded(self) -> None:
        self.assertEqual(self.database.next_employee_id(), "001")

    def test_next_id_follows_the_highest(self) -> None:
        self.database.add_employee("001", "A", [sample(1)])
        self.database.add_employee("005", "B", [sample(2)])
        self.assertEqual(self.database.next_employee_id(), "006")

    def test_deleted_ids_are_not_reused(self) -> None:
        self.database.add_employee("001", "A", [sample(1)])
        self.database.add_employee("002", "B", [sample(2)])
        self.database.delete_employee("002")
        self.assertEqual(self.database.next_employee_id(), "003")

    def test_non_numeric_ids_do_not_affect_generation(self) -> None:
        self.database.add_employee("staff-x", "A", [sample(1)])
        self.assertEqual(self.database.next_employee_id(), "001")

    def test_numeric_ids_are_zero_padded_on_open(self) -> None:
        self.database.add_employee("007", "A", [sample(1)])
        self.database.close()
        connection = sqlite3.connect(self.path)
        with connection:
            connection.execute("UPDATE employees SET employee_id = '7'")
        connection.close()

        reopened = FaceDatabase(self.path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.next_employee_id(), "008")
        self.assertIsNotNone(reopened.get_employee("007"))


class GalleryTests(DatabaseTestCase):
    def test_embeddings_round_trip_exactly(self) -> None:
        original = unit_vector(4)
        self.database.add_employee("001", "Ana", [FaceSample(embedding=original)])
        gallery = self.database.load_gallery()
        np.testing.assert_allclose(gallery[0].embeddings[0], original, rtol=1e-6)

    def test_groups_by_identity(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1), sample(2), sample(3)])
        gallery = self.database.load_gallery()
        self.assertEqual(len(gallery), 1)
        self.assertEqual(gallery[0].embeddings.shape, (3, 8))

    def test_empty_database_returns_empty_gallery(self) -> None:
        self.assertEqual(self.database.load_gallery(), [])

    def test_mixed_dimensions_are_rejected(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1, dim=8)])
        with self.assertRaises(DatabaseError):
            self.database.add_employee("002", "Bob", [sample(2, dim=16)])

    def test_different_model_signature_is_rejected(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1, signature="buffalo_s:w600k_mbf")])
        with self.assertRaises(DatabaseError):
            self.database.add_employee("002", "Bob", [sample(2, signature="buffalo_l:w600k_r50")])

    def test_empty_sample_list_is_rejected(self) -> None:
        with self.assertRaises(DatabaseError):
            self.database.add_employee("001", "Ana", [])


class LifecycleTests(DatabaseTestCase):
    def test_closed_database_refuses_queries(self) -> None:
        self.database.close()
        with self.assertRaises(DatabaseError):
            self.database.list_employees()

    def test_context_manager_closes(self) -> None:
        with FaceDatabase(self.path) as database:
            database.add_employee("001", "Ana", [sample(1)])
        with self.assertRaises(DatabaseError):
            database.list_employees()

    def test_parent_directory_is_created(self) -> None:
        nested = Path(self._directory.name) / "deep" / "faces.db"
        with FaceDatabase(nested) as database:
            self.assertEqual(database.count_employees(), 0)

    def test_summary_line_mentions_the_sample_count(self) -> None:
        record = self.database.add_employee("001", "Ana", [sample(1), sample(2)])
        self.assertIn("2 samples", record.summary)


class ThreadTests(DatabaseTestCase):
    """A sqlite3 connection belongs to the thread that made it.

    The app reads and writes from the GUI thread, the enrollment worker, and the
    inference thread that records attendance, so the database has to hand each
    thread its own connection instead of sharing one.
    """

    def _run_in_thread(self, action) -> dict[str, object]:
        outcome: dict[str, object] = {}

        def target() -> None:
            try:
                outcome["value"] = action()
            except Exception as exc:  # noqa: BLE001 - reported to the assertion below
                outcome["error"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        return outcome

    def test_a_worker_thread_can_read_the_database(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1)])
        self.database.next_employee_id()  # the GUI thread opens its connection first

        outcome = self._run_in_thread(
            lambda: (self.database.count_employees(), self.database.next_employee_id())
        )

        self.assertNotIn("error", outcome, outcome.get("error"))
        self.assertEqual(outcome["value"], (1, "002"))

    def test_a_worker_thread_can_write_the_database(self) -> None:
        outcome = self._run_in_thread(
            lambda: self.database.add_employee("002", "Budi", [sample(2)]).full_name
        )

        self.assertNotIn("error", outcome, outcome.get("error"))
        self.assertEqual(outcome["value"], "Budi")
        self.assertEqual(self.database.count_employees(), 1)

    def test_closing_releases_the_connection_of_a_worker_thread(self) -> None:
        it_worked = self._run_in_thread(self.database.count_employees)
        self.assertNotIn("error", it_worked, it_worked.get("error"))

        self.database.close()

        with self.assertRaises(DatabaseError):
            self.database.count_employees()
        after_close = self._run_in_thread(self.database.count_employees)
        self.assertIn("error", after_close)


class WriteLockTests(unittest.TestCase):
    """No path may leave a transaction open, or the next save reports a locked file.

    A sighting committed the attendance row and then wrote the audit entry outside
    any transaction. That insert opened one and nothing closed it, so the connection
    held the write lock for the life of the process, and the next identity could not
    be saved: "database is locked".
    """

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        # A short timeout turns a regression into a fast failure instead of a hang.
        self.database = FaceDatabase(
            Path(self._directory.name) / "faces.db", busy_timeout_ms=250
        )
        self.addCleanup(self.database.close)
        self.database.add_employee("001", "Ana", [sample(1)])

    def _run_in_thread(self, action) -> dict[str, object]:
        outcome: dict[str, object] = {}

        def target() -> None:
            try:
                outcome["value"] = action()
            except Exception as exc:  # noqa: BLE001 - reported to the assertion below
                outcome["error"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        return outcome

    def test_marking_attendance_commits_its_audit_entry(self) -> None:
        marked = self._run_in_thread(
            lambda: self.database.mark_present("001", "2026-11-19", "08:00")
        )

        self.assertNotIn("error", marked, marked.get("error"))
        self.assertFalse(
            self.database._connection.in_transaction,
            "an open transaction would hold the write lock",
        )

    def test_a_sighting_does_not_block_the_next_identity(self) -> None:
        sighting = self._run_in_thread(
            lambda: self.database.mark_present("001", "2026-11-19", "08:00")
        )
        self.assertNotIn("error", sighting, sighting.get("error"))

        saved = self._run_in_thread(
            lambda: self.database.add_employee("002", "Budi", [sample(2)]).full_name
        )

        self.assertNotIn("error", saved, saved.get("error"))
        self.assertEqual(saved["value"], "Budi")

    def test_finished_threads_do_not_accumulate_connections(self) -> None:
        for _ in range(4):
            self.assertNotIn("error", self._run_in_thread(self.database.count_employees))

        # The GUI thread plus at most the most recent worker.
        self.assertLessEqual(len(self.database._connections), 2)


class EncryptionTests(unittest.TestCase):
    """Values are encrypted at rest and need the key to be read back."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.key_file = self.root / "face_key.bin"
        self.cipher = FieldCipher.from_key_file(self.key_file)
        self.database = FaceDatabase(self.root / "faces.db", cipher=self.cipher)
        self.addCleanup(self.database.close)

    def test_name_and_embeddings_survive_a_reopen(self) -> None:
        self.database.add_employee("001", "Ana Maria", [sample(1), sample(2)])
        self.database.close()
        again = FaceDatabase(self.root / "faces.db", cipher=self.cipher)
        self.addCleanup(again.close)
        record = again.get_employee("001")
        assert record is not None
        self.assertEqual(record.full_name, "Ana Maria")
        gallery = again.load_gallery()
        self.assertEqual(gallery[0].full_name, "Ana Maria")
        self.assertEqual(gallery[0].embeddings.shape[0], 2)

    def test_the_file_does_not_contain_the_name(self) -> None:
        self.database.add_employee("001", "Ana Maria", [sample(1)])
        self.database.close()
        raw = (self.root / "faces.db").read_bytes()
        self.assertNotIn(b"Ana Maria", raw)
        self.assertNotIn(b"Ana", raw)

    def test_the_file_does_not_contain_the_embedding(self) -> None:
        vector = unit_vector(7)
        self.database.add_employee("001", "Ana", [FaceSample(embedding=vector)])
        self.database.close()
        raw = (self.root / "faces.db").read_bytes()
        self.assertNotIn(vector.astype(np.float32).tobytes(), raw)

    def test_a_missing_key_is_reported(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1)])
        self.database.close()
        with self.assertRaises(DatabaseError):
            FaceDatabase(self.root / "faces.db")

    def test_a_different_key_cannot_read_the_names(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1)])
        self.database.close()
        other = FieldCipher.from_key_file(self.root / "other_key.bin")
        database = FaceDatabase(self.root / "faces.db", cipher=other)
        self.addCleanup(database.close)
        with self.assertRaises(DatabaseError):
            database.list_employees()

    def test_a_plain_text_database_is_encrypted_on_open(self) -> None:
        plain_path = self.root / "legacy.db"
        legacy = FaceDatabase(plain_path)
        legacy.add_employee("001", "Legacy Person", [sample(1)])
        legacy.close()
        self.assertIn(b"Legacy Person", plain_path.read_bytes())

        upgraded = FaceDatabase(plain_path, cipher=self.cipher)
        self.addCleanup(upgraded.close)
        record = upgraded.get_employee("001")
        assert record is not None
        self.assertEqual(record.full_name, "Legacy Person")
        self.assertTrue(upgraded.encrypted)
        upgraded.close()
        self.assertNotIn(b"Legacy Person", plain_path.read_bytes())

    def test_backup_opens_with_the_same_key(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1)])
        target = self.database.backup_to(self.root / "copy" / "faces.db")
        copy = FaceDatabase(target, cipher=self.cipher)
        self.addCleanup(copy.close)
        self.assertEqual(copy.count_employees(), 1)


class ActivationTests(DatabaseTestCase):
    def test_deactivating_hides_the_identity_from_matching(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1)])
        self.database.set_employee_active("001", False)
        self.assertEqual(self.database.load_gallery(), [])
        self.assertEqual(len(self.database.load_gallery(only_active=False)), 1)

    def test_inactive_identities_can_be_filtered_out_of_the_list(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1)])
        self.database.set_employee_active("001", False)
        self.assertEqual(self.database.list_employees(include_inactive=False), [])
        self.assertEqual(len(self.database.list_employees()), 1)

    def test_reactivating_restores_matching(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1)])
        self.database.set_employee_active("001", False)
        record = self.database.set_employee_active("001", True)
        self.assertTrue(record.active)
        self.assertEqual(len(self.database.load_gallery()), 1)

    def test_the_state_shows_in_the_summary(self) -> None:
        self.database.add_employee("001", "Ana", [sample(1)])
        record = self.database.set_employee_active("001", False)
        self.assertIn("inactive", record.summary)


class DerivedFilePermissionTests(unittest.TestCase):
    """Every file that ends up holding biometric rows must be owner-only.

    The database file was restricted but the files SQLite creates beside it were not,
    so the WAL and SHM were readable by other accounts on the machine. A backup and a
    CSV export are full copies of the same data and had the same gap.
    """

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.path = Path(self._directory.name) / "faces.db"

    def test_the_database_and_its_sidecar_files_are_restricted(self) -> None:
        with mock.patch.object(database_module, "restrict_to_owner") as restrict:
            database = FaceDatabase(self.path, busy_timeout_ms=500)
            self.addCleanup(database.close)
            database.add_employee("001", "Ana", [sample(1)])

        restricted = {Path(call.args[0]).name for call in restrict.call_args_list}
        self.assertIn("faces.db", restricted)
        self.assertIn("faces.db-wal", restricted)
        self.assertIn("faces.db-shm", restricted)

    def test_a_backup_is_created_owner_only_before_it_is_written(self) -> None:
        database = FaceDatabase(self.path, busy_timeout_ms=500)
        self.addCleanup(database.close)
        database.add_employee("001", "Ana", [sample(1)])
        target = Path(self._directory.name) / "backups" / "copy.db"

        with mock.patch.object(
            database_module, "prepare_private_file", wraps=database_module.prepare_private_file
        ) as prepare:
            database.backup_to(target)

        prepare.assert_called_once_with(target)
        self.assertTrue(target.is_file())

    def test_the_csv_export_is_created_owner_only(self) -> None:
        from src.attendance import AttendanceService

        database = FaceDatabase(self.path, busy_timeout_ms=500)
        self.addCleanup(database.close)
        database.add_employee("001", "Ana", [sample(1)])
        database.mark_present("001", "2026-11-19", "08:00")
        service = AttendanceService(database, load_config().attendance)
        target = Path(self._directory.name) / "attendance" / "2026-11.csv"

        with mock.patch.object(
            attendance_module, "prepare_private_file", wraps=attendance_module.prepare_private_file
        ) as prepare:
            service.export_csv(target, date(2026, 11, 19), date(2026, 11, 19))

        prepare.assert_called_once_with(target)
        self.assertIn("Ana", target.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
