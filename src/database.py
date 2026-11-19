"""SQLite storage for identities, embeddings, attendance, and the audit trail.

Design notes:

* Embeddings and full names are encrypted with AES-GCM before they are written (see
  `src/crypto.py`). Without a cipher the database stays in plain text, which is only
  used by tests and by an explicit configuration.
* ``schema_meta`` records the embedding dimension, the recognition model signature,
  the id sequence, and the encryption mode. Two different model packs can produce
  512-dimensional vectors that are not comparable, so a dimension check alone would
  let mismatched embeddings silently produce meaningless scores.
* Employee IDs are generated (`001`, `002`, ...) and never reused, so an attendance
  record can never be attributed to a different person later.
* Attendance has a ``UNIQUE(employee_pk, work_date)`` constraint, so a duplicate for
  the same person and the same day is impossible even if two callers race.
* Identities are deactivated rather than deleted by default: the person stops being
  recognised, while their attendance history stays intact.
* One SQLite connection is opened per thread. A ``sqlite3.Connection`` may only be
  used by the thread that created it, and this database is touched from the GUI
  thread, the enrollment worker, and the inference thread that records attendance.
  WAL mode plus the busy timeout let those connections share one file.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .crypto import CryptoError, FieldCipher
from .permissions import restrict_to_owner
from .similarity import IdentityRecord

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3
ENCRYPTION_MODE = "aesgcm:v1"

_EMPLOYEE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
_MAX_NAME_LENGTH = 64
_EMPLOYEE_ID_WIDTH = 3

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS employees (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT    NOT NULL UNIQUE,
        full_name   BLOB    NOT NULL,
        created_at  TEXT    NOT NULL,
        updated_at  TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_pk   INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
        embedding     BLOB    NOT NULL,
        embedding_dim INTEGER NOT NULL,
        det_score     REAL    NOT NULL DEFAULT 0.0,
        quality_score REAL    NOT NULL DEFAULT 0.0,
        yaw_ratio     REAL    NOT NULL DEFAULT 0.0,
        roll_deg      REAL    NOT NULL DEFAULT 0.0,
        created_at    TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_embeddings_owner ON face_embeddings(employee_pk)",
    """
    CREATE TABLE IF NOT EXISTS attendance (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_pk INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
        work_date   TEXT    NOT NULL,
        first_seen  TEXT    NOT NULL,
        last_seen   TEXT    NOT NULL,
        sightings   INTEGER NOT NULL DEFAULT 1,
        status      TEXT    NOT NULL,
        source      TEXT    NOT NULL DEFAULT 'recognition',
        UNIQUE (employee_pk, work_date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(work_date)",
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id     INTEGER PRIMARY KEY AUTOINCREMENT,
        at     TEXT NOT NULL,
        action TEXT NOT NULL,
        detail TEXT NOT NULL
    )
    """,
)


class DatabaseError(RuntimeError):
    """Storage failed or the data is inconsistent."""


class DuplicateEmployeeError(DatabaseError):
    """The employee ID or the face is already registered."""


class EmployeeNotFoundError(DatabaseError):
    """No identity with the given ID exists."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _restrict_permissions(path: Path, mode: int) -> None:
    """Restrict a file or directory to its owner: POSIX modes, Windows ACL."""
    restrict_to_owner(path, directory=mode != 0o600)


def validate_employee_id(employee_id: str) -> str:
    cleaned = employee_id.strip()
    if not _EMPLOYEE_ID_PATTERN.match(cleaned):
        raise DatabaseError(
            f"Invalid employee ID {employee_id!r}. Use 1-32 characters: letters, digits, "
            "dot, underscore, or hyphen, starting with a letter or digit."
        )
    return cleaned


def validate_full_name(full_name: str) -> str:
    cleaned = " ".join(full_name.split())
    if not cleaned:
        raise DatabaseError("Name must not be empty.")
    if len(cleaned) > _MAX_NAME_LENGTH:
        raise DatabaseError(f"Name is longer than {_MAX_NAME_LENGTH} characters.")
    return cleaned


def _name_context(employee_id: str) -> str:
    return f"employees:{employee_id}:full_name"


def _embedding_context(employee_id: str) -> str:
    return f"face_embeddings:{employee_id}"


@dataclass(frozen=True)
class FaceSample:
    """One embedding plus the conditions it was captured under."""

    embedding: np.ndarray
    det_score: float = 0.0
    quality_score: float = 0.0
    yaw_ratio: float = 0.0
    roll_deg: float = 0.0
    model_signature: str = ""

    def __post_init__(self) -> None:
        vector = np.asarray(self.embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise DatabaseError("Sample embedding is empty.")
        if not np.all(np.isfinite(vector)):
            raise DatabaseError("Sample embedding contains non-finite values.")
        if float(np.linalg.norm(vector)) <= 1e-6:
            raise DatabaseError("Sample embedding has zero length and can never be matched.")
        object.__setattr__(self, "embedding", vector)

    @property
    def dimension(self) -> int:
        return int(self.embedding.size)


@dataclass(frozen=True)
class EmployeeRecord:
    employee_id: str
    full_name: str
    created_at: str
    sample_count: int
    active: bool = True

    @property
    def summary(self) -> str:
        state = "" if self.active else "  [inactive]"
        return (
            f"{self.employee_id:<8} {self.full_name[:28]:<28} "
            f"{self.sample_count:>3} samples   registered {self.created_at}{state}"
        )


@dataclass(frozen=True)
class AttendanceEntry:
    """One person's attendance row for one day."""

    attendance_id: int
    employee_id: str
    full_name: str
    work_date: str
    first_seen: str
    last_seen: str
    sightings: int
    status: str
    source: str

    @property
    def summary(self) -> str:
        return (
            f"{self.employee_id:<8} {self.full_name[:26]:<26} {self.work_date} "
            f"in {self.first_seen[11:19]}  last {self.last_seen[11:19]}  "
            f"{self.status:<8} {self.sightings:>4} sighting(s)"
        )


@dataclass(frozen=True)
class AttendanceMark:
    """Result of marking one sighting."""

    entry: AttendanceEntry
    created: bool


class FaceDatabase:
    """Thin, strictly validated layer over one SQLite file.

    Connections are per thread: SQLite refuses to let a connection leave the thread
    that created it, and the enrollment worker and the inference thread both read
    and write here. Every connection points at the same file and shares the WAL, so
    the busy timeout settles the write lock instead of raising.
    """

    def __init__(
        self,
        path: str | Path,
        busy_timeout_ms: int = 5000,
        cipher: FieldCipher | None = None,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _restrict_permissions(self._path.parent, 0o700)
        self._embedding_dimension: int | None = None
        self._model_signature: str | None = None
        self._cipher = cipher
        self._busy_timeout_s = busy_timeout_ms / 1000.0
        self._lock = threading.Lock()
        # Thread id -> (owning thread, connection). The owner is kept so a finished
        # thread's connection can be closed instead of holding the write lock.
        self._connections: dict[int, tuple[threading.Thread, sqlite3.Connection]] = {}
        self._local = threading.local()
        self._closed = False

        try:
            connection = self._open_connection()
        except sqlite3.Error as exc:  # pragma: no cover - depends on the filesystem
            raise DatabaseError(f"Could not open database {self._path}: {exc}") from exc
        self._local.connection = connection
        try:
            self._ensure_schema()
        except Exception:
            # A wrong key or a corrupt file must not leave the handle open: on
            # Windows an open handle also keeps the file locked for the next attempt.
            self.close()
            raise

    def _open_connection(self) -> sqlite3.Connection:
        """A fresh connection for the calling thread.

        ``check_same_thread=False`` only lifts the ownership check so that
        :meth:`close` can shut down connections owned by threads that already
        finished; each connection is still used by one thread at a time.
        """
        connection = sqlite3.connect(
            str(self._path), timeout=self._busy_timeout_s, check_same_thread=False
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            _restrict_permissions(self._path, 0o600)
        except sqlite3.Error:
            connection.close()
            raise
        self._prune_finished_threads()
        with self._lock:
            if self._closed:
                connection.close()
                raise DatabaseError(
                    f"Database {self._path} is closed. Create a new FaceDatabase to use it again."
                )
            self._connections[threading.get_ident()] = (threading.current_thread(), connection)
        return connection

    def _prune_finished_threads(self) -> None:
        """Close the connections of threads that have ended.

        A worker thread that stops while a statement is uncommitted would otherwise
        keep the database write lock for the life of the process, and the next save
        anywhere in the app fails with "database is locked". Closing the connection
        rolls that transaction back.
        """
        current = threading.current_thread()
        with self._lock:
            stale = [
                ident
                for ident, (owner, _) in self._connections.items()
                if owner is not current and not owner.is_alive()
            ]
            expired = [self._connections.pop(ident)[1] for ident in stale]
        for connection in expired:
            connection.close()

    @property
    def _connection(self) -> sqlite3.Connection:
        """This thread's connection, opened on first use."""
        if self._closed:
            raise DatabaseError(
                f"Database {self._path} is closed. Create a new FaceDatabase to use it again."
            )
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            return connection
        try:
            connection = self._open_connection()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not open database {self._path}: {exc}") from exc
        self._local.connection = connection
        return connection

    def __enter__(self) -> FaceDatabase:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close every connection, including the ones opened by other threads."""
        with self._lock:
            owned = list(self._connections.values())
            self._connections = {}
            self._closed = True
        for _, connection in owned:
            connection.close()
        self._local = threading.local()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def encrypted(self) -> bool:
        return self._cipher is not None

    @property
    def model_signature(self) -> str | None:
        """Recognition model recorded at registration time."""
        return self._model_signature

    def _ensure_schema(self) -> None:
        try:
            with self._connection:
                for statement in _SCHEMA_STATEMENTS:
                    self._connection.execute(statement)
                self._connection.execute(
                    "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)",
                    ("schema_version", str(SCHEMA_VERSION)),
                )
            self._migrate_schema()
            self._embedding_dimension = self._read_int_meta("embedding_dimension")
            self._model_signature = self._meta("model_signature")
            self._migrate_employee_ids()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not prepare the database schema: {exc}") from exc

    def _meta(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM schema_meta WHERE key = ?", (key,)
        ).fetchone()
        return str(row["value"]) if row is not None else None

    def _read_int_meta(self, key: str) -> int | None:
        value = self._meta(key)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:  # pragma: no cover - only a corrupt file
            return None

    def _set_meta(self, key: str, value: str) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)", (key, value)
        )

    def _migrate_schema(self) -> None:
        """Add columns and encryption to databases written by an earlier version."""
        columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(employees)").fetchall()
        }
        if "active" not in columns:
            with self._connection:
                self._connection.execute(
                    "ALTER TABLE employees ADD COLUMN active INTEGER NOT NULL DEFAULT 1"
                )
            logger.info("Added the 'active' column to employees.")

        recorded = self._meta("encryption")
        if recorded == ENCRYPTION_MODE:
            if self._cipher is None:
                raise DatabaseError(
                    f"{self._path} is encrypted, but no key was supplied. Keep the key file "
                    "(database/face_key.bin) with the database, or turn encryption off "
                    "deliberately in Settings."
                )
            return
        if recorded not in (None, ""):
            raise DatabaseError(f"{self._path} uses an unknown encryption mode {recorded!r}.")
        if self._cipher is None:
            return
        self._encrypt_existing_rows()

    def _encrypt_existing_rows(self) -> None:
        """Encrypt rows written by an older plain-text version."""
        cipher = self._cipher
        if cipher is None:  # pragma: no cover - guarded by the caller
            return
        rows = self._connection.execute(
            "SELECT id, employee_id, full_name FROM employees"
        ).fetchall()
        sample_rows = self._connection.execute(
            "SELECT face_embeddings.id, employees.employee_id, face_embeddings.embedding "
            "FROM face_embeddings JOIN employees ON employees.id = face_embeddings.employee_pk"
        ).fetchall()
        if not rows and not sample_rows:
            with self._connection:
                self._set_meta("encryption", ENCRYPTION_MODE)
            return

        logger.warning(
            "Encrypting %d identities and %d embeddings already in %s",
            len(rows),
            len(sample_rows),
            self._path,
        )
        try:
            with self._connection:
                for row in rows:
                    employee_id = str(row["employee_id"])
                    raw = row["full_name"]
                    payload = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
                    self._connection.execute(
                        "UPDATE employees SET full_name = ? WHERE id = ?",
                        (cipher.encrypt(payload, _name_context(employee_id)), int(row["id"])),
                    )
                for row in sample_rows:
                    employee_id = str(row["employee_id"])
                    raw = row["embedding"]
                    payload = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
                    self._connection.execute(
                        "UPDATE face_embeddings SET embedding = ? WHERE id = ?",
                        (cipher.encrypt(payload, _embedding_context(employee_id)), int(row["id"])),
                    )
                self._set_meta("encryption", ENCRYPTION_MODE)
        except (sqlite3.Error, CryptoError) as exc:
            raise DatabaseError(f"Could not encrypt the existing database: {exc}") from exc
        # An UPDATE leaves the old plain-text pages in the file and the WAL; checkpoint
        # and VACUUM rewrite both, so no readable copy of the names survives.
        try:
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._connection.execute("VACUUM")
        except sqlite3.Error as exc:  # pragma: no cover - disk full or locked
            logger.warning("Could not compact the database after encrypting it: %s", exc)

    def _migrate_employee_ids(self) -> None:
        """Zero-pad numeric IDs written by earlier versions (``1`` -> ``001``).

        Only fully numeric IDs are renamed, and only when the target is free;
        otherwise the row is left alone so no identity is overwritten.
        """
        rows = self._connection.execute("SELECT id, employee_id FROM employees").fetchall()
        existing = {str(row["employee_id"]) for row in rows}
        renames: list[tuple[str, int]] = []
        for row in rows:
            current = str(row["employee_id"])
            if not current.isdigit():
                continue
            normalized = current.zfill(_EMPLOYEE_ID_WIDTH)
            if normalized == current:
                continue
            if normalized in existing:
                logger.warning("Left ID %r unchanged: %r is already taken.", current, normalized)
                continue
            renames.append((normalized, int(row["id"])))

        if renames:
            with self._connection:
                for normalized, pk in renames:
                    self._connection.execute(
                        "UPDATE employees SET employee_id = ? WHERE id = ?", (normalized, pk)
                    )
            logger.info("Normalised %d employee ID(s).", len(renames))
        with self._connection:
            self._set_meta("schema_version", str(SCHEMA_VERSION))

    def next_employee_id(self) -> str:
        """Next free generated ID (``001``, ``002``, ...).

        The highest ID ever issued is tracked in ``schema_meta``, so deleting the
        newest identity does not free its ID for somebody else.
        """
        rows = self._connection.execute("SELECT employee_id FROM employees").fetchall()
        taken = {str(row["employee_id"]) for row in rows}
        highest = max((int(value) for value in taken if value.isdigit()), default=0)
        highest = max(highest, self._id_sequence())
        candidate = highest + 1
        while f"{candidate:0{_EMPLOYEE_ID_WIDTH}d}" in taken:
            candidate += 1
        return f"{candidate:0{_EMPLOYEE_ID_WIDTH}d}"

    def _id_sequence(self) -> int:
        return self._read_int_meta("employee_id_sequence") or 0

    def _bump_id_sequence(self, numeric_id: int) -> None:
        if numeric_id <= self._id_sequence():
            return
        self._set_meta("employee_id_sequence", str(numeric_id))

    def _record_model_identity(self, dimension: int, signature: str) -> None:
        """Store or verify the embedding dimension and model signature."""
        if self._embedding_dimension is None:
            with self._connection:
                self._set_meta("embedding_dimension", str(dimension))
                if signature:
                    self._set_meta("model_signature", signature)
            self._embedding_dimension = dimension
            if signature:
                self._model_signature = signature
            return

        if dimension != self._embedding_dimension:
            raise DatabaseError(
                f"This database holds {self._embedding_dimension}-dimensional embeddings, but "
                f"the current model produces {dimension}. Use the model used for registration, "
                "or start a new database."
            )
        if signature and self._model_signature and signature != self._model_signature:
            raise DatabaseError(
                f"This database holds embeddings from '{self._model_signature}', but the current "
                f"model is '{signature}'. Embeddings from different models cannot be compared. "
                "Restore model.name in config.yaml, or delete the database and register again."
            )

    def _encrypt_name(self, employee_id: str, full_name: str) -> bytes | str:
        if self._cipher is None:
            return full_name
        return self._cipher.encrypt_text(full_name, _name_context(employee_id))

    def _decrypt_name(self, employee_id: str, value: object) -> str:
        if self._cipher is None:
            return str(value)
        if isinstance(value, str):  # pragma: no cover - only a half-migrated row
            return value
        try:
            return self._cipher.decrypt_text(bytes(value), _name_context(employee_id))  # type: ignore[arg-type]
        except CryptoError as exc:
            raise DatabaseError(f"Could not read the name of {employee_id}: {exc}") from exc

    def _encode_embedding(self, employee_id: str, embedding: np.ndarray) -> bytes:
        raw = embedding.astype(np.float32).tobytes()
        if self._cipher is None:
            return raw
        return self._cipher.encrypt(raw, _embedding_context(employee_id))

    def _decode_embedding(self, employee_id: str, value: object) -> np.ndarray:
        if self._cipher is None:
            return np.frombuffer(value, dtype=np.float32).copy()  # type: ignore[arg-type]
        try:
            raw = self._cipher.decrypt(bytes(value), _embedding_context(employee_id))  # type: ignore[arg-type]
        except CryptoError as exc:
            raise DatabaseError(f"Could not read an embedding of {employee_id}: {exc}") from exc
        return np.frombuffer(raw, dtype=np.float32).copy()

    def _row_to_record(self, row: sqlite3.Row) -> EmployeeRecord:
        employee_id = str(row["employee_id"])
        return EmployeeRecord(
            employee_id=employee_id,
            full_name=self._decrypt_name(employee_id, row["full_name"]),
            created_at=str(row["created_at"]),
            sample_count=int(row["sample_count"]),
            active=bool(row["active"]),
        )

    def list_employees(self, include_inactive: bool = True) -> list[EmployeeRecord]:
        clause = "" if include_inactive else "WHERE e.active = 1"
        rows = self._connection.execute(
            f"""
            SELECT e.employee_id, e.full_name, e.created_at, e.active,
                   COUNT(f.id) AS sample_count
            FROM employees AS e
            LEFT JOIN face_embeddings AS f ON f.employee_pk = e.id
            {clause}
            GROUP BY e.id
            ORDER BY e.employee_id COLLATE NOCASE
            """
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_employee(self, employee_id: str) -> EmployeeRecord | None:
        cleaned = validate_employee_id(employee_id)
        row = self._connection.execute(
            """
            SELECT e.employee_id, e.full_name, e.created_at, e.active,
                   COUNT(f.id) AS sample_count
            FROM employees AS e
            LEFT JOIN face_embeddings AS f ON f.employee_pk = e.id
            WHERE e.employee_id = ?
            GROUP BY e.id
            """,
            (cleaned,),
        ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def load_gallery(self, only_active: bool = True) -> list[IdentityRecord]:
        """All identities with every embedding, ready for matching."""
        clause = "WHERE e.active = 1" if only_active else ""
        rows = self._connection.execute(
            f"""
            SELECT e.employee_id, e.full_name, f.embedding, f.embedding_dim
            FROM employees AS e
            JOIN face_embeddings AS f ON f.employee_pk = e.id
            {clause}
            ORDER BY e.employee_id COLLATE NOCASE, f.id
            """
        ).fetchall()
        if not rows:
            return []

        grouped: dict[str, tuple[str, list[np.ndarray]]] = {}
        for row in rows:
            dimension = int(row["embedding_dim"])
            if self._embedding_dimension is None:
                self._embedding_dimension = dimension
            elif dimension != self._embedding_dimension:
                raise DatabaseError(
                    f"Inconsistent embedding sizes in the database: {dimension} vs "
                    f"{self._embedding_dimension}."
                )
            employee_id = str(row["employee_id"])
            vector = self._decode_embedding(employee_id, row["embedding"])
            if vector.size != dimension:
                raise DatabaseError(
                    f"Corrupt embedding for {employee_id!r}: "
                    f"{vector.size} values, expected {dimension}."
                )
            if employee_id not in grouped:
                grouped[employee_id] = (self._decrypt_name(employee_id, row["full_name"]), [])
            grouped[employee_id][1].append(vector)

        return [
            IdentityRecord(employee_id, full_name, np.stack(vectors).astype(np.float32))
            for employee_id, (full_name, vectors) in grouped.items()
        ]

    def sample_count(self, employee_id: str) -> int:
        record = self.get_employee(employee_id)
        return record.sample_count if record is not None else 0

    def count_employees(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS total FROM employees").fetchone()
        return int(row["total"]) if row is not None else 0

    def _validate_samples(self, samples: Sequence[FaceSample]) -> list[FaceSample]:
        if not samples:
            raise DatabaseError("At least one face sample is required.")
        dimensions = {sample.dimension for sample in samples}
        if len(dimensions) != 1:
            raise DatabaseError(f"Samples have mixed embedding sizes: {sorted(dimensions)}.")
        dimension = dimensions.pop()
        signatures = {s.model_signature for s in samples if s.model_signature}
        if len(signatures) > 1:
            raise DatabaseError(f"Samples come from different models: {sorted(signatures)}.")
        signature = signatures.pop() if signatures else ""
        self._record_model_identity(dimension, signature)
        return list(samples)

    def _insert_samples(
        self, employee_pk: int, employee_id: str, samples: Iterable[FaceSample]
    ) -> None:
        timestamp = _utc_now()
        self._connection.executemany(
            """
            INSERT INTO face_embeddings
                (employee_pk, embedding, embedding_dim, det_score, quality_score,
                 yaw_ratio, roll_deg, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    employee_pk,
                    self._encode_embedding(employee_id, sample.embedding),
                    sample.dimension,
                    float(sample.det_score),
                    float(sample.quality_score),
                    float(sample.yaw_ratio),
                    float(sample.roll_deg),
                    timestamp,
                )
                for sample in samples
            ],
        )

    def _reload_record(self, employee_id: str) -> EmployeeRecord:
        """Read back a row that was just written, so callers get real sample counts."""
        record = self.get_employee(employee_id)
        if record is None:  # pragma: no cover - only if the row vanished mid-call
            raise DatabaseError(f"Identity {employee_id!r} was not found after being saved.")
        return record

    def add_employee(
        self, employee_id: str, full_name: str, samples: Sequence[FaceSample]
    ) -> EmployeeRecord:
        cleaned_id = validate_employee_id(employee_id)
        cleaned_name = validate_full_name(full_name)
        prepared = self._validate_samples(samples)
        if self.get_employee(cleaned_id) is not None:
            raise DuplicateEmployeeError(
                f"ID {cleaned_id!r} is already registered. Use 'Re-record face' to replace its "
                "samples, or pick another ID."
            )

        timestamp = _utc_now()
        try:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    INSERT INTO employees (employee_id, full_name, created_at, updated_at, active)
                    VALUES (?, ?, ?, ?, 1)
                    """,
                    (cleaned_id, self._encrypt_name(cleaned_id, cleaned_name), timestamp, timestamp),
                )
                new_row_id = cursor.lastrowid
                if new_row_id is None:  # pragma: no cover - SQLite always returns one
                    raise DatabaseError(f"Database returned no row ID for {cleaned_id!r}.")
                self._insert_samples(int(new_row_id), cleaned_id, prepared)
                if cleaned_id.isdigit():
                    self._bump_id_sequence(int(cleaned_id))
                self._audit("employee.add", f"{cleaned_id} with {len(prepared)} sample(s)")
        except sqlite3.IntegrityError as exc:
            raise DuplicateEmployeeError(f"ID {cleaned_id!r} already exists: {exc}") from exc
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not save identity {cleaned_id!r}: {exc}") from exc

        return self._reload_record(cleaned_id)

    def replace_samples(
        self, employee_id: str, samples: Sequence[FaceSample]
    ) -> EmployeeRecord:
        cleaned_id = validate_employee_id(employee_id)
        if self.get_employee(cleaned_id) is None:
            raise EmployeeNotFoundError(f"ID {cleaned_id!r} is not registered.")
        prepared = self._validate_samples(samples)
        try:
            with self._connection:
                row = self._connection.execute(
                    "SELECT id FROM employees WHERE employee_id = ?", (cleaned_id,)
                ).fetchone()
                employee_pk = int(row["id"])
                self._connection.execute(
                    "DELETE FROM face_embeddings WHERE employee_pk = ?", (employee_pk,)
                )
                self._connection.execute(
                    "UPDATE employees SET updated_at = ? WHERE employee_id = ?",
                    (_utc_now(), cleaned_id),
                )
                self._insert_samples(employee_pk, cleaned_id, prepared)
                self._audit("employee.record", f"{cleaned_id} with {len(prepared)} sample(s)")
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not replace samples for {cleaned_id!r}: {exc}") from exc

        return self._reload_record(cleaned_id)

    def rename_employee(self, employee_id: str, full_name: str) -> EmployeeRecord:
        cleaned_id = validate_employee_id(employee_id)
        cleaned_name = validate_full_name(full_name)
        if self.get_employee(cleaned_id) is None:
            raise EmployeeNotFoundError(f"ID {cleaned_id!r} is not registered.")
        try:
            with self._connection:
                self._connection.execute(
                    "UPDATE employees SET full_name = ?, updated_at = ? WHERE employee_id = ?",
                    (self._encrypt_name(cleaned_id, cleaned_name), _utc_now(), cleaned_id),
                )
                self._audit("employee.rename", f"{cleaned_id} renamed")
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not rename {cleaned_id!r}: {exc}") from exc
        return self._reload_record(cleaned_id)

    def set_employee_active(self, employee_id: str, active: bool) -> EmployeeRecord:
        """Activate or deactivate an identity without touching its history."""
        cleaned_id = validate_employee_id(employee_id)
        if self.get_employee(cleaned_id) is None:
            raise EmployeeNotFoundError(f"ID {cleaned_id!r} is not registered.")
        try:
            with self._connection:
                self._connection.execute(
                    "UPDATE employees SET active = ?, updated_at = ? WHERE employee_id = ?",
                    (1 if active else 0, _utc_now(), cleaned_id),
                )
                self._audit("employee.activate" if active else "employee.deactivate", cleaned_id)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not update {cleaned_id!r}: {exc}") from exc
        return self._reload_record(cleaned_id)

    def delete_employee(self, employee_id: str) -> bool:
        cleaned_id = validate_employee_id(employee_id)
        try:
            with self._connection:
                cursor = self._connection.execute(
                    "DELETE FROM employees WHERE employee_id = ?", (cleaned_id,)
                )
                deleted = cursor.rowcount > 0
                if deleted:
                    self._audit("employee.delete", f"{cleaned_id} and its attendance history")
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not delete {cleaned_id!r}: {exc}") from exc

        if deleted and self.count_employees() == 0:
            # Clearing the model metadata lets a different model be used next time.
            try:
                with self._connection:
                    self._connection.execute(
                        "DELETE FROM schema_meta WHERE key IN (?, ?)",
                        ("embedding_dimension", "model_signature"),
                    )
            except sqlite3.Error as exc:  # pragma: no cover - rare
                raise DatabaseError(f"Could not clear database metadata: {exc}") from exc
            self._embedding_dimension = None
            self._model_signature = None
        return deleted

    def _audit(self, action: str, detail: str) -> None:
        """Append to the local audit trail (best effort, never blocks a write).

        The row is committed on its own. An audit entry written while no other
        transaction is open would leave one open, and that holds the database write
        lock until the connection is closed.
        """
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO audit_log(at, action, detail) VALUES (?, ?, ?)",
                    (_utc_now(), action, detail),
                )
        except sqlite3.Error:  # pragma: no cover - the trail must not break a save
            logger.warning("Could not write an audit entry for %s", action)

    def audit_entries(self, limit: int = 50) -> list[tuple[str, str, str]]:
        rows = self._connection.execute(
            "SELECT at, action, detail FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [(str(row["at"]), str(row["action"]), str(row["detail"])) for row in rows]

    def mark_present(
        self,
        employee_id: str,
        work_date: str,
        when: str,
        status: str = "present",
        source: str = "recognition",
    ) -> AttendanceMark:
        """Record one sighting for a day, creating the row only once.

        The unique constraint on (employee, date) is the guarantee: a second call
        updates ``last_seen`` and the counter instead of inserting a duplicate.
        """
        cleaned_id = validate_employee_id(employee_id)
        row = self._connection.execute(
            "SELECT id FROM employees WHERE employee_id = ?", (cleaned_id,)
        ).fetchone()
        if row is None:
            raise EmployeeNotFoundError(f"ID {cleaned_id!r} is not registered.")
        employee_pk = int(row["id"])

        try:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    INSERT INTO attendance
                        (employee_pk, work_date, first_seen, last_seen, sightings, status, source)
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                    ON CONFLICT(employee_pk, work_date) DO NOTHING
                    """,
                    (employee_pk, work_date, when, when, status, source),
                )
                created = cursor.rowcount > 0
                if not created:
                    self._connection.execute(
                        """
                        UPDATE attendance
                        SET last_seen = ?, sightings = sightings + 1
                        WHERE employee_pk = ? AND work_date = ?
                        """,
                        (when, employee_pk, work_date),
                    )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not record attendance for {cleaned_id!r}: {exc}") from exc

        entry = self._attendance_entry(employee_pk, work_date)
        if entry is None:  # pragma: no cover - the row was just written
            raise DatabaseError(f"Attendance for {cleaned_id!r} on {work_date} disappeared.")
        if created:
            self._audit("attendance.mark", f"{cleaned_id} {work_date} {status}")
        return AttendanceMark(entry=entry, created=created)

    def _attendance_entry(self, employee_pk: int, work_date: str) -> AttendanceEntry | None:
        row = self._connection.execute(
            """
            SELECT a.id, a.work_date, a.first_seen, a.last_seen, a.sightings, a.status,
                   a.source, e.employee_id, e.full_name
            FROM attendance AS a
            JOIN employees AS e ON e.id = a.employee_pk
            WHERE a.employee_pk = ? AND a.work_date = ?
            """,
            (employee_pk, work_date),
        ).fetchone()
        return self._row_to_attendance(row) if row is not None else None

    def _row_to_attendance(self, row: sqlite3.Row) -> AttendanceEntry:
        employee_id = str(row["employee_id"])
        return AttendanceEntry(
            attendance_id=int(row["id"]),
            employee_id=employee_id,
            full_name=self._decrypt_name(employee_id, row["full_name"]),
            work_date=str(row["work_date"]),
            first_seen=str(row["first_seen"]),
            last_seen=str(row["last_seen"]),
            sightings=int(row["sightings"]),
            status=str(row["status"]),
            source=str(row["source"]),
        )

    def attendance_for(self, work_date: str) -> list[AttendanceEntry]:
        rows = self._connection.execute(
            """
            SELECT a.id, a.work_date, a.first_seen, a.last_seen, a.sightings, a.status,
                   a.source, e.employee_id, e.full_name
            FROM attendance AS a
            JOIN employees AS e ON e.id = a.employee_pk
            WHERE a.work_date = ?
            ORDER BY a.first_seen
            """,
            (work_date,),
        ).fetchall()
        return [self._row_to_attendance(row) for row in rows]

    def attendance_range(self, start_date: str, end_date: str) -> list[AttendanceEntry]:
        rows = self._connection.execute(
            """
            SELECT a.id, a.work_date, a.first_seen, a.last_seen, a.sightings, a.status,
                   a.source, e.employee_id, e.full_name
            FROM attendance AS a
            JOIN employees AS e ON e.id = a.employee_pk
            WHERE a.work_date BETWEEN ? AND ?
            ORDER BY a.work_date, a.first_seen
            """,
            (start_date, end_date),
        ).fetchall()
        return [self._row_to_attendance(row) for row in rows]

    def attendance_dates(self, limit: int = 30) -> list[str]:
        rows = self._connection.execute(
            "SELECT DISTINCT work_date FROM attendance ORDER BY work_date DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [str(row["work_date"]) for row in rows]

    def delete_attendance(self, attendance_id: int) -> bool:
        """Remove one attendance row (a mistake or a correction by an operator)."""
        try:
            with self._connection:
                cursor = self._connection.execute(
                    "DELETE FROM attendance WHERE id = ?", (int(attendance_id),)
                )
                deleted = cursor.rowcount > 0
                if deleted:
                    self._audit("attendance.delete", f"row {attendance_id}")
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not delete attendance {attendance_id}: {exc}") from exc
        return deleted

    def delete_attendance_for(self, employee_id: str, work_date: str) -> bool:
        cleaned_id = validate_employee_id(employee_id)
        try:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    DELETE FROM attendance
                    WHERE work_date = ?
                      AND employee_pk = (SELECT id FROM employees WHERE employee_id = ?)
                    """,
                    (work_date, cleaned_id),
                )
                deleted = cursor.rowcount > 0
                if deleted:
                    self._audit("attendance.delete", f"{cleaned_id} {work_date}")
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not delete attendance for {cleaned_id!r}: {exc}") from exc
        return deleted

    def backup_to(self, destination: str | Path) -> Path:
        """Write a consistent copy of the database (WAL included) to ``destination``."""
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        # `with sqlite3.connect(...)` commits but does not close, which would leave
        # the copy locked on Windows, so the handle is closed explicitly.
        copy = None
        try:
            copy = sqlite3.connect(str(target))
            self._connection.backup(copy)
            copy.commit()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not back up to {target}: {exc}") from exc
        finally:
            if copy is not None:
                copy.close()
        return target
