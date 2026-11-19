"""Owner-only files: the key and the database hold personal data.

POSIX has file modes; Windows does not, so the rule is expressed with an ACL built
from the shipped ``icacls``. These tests pin the command that is issued, because a
silent no-op would read as "protected" in the docs while protecting nothing.
"""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import permissions


class PosixModeTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "POSIX modes only")
    def test_a_file_becomes_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key.bin"
            path.write_bytes(b"secret")

            permissions.restrict_to_owner(path)

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    @unittest.skipIf(os.name == "nt", "POSIX modes only")
    def test_a_directory_becomes_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "database"
            path.mkdir()

            permissions.restrict_to_owner(path, directory=True)

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)


class WindowsAclTests(unittest.TestCase):
    """The Windows path issues one icacls command that drops inherited entries."""

    def _run(self, path: Path, *, directory: bool = False) -> mock.Mock:
        with mock.patch.object(permissions.os, "name", "nt"), mock.patch.dict(
            os.environ, {"USERNAME": "owner"}, clear=False
        ), mock.patch.object(permissions.subprocess, "run") as runner:
            runner.return_value = subprocess.CompletedProcess([], 0, b"", b"")
            permissions.restrict_to_owner(path, directory=directory)
        return runner

    def test_a_file_grants_only_the_current_user(self) -> None:
        runner = self._run(Path("D:/data/face_key.bin"))

        runner.assert_called_once()
        command = runner.call_args.args[0]
        self.assertEqual(command[0], "icacls")
        self.assertIn("/inheritance:r", command)
        self.assertIn("owner:F", command)
        self.assertTrue(command[1].endswith("face_key.bin"))
        self.assertNotIn("shell", runner.call_args.kwargs)

    def test_directories_are_left_alone(self) -> None:
        runner = self._run(Path("D:/data"), directory=True)

        runner.assert_not_called()

    def test_a_missing_username_changes_nothing(self) -> None:
        with mock.patch.object(permissions.os, "name", "nt"), mock.patch.dict(
            os.environ, {"USERNAME": ""}, clear=False
        ), mock.patch.object(permissions.subprocess, "run") as runner:
            permissions.restrict_to_owner(Path("D:/data/face_key.bin"))

        runner.assert_not_called()

    def test_a_failing_icacls_is_swallowed(self) -> None:
        with mock.patch.object(permissions.os, "name", "nt"), mock.patch.dict(
            os.environ, {"USERNAME": "owner"}, clear=False
        ), mock.patch.object(
            permissions.subprocess, "run", side_effect=OSError("no icacls")
        ):
            permissions.restrict_to_owner(Path("D:/data/face_key.bin"))


class PrivateWriteTests(unittest.TestCase):
    def test_the_content_lands_and_no_temp_file_survives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key.bin"

            permissions.write_private_file(path, b"secret")

            self.assertEqual(path.read_bytes(), b"secret")
            self.assertEqual([entry.name for entry in path.parent.iterdir()], ["key.bin"])

    def test_an_existing_file_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key.bin"
            path.write_bytes(b"old")

            permissions.write_private_file(path, b"new")

            self.assertEqual(path.read_bytes(), b"new")

    def test_the_temporary_file_is_restricted_before_it_is_moved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key.bin"
            restricted: list[str] = []

            with mock.patch.object(
                permissions, "restrict_to_owner", side_effect=lambda target, **_: restricted.append(
                    Path(target).name
                )
            ):
                permissions.write_private_file(path, b"secret")

            self.assertIn("key.bin.tmp", restricted, "permissions were set after the move")
            self.assertEqual(restricted[0], "key.bin.tmp")


if __name__ == "__main__":
    unittest.main()
