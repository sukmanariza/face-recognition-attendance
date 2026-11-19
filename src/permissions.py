"""Keep the files that hold biometric data readable by their owner only.

The key file and the database are personal data. POSIX has file modes; Windows does
not, so the same rule is expressed with an ACL instead of silently skipping it. Both
paths are best effort: a failure is logged and never stops the app, because a broken
ACL must not make the program unusable.

Only files are restricted on Windows. Changing the ACL of the folders would also hide
the project from tools the owner runs, and a folder is not what leaks here: the key
and the database are.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_FILE_MODE = 0o600
_DIRECTORY_MODE = 0o700


def restrict_to_owner(path: Path, *, directory: bool = False) -> None:
    """Limit a file (or directory) to the current user, on POSIX and on Windows."""
    if os.name == "posix":
        with contextlib.suppress(OSError):
            os.chmod(path, _DIRECTORY_MODE if directory else _FILE_MODE)
        return
    if directory:
        return
    _restrict_windows(path)


def _restrict_windows(path: Path) -> None:
    """Drop inherited ACEs and grant the current user, using the shipped icacls.

    ``/inheritance:r`` removes what the parent folder grants, so a permissive
    profile or a shared drive no longer exposes the key; ``/grant:r`` replaces the
    user's existing entries instead of adding a second one.
    """
    user = os.environ.get("USERNAME", "").strip()
    if not user:
        logger.debug("No USERNAME in the environment; leaving %s unchanged", path)
        return
    command = ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Could not restrict %s to its owner: %s", path, exc)
        return
    if result.returncode != 0:  # pragma: no cover - depends on the file system
        logger.debug(
            "icacls returned %d for %s: %s",
            result.returncode,
            path,
            result.stderr.decode("utf-8", "replace").strip(),
        )


def write_private_file(path: Path, payload: bytes) -> None:
    """Write ``payload`` so the file is never readable with default permissions.

    The bytes go to a sibling temp file that is restricted first and moved into place
    afterwards, so there is no window where the content exists with a permissive ACL
    (or a world-readable mode, on POSIX).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with contextlib.suppress(OSError):
        temporary.unlink()
    try:
        temporary.write_bytes(payload)
        restrict_to_owner(temporary)
        os.replace(temporary, path)
    except OSError:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    restrict_to_owner(path)
