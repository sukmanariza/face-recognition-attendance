"""Field-level encryption for the values that must not sit in the clear.

The database holds two kinds of sensitive value: the embeddings (biometric) and the
full name (personal data). Both are encrypted with AES-GCM before they reach SQLite,
so the file on disk never contains them readable, and a copy of the file alone is
useless without the key. Row ids, timestamps, scores and the attendance calendar stay
readable so the app can query and sort without decrypting everything.

Each value is bound to its place with associated data (for example
``employees:001:full_name``), so a ciphertext copied from one row into another fails
to decrypt instead of silently moving a name to a different person.

Key handling:

* the master key is 32 random bytes in ``database/face_key.bin``, owner-only;
* if a passphrase is supplied (``FACEID_PASSPHRASE`` or the ``--passphrase`` flag),
  the key file stores that key wrapped with AES-GCM under a scrypt-derived key, so
  copying the file is not enough;
* the key is never written to the database, never logged, and never printed.

SQLite itself is not encrypted; only these columns are. That keeps the dependency
list small (no SQLCipher build) and the queries plain, at the cost of the structure
being visible: an attacker with the file can see how many identities exist and when
they were registered, but not who they are or what their faces look like.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any

from .permissions import write_private_file

KEY_BYTES = 32
NONCE_BYTES = 12
SALT_BYTES = 16
_WRAP_MAGIC = b"FID1"
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
PASSPHRASE_ENV = "FACEID_PASSPHRASE"


class CryptoError(RuntimeError):
    """The key is missing, wrong, or the ciphertext is not usable."""


def _load_aead() -> tuple[Any, Any]:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise CryptoError(
            "Database encryption needs the 'cryptography' package. "
            "Run `pip install -r requirements.txt` (or `python install.py`)."
        ) from exc
    return AESGCM, Scrypt


def passphrase_from_environment() -> str | None:
    """Passphrase from ``FACEID_PASSPHRASE``, if the operator set one."""
    value = os.environ.get(PASSPHRASE_ENV, "").strip()
    return value or None


def _derive(passphrase: str, salt: bytes) -> bytes:
    _AESGCM, Scrypt = _load_aead()
    kdf = Scrypt(salt=salt, length=KEY_BYTES, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return kdf.derive(passphrase.encode("utf-8"))


class FieldCipher:
    """Encrypt and decrypt single values with AES-GCM."""

    def __init__(self, key: bytes) -> None:
        if len(key) != KEY_BYTES:
            raise CryptoError(f"Key must be {KEY_BYTES} bytes, got {len(key)}.")
        AESGCM, _Scrypt = _load_aead()
        self._aead = AESGCM(key)  # type: ignore[operator]

    @classmethod
    def from_key_file(
        cls,
        path: Path,
        passphrase: str | None = None,
        create: bool = True,
    ) -> FieldCipher:
        """Load the key file, creating it when it does not exist yet."""
        path = Path(path)
        if path.is_file():
            return cls(_read_key_file(path, passphrase))
        if not create:
            raise CryptoError(
                f"Encryption key not found at {path}. Restore your backup of the key, "
                "or start a new database."
            )
        return cls(_create_key_file(path, passphrase))

    @classmethod
    def from_key(cls, key: bytes) -> FieldCipher:
        return cls(key)

    def encrypt(self, data: bytes, context: str) -> bytes:
        """Return ``nonce || ciphertext`` bound to ``context``."""
        nonce = secrets.token_bytes(NONCE_BYTES)
        sealed = self._aead.encrypt(nonce, data, context.encode("utf-8"))  # type: ignore[attr-defined]
        return nonce + sealed

    def decrypt(self, blob: bytes, context: str) -> bytes:
        if len(blob) <= NONCE_BYTES:
            raise CryptoError("Value is too short to be encrypted data.")
        nonce, sealed = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
        try:
            return self._aead.decrypt(nonce, sealed, context.encode("utf-8"))  # type: ignore[attr-defined]
        except Exception as exc:
            raise CryptoError(
                "Value could not be decrypted: the key does not match this database, "
                "or the file was modified."
            ) from exc

    def encrypt_text(self, text: str, context: str) -> bytes:
        return self.encrypt(text.encode("utf-8"), context)

    def decrypt_text(self, blob: bytes, context: str) -> str:
        try:
            return self.decrypt(blob, context).decode("utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - only a corrupt value
            raise CryptoError("Decrypted value is not valid text.") from exc


def _read_key_file(path: Path, passphrase: str | None) -> bytes:
    raw = path.read_bytes()
    if len(raw) == KEY_BYTES:
        return raw
    if not raw.startswith(_WRAP_MAGIC):
        raise CryptoError(
            f"{path} is not a valid key file. Restore it from a backup, or remove it and "
            "re-register every face."
        )
    if not passphrase:
        raise CryptoError(
            "This database key is protected by a passphrase. Set "
            f"{PASSPHRASE_ENV} or pass --passphrase."
        )
    salt = raw[len(_WRAP_MAGIC) : len(_WRAP_MAGIC) + SALT_BYTES]
    wrapped = raw[len(_WRAP_MAGIC) + SALT_BYTES :]
    AESGCM, _Scrypt = _load_aead()
    cipher = AESGCM(_derive(passphrase, salt))  # type: ignore[operator]
    try:
        return cipher.decrypt(wrapped[:NONCE_BYTES], wrapped[NONCE_BYTES:], _WRAP_MAGIC)
    except Exception as exc:
        raise CryptoError("Wrong passphrase for this database key.") from exc


def _create_key_file(path: Path, passphrase: str | None) -> bytes:
    key = secrets.token_bytes(KEY_BYTES)
    payload = key if not passphrase else _wrap(key, passphrase)
    if path.exists():
        # Another process created it between the check and here; its key wins, because
        # overwriting it would strand every value already encrypted with that key.
        return _read_key_file(path, passphrase)
    try:
        write_private_file(path, payload)
    except OSError as exc:
        raise CryptoError(f"Could not create the key file {path}: {exc}") from exc
    return key


def _wrap(key: bytes, passphrase: str) -> bytes:
    salt = secrets.token_bytes(SALT_BYTES)
    AESGCM, _Scrypt = _load_aead()
    cipher = AESGCM(_derive(passphrase, salt))  # type: ignore[operator]
    nonce = secrets.token_bytes(NONCE_BYTES)
    sealed = cipher.encrypt(nonce, key, _WRAP_MAGIC)
    return _WRAP_MAGIC + salt + nonce + sealed


def rotate_key(key_path: Path, new_passphrase: str | None, old_passphrase: str | None) -> None:
    """Re-wrap the key file with a new passphrase (no database change needed).

    The key itself stays the same, so every stored value remains readable; only the
    protection of the key file changes.
    """
    key = _read_key_file(Path(key_path), old_passphrase)
    payload = key if not new_passphrase else _wrap(key, new_passphrase)
    write_private_file(Path(key_path), payload)
