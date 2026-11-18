"""Field encryption: key handling, round trips, and failure modes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.crypto import CryptoError, FieldCipher, rotate_key


class CipherTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def test_round_trip(self) -> None:
        cipher = FieldCipher.from_key_file(self.root / "key.bin")
        blob = cipher.encrypt_text("Ana Maria", "employees:001:full_name")
        self.assertNotIn(b"Ana Maria", blob)
        self.assertEqual(cipher.decrypt_text(blob, "employees:001:full_name"), "Ana Maria")

    def test_ciphertext_is_bound_to_its_place(self) -> None:
        cipher = FieldCipher.from_key_file(self.root / "key.bin")
        blob = cipher.encrypt_text("Ana", "employees:001:full_name")
        with self.assertRaises(CryptoError):
            cipher.decrypt_text(blob, "employees:002:full_name")

    def test_two_encryptions_differ(self) -> None:
        cipher = FieldCipher.from_key_file(self.root / "key.bin")
        first = cipher.encrypt_text("Ana", "ctx")
        second = cipher.encrypt_text("Ana", "ctx")
        self.assertNotEqual(first, second)

    def test_key_file_is_created_once_and_reused(self) -> None:
        path = self.root / "key.bin"
        first = FieldCipher.from_key_file(path)
        second = FieldCipher.from_key_file(path)
        blob = first.encrypt_text("Ana", "ctx")
        self.assertEqual(second.decrypt_text(blob, "ctx"), "Ana")

    def test_missing_key_can_refuse_to_be_created(self) -> None:
        with self.assertRaises(CryptoError):
            FieldCipher.from_key_file(self.root / "absent.bin", create=False)

    def test_short_key_is_rejected(self) -> None:
        with self.assertRaises(CryptoError):
            FieldCipher.from_key(b"too-short")

    def test_garbage_key_file_is_rejected(self) -> None:
        path = self.root / "bad.bin"
        path.write_bytes(b"not a key")
        with self.assertRaises(CryptoError):
            FieldCipher.from_key_file(path)

    def test_wrong_context_or_truncated_value_is_rejected(self) -> None:
        cipher = FieldCipher.from_key_file(self.root / "key.bin")
        with self.assertRaises(CryptoError):
            cipher.decrypt_text(b"short", "ctx")

    def test_passphrase_protected_key_needs_the_passphrase(self) -> None:
        path = self.root / "wrapped.bin"
        cipher = FieldCipher.from_key_file(path, passphrase="open sesame")
        blob = cipher.encrypt_text("Ana", "ctx")
        with self.assertRaises(CryptoError):
            FieldCipher.from_key_file(path)
        with self.assertRaises(CryptoError):
            FieldCipher.from_key_file(path, passphrase="wrong")
        again = FieldCipher.from_key_file(path, passphrase="open sesame")
        self.assertEqual(again.decrypt_text(blob, "ctx"), "Ana")

    def test_rotating_the_passphrase_keeps_the_data_readable(self) -> None:
        path = self.root / "rotating.bin"
        cipher = FieldCipher.from_key_file(path, passphrase="first")
        blob = cipher.encrypt_text("Ana", "ctx")
        rotate_key(path, "second", "first")
        rotated = FieldCipher.from_key_file(path, passphrase="second")
        self.assertEqual(rotated.decrypt_text(blob, "ctx"), "Ana")

    def test_rotating_to_no_passphrase_stores_a_plain_key(self) -> None:
        path = self.root / "plain.bin"
        FieldCipher.from_key_file(path, passphrase="first")
        rotate_key(path, None, "first")
        self.assertEqual(len(path.read_bytes()), 32)
        FieldCipher.from_key_file(path)


if __name__ == "__main__":
    unittest.main()
