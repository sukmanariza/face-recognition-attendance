"""Derived hardening checks: model names, archives, and configuration guards."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from config import ConfigError, build_config
from src.face_detector import (
    MODEL_ZOO_URL_TEMPLATE,
    ModelLoadError,
    _extract_archive,
    _validate_model_name,
)


class ModelNameTests(unittest.TestCase):
    def test_plain_names_are_accepted(self) -> None:
        for name in ("buffalo_s", "buffalo_l", "antelopev2", "my-pack.v2"):
            self.assertEqual(_validate_model_name(name), name)

    def test_traversal_is_rejected(self) -> None:
        for name in ("../evil", "..", "a/b", "a\\b", "/etc/passwd"):
            with self.assertRaises(ModelLoadError):
                _validate_model_name(name)

    def test_empty_name_is_rejected(self) -> None:
        with self.assertRaises(ModelLoadError):
            _validate_model_name("   ")

    def test_url_template_uses_the_name_only(self) -> None:
        self.assertIn("{name}", MODEL_ZOO_URL_TEMPLATE)
        self.assertTrue(MODEL_ZOO_URL_TEMPLATE.startswith("https://"))


def build_archive(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)


class ArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def test_normal_archive_is_extracted(self) -> None:
        archive = self.root / "model.zip"
        build_archive(archive, {"model.onnx": b"weights"})
        target = self.root / "extracted"
        _extract_archive(archive, target)
        self.assertTrue((target / "model.onnx").is_file())

    def test_traversal_entry_is_rejected(self) -> None:
        archive = self.root / "evil.zip"
        build_archive(archive, {"../escaped.onnx": b"x"})
        target = self.root / "safe"
        with self.assertRaises(ModelLoadError):
            _extract_archive(archive, target)
        self.assertFalse((self.root / "escaped.onnx").exists())

    def test_absolute_path_entry_is_rejected(self) -> None:
        archive = self.root / "absolute.zip"
        build_archive(archive, {"/tmp/escaped.onnx": b"x"})
        with self.assertRaises(ModelLoadError):
            _extract_archive(archive, self.root / "safe")

    def test_symlink_entry_is_rejected(self) -> None:
        archive = self.root / "link.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            info = zipfile.ZipInfo("link.onnx")
            info.external_attr = (0o120777 << 16)
            bundle.writestr(info, "/etc/passwd")
        with self.assertRaises(ModelLoadError):
            _extract_archive(archive, self.root / "safe")

    def test_too_many_members_are_rejected(self) -> None:
        archive = self.root / "many.zip"
        build_archive(archive, {f"file{index}.onnx": b"x" for index in range(80)})
        with self.assertRaises(ModelLoadError):
            _extract_archive(archive, self.root / "safe")

    def test_executable_entry_is_rejected(self) -> None:
        for suffix in (".dll", ".exe", ".sh", ".py"):
            archive = self.root / f"payload{suffix}.zip"
            build_archive(archive, {f"helper{suffix}": b"MZ", "det_500m.onnx": b"weights"})
            with self.assertRaises(ModelLoadError, msg=suffix):
                _extract_archive(archive, self.root / "safe")

    def test_corrupt_archive_is_rejected(self) -> None:
        archive = self.root / "broken.zip"
        archive.write_bytes(b"not a zip file")
        with self.assertRaises(zipfile.BadZipFile):
            _extract_archive(archive, self.root / "safe")

    def test_drive_letter_entry_cannot_escape_the_target(self) -> None:
        """A Windows absolute path is rejected; on POSIX it stays inside the target."""
        archive = self.root / "drive.zip"
        build_archive(archive, {"C:/escaped.onnx": b"x"})
        target = self.root / "safe"
        try:
            _extract_archive(archive, target)
        except ModelLoadError:
            self.assertFalse((self.root / "escaped.onnx").exists())
            return

        resolved_target = target.resolve()
        for written in target.rglob("*"):
            resolved = written.resolve()
            self.assertTrue(
                resolved == resolved_target or resolved_target in resolved.parents,
                f"{resolved} escaped {resolved_target}",
            )


class ConfigGuardTests(unittest.TestCase):
    def test_liveness_mode_is_validated(self) -> None:
        with self.assertRaises(ConfigError):
            build_config({"liveness": {"mode": "trust-me"}})

    def test_removed_liveness_key_is_rejected(self) -> None:
        # The old boolean is gone; a stale settings.json must fail loudly instead
        # of silently leaving the check off.
        with self.assertRaises(ConfigError):
            build_config({"liveness": {"enabled": True}})

    def test_logging_level_is_validated(self) -> None:
        with self.assertRaises(ConfigError):
            build_config({"logging": {"level": "verbose"}})

    def test_ui_style_is_validated(self) -> None:
        with self.assertRaises(ConfigError):
            build_config({"ui": {"style": "neon"}})

    def test_registration_sample_count_is_bounded(self) -> None:
        with self.assertRaises(ConfigError):
            build_config({"registration": {"samples": 50}})

    def test_camera_index_must_be_non_negative(self) -> None:
        with self.assertRaises(ConfigError):
            build_config({"camera": {"index": -1}})

    def test_provider_list_must_be_a_list(self) -> None:
        with self.assertRaises(ConfigError):
            build_config({"model": {"providers": "CPUExecutionProvider"}})


if __name__ == "__main__":
    unittest.main()
