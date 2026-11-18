"""Configuration loading, validation, and settings precedence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from config import ConfigError, build_config, load_config
from src.hardware import Hardware
from src.performance import apply_profile, choose_tier
from src.settings import SettingsStore, merge_mappings


def make_hardware(cores: int = 4, ram_mb: int = 8000, single_board: bool = False) -> Hardware:
    return Hardware(
        cpu_cores=cores,
        total_ram_mb=ram_mb,
        platform_tag="linux" if single_board else "win32",
        machine="armv7l" if single_board else "AMD64",
        board_model="Raspberry Pi 4 Model B" if single_board else "",
    )


class ConfigTests(unittest.TestCase):
    def test_baseline_file_loads(self) -> None:
        config = load_config()
        self.assertEqual(config.model.name, "buffalo_s")
        self.assertEqual(config.recognition.similarity_threshold, 0.45)
        self.assertEqual(config.registration.samples, 5)
        self.assertTrue(config.ui.animations)

    def test_unknown_section_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            build_config({"nonsense": {}})

    def test_unknown_key_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            build_config({"recognition": {"similarty_threshold": 0.4}})

    def test_out_of_range_value_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            build_config({"recognition": {"similarity_threshold": 3.0}})

    def test_camera_backend_choices_include_linux(self) -> None:
        config = build_config({"camera": {"backend": "v4l2"}})
        self.assertEqual(config.camera.backend, "v4l2")

    def test_mapping_round_trip_rebuilds_the_same_config(self) -> None:
        config = load_config()
        rebuilt = build_config(config.to_mapping(), config.source_path)
        self.assertEqual(config.to_mapping(), rebuilt.to_mapping())

    def test_paths_resolve_against_the_config_directory(self) -> None:
        config = load_config()
        self.assertTrue(config.database_path.is_absolute())
        self.assertEqual(config.model_dir.name, config.model.name)


class PerformanceTierTests(unittest.TestCase):
    def test_single_board_gets_the_low_tier(self) -> None:
        self.assertEqual(choose_tier(make_hardware(single_board=True), "auto"), "low")

    def test_two_cores_gets_the_low_tier(self) -> None:
        self.assertEqual(choose_tier(make_hardware(cores=2), "auto"), "low")

    def test_four_cores_gets_the_balanced_tier(self) -> None:
        self.assertEqual(choose_tier(make_hardware(cores=4, ram_mb=8000), "auto"), "balanced")

    def test_many_cores_gets_the_high_tier(self) -> None:
        self.assertEqual(choose_tier(make_hardware(cores=8, ram_mb=16000), "auto"), "high")

    def test_explicit_choice_wins_over_detection(self) -> None:
        self.assertEqual(choose_tier(make_hardware(cores=16, ram_mb=32000), "low"), "low")

    def test_one_gigabyte_gets_the_minimal_tier(self) -> None:
        self.assertEqual(choose_tier(make_hardware(cores=4, ram_mb=1024), "auto"), "minimal")

    def test_explicit_minimal_tier_is_accepted(self) -> None:
        self.assertEqual(choose_tier(make_hardware(cores=16, ram_mb=32000), "minimal"), "minimal")

    def test_applying_a_profile_changes_only_performance_fields(self) -> None:
        baseline = load_config()
        tuned, tier = apply_profile(baseline, make_hardware(cores=2, ram_mb=4096))
        self.assertEqual(tier, "low")
        self.assertEqual(tuned.model.det_size, (320, 320))
        self.assertEqual(tuned.model.intra_op_threads, 1)
        self.assertEqual(tuned.recognition.detection_interval, 3)
        # Accuracy settings must not be touched by a tier change.
        self.assertEqual(
            tuned.recognition.similarity_threshold,
            baseline.recognition.similarity_threshold,
        )
        self.assertEqual(tuned.registration.samples, baseline.registration.samples)

    def test_minimal_tier_uses_a_smaller_detector(self) -> None:
        tuned, tier = apply_profile(load_config(), make_hardware(cores=1, ram_mb=512))
        self.assertEqual(tier, "minimal")
        self.assertEqual(tuned.model.det_size, (256, 256))
        self.assertEqual(tuned.recognition.detection_interval, 5)


class SettingsStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.path = Path(self._directory.name) / "settings.json"
        self.store = SettingsStore(self.path, make_hardware())
        self.baseline = load_config()

    def test_effective_config_falls_back_to_the_baseline(self) -> None:
        self.assertEqual(
            self.store.effective_config(self.baseline).recognition.similarity_threshold,
            0.45,
        )

    def test_saved_override_wins(self) -> None:
        self.store.save({"recognition": {"similarity_threshold": 0.6}})
        config = self.store.effective_config(self.baseline)
        self.assertAlmostEqual(config.recognition.similarity_threshold, 0.6)

    def test_overrides_are_merged_not_replaced(self) -> None:
        self.store.save({"ui": {"style": "light"}})
        config = self.store.effective_config(self.baseline)
        self.assertEqual(config.ui.style, "light")
        self.assertEqual(config.ui.display_max_width, self.baseline.ui.display_max_width)

    def test_saved_file_carries_a_version(self) -> None:
        self.store.save({"ui": {"show_fps": False}})
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], 1)

    def test_reset_removes_the_file(self) -> None:
        self.store.save({"ui": {"style": "light"}})
        self.store.reset()
        self.assertFalse(self.path.exists())

    def test_broken_file_is_ignored(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.store.load(), {})
        self.assertAlmostEqual(
            self.store.effective_config(self.baseline).recognition.similarity_threshold, 0.45
        )

    def test_invalid_override_falls_back_to_the_tuned_config(self) -> None:
        self.path.write_text(
            json.dumps({"version": 1, "recognition": {"similarity_threshold": 9.0}}),
            encoding="utf-8",
        )
        config = self.store.effective_config(self.baseline)
        self.assertAlmostEqual(config.recognition.similarity_threshold, 0.45)

    def test_merge_mappings_is_recursive_and_does_not_mutate(self) -> None:
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        merged = merge_mappings(base, {"a": {"c": 9}, "e": 4})
        self.assertEqual(merged, {"a": {"b": 1, "c": 9}, "d": 3, "e": 4})
        self.assertEqual(base["a"]["c"], 2)


if __name__ == "__main__":
    unittest.main()
