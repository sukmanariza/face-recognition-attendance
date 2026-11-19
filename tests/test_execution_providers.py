"""Execution provider selection: auto order, shortcuts, and the CPU fallback."""

from __future__ import annotations

import unittest

from src import execution_providers

CPU = "CPUExecutionProvider"
DML = "DmlExecutionProvider"
CUDA = "CUDAExecutionProvider"
OPENVINO = "OpenVINOExecutionProvider"


class ResolveTests(unittest.TestCase):
    def test_auto_prefers_a_gpu_when_the_build_has_one(self) -> None:
        chosen = execution_providers.resolve(["auto"], [CPU, DML])
        self.assertEqual(chosen, [DML, CPU])

    def test_auto_keeps_cuda_ahead_of_directml(self) -> None:
        chosen = execution_providers.resolve(["auto"], [CPU, DML, CUDA])
        self.assertEqual(chosen, [CUDA, DML, CPU])

    def test_auto_falls_back_to_the_cpu_when_no_gpu_is_compiled_in(self) -> None:
        self.assertEqual(execution_providers.resolve(["auto"], [CPU]), [CPU])

    def test_an_explicit_choice_is_kept_and_cpu_is_appended(self) -> None:
        chosen = execution_providers.resolve(["DmlExecutionProvider"], [CPU, DML])
        self.assertEqual(chosen, [DML, CPU])

    def test_cpu_only_request_stays_on_the_cpu(self) -> None:
        self.assertEqual(execution_providers.resolve(["cpu"], [CPU, DML]), [CPU])

    def test_shortcuts_expand_to_provider_names(self) -> None:
        self.assertEqual(execution_providers.expand(["directml"]), [DML, CPU])
        self.assertEqual(execution_providers.expand(["cuda"]), [CUDA, CPU])
        self.assertEqual(execution_providers.expand(["openvino"]), [OPENVINO, CPU])
        self.assertEqual(execution_providers.expand(["auto"]), ["auto"])

    def test_a_missing_provider_is_skipped_without_losing_the_cpu(self) -> None:
        with self.assertLogs("src.execution_providers", level="WARNING"):
            chosen = execution_providers.resolve(["CUDAExecutionProvider"], [CPU])
        self.assertEqual(chosen, [CPU])

    def test_duplicates_are_removed(self) -> None:
        chosen = execution_providers.resolve([DML, DML, CPU], [CPU, DML])
        self.assertEqual(chosen, [DML, CPU])

    def test_no_provider_at_all_resolves_to_nothing(self) -> None:
        self.assertEqual(execution_providers.resolve(["auto"], []), [])


class DescriptionTests(unittest.TestCase):
    def test_gpu_label_names_the_backend(self) -> None:
        self.assertEqual(execution_providers.describe([DML, CPU]), "DirectML (GPU)")
        self.assertEqual(execution_providers.describe([CPU]), "CPU")

    def test_an_empty_list_is_not_a_backend(self) -> None:
        self.assertEqual(execution_providers.describe([]), "not loaded")

    def test_gpu_detection_follows_the_first_provider(self) -> None:
        self.assertTrue(execution_providers.is_gpu([DML, CPU]))
        self.assertFalse(execution_providers.is_gpu([CPU]))
        self.assertFalse(execution_providers.is_gpu([]))

    def test_directml_needs_serial_execution_without_memory_patterns(self) -> None:
        self.assertTrue(execution_providers.needs_serial_execution([DML, CPU]))
        self.assertFalse(execution_providers.needs_serial_execution([CPU]))

    def test_known_names_cover_the_shortcuts(self) -> None:
        for name in ("auto", "cpu", "cuda", "directml", "openvino", CPU, DML):
            self.assertIn(name, execution_providers.known_names())


if __name__ == "__main__":
    unittest.main()
