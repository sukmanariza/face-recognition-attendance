"""Model loader behaviour around execution providers.

The sessions cannot be built without the real model pack, so these tests check the
decisions around them: which providers reach the session options, what DirectML
requires, and that a GPU failure degrades to the CPU instead of losing the model.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from config import ModelConfig
from src import execution_providers
from src.face_detector import FaceDetector, ModelLoadError

CPU = "CPUExecutionProvider"
DML = "DmlExecutionProvider"


def detector(providers: tuple[str, ...] = ("auto",)) -> FaceDetector:
    return FaceDetector(ModelConfig(providers=providers), Path("models"))


class ProviderSelectionTests(unittest.TestCase):
    def test_auto_picks_directml_when_the_build_has_it(self) -> None:
        with mock.patch.object(
            execution_providers, "available_providers", return_value=(CPU, DML)
        ):
            self.assertEqual(detector()._select_providers(), [DML, CPU])

    def test_auto_stays_on_the_cpu_without_a_gpu_build(self) -> None:
        with mock.patch.object(
            execution_providers, "available_providers", return_value=(CPU,)
        ):
            self.assertEqual(detector()._select_providers(), [CPU])

    def test_no_provider_at_all_is_a_clear_error(self) -> None:
        with (
            mock.patch.object(execution_providers, "available_providers", return_value=()),
            self.assertRaises(ModelLoadError),
        ):
            detector()._select_providers()


class SessionOptionTests(unittest.TestCase):
    def _mem_pattern(self, providers: list[str]) -> object:
        options = detector()._build_session_options(providers)
        value = getattr(options, "enable_mem_pattern", None)
        self.assertIsNotNone(value, "SessionOptions lost enable_mem_pattern")
        return value

    def test_directml_disables_memory_patterns(self) -> None:
        self.assertFalse(self._mem_pattern([DML, CPU]))

    def test_the_cpu_keeps_memory_patterns(self) -> None:
        self.assertTrue(self._mem_pattern([CPU]))


class FallbackTests(unittest.TestCase):
    def test_a_gpu_failure_retries_on_the_cpu(self) -> None:
        subject = detector()
        used: list[tuple[str, ...]] = []

        def fake_build(factory: object, model_dir: object, providers: object) -> object:
            used.append(tuple(providers))  # type: ignore[arg-type]
            if next(iter(providers), CPU) != CPU:  # type: ignore[call-overload]
                raise ModelLoadError("device lost")
            return mock.Mock(models={"detection": object(), "recognition": object()})

        with (
            mock.patch.object(
                execution_providers, "available_providers", return_value=(CPU, DML)
            ),
            mock.patch.object(subject, "_build_analysis", side_effect=fake_build),
            mock.patch.object(subject, "warmup"),
        ):
            subject.load()

        self.assertEqual(used, [(DML, CPU), (CPU,)])
        self.assertEqual(subject.providers, (CPU,))

    def test_a_cpu_failure_is_not_retried(self) -> None:
        subject = detector(("cpu",))
        calls: list[tuple[str, ...]] = []

        def fake_build(factory: object, model_dir: object, providers: object) -> object:
            calls.append(tuple(providers))  # type: ignore[arg-type]
            raise ModelLoadError("corrupt model")

        with (
            mock.patch.object(
                execution_providers, "available_providers", return_value=(CPU, DML)
            ),
            mock.patch.object(subject, "_build_analysis", side_effect=fake_build),
            self.assertRaises(ModelLoadError),
        ):
            subject.load()
        self.assertEqual(calls, [(CPU,)])


if __name__ == "__main__":
    unittest.main()
