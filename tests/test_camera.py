"""Camera helpers that can be tested without hardware."""

from __future__ import annotations

import unittest

import numpy as np

from config import CameraConfig
from src.camera import CameraError, Frame, probe_cameras, resolve_backend
from src.pipeline import CameraSource


class FrameTests(unittest.TestCase):
    def test_frame_carries_image_and_sequence(self) -> None:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        frame = Frame(image, 7)
        self.assertEqual(frame.sequence, 7)
        self.assertIs(frame.image, image)

    def test_image_shape_is_reachable_through_the_frame(self) -> None:
        frame = Frame(np.zeros((10, 20, 3), dtype=np.uint8), 1)
        self.assertEqual(frame.image.shape, (10, 20, 3))


class BackendTests(unittest.TestCase):
    def test_auto_resolves_to_a_real_backend(self) -> None:
        self.assertIsInstance(resolve_backend("auto"), int)

    def test_known_backends_resolve(self) -> None:
        for name in ("any", "dshow", "msmf", "v4l2"):
            self.assertIsInstance(resolve_backend(name), int)

    def test_unknown_backend_is_reported(self) -> None:
        # config.yaml validation rejects unknown names first; this is the second line.
        with self.assertRaises(CameraError):
            resolve_backend("nonsense")


class ProbeTests(unittest.TestCase):
    def test_probe_with_no_indices_returns_empty(self) -> None:
        self.assertEqual(probe_cameras(max_index=-1), [])


class ConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        config = CameraConfig()
        self.assertEqual(config.index, 0)
        self.assertEqual((config.width, config.height), (640, 480))

    def test_camera_source_reports_describe(self) -> None:
        class _StubCamera:
            def read(self):
                return Frame(np.zeros((4, 4, 3), dtype=np.uint8), 1)

            def describe(self) -> str:
                return "Camera 0: 640x480"

            def close(self) -> None:
                pass

        source = CameraSource(_StubCamera())  # type: ignore[arg-type]
        self.assertIn("640x480", source.describe())
        self.assertIsNotNone(source.read())

    def test_camera_source_turns_errors_into_missing_frames(self) -> None:
        class _FailingCamera:
            def read(self):
                raise CameraError("no frame")

            def describe(self) -> str:
                return "broken"

            def close(self) -> None:
                pass

        source = CameraSource(_FailingCamera())  # type: ignore[arg-type]
        self.assertIsNone(source.read())


if __name__ == "__main__":
    unittest.main()
