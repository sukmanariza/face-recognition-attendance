"""Benchmark aggregation and report writing (no camera involved)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.benchmark import BenchmarkReport, BenchmarkSample


def sample(index: int, fps: float = 20.0, inference_ms: float = 45.0) -> BenchmarkSample:
    return BenchmarkSample(
        elapsed_seconds=float(index),
        capture_fps=30.0,
        inference_fps=fps,
        display_fps=25.0,
        dropped_frames=index,
        inference_ms=inference_ms,
        faces=1,
        process_cpu_percent=80.0,
        machine_cpu_percent=90.0,
        process_ram_mb=320.0,
    )


def report(*samples: BenchmarkSample, duration: float = 30.0) -> BenchmarkReport:
    return BenchmarkReport(
        started_at="2026-01-02T03:04:05+00:00",
        duration_seconds=duration,
        warmup_seconds=5.0,
        tier="balanced",
        hardware="4 cores",
        model="buffalo_s det 320x320",
        samples=samples,
    )


class SummaryTests(unittest.TestCase):
    def test_summary_aggregates_the_samples(self) -> None:
        data = report(sample(0, 20.0, 40.0), sample(1, 10.0, 60.0)).summary()
        self.assertEqual(data["samples"], 2.0)
        self.assertAlmostEqual(data["inference_fps_mean"], 15.0)
        self.assertAlmostEqual(data["inference_fps_min"], 10.0)
        self.assertAlmostEqual(data["inference_ms_mean"], 50.0)

    def test_empty_report_has_no_summary(self) -> None:
        self.assertEqual(report().summary(), {})

    def test_describe_mentions_the_dropped_frames(self) -> None:
        text = report(sample(0), sample(3)).describe()
        self.assertIn("dropped frames", text)
        self.assertIn("balanced", text)

    def test_describe_handles_an_empty_report(self) -> None:
        self.assertIn("no samples", report().describe().lower())


class VerdictTests(unittest.TestCase):
    def test_fast_enough_passes(self) -> None:
        self.assertIn("PASS", report(sample(0, fps=30.0)).verdict(25.0))

    def test_too_slow_recommends_the_low_profile(self) -> None:
        text = report(sample(0, fps=8.0)).verdict(25.0)
        self.assertIn("BELOW TARGET", text)
        self.assertIn("low", text.lower())

    def test_empty_report_is_explained(self) -> None:
        self.assertIn("no samples", report().verdict(25.0).lower())


class WriteTests(unittest.TestCase):
    def test_report_is_written_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = report(sample(0)).write(Path(directory))
            self.assertTrue(path.is_file())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["tier"], "balanced")
            self.assertEqual(len(payload["samples"]), 1)
            self.assertIn("summary", payload)

    def test_directory_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "reports" / "daily"
            path = report(sample(0)).write(nested)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
