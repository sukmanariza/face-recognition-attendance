"""Timing helpers used by the pipeline, the HUD, and the benchmark."""

from __future__ import annotations

import time
import unittest

from src.metrics import LatencyTracker, PipelineStats, RateCounter, StageTimings


class RateCounterTests(unittest.TestCase):
    def test_no_ticks_means_zero(self) -> None:
        self.assertEqual(RateCounter().rate(), 0.0)

    def test_rate_reflects_the_tick_count(self) -> None:
        counter = RateCounter(window_seconds=10.0)
        for _ in range(5):
            counter.tick()
        self.assertGreater(counter.rate(), 0.0)
        self.assertEqual(counter.total, 5)

    def test_reset_clears_everything(self) -> None:
        counter = RateCounter()
        counter.tick()
        counter.reset()
        self.assertEqual(counter.total, 0)
        self.assertEqual(counter.rate(), 0.0)

    def test_tick_accepts_a_count(self) -> None:
        counter = RateCounter()
        counter.tick(4)
        self.assertEqual(counter.total, 4)

    def test_old_ticks_leave_the_window(self) -> None:
        counter = RateCounter(window_seconds=0.01)
        counter.tick()
        time.sleep(0.05)
        self.assertEqual(counter.rate(), 0.0)


class LatencyTrackerTests(unittest.TestCase):
    def test_empty_tracker_reports_zero(self) -> None:
        tracker = LatencyTracker()
        self.assertEqual(tracker.average_ms, 0.0)
        self.assertEqual(tracker.p95_ms, 0.0)
        self.assertEqual(tracker.max_ms, 0.0)

    def test_average_and_max(self) -> None:
        tracker = LatencyTracker(capacity=8)
        for value in (10.0, 20.0, 30.0):
            tracker.add(value)
        self.assertAlmostEqual(tracker.average_ms, 20.0)
        self.assertAlmostEqual(tracker.max_ms, 30.0)

    def test_p95_is_within_the_range(self) -> None:
        tracker = LatencyTracker(capacity=32)
        for value in range(1, 101):
            tracker.add(float(value))
        self.assertGreaterEqual(tracker.p95_ms, 90.0)
        self.assertLessEqual(tracker.p95_ms, 100.0)

    def test_capacity_is_respected(self) -> None:
        tracker = LatencyTracker(capacity=2)
        for value in (1.0, 2.0, 3.0):
            tracker.add(value)
        self.assertEqual(tracker.count, 2)


class StageTimingsTests(unittest.TestCase):
    def test_records_detection_time(self) -> None:
        timings = StageTimings()
        timings.record_detection(0.010)
        timings.record_detection(0.030)
        self.assertEqual(timings.detection_calls, 2)
        self.assertAlmostEqual(timings.detection_ms, 20.0)

    def test_records_recognition_time(self) -> None:
        timings = StageTimings()
        timings.record_recognition(0.005)
        self.assertAlmostEqual(timings.recognition_ms, 5.0)
        self.assertEqual(timings.recognition_calls, 1)

    def test_reset_clears_counters(self) -> None:
        timings = StageTimings()
        timings.record_detection(0.01)
        timings.reset()
        self.assertEqual(timings.detection_calls, 0)
        self.assertEqual(timings.detection_ms, 0.0)

    def test_empty_timings_report_zero(self) -> None:
        timings = StageTimings()
        self.assertEqual(timings.detection_ms, 0.0)
        self.assertEqual(timings.recognition_ms, 0.0)


class PipelineStatsTests(unittest.TestCase):
    def test_describe_mentions_the_rates(self) -> None:
        stats = PipelineStats(capture_fps=30.0, inference_fps=12.5)
        text = stats.describe()
        self.assertIn("30.0", text)
        self.assertIn("12.5", text)

    def test_as_dict_is_serialisable(self) -> None:
        payload = PipelineStats(capture_fps=1.0).as_dict()
        self.assertIsInstance(payload, dict)
        self.assertIn("capture_fps", payload)


if __name__ == "__main__":
    unittest.main()
