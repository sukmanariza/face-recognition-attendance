"""Measurement primitives for FPS and latency.

All counters are safe to use across threads: the inference and display threads
write, while the UI and the benchmark read.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


class RateCounter:
    """Events per second over a rolling window.

    A rolling window is used instead of a cumulative average so the number
    reflects the last few seconds and reacts when performance changes.
    """

    def __init__(self, window_seconds: float = 2.0) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than 0.")
        self._window = float(window_seconds)
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()
        self._total = 0

    def tick(self, count: int = 1) -> None:
        now = time.monotonic()
        with self._lock:
            for _ in range(max(1, count)):
                self._timestamps.append(now)
            self._total += max(1, count)
            self._prune(now)

    def rate(self) -> float:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            if len(self._timestamps) < 2:
                return 0.0
            span = self._timestamps[-1] - self._timestamps[0]
            return (len(self._timestamps) - 1) / span if span > 0 else 0.0

    def reset(self) -> None:
        with self._lock:
            self._timestamps.clear()
            self._total = 0

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()


class LatencyTracker:
    """Average, p95, and max of the most recent latency samples in milliseconds."""

    def __init__(self, capacity: int = 240) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1.")
        self._lock = threading.Lock()
        self._samples: deque[float] = deque(maxlen=capacity)

    def add(self, milliseconds: float) -> None:
        with self._lock:
            self._samples.append(float(milliseconds))

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._samples)

    @property
    def average_ms(self) -> float:
        with self._lock:
            return sum(self._samples) / len(self._samples) if self._samples else 0.0

    @property
    def p95_ms(self) -> float:
        with self._lock:
            if not self._samples:
                return 0.0
            ordered = sorted(self._samples)
            index = min(len(ordered) - 1, round(len(ordered) * 0.95) - 1)
            return ordered[max(0, index)]

    @property
    def max_ms(self) -> float:
        with self._lock:
            return max(self._samples) if self._samples else 0.0


@dataclass(frozen=True)
class PipelineStats:
    """A snapshot of pipeline metrics at one point in time."""

    capture_fps: float = 0.0
    inference_fps: float = 0.0
    display_fps: float = 0.0
    detection_fps: float = 0.0
    recognition_fps: float = 0.0
    detection_ms: float = 0.0
    recognition_ms: float = 0.0
    inference_ms: float = 0.0
    inference_p95_ms: float = 0.0
    frames_captured: int = 0
    frames_inferred: int = 0
    frames_displayed: int = 0
    frames_skipped: int = 0

    def describe(self) -> str:
        return (
            f"camera {self.capture_fps:5.1f} FPS | inference {self.inference_fps:5.1f} FPS "
            f"| display {self.display_fps:5.1f} FPS"
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "capture_fps": round(self.capture_fps, 2),
            "inference_fps": round(self.inference_fps, 2),
            "display_fps": round(self.display_fps, 2),
            "detection_fps": round(self.detection_fps, 2),
            "recognition_fps": round(self.recognition_fps, 2),
            "detection_ms": round(self.detection_ms, 2),
            "recognition_ms": round(self.recognition_ms, 2),
            "inference_ms": round(self.inference_ms, 2),
            "inference_p95_ms": round(self.inference_p95_ms, 2),
            "frames_captured": self.frames_captured,
            "frames_inferred": self.frames_inferred,
            "frames_displayed": self.frames_displayed,
            "frames_skipped": self.frames_skipped,
        }


@dataclass
class StageTimings:
    """Accumulated per-stage inference time, recorded by the detector."""

    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    detection_calls: int = 0
    detection_seconds: float = 0.0
    recognition_calls: int = 0
    recognition_seconds: float = 0.0

    def record_detection(self, seconds: float) -> None:
        with self.lock:
            self.detection_calls += 1
            self.detection_seconds += seconds

    def record_recognition(self, seconds: float) -> None:
        with self.lock:
            self.recognition_calls += 1
            self.recognition_seconds += seconds

    @property
    def detection_ms(self) -> float:
        with self.lock:
            return (self.detection_seconds / self.detection_calls) * 1000.0 if self.detection_calls else 0.0

    @property
    def recognition_ms(self) -> float:
        with self.lock:
            return (
                (self.recognition_seconds / self.recognition_calls) * 1000.0
                if self.recognition_calls
                else 0.0
            )

    def reset(self) -> None:
        with self.lock:
            self.detection_calls = 0
            self.detection_seconds = 0.0
            self.recognition_calls = 0
            self.recognition_seconds = 0.0
