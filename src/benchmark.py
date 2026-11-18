"""Measured performance report, so numbers are recorded instead of assumed.

The benchmark runs the real pipeline (capture thread, inference thread, the
configured intervals) for a fixed time and records one sample per interval. It
stops early if dropped frames climb, which means the machine cannot keep up with
the current tier: the practical signal for switching to a lower profile.

Output: one JSON file per run in ``benchmark.output_directory`` plus a short
console table.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .application import Application
from .pipeline import CameraSource, Mode, RecognitionPipeline
from .sysmetrics import SystemMetrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkSample:
    """One snapshot taken during the run."""

    elapsed_seconds: float
    capture_fps: float
    inference_fps: float
    display_fps: float
    dropped_frames: int
    inference_ms: float
    faces: int
    process_cpu_percent: float
    machine_cpu_percent: float
    process_ram_mb: float


@dataclass(frozen=True)
class BenchmarkReport:
    """Aggregated result of one benchmark run."""

    started_at: str
    duration_seconds: float
    warmup_seconds: float
    tier: str
    hardware: str
    model: str
    samples: tuple[BenchmarkSample, ...]

    def _values(self, attribute: str) -> list[float]:
        return [float(getattr(sample, attribute)) for sample in self.samples]

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {}
        inference = self._values("inference_ms")
        return {
            "samples": float(len(self.samples)),
            "duration_seconds": self.duration_seconds,
            "inference_fps_mean": round(statistics.fmean(self._values("inference_fps")), 2),
            "inference_fps_min": round(min(self._values("inference_fps")), 2),
            "capture_fps_mean": round(statistics.fmean(self._values("capture_fps")), 2),
            "inference_ms_mean": round(statistics.fmean(inference), 2),
            "inference_ms_p95": round(float(np.percentile(inference, 95)), 2),
            "dropped_frames": float(max(self._values("dropped_frames"))),
            "process_cpu_percent_mean": round(
                statistics.fmean(self._values("process_cpu_percent")), 2
            ),
            "process_ram_mb_mean": round(statistics.fmean(self._values("process_ram_mb")), 2),
            "faces_mean": round(statistics.fmean(self._values("faces")), 2),
        }

    def verdict(self, target_fps: float) -> str:
        """Rule of thumb used by the CLI: is this tier good enough?"""
        data = self.summary()
        if not data:
            return "No samples were collected; the benchmark did not run for long enough."
        mean_fps = data["inference_fps_mean"]
        if mean_fps >= target_fps:
            return f"PASS: {mean_fps:.1f} FPS meets the {target_fps:.0f} FPS target."
        return (
            f"BELOW TARGET: {mean_fps:.1f} FPS against a {target_fps:.0f} FPS target. "
            "Choose the 'low' performance profile in Settings, or lower "
            "recognition.embedding_interval_seconds."
        )

    def describe(self) -> str:
        data = self.summary()
        if not data:
            return "No samples collected."
        lines = [
            f"Benchmark {self.started_at} ({self.duration_seconds:.0f} s, tier {self.tier})",
            f"  model            {self.model}",
            f"  hardware         {self.hardware}",
            f"  inference        {data['inference_fps_mean']:.1f} FPS mean, "
            f"{data['inference_fps_min']:.1f} FPS min",
            f"  frame time       {data['inference_ms_mean']:.1f} ms mean, "
            f"{data['inference_ms_p95']:.1f} ms p95",
            f"  capture          {data['capture_fps_mean']:.1f} FPS",
            f"  dropped frames   {int(data['dropped_frames'])}",
            f"  process CPU      {data['process_cpu_percent_mean']:.0f}% "
            f"(whole machine {max(self._values('machine_cpu_percent')):.0f}% peak)",
            f"  process RAM      {data['process_ram_mb_mean']:.0f} MB",
            f"  faces per frame  {data['faces_mean']:.2f}",
        ]
        return "\n".join(lines)

    def to_mapping(self) -> dict[str, object]:
        return {
            "started_at": self.started_at,
            "duration_seconds": self.duration_seconds,
            "warmup_seconds": self.warmup_seconds,
            "tier": self.tier,
            "hardware": self.hardware,
            "model": self.model,
            "summary": self.summary(),
            "samples": [asdict(sample) for sample in self.samples],
        }

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.replace(":", "").replace("-", "").replace("+0000", "")
        path = directory / f"benchmark-{stamp}.json"
        path.write_text(
            json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path


ProgressCallback = Callable[[float, float], None]


def run_benchmark(
    application: Application,
    *,
    duration: float | None = None,
    warmup: float | None = None,
    progress: ProgressCallback | None = None,
) -> BenchmarkReport:
    """Run the real pipeline for a fixed time and return the measurements."""
    settings = application.config.benchmark
    total = settings.duration_seconds if duration is None else duration
    warm_seconds = settings.warmup_seconds if warmup is None else warmup

    detector = application.detector  # loads the model and warms it up
    model_description = detector.describe()
    session = application.create_recognition_session()

    camera = application.create_camera()
    pipeline = RecognitionPipeline(application.config, CameraSource(camera))
    pipeline.set_recognition_session(session)
    pipeline.set_mode(Mode.RECOGNIZE)

    metrics = SystemMetrics()
    samples: list[BenchmarkSample] = []
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    next_sample_at = started + warm_seconds
    end_at = started + warm_seconds + total
    inference_ms: list[float] = []
    last_faces = 0

    camera.open()
    pipeline.start()
    try:
        while time.monotonic() < end_at:
            output = pipeline.wait_output(timeout=0.2)
            if output is not None:
                if output.inference_ms > 0.0:
                    inference_ms.append(output.inference_ms)
                if output.recognition is not None:
                    last_faces = output.recognition.face_count
            now = time.monotonic()
            if progress is not None:
                progress(min(total, max(0.0, now - next_sample_at)), total)
            if now < next_sample_at:
                continue

            stats = pipeline.stats()
            system = metrics.sample()
            recent = inference_ms[-30:] or [0.0]
            samples.append(
                BenchmarkSample(
                    elapsed_seconds=round(now - started - warm_seconds, 2),
                    capture_fps=round(stats.capture_fps, 2),
                    inference_fps=round(stats.inference_fps, 2),
                    display_fps=round(stats.display_fps, 2),
                    dropped_frames=stats.dropped_frames,
                    inference_ms=round(statistics.fmean(recent), 2),
                    faces=last_faces,
                    process_cpu_percent=round(system.process_cpu_percent, 2),
                    machine_cpu_percent=round(system.machine_cpu_percent, 2),
                    process_ram_mb=round(system.process_ram_mb, 2),
                )
            )
            next_sample_at = now + settings.sample_interval_seconds
    finally:
        pipeline.close()

    return BenchmarkReport(
        started_at=started_at,
        duration_seconds=round(total, 1),
        warmup_seconds=round(warm_seconds, 1),
        tier=str(application.tier),
        hardware=application.hardware.summary,
        model=model_description,
        samples=tuple(samples),
    )
