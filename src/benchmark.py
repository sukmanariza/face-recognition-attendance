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
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import execution_providers
from .application import Application
from .face_detector import DetectedFace, FaceDetector
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


@dataclass(frozen=True)
class ProviderTiming:
    """Measured cost of one execution provider on this machine."""

    provider: str
    device: str
    detect_ms: float
    embed_ms: float
    frames: int
    error: str = ""

    @property
    def total_ms(self) -> float:
        """Detection plus embedding for one face."""
        return self.detect_ms + self.embed_ms

    def describe(self) -> str:
        if self.error:
            return f"{self.provider:<26} {self.device:<4} failed: {self.error}"
        return (
            f"{self.provider:<26} {self.device:<4} detect {self.detect_ms:7.2f} ms  "
            f"embed {self.embed_ms:7.2f} ms  per face {self.total_ms:7.2f} ms"
        )


def _synthetic_scene(width: int = 640, height: int = 480) -> np.ndarray:
    """A deterministic frame with texture, so detection does real convolution work."""
    rows = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    columns = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    frame = np.empty((height, width, 3), dtype=np.float32)
    frame[..., 0] = (rows + columns) * 0.5
    frame[..., 1] = rows
    frame[..., 2] = columns
    noise = np.random.default_rng(7).integers(0, 24, size=(height, width, 3), dtype=np.uint8)
    return (np.clip(frame, 0, 255).astype(np.uint8) + noise).astype(np.uint8)


def _synthetic_face(width: int, height: int) -> DetectedFace:
    """A face-shaped box with the five landmarks the aligner needs."""
    left, top = width * 0.35, height * 0.25
    right, bottom = width * 0.65, height * 0.75
    span_x, span_y = right - left, bottom - top
    landmarks = np.array(
        [
            [left + span_x * 0.30, top + span_y * 0.38],
            [left + span_x * 0.70, top + span_y * 0.38],
            [left + span_x * 0.50, top + span_y * 0.55],
            [left + span_x * 0.35, top + span_y * 0.75],
            [left + span_x * 0.65, top + span_y * 0.75],
        ],
        dtype=np.float32,
    )
    return DetectedFace(
        bbox=np.array([left, top, right, bottom], dtype=np.float32),
        det_score=0.9,
        landmarks=landmarks,
    )


def compare_providers(
    application: Application,
    *,
    frames: int = 15,
    providers: Sequence[str] | None = None,
    det_size: int | None = None,
) -> tuple[ProviderTiming, ...]:
    """Time detection and embedding on every provider this build offers.

    Runs on a synthetic frame, so no camera and no face are needed. The point is a
    decision backed by numbers: a small model on a slow integrated GPU can lose to
    the CPU, and only the actual machine can answer that.
    """
    present = execution_providers.available_providers()
    candidates = list(providers or [])
    if not candidates:
        candidates = [name for name in execution_providers.PREFERENCE if name in present]
    frame = _synthetic_scene()
    height, width = frame.shape[:2]
    model = application.config.model
    if det_size is not None:
        model = replace(model, det_size=(det_size, det_size))
    timings: list[ProviderTiming] = []

    for name in candidates:
        device = "GPU" if name in execution_providers.GPU_PROVIDERS else "CPU"
        detector = FaceDetector(
            replace(model, providers=(name,), warmup=False),
            application.config.model_dir,
        )
        try:
            detector.load()
            detector.detect(frame, max_faces=1)  # first call builds the kernels
            detection: list[float] = []
            embedding: list[float] = []
            for _ in range(max(1, frames)):
                started = time.perf_counter()
                detector.detect(frame, max_faces=1)
                detection.append((time.perf_counter() - started) * 1000.0)
                started = time.perf_counter()
                detector.embed(frame, [_synthetic_face(width, height)])
                embedding.append((time.perf_counter() - started) * 1000.0)
            timings.append(
                ProviderTiming(
                    provider=name,
                    device=device,
                    detect_ms=round(statistics.median(detection), 2),
                    embed_ms=round(statistics.median(embedding), 2),
                    frames=len(detection),
                )
            )
        except Exception as exc:  # noqa: BLE001 - one provider must not abort the run
            timings.append(
                ProviderTiming(
                    provider=name,
                    device=device,
                    detect_ms=0.0,
                    embed_ms=0.0,
                    frames=0,
                    error=str(exc),
                )
            )
        finally:
            detector.close()
    return tuple(timings)


def describe_provider_comparison(timings: Sequence[ProviderTiming]) -> str:
    """Console table plus the winner, so the choice is not a guess."""
    if not timings:
        return "No execution provider to compare."
    lines = ["Execution provider comparison (median per frame, lower is better)"]
    lines.extend(f"  {timing.describe()}" for timing in timings)
    working = [timing for timing in timings if not timing.error]
    if working:
        fastest = min(working, key=lambda timing: timing.total_ms)
        lines.append(
            f"  fastest: {fastest.provider} at {fastest.total_ms:.2f} ms per face "
            f"({fastest.frames} frames)"
        )
    lines.append(
        "  Set model.providers in config.yaml to pin a provider; [auto] takes the "
        "fastest backend the installed onnxruntime build provides."
    )
    return "\n".join(lines)
