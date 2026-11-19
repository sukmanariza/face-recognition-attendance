"""InsightFace wrapper: face detection, alignment, and embedding extraction.

Uses the pretrained InsightFace model pack. The default is `buffalo_s` (SCRFD
`det_500m` detector + `w600k_mbf` ArcFace recognition head) because on a 4-core CPU
at 320x320 it costs about 15 ms per frame, while `buffalo_l` costs about 250 ms.

Rules kept in this module:

* Detection and embedding are separate calls, so detection-only work (frames with
  no face, or pose gating) never pays for embedding.
* Embeddings are always L2-normalised, and alignment uses the same ArcFace 5-point
  `norm_crop` at registration and recognition. This is what makes the cosine
  similarity between the two comparable.
* Each stage is timed (`stats()`) so performance is measured, not assumed.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.request
import warnings
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import ModelConfig

from . import execution_providers
from .metrics import RateCounter, StageTimings
from .similarity import normalize_embedding

logger = logging.getLogger(__name__)

MODEL_ZOO_URL_TEMPLATE = (
    "https://github.com/deepinsight/insightface/releases/download/model-zoo/{name}.zip"
)
_LEGACY_MODEL_ROOT = Path.home() / ".insightface" / "models"
_DOWNLOAD_CHUNK_BYTES = 1 << 20
_USER_AGENT = "face-id-local/1.0"

# The model name forms a path and a URL, so its character set is constrained.
_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MAX_ARCHIVE_MEMBERS = 64
_MAX_EXTRACTED_BYTES = 1 << 30  # 1 GiB, far above the official packs (~300 MB)
# A model pack is data. An executable beside it could be loaded by some other
# tool that treats the folder as trusted, so those entries are refused.
_FORBIDDEN_SUFFIXES = frozenset(
    {".exe", ".dll", ".so", ".dylib", ".bat", ".cmd", ".ps1", ".sh", ".vbs", ".py", ".pyd"}
)

# scikit-image 0.26 marks `SimilarityTransform.estimate` as deprecated and
# insightface calls it on every alignment. It is not our warning and it floods logs.
warnings.filterwarnings("ignore", message=r".*estimate.*deprecated.*", category=FutureWarning)

ProgressCallback = Callable[[int, int], None]


class ModelLoadError(RuntimeError):
    """The model is unavailable, could not be downloaded, or failed to load."""


class DetectionError(RuntimeError):
    """Detection or embedding failed for a frame."""


@dataclass(frozen=True)
class DetectorStats:
    """Per-stage inference metrics used by the pipeline and the benchmark."""

    detection_calls: int
    recognition_calls: int
    detection_ms: float
    recognition_ms: float
    detection_fps: float
    recognition_fps: float

    def describe(self) -> str:
        return (
            f"detection {self.detection_fps:5.1f} FPS ({self.detection_ms:5.1f} ms) | "
            f"embedding {self.recognition_fps:5.1f} FPS ({self.recognition_ms:5.1f} ms)"
        )


@dataclass
class DetectedFace:
    """One face in a frame, in original pixel coordinates."""

    bbox: np.ndarray  # (4,) float32: x1, y1, x2, y2
    det_score: float
    landmarks: np.ndarray | None  # (5, 2) float32: eyes, nose, mouth corners
    embedding: np.ndarray | None = None  # (D,) float32, L2-normalised

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def min_side(self) -> float:
        return min(self.width, self.height)

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    def integer_bbox(self, frame_shape: Sequence[int] | None = None) -> tuple[int, int, int, int]:
        """Integer box, clipped to the frame when its shape is given."""
        x1, y1, x2, y2 = (float(value) for value in self.bbox[:4])
        if frame_shape is not None:
            height, width = int(frame_shape[0]), int(frame_shape[1])
            x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
            y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        return round(x1), round(y1), round(x2), round(y2)

    def crop(self, frame: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = self.integer_bbox(frame.shape)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return frame[y1:y2, x1:x2]


def _has_onnx_files(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.onnx"))


def _validate_model_name(name: str) -> str:
    """Reject names that could escape the model directory (e.g. ``../evil``)."""
    cleaned = name.strip()
    if not _MODEL_NAME_PATTERN.match(cleaned):
        raise ModelLoadError(
            f"model.name {name!r} is not valid. Use a plain InsightFace pack name "
            "(letters, digits, dot, underscore, hyphen), for example buffalo_s."
        )
    return cleaned


def _extract_archive(archive: Path, target: Path) -> None:
    """Extract a model archive with safety limits.

    ``extractall`` strips absolute paths and ``..``, but it neither rejects symlink
    entries nor bounds the total size. Both allow writing outside the target or
    filling the disk, so entries are checked first.
    """
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > _MAX_ARCHIVE_MEMBERS:
            raise ModelLoadError(
                f"Model archive contains {len(members)} files, above the limit of "
                f"{_MAX_ARCHIVE_MEMBERS}; refusing to extract: {archive}"
            )

        resolved_target = target.resolve()
        total_bytes = 0
        for member in members:
            if member.is_dir():
                continue
            # MODE symlink -> the entry could point outside the target.
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ModelLoadError(
                    f"Model archive contains a symlink ({member.filename}); "
                    f"refusing to extract: {archive}"
                )
            parts = Path(member.filename).parts
            if member.filename.startswith(("/", "\\")) or ".." in parts:
                raise ModelLoadError(
                    f"Model archive contains an unsafe path ({member.filename}); "
                    f"refusing to extract: {archive}"
                )
            if Path(member.filename).suffix.lower() in _FORBIDDEN_SUFFIXES:
                raise ModelLoadError(
                    f"Model archive contains an executable entry ({member.filename}); "
                    f"refusing to extract: {archive}"
                )
            extracted = (resolved_target / member.filename).resolve()
            if extracted != resolved_target and resolved_target not in extracted.parents:
                raise ModelLoadError(
                    f"Model archive writes outside the target ({member.filename}): {archive}"
                )
            total_bytes += member.file_size
            if total_bytes > _MAX_EXTRACTED_BYTES:
                raise ModelLoadError(
                    f"Model archive expands beyond "
                    f"{_MAX_EXTRACTED_BYTES // (1024 * 1024)} MB; refusing to extract: {archive}"
                )

        broken = bundle.testzip()
        if broken is not None:
            raise ModelLoadError(f"Model archive is corrupt at {broken}: {archive}")
        target.mkdir(parents=True, exist_ok=True)
        bundle.extractall(target)


def ensure_model_available(
    model_dir: Path, name: str, progress: ProgressCallback | None = None
) -> Path:
    """Return a directory containing model files, downloading the pack if needed."""
    name = _validate_model_name(name)
    if _has_onnx_files(model_dir):
        return model_dir

    legacy = _LEGACY_MODEL_ROOT / name
    if _has_onnx_files(legacy):
        logger.info("Using the existing model at %s", legacy)
        return legacy

    model_root = model_dir.parent
    try:
        model_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelLoadError(f"Could not create the model directory {model_root}: {exc}") from exc

    url = MODEL_ZOO_URL_TEMPLATE.format(name=name)
    archive = model_root / f"{name}.zip"
    logger.info("Downloading model %s from %s", name, url)
    _download_with_resume(url, archive, progress)

    try:
        _extract_archive(archive, model_dir)
    except zipfile.BadZipFile as exc:
        archive.unlink(missing_ok=True)
        raise ModelLoadError(
            f"Model download is incomplete or corrupt: {archive}. Delete it and run again."
        ) from exc
    except ModelLoadError:
        archive.unlink(missing_ok=True)
        raise

    if not _has_onnx_files(model_dir):
        raise ModelLoadError(f"Archive extracted but contains no .onnx files: {model_dir}")

    archive.unlink(missing_ok=True)
    logger.info("Model %s ready at %s", name, model_dir)
    return model_dir


def _download_with_resume(
    url: str, destination: Path, progress: ProgressCallback | None = None
) -> None:
    """Download with resume support and completeness verification."""
    part = destination.with_name(destination.name + ".part")
    existing = part.stat().st_size if part.is_file() else 0
    headers = {"User-Agent": _USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        logger.info("Resuming download at %.1f MB", existing / 1048576)

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = int(getattr(response, "status", 200))
            if existing and status != 206:
                existing = 0  # server ignored Range: start over
                mode = "wb"
            else:
                mode = "ab" if existing else "wb"
            total = int(response.headers.get("Content-Length") or 0) + existing
            downloaded = existing
            with part.open(mode) as handle:
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, total)
            if total and downloaded != total:
                raise ModelLoadError(
                    f"Model download incomplete ({downloaded} of {total} bytes). "
                    "Run again to resume."
                )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ModelLoadError(
            f"Model download failed: {exc}. The partial file is kept at {part.name} "
            "and the download resumes on the next run."
        ) from exc

    part.replace(destination)


class FaceDetector:
    """Detection and embedding with an InsightFace model. Not thread-safe.

    Use it from a single inference thread.
    """

    def __init__(
        self,
        config: ModelConfig,
        model_dir: Path,
        progress: ProgressCallback | None = None,
    ) -> None:
        self._config = config
        self._model_dir = Path(model_dir)
        self._progress = progress
        self._analysis: object | None = None
        self._recognition: object | None = None
        self._face_factory: Callable[..., object] | None = None
        self._providers: list[str] = []
        self._model_signature = ""
        self._timings = StageTimings()
        self._detection_rate = RateCounter()
        self._recognition_rate = RateCounter()

    @property
    def is_loaded(self) -> bool:
        return self._analysis is not None and self._recognition is not None

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(self._providers)

    @property
    def model_signature(self) -> str:
        """Recognition model identity, e.g. ``buffalo_s:w600k_mbf``.

        Stored with the embeddings. Two packs can both produce 512-dimensional
        vectors, so the dimension alone cannot prevent mixing them.
        """
        return self._model_signature

    def describe(self) -> str:
        """Short model summary for the UI and logs (no "Model" prefix)."""
        if not self.is_loaded:
            return f"{self._config.name} (not loaded)"
        width, height = self._config.det_size
        return f"{self._config.name} det {width}x{height} ({', '.join(self._providers)})"

    def stats(self) -> DetectorStats:
        return DetectorStats(
            detection_calls=self._timings.detection_calls,
            recognition_calls=self._timings.recognition_calls,
            detection_ms=self._timings.detection_ms,
            recognition_ms=self._timings.recognition_ms,
            detection_fps=self._detection_rate.rate(),
            recognition_fps=self._recognition_rate.rate(),
        )

    def _select_providers(self) -> list[str]:
        """Providers for the session, chosen from what this build can actually run."""
        available = execution_providers.available_providers()
        if not available:
            raise ModelLoadError(
                "onnxruntime is not usable: it either failed to import or reports no "
                "execution provider. Reinstall it with `pip install onnxruntime` "
                "(or onnxruntime-directml for an AMD or Intel GPU on Windows)."
            )
        usable = execution_providers.resolve(self._config.providers, available)
        if not usable:
            raise ModelLoadError(
                f"onnxruntime provides no usable execution provider (available: "
                f"{', '.join(sorted(available))}). Reinstall it with "
                "`pip install --force-reinstall onnxruntime`."
            )
        logger.info(
            "Execution provider: %s (requested %s, available %s)",
            execution_providers.describe(usable),
            list(self._config.providers),
            sorted(available),
        )
        return usable

    @staticmethod
    def _ctx_id(providers: Sequence[str]) -> int:
        # ctx_id = -1 means CPU in the InsightFace API; a GPU device uses 0.
        return -1 if providers and providers[0] == "CPUExecutionProvider" else 0

    def _build_session_options(self, providers: Sequence[str]) -> object:
        """ONNX Runtime graph optimisation and thread settings.

        DirectML (and TensorRT) refuse to build a session with memory patterns or
        parallel execution enabled, so the options follow the chosen provider.
        """
        import onnxruntime

        options = onnxruntime.SessionOptions()
        levels = {
            "disabled": onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        options.graph_optimization_level = levels[self._config.graph_optimization]
        if self._config.intra_op_threads > 0:
            options.intra_op_num_threads = self._config.intra_op_threads
        if self._config.inter_op_threads > 0:
            options.inter_op_num_threads = self._config.inter_op_threads
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        if execution_providers.needs_serial_execution(providers):
            options.enable_mem_pattern = False
        logger.info(
            "ONNX Runtime: %s, graph_optimization=%s, intra_op=%s, inter_op=%s",
            execution_providers.describe(providers),
            self._config.graph_optimization,
            self._config.intra_op_threads or "auto",
            self._config.inter_op_threads or "auto",
        )
        return options

    def _build_model_signature(self, recognition_model: object) -> str:
        model_file = getattr(recognition_model, "model_file", None)
        stem = Path(str(model_file)).stem if model_file else self._config.name
        return f"{self._config.name}:{stem}"

    def load(self) -> None:
        """Load the model, downloading the pack if needed. Safe to call twice."""
        if self.is_loaded:
            return
        try:
            from insightface.app import FaceAnalysis
            from insightface.app.common import Face
        except ImportError as exc:
            raise ModelLoadError(
                "The insightface package is missing or failed to import. "
                "Run `python install.py` or `pip install -r requirements.txt`."
            ) from exc

        model_dir = ensure_model_available(self._model_dir, self._config.name, self._progress)
        providers = self._select_providers()

        try:
            analysis = self._build_analysis(FaceAnalysis, model_dir, providers)
        except ModelLoadError as exc:
            if not execution_providers.is_gpu(providers):
                raise
            # A missing driver, an unsupported operator, or too little video memory
            # must degrade to the CPU instead of leaving the app without a model.
            logger.warning(
                "GPU inference (%s) failed: %s. Retrying on the CPU.",
                execution_providers.describe(providers),
                exc,
            )
            providers = [execution_providers.CPU_PROVIDER]
            analysis = self._build_analysis(FaceAnalysis, model_dir, providers)

        models = getattr(analysis, "models", {})
        if "detection" not in models or "recognition" not in models:
            raise ModelLoadError(
                f"Model pack at {model_dir} is incomplete: found {sorted(models)}; "
                "needs 'detection' and 'recognition'."
            )

        self._analysis = analysis
        self._recognition = models["recognition"]
        self._face_factory = Face
        self._providers = providers
        self._model_signature = self._build_model_signature(models["recognition"])
        logger.info(
            "Model loaded from %s (providers %s, signature %s)",
            model_dir,
            providers,
            self._model_signature,
        )

        if self._config.warmup:
            self.warmup()

    def _build_analysis(
        self, factory: object, model_dir: Path, providers: Sequence[str]
    ) -> object:
        """Create the InsightFace analysis object for one provider list."""
        try:
            analysis = factory(  # type: ignore[operator]
                name=str(model_dir),
                allowed_modules=["detection", "recognition"],
                providers=list(providers),
                sess_options=self._build_session_options(providers),
            )
            analysis.prepare(  # type: ignore[attr-defined]
                ctx_id=self._ctx_id(providers),
                det_thresh=self._config.det_thresh,
                det_size=tuple(self._config.det_size),
            )
        except Exception as exc:
            raise ModelLoadError(
                f"InsightFace model failed to load from {model_dir}: {exc}. "
                "Check that the .onnx files are complete (det_500m.onnx and w600k_mbf.onnx "
                "for buffalo_s; det_10g.onnx and w600k_r50.onnx for buffalo_l)."
            ) from exc
        return analysis

    def _require_loaded(self) -> object:
        if self._analysis is None:
            raise ModelLoadError("Model is not loaded. Call FaceDetector.load() first.")
        return self._analysis

    def warmup(self) -> None:
        """One dummy inference so the first real frame is not slow."""
        self._require_loaded()
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            self.detect(dummy, max_faces=1)
            logger.debug("Model warmup finished")
        except DetectionError as exc:  # pragma: no cover - rare
            logger.warning("Model warmup failed (ignored): %s", exc)

    def close(self) -> None:
        self._analysis = None
        self._recognition = None
        self._face_factory = None

    def detect(self, frame: np.ndarray, max_faces: int = 0) -> list[DetectedFace]:
        """Detect faces without extracting embeddings."""
        analysis = self._require_loaded()
        if frame is None or getattr(frame, "size", 0) == 0:
            raise DetectionError("Cannot detect on an empty frame.")
        try:
            started = time.perf_counter()
            boxes, landmarks = analysis.det_model.detect(  # type: ignore[attr-defined]
                frame, max_num=max_faces, metric="default"
            )
            self._timings.record_detection(time.perf_counter() - started)
            self._detection_rate.tick()
        except DetectionError:
            raise
        except Exception as exc:
            raise DetectionError(f"Face detection failed: {exc}") from exc

        faces: list[DetectedFace] = []
        if boxes is None or len(boxes) == 0:
            return faces
        for index in range(len(boxes)):
            landmarks_for_face = None
            if landmarks is not None and index < len(landmarks):
                landmarks_for_face = np.asarray(landmarks[index], dtype=np.float32)
            faces.append(
                DetectedFace(
                    bbox=np.asarray(boxes[index, 0:4], dtype=np.float32),
                    det_score=float(boxes[index, 4]),
                    landmarks=landmarks_for_face,
                )
            )
        return faces

    def embed(self, frame: np.ndarray, faces: Iterable[DetectedFace]) -> None:
        """Fill ``face.embedding`` (L2-normalised) for every face."""
        self._require_loaded()
        if self._recognition is None or self._face_factory is None:
            raise ModelLoadError("Recognition model is not loaded.")
        for face in faces:
            if face.landmarks is None:
                raise DetectionError("Alignment is impossible without face landmarks.")
            wrapper = self._face_factory(
                bbox=face.bbox, kps=face.landmarks, det_score=face.det_score
            )
            try:
                started = time.perf_counter()
                raw = self._recognition.get(frame, wrapper)  # type: ignore[attr-defined]
                self._timings.record_recognition(time.perf_counter() - started)
                self._recognition_rate.tick()
            except Exception as exc:
                raise DetectionError(f"Embedding extraction failed: {exc}") from exc
            if raw is None or len(raw) == 0:
                raise DetectionError("The recognition model returned an empty embedding.")
            face.embedding = normalize_embedding(raw)
