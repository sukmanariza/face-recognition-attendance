"""Pick the ONNX Runtime execution provider that is actually usable.

Both InsightFace models are small, so the provider decides most of the speed:

* ``DmlExecutionProvider`` runs on any DirectX 12 GPU on Windows, which is the
  only route to an AMD or Intel integrated GPU (``onnxruntime-directml``).
* ``CUDAExecutionProvider`` and ``TensorrtExecutionProvider`` need
  ``onnxruntime-gpu`` and an NVIDIA card with a matching CUDA runtime.
* ``OpenVINOExecutionProvider`` covers Intel graphics and CPUs.
* ``CPUExecutionProvider`` is always present and always the last resort.

``model.providers: [auto]`` takes the fastest backend the installed build
provides, so a machine without a usable GPU behaves exactly as before.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

AUTO = "auto"
CPU_PROVIDER = "CPUExecutionProvider"

# Ordered by expected speed for the two InsightFace models. A provider is only
# used when the installed onnxruntime build contains it.
PREFERENCE: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
    CPU_PROVIDER,
)

GPU_PROVIDERS = frozenset(PREFERENCE[:-1])

# DirectML cannot pre-allocate memory patterns or run inferences in parallel, and ONNX
# Runtime refuses the session otherwise. TensorRT has the same restriction.
SERIAL_PROVIDERS = frozenset({"DmlExecutionProvider", "TensorrtExecutionProvider"})

LABELS = {
    "TensorrtExecutionProvider": "TensorRT",
    "CUDAExecutionProvider": "CUDA",
    "DmlExecutionProvider": "DirectML",
    "OpenVINOExecutionProvider": "OpenVINO",
    "CoreMLExecutionProvider": "CoreML",
    CPU_PROVIDER: "CPU",
}

# What a user may write in model.providers, beyond the concrete provider names.
SHORTCUTS = {
    "cpu": (CPU_PROVIDER,),
    "cuda": ("CUDAExecutionProvider", CPU_PROVIDER),
    "directml": ("DmlExecutionProvider", CPU_PROVIDER),
    "openvino": ("OpenVINOExecutionProvider", CPU_PROVIDER),
}


def known_names() -> tuple[str, ...]:
    """Every value ``model.providers`` accepts."""
    return (AUTO, *SHORTCUTS, *PREFERENCE)


def available_providers() -> tuple[str, ...]:
    """Providers compiled into the installed onnxruntime build."""
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - reported by the model loader
        logger.debug("onnxruntime is not importable: %s", exc)
        return ()
    try:
        return tuple(onnxruntime.get_available_providers())
    except RuntimeError as exc:  # pragma: no cover - defensive
        logger.debug("onnxruntime.get_available_providers() failed: %s", exc)
        return ()


def expand(requested: Sequence[str]) -> list[str]:
    """Turn config shortcuts (``cpu``, ``directml``) into provider names."""
    expanded: list[str] = []
    for name in requested or [AUTO]:
        expanded.extend(SHORTCUTS.get(name, (name,)))
    return expanded


def resolve(requested: Sequence[str], available: Iterable[str]) -> list[str]:
    """Ordered provider list for the session, CPU kept as the last resort."""
    present = set(available)
    wanted = expand(requested)
    if AUTO in wanted:
        candidates = [name for name in PREFERENCE if name in present]
    else:
        candidates = []
        seen: set[str] = set()
        for name in wanted:
            if name in seen:
                continue
            seen.add(name)
            if name in present:
                candidates.append(name)
            else:
                logger.warning(
                    "model.providers asks for %s, which this onnxruntime build does not "
                    "provide (%s); skipping it.",
                    name,
                    ", ".join(sorted(present)) or "none",
                )
    if CPU_PROVIDER in present and CPU_PROVIDER not in candidates:
        candidates.append(CPU_PROVIDER)
    return candidates


def plan(requested: Sequence[str]) -> tuple[list[str], tuple[str, ...]]:
    """The providers this machine would use, and everything it has available."""
    available = available_providers()
    return resolve(requested, available), available


def is_gpu(providers: Sequence[str]) -> bool:
    """True when the first provider sends the work to a GPU."""
    return bool(providers) and providers[0] in GPU_PROVIDERS


def needs_serial_execution(providers: Sequence[str]) -> bool:
    """True when the session must run one inference at a time without mem patterns."""
    return any(name in SERIAL_PROVIDERS for name in providers)


def describe(providers: Sequence[str]) -> str:
    """Short label for the UI, e.g. ``DirectML (GPU)`` or ``CPU``."""
    if not providers:
        return "not loaded"
    primary = providers[0]
    label = LABELS.get(primary, primary)
    if primary == CPU_PROVIDER:
        return label
    return f"{label} (GPU)"
