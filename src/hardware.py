"""Runtime hardware detection used to pick a performance tier.

Deliberately dependency-free: `psutil` would add a build dependency for values
the platform already exposes.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_UNKNOWN_RAM_MB = 0


def _windows_total_ram_mb() -> int:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        return _UNKNOWN_RAM_MB
    return int(status.ullTotalPhys // (1024 * 1024))


def _linux_total_ram_mb() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except (OSError, IndexError, ValueError):
        pass
    return _UNKNOWN_RAM_MB


def _macos_total_ram_mb() -> int:
    # No shell: the argv is fixed, and nothing here can be influenced by input.
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return int(result.stdout.strip()) // (1024 * 1024)
    except (OSError, ValueError, subprocess.SubprocessError):  # pragma: no cover - macOS only
        return _UNKNOWN_RAM_MB


def total_ram_mb() -> int:
    if sys.platform.startswith("win"):
        return _windows_total_ram_mb()
    if sys.platform == "darwin":
        return _macos_total_ram_mb()  # pragma: no cover
    return _linux_total_ram_mb()


_SBC_MARKERS = (
    "raspberry",
    "orange pi",
    "orangepi",
    "banana pi",
    "rock pi",
    "rockchip",
    "jetson",
    "odroid",
    "libre computer",
)


def _single_board_model() -> str:
    """Read the board model string exposed by SBC device trees."""
    for candidate in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            text = Path(candidate).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        cleaned = text.replace("\x00", "").strip()
        if cleaned:
            return cleaned
    return ""


@dataclass(frozen=True)
class Hardware:
    """Facts about the machine the app runs on."""

    cpu_cores: int
    total_ram_mb: int
    platform_tag: str
    machine: str
    board_model: str = ""

    @property
    def is_single_board(self) -> bool:
        lowered = self.board_model.lower()
        if any(marker in lowered for marker in _SBC_MARKERS):
            return True
        return self.machine.lower().startswith(("arm", "aarch64"))

    @property
    def summary(self) -> str:
        ram = f"{self.total_ram_mb} MB" if self.total_ram_mb else "unknown RAM"
        board = f" · {self.board_model}" if self.board_model else ""
        return f"{self.cpu_cores} cores · {ram} · {self.platform_tag}/{self.machine}{board}"


def detect_hardware() -> Hardware:
    return Hardware(
        cpu_cores=max(1, os.cpu_count() or 1),
        total_ram_mb=total_ram_mb(),
        platform_tag=sys.platform,
        machine=platform.machine() or "unknown",
        board_model=_single_board_model(),
    )
