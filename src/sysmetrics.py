"""Process and machine CPU/RAM sampling without third-party dependencies.

`psutil` would add a build dependency for values the OS already exposes. Two CPU
figures are reported deliberately, because they answer different questions:

* ``process_cpu_percent``: share of the whole machine used by this application
  (0-100). 100% means the app saturates every core.
* ``machine_cpu_percent``: total system load (0-100). Useful to explain a
  dropped frame rate caused by something else on the machine.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_FILETIME_UNITS_PER_SECOND = 10_000_000.0


@dataclass(frozen=True)
class SystemSample:
    """One sample of system load."""

    process_cpu_percent: float
    machine_cpu_percent: float
    process_ram_mb: float
    cpu_cores: int

    def describe(self) -> str:
        return (
            f"app CPU {self.process_cpu_percent:5.1f}% | "
            f"system CPU {self.machine_cpu_percent:5.1f}% | "
            f"RAM {self.process_ram_mb:6.0f} MB"
        )


class _WindowsCounters:
    """CPU time and working set through the Win32 API."""

    def __init__(self) -> None:
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = [
                ("dwLowDateTime", wintypes.DWORD),
                ("dwHighDateTime", wintypes.DWORD),
            ]

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        self._filetime = FILETIME
        self._memory_counters = PROCESS_MEMORY_COUNTERS
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._psapi = ctypes.WinDLL("psapi", use_last_error=True)

        # Explicit prototypes are required: HANDLE is pointer sized, and an
        # untyped call passes it as a 32-bit int, which makes the calls fail.
        self._kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        self._kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
        ]
        self._kernel32.GetProcessTimes.restype = wintypes.BOOL
        self._kernel32.GetSystemTimes.argtypes = [
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
        ]
        self._kernel32.GetSystemTimes.restype = wintypes.BOOL
        self._psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        self._psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        self._handle = self._kernel32.GetCurrentProcess()

    def _ft_seconds(self, filetime) -> float:
        value = (filetime.dwHighDateTime << 32) | filetime.dwLowDateTime
        return value / _FILETIME_UNITS_PER_SECOND

    def process_cpu_seconds(self) -> float:
        creation, exit_, kernel, user = (
            self._filetime(),
            self._filetime(),
            self._filetime(),
            self._filetime(),
        )
        if not self._kernel32.GetProcessTimes(
            self._handle,
            ctypes.byref(creation),
            ctypes.byref(exit_),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise OSError(ctypes.get_last_error(), "GetProcessTimes failed")
        return self._ft_seconds(kernel) + self._ft_seconds(user)

    def system_times(self) -> tuple[float, float]:
        """Return (idle_seconds, total_seconds) for the machine."""
        idle, kernel, user = self._filetime(), self._filetime(), self._filetime()
        if not self._kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            raise OSError(ctypes.get_last_error(), "GetSystemTimes failed")
        return self._ft_seconds(idle), self._ft_seconds(kernel) + self._ft_seconds(user)

    def working_set_bytes(self) -> int:
        counters = self._memory_counters()
        counters.cb = ctypes.sizeof(counters)
        if not self._psapi.GetProcessMemoryInfo(
            self._handle, ctypes.byref(counters), counters.cb
        ):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)


class _PosixCounters:
    """CPU time and memory through ``resource`` and ``/proc`` (Linux, macOS)."""

    def __init__(self) -> None:
        import resource

        self._resource = resource

    def process_cpu_seconds(self) -> float:
        usage = self._resource.getrusage(self._resource.RUSAGE_SELF)
        return float(usage.ru_utime + usage.ru_stime)

    def system_times(self) -> tuple[float, float]:
        try:
            with open("/proc/stat", encoding="utf-8") as handle:
                fields = handle.readline().split()[1:]
            values = [float(value) for value in fields]
            tick = float(os.sysconf("SC_CLK_TCK"))
            idle = values[3] + (values[4] if len(values) > 4 else 0.0)
            return idle / tick, sum(values) / tick
        except (OSError, IndexError, ValueError, AttributeError):
            return 0.0, 0.0

    def working_set_bytes(self) -> int:
        try:
            with open("/proc/self/statm", encoding="utf-8") as handle:
                pages = int(handle.read().split()[1])
            return pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, IndexError, ValueError, AttributeError):
            usage = self._resource.getrusage(self._resource.RUSAGE_SELF)
            return int(usage.ru_maxrss) * 1024


class SystemMetrics:
    """CPU and RAM sampler; call ``sample()`` once per interval."""

    def __init__(self) -> None:
        self._cores = max(1, os.cpu_count() or 1)
        if sys.platform.startswith("win"):
            self._counters: _WindowsCounters | _PosixCounters = _WindowsCounters()
        else:
            self._counters = _PosixCounters()
        self._last_wall = time.monotonic()
        self._last_process_cpu = self._counters.process_cpu_seconds()
        self._last_system = self._counters.system_times()

    @property
    def cpu_cores(self) -> int:
        return self._cores

    def sample(self) -> SystemSample:
        """Measure usage since the previous call."""
        now = time.monotonic()
        process_cpu = self._counters.process_cpu_seconds()
        system = self._counters.system_times()

        elapsed = max(now - self._last_wall, 1e-6)
        process_delta = max(process_cpu - self._last_process_cpu, 0.0)
        idle_delta = max(system[0] - self._last_system[0], 0.0)
        total_delta = max(system[1] - self._last_system[1], 0.0)

        process_percent = (process_delta / elapsed) / self._cores * 100.0
        machine_percent = 0.0
        if total_delta > 0.0:
            machine_percent = max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))

        ram_mb = self._counters.working_set_bytes() / (1024.0 * 1024.0)
        self._last_wall = now
        self._last_process_cpu = process_cpu
        self._last_system = system
        return SystemSample(
            process_cpu_percent=round(process_percent, 2),
            machine_cpu_percent=round(machine_percent, 2),
            process_ram_mb=round(ram_mb, 1),
            cpu_cores=self._cores,
        )
