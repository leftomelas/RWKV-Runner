from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import time
from typing import Iterator


@dataclass
class TimerValue:
    count: int = 0
    total_ns: int = 0

    def add(self, elapsed_ns: int) -> None:
        self.count += 1
        self.total_ns += elapsed_ns

    def snapshot(self) -> dict:
        total_ms = self.total_ns / 1_000_000
        return {
            "count": self.count,
            "total_ms": total_ms,
            "avg_ms": total_ms / self.count if self.count else 0.0,
        }


@dataclass
class ProfileAccumulator:
    enabled: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _counters: dict[str, int] = field(default_factory=dict)
    _timers: dict[str, TimerValue] = field(default_factory=dict)

    def add(self, name: str, value: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - started
            with self._lock:
                self._timers.setdefault(name, TimerValue()).add(elapsed)

    def snapshot(self, reset: bool = False) -> dict:
        with self._lock:
            data = {
                "enabled": self.enabled,
                "counters": dict(self._counters),
                "timers": {
                    name: timer.snapshot()
                    for name, timer in self._timers.items()
                },
            }
            if reset:
                self._counters.clear()
                self._timers.clear()
            return data
