import os
import time
from typing import Callable, Optional


DEFAULT_LOG_INTERVAL_SECONDS = 5.0


def parse_log_interval(
    raw_value: Optional[str],
    default: float = DEFAULT_LOG_INTERVAL_SECONDS,
) -> float:
    if raw_value is None or raw_value == "":
        return default
    try:
        value = float(raw_value)
    except ValueError:
        return default
    return max(0.0, value)


def get_log_interval_from_env() -> float:
    return parse_log_interval(os.environ.get("ALBATROSS_TPS_LOG_INTERVAL"))


class ThroughputReporter:
    def __init__(
        self,
        worker_id: str,
        interval_seconds: float,
        clock: Callable[[], float] = time.perf_counter,
        sink: Callable[[str], None] = print,
    ):
        self.worker_id = worker_id
        self.interval_seconds = max(0.0, interval_seconds)
        self.clock = clock
        self.sink = sink
        self.interval_started_at: Optional[float] = None
        self.interval_tokens = 0

    @property
    def enabled(self) -> bool:
        return self.interval_seconds > 0.0

    def observe(self, decode_tokens: int, active_batch: int) -> None:
        if not self.enabled or decode_tokens <= 0:
            return

        now = self.clock()
        if self.interval_started_at is None:
            self.interval_started_at = now

        self.interval_tokens += decode_tokens
        elapsed = now - self.interval_started_at
        if elapsed < self.interval_seconds or elapsed <= 0:
            return

        tokens_per_second = self.interval_tokens / elapsed
        self.sink(
            f"[{self.worker_id}] Albatross decode throughput: "
            f"{tokens_per_second:.2f} tok/s, "
            f"active_batch={active_batch}, "
            f"interval_tokens={self.interval_tokens}"
        )
        self.interval_started_at = now
        self.interval_tokens = 0
