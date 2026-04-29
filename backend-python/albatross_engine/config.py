from dataclasses import dataclass


@dataclass(frozen=True)
class AlbatrossBackendConfig:
    worker_num: int = 1
    batch_size: int = 32


def is_albatross_strategy(strategy: str | None) -> bool:
    if not strategy:
        return False
    first = strategy.strip().lower().split(maxsplit=1)[0]
    return first in {"albatross", "chirrup"}


def parse_albatross_strategy(strategy: str | None) -> AlbatrossBackendConfig:
    worker_num = 1
    batch_size = 32
    if not strategy:
        return AlbatrossBackendConfig(worker_num=worker_num, batch_size=batch_size)

    for part in strategy.lower().split()[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            parsed = int(value)
        except ValueError:
            continue
        if parsed < 1:
            continue
        if key in {"workers", "worker", "worker_num"}:
            worker_num = parsed
        elif key in {"batch", "batch_size"}:
            batch_size = parsed

    return AlbatrossBackendConfig(worker_num=worker_num, batch_size=batch_size)
