import time
import asyncio
import queue

from albatross_engine.profiling import ProfileAccumulator
from albatross_engine.task import Task
from albatross_engine.worker import StateCategory, Worker


class FakeTokenizer:
    def decode(self, tokens, utf8_errors="ignore"):
        return f"token-{tokens[0]}"


def test_profile_accumulator_records_timer_and_counter():
    profile = ProfileAccumulator(enabled=True)

    with profile.time("sampling"):
        time.sleep(0.001)
    profile.add("decode_tokens", 3)

    snapshot = profile.snapshot(reset=False)

    assert snapshot["enabled"] is True
    assert snapshot["counters"]["decode_tokens"] == 3
    assert snapshot["timers"]["sampling"]["count"] == 1
    assert snapshot["timers"]["sampling"]["total_ms"] > 0


def test_profile_accumulator_reset_clears_values():
    profile = ProfileAccumulator(enabled=True)
    profile.add("worker_loops", 2)

    first = profile.snapshot(reset=True)
    second = profile.snapshot(reset=False)

    assert first["counters"]["worker_loops"] == 2
    assert second["counters"] == {}
    assert second["timers"] == {}


def test_disabled_profile_has_empty_snapshot():
    profile = ProfileAccumulator(enabled=False)

    with profile.time("sampling"):
        pass
    profile.add("decode_tokens", 3)

    snapshot = profile.snapshot(reset=False)

    assert snapshot == {"enabled": False, "counters": {}, "timers": {}}


def test_profile_payload_shape_is_stable():
    profile = ProfileAccumulator(enabled=True)
    profile.add("worker_loops", 1)

    payload = profile.snapshot(reset=False)

    assert set(payload) == {"enabled", "counters", "timers"}
    assert payload["counters"]["worker_loops"] == 1


def test_worker_decode_phase_profiles_token_decode_and_output_enqueue():
    worker = Worker.__new__(Worker)
    worker.profile = ProfileAccumulator(enabled=True)
    worker.tokenizer = FakeTokenizer()
    worker.no_penalty_token_ids = set()

    task = Task(
        output_queue=asyncio.Queue(),
        task_event_queue=queue.Queue(),
        prompt_str="",
        prefill_tokens=[],
        state=None,
        max_tokens=2,
        stop_tokens=[],
    )
    task_data = {
        "task": task,
        "is_prefilling": False,
        "new_token": 42,
        "next_input_token": None,
        "state_category": StateCategory.FORWARD_ONE_DECODE,
        "prefilled_tokens": [],
        "prefill_cached": False,
    }

    update_info = worker._handle_forward_one_decode_phase(task_data, 3)
    snapshot = worker.profile.snapshot(reset=False)

    assert update_info == (3, 42, 1.0)
    assert snapshot["timers"]["decode_tokenizer_decode"]["count"] == 1
    assert snapshot["timers"]["decode_output_enqueue"]["count"] == 1
