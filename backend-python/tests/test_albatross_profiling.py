import time

from albatross_engine.profiling import ProfileAccumulator


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
