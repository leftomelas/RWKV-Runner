import urllib.error

from bench.albatross_http_client import (
    is_connection_refused,
    run_with_connect_retries,
)


def refused_error():
    reason = OSError("connection refused")
    reason.winerror = 10061
    return urllib.error.URLError(reason)


def test_connection_refused_detection_matches_windows_error():
    assert is_connection_refused(refused_error()) is True
    assert is_connection_refused(RuntimeError("different failure")) is False


def test_run_with_connect_retries_retries_refused_connections_only():
    calls = []
    sleeps = []

    def operation():
        calls.append("call")
        if len(calls) < 3:
            raise refused_error()
        return {"ok": True, "tokens": 7}

    result = run_with_connect_retries(
        operation,
        connect_retries=3,
        connect_retry_delay=0.05,
        sleep=sleeps.append,
    )

    assert result == {"ok": True, "tokens": 7, "attempts": 3}
    assert sleeps == [0.05, 0.10]


def test_run_with_connect_retries_does_not_retry_other_errors():
    calls = []

    def operation():
        calls.append("call")
        raise ValueError("bad payload")

    try:
        run_with_connect_retries(
            operation,
            connect_retries=3,
            connect_retry_delay=0.05,
            sleep=lambda _: None,
        )
    except ValueError as error:
        assert str(error) == "bad payload"
        assert getattr(error, "attempts") == 1
    else:
        raise AssertionError("expected ValueError")

    assert calls == ["call"]
