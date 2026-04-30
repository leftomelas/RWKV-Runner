import time
import urllib.error


def is_connection_refused(exc: Exception) -> bool:
    if not isinstance(exc, urllib.error.URLError):
        return False
    reason = exc.reason
    if not isinstance(reason, OSError):
        return False
    return getattr(reason, "winerror", None) == 10061


def run_with_connect_retries(
    operation,
    *,
    connect_retries: int,
    connect_retry_delay: float,
    sleep=time.sleep,
):
    attempts = 0
    while True:
        attempts += 1
        try:
            result = operation()
            if isinstance(result, dict):
                result = dict(result)
                result["attempts"] = attempts
            return result
        except Exception as exc:
            exc.attempts = attempts
            if attempts > connect_retries or not is_connection_refused(exc):
                raise
            sleep(connect_retry_delay * attempts)
