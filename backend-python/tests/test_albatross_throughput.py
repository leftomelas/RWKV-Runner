import unittest

from albatross_engine.throughput import ThroughputReporter, parse_log_interval


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class ThroughputReporterTests(unittest.TestCase):
    def test_parse_log_interval_uses_default_for_missing_or_invalid_values(self):
        self.assertEqual(parse_log_interval(None, default=5.0), 5.0)
        self.assertEqual(parse_log_interval("", default=5.0), 5.0)
        self.assertEqual(parse_log_interval("bad", default=5.0), 5.0)

    def test_parse_log_interval_allows_zero_to_disable_logging(self):
        self.assertEqual(parse_log_interval("0", default=5.0), 0.0)
        self.assertEqual(parse_log_interval("-1", default=5.0), 0.0)

    def test_reporter_does_not_log_before_interval(self):
        clock = FakeClock()
        messages = []
        reporter = ThroughputReporter(
            worker_id="0",
            interval_seconds=5.0,
            clock=clock,
            sink=messages.append,
        )

        reporter.observe(decode_tokens=100, active_batch=32)
        clock.advance(4.9)
        reporter.observe(decode_tokens=100, active_batch=32)

        self.assertEqual(messages, [])

    def test_reporter_logs_decode_tokens_per_second_after_interval(self):
        clock = FakeClock()
        messages = []
        reporter = ThroughputReporter(
            worker_id="0",
            interval_seconds=5.0,
            clock=clock,
            sink=messages.append,
        )

        reporter.observe(decode_tokens=100, active_batch=32)
        clock.advance(5.0)
        reporter.observe(decode_tokens=150, active_batch=64)

        self.assertEqual(len(messages), 1)
        self.assertIn("[0] Albatross decode throughput: 50.00 tok/s", messages[0])
        self.assertIn("active_batch=64", messages[0])
        self.assertIn("interval_tokens=250", messages[0])

    def test_reporter_ignores_zero_tokens_and_can_be_disabled(self):
        clock = FakeClock()
        messages = []
        reporter = ThroughputReporter(
            worker_id="0",
            interval_seconds=0.0,
            clock=clock,
            sink=messages.append,
        )

        reporter.observe(decode_tokens=100, active_batch=32)
        clock.advance(10.0)
        reporter.observe(decode_tokens=100, active_batch=32)

        self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()
