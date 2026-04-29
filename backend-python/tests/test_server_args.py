import unittest

from main import get_args


class ServerArgsTests(unittest.TestCase):
    def test_high_concurrency_server_args_have_safe_defaults(self):
        args = get_args([])

        self.assertEqual(args.backlog, 2048)
        self.assertEqual(args.timeout_keep_alive, 120)
        self.assertFalse(args.no_access_log)

    def test_high_concurrency_server_args_can_be_overridden(self):
        args = get_args(
            [
                "--backlog",
                "4096",
                "--timeout-keep-alive",
                "240",
                "--no-access-log",
            ]
        )

        self.assertEqual(args.backlog, 4096)
        self.assertEqual(args.timeout_keep_alive, 240)
        self.assertTrue(args.no_access_log)


if __name__ == "__main__":
    unittest.main()
