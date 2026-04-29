import unittest

from albatross_engine.config import (
    AlbatrossBackendConfig,
    is_albatross_strategy,
    parse_albatross_strategy,
)


class AlbatrossStrategyTests(unittest.TestCase):
    def test_is_albatross_strategy_accepts_new_and_legacy_names(self):
        self.assertTrue(is_albatross_strategy("albatross"))
        self.assertTrue(is_albatross_strategy("albatross workers=2 batch=64"))
        self.assertTrue(is_albatross_strategy("chirrup workers=1 batch=32"))
        self.assertFalse(is_albatross_strategy("cuda fp16"))
        self.assertFalse(is_albatross_strategy(""))

    def test_parse_albatross_strategy_defaults(self):
        self.assertEqual(
            parse_albatross_strategy("albatross"),
            AlbatrossBackendConfig(worker_num=1, batch_size=32),
        )

    def test_parse_albatross_strategy_custom_values(self):
        self.assertEqual(
            parse_albatross_strategy("albatross workers=2 batch=64"),
            AlbatrossBackendConfig(worker_num=2, batch_size=64),
        )

    def test_parse_albatross_strategy_ignores_invalid_values(self):
        self.assertEqual(
            parse_albatross_strategy("albatross workers=nope batch=-1"),
            AlbatrossBackendConfig(worker_num=1, batch_size=32),
        )


if __name__ == "__main__":
    unittest.main()
