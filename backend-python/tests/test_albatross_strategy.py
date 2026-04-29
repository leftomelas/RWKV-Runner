import unittest
from unittest.mock import Mock, patch

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


class AlbatrossSwitchModelTests(unittest.TestCase):
    def setUp(self):
        import global_var

        global_var.init()

    def test_switch_model_selects_albatross_for_pth_albatross_strategy(self):
        import global_var
        from fastapi import Response
        from routes import config
        from utils.rwkv import ModelConfigBody

        fake_albatross = object()

        with (
            patch.object(config, "torch_gc"),
            patch.object(config.state_cache, "enable_state_cache"),
            patch.object(config, "RWKV") as rwkv_factory,
            patch.object(config, "Llama") as llama_factory,
            patch.object(
                config,
                "AlbatrossRWKV",
                Mock(return_value=fake_albatross),
                create=True,
            ) as albatross_factory,
            patch.object(config, "get_rwkv_config", return_value=ModelConfigBody()),
            patch.object(config, "get_llama_config", return_value=ModelConfigBody()),
        ):
            result = config.switch_model(
                config.SwitchModelBody(
                    model="models/rwkv7-test.pth",
                    strategy="albatross workers=2 batch=64",
                    tokenizer="",
                    customCuda=False,
                ),
                Response(),
                None,
            )

        self.assertEqual(result, "success")
        albatross_factory.assert_called_once_with(
            model_path="models/rwkv7-test.pth",
            worker_num=2,
            batch_size=64,
            tokenizer="",
        )
        rwkv_factory.assert_not_called()
        llama_factory.assert_not_called()
        self.assertIs(global_var.get(global_var.Model), fake_albatross)

    def test_switch_model_keeps_gguf_on_llama_even_with_albatross_strategy(self):
        import global_var
        from fastapi import Response
        from routes import config
        from utils.rwkv import ModelConfigBody

        fake_llama = object()

        with (
            patch.object(config, "torch_gc"),
            patch.object(config.state_cache, "enable_state_cache"),
            patch.object(config, "RWKV") as rwkv_factory,
            patch.object(
                config, "Llama", Mock(return_value=fake_llama)
            ) as llama_factory,
            patch.object(
                config,
                "AlbatrossRWKV",
                Mock(),
                create=True,
            ) as albatross_factory,
            patch.object(config, "get_rwkv_config", return_value=ModelConfigBody()),
            patch.object(config, "get_llama_config", return_value=ModelConfigBody()),
        ):
            result = config.switch_model(
                config.SwitchModelBody(
                    model="models/rwkv7-test.gguf",
                    strategy="albatross workers=2 batch=64",
                    tokenizer="",
                    customCuda=False,
                ),
                Response(),
                None,
            )

        self.assertEqual(result, "success")
        llama_factory.assert_called_once_with(
            model_path="models/rwkv7-test.gguf",
            strategy="albatross workers=2 batch=64",
        )
        albatross_factory.assert_not_called()
        rwkv_factory.assert_not_called()
        self.assertIs(global_var.get(global_var.Model), fake_llama)


if __name__ == "__main__":
    unittest.main()
