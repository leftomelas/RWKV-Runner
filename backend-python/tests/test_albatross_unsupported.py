import unittest
from unittest.mock import patch

from albatross_engine.adapter import AlbatrossRWKV


class AlbatrossUnsupportedTests(unittest.TestCase):
    def test_albatross_embedding_is_explicitly_unsupported(self):
        with patch.object(AlbatrossRWKV, "_init_engine", lambda self: None):
            model = AlbatrossRWKV("models/example-rwkv7.pth")

        with self.assertRaisesRegex(NotImplementedError, "does not support embeddings"):
            model.get_embedding("hello", fast_mode=False)

    def test_albatross_run_rnn_is_explicitly_unsupported(self):
        with patch.object(AlbatrossRWKV, "_init_engine", lambda self: None):
            model = AlbatrossRWKV("models/example-rwkv7.pth")

        with self.assertRaisesRegex(NotImplementedError, "batch inference"):
            model.run_rnn([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
