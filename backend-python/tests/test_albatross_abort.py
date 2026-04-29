import unittest

from routes import completion
from tests.test_albatross_completion_contract import FakeAlbatross, FakeRequest


class AlbatrossAbortTests(unittest.IsolatedAsyncioTestCase):
    async def test_eval_albatross_aborts_on_disconnect_after_first_token(self):
        body = completion.CompletionBody(prompt="prompt")
        model = FakeAlbatross()

        chunks = []
        async for chunk in completion.eval_albatross(
            model, FakeRequest(disconnect_after=2), body, "prompt", True, None, None, False
        ):
            chunks.append(chunk)

        self.assertEqual(model.completion.abort_calls, 1)
        self.assertEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
