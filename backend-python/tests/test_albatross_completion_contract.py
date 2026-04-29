import asyncio
import json
import unittest

from routes import completion


class FakeRequest:
    def __init__(self, disconnect_after=None):
        self.client = "test-client"
        self._calls = 0
        self._disconnect_after = disconnect_after

    async def is_disconnected(self):
        self._calls += 1
        return (
            self._disconnect_after is not None
            and self._calls >= self._disconnect_after
        )


class FakeAlbatrossCompletion:
    def __init__(self, events):
        self._events = iter(events)
        self.abort_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._events)

    def abort(self):
        self.abort_calls += 1


class FakeAlbatross:
    name = "albatross"

    def __init__(self, events=None):
        self.completion = FakeAlbatrossCompletion(
            events
            or [
                ("text", "Hello", "Hello", 3, 1),
                ("text", "Hello world", " world", 3, 2),
            ]
        )

    def generate(self, body, prompt, stop=None, stop_token_ids=None):
        self.generate_args = (body, prompt, stop, stop_token_ids)
        return self.completion


class FakeAsyncAlbatross:
    name = "albatross"

    def __init__(self):
        self.async_generate_args = None

    def generate(self, body, prompt, stop=None, stop_token_ids=None):
        raise AssertionError("blocking generate should not be used")

    async def async_generate(self, body, prompt, stop=None, stop_token_ids=None):
        self.async_generate_args = (body, prompt, stop, stop_token_ids)
        yield ("text", "Hello", "Hello", 3, 1)
        yield ("text", "Hello async", " async", 3, 2)


class AlbatrossCompletionContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_eval_albatross_non_streaming_chat_response(self):
        body = completion.ChatCompletionBody(messages=[])
        model = FakeAlbatross()

        result = await completion.eval_albatross(
            model, FakeRequest(), body, "prompt", False, None, None, True
        ).__anext__()

        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(result["model"], "albatross")
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(result["usage"]["prompt_tokens"], 3)
        self.assertEqual(result["usage"]["completion_tokens"], 2)
        self.assertEqual(result["usage"]["total_tokens"], 5)

    async def test_eval_albatross_streaming_chat_response(self):
        body = completion.ChatCompletionBody(messages=[])
        model = FakeAlbatross()

        chunks = []
        async for chunk in completion.eval_albatross(
            model, FakeRequest(), body, "prompt", True, None, None, True
        ):
            chunks.append(chunk)

        first = json.loads(chunks[0])
        second = json.loads(chunks[1])
        final = json.loads(chunks[2])
        self.assertEqual(first["choices"][0]["delta"]["content"], "Hello")
        self.assertEqual(second["choices"][0]["delta"]["content"], " world")
        self.assertEqual(final["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1], "[DONE]")

    async def test_eval_albatross_prefers_async_generation_path(self):
        body = completion.ChatCompletionBody(messages=[])
        model = FakeAsyncAlbatross()

        chunks = []
        async for chunk in completion.eval_albatross(
            model, FakeRequest(), body, "prompt", True, None, None, True
        ):
            chunks.append(chunk)

        self.assertEqual(model.async_generate_args, (body, "prompt", None, None))
        self.assertEqual(json.loads(chunks[1])["choices"][0]["delta"]["content"], " async")

    async def test_eval_albatross_throttles_stream_disconnect_polling(self):
        body = completion.ChatCompletionBody(messages=[])
        events = [
            ("text", "x" * index, "x", 3, index)
            for index in range(1, 11)
        ]
        model = FakeAlbatross(events)
        request = FakeRequest()

        chunks = []
        async for chunk in completion.eval_albatross(
            model, request, body, "prompt", True, None, None, True
        ):
            chunks.append(chunk)

        self.assertLessEqual(request._calls, 4)
        self.assertEqual(chunks[-1], "[DONE]")

    async def test_eval_dispatches_albatross_without_waiting_on_completion_lock(self):
        body = completion.CompletionBody(prompt="prompt")
        model = FakeAlbatross()
        completion.completion_lock.acquire()
        original_albatross_cls = getattr(completion, "AlbatrossRWKV", None)
        completion.AlbatrossRWKV = FakeAlbatross
        try:
            result = await asyncio.wait_for(
                completion.eval(
                    model, FakeRequest(), body, "prompt", False, None, None, False
                ).__anext__(),
                timeout=0.2,
            )
        finally:
            completion.completion_lock.release()
            if original_albatross_cls is not None:
                completion.AlbatrossRWKV = original_albatross_cls

        self.assertEqual(result["object"], "text_completion")
        self.assertEqual(result["choices"][0]["text"], "Hello world")


if __name__ == "__main__":
    unittest.main()
