import asyncio
import unittest

from albatross_engine.core import ThreadSafeAsyncQueue


class ThreadSafeAsyncQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_put_nowait_drops_item_when_queue_is_full(self):
        loop = asyncio.get_running_loop()
        errors = []
        original_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        try:
            queue = asyncio.Queue(maxsize=1)
            queue.put_nowait("existing")
            thread_safe_queue = ThreadSafeAsyncQueue(loop, queue)

            thread_safe_queue.put_nowait("dropped")
            await asyncio.sleep(0)

            self.assertEqual(queue.qsize(), 1)
            self.assertEqual(await queue.get(), "existing")
            self.assertEqual(errors, [])
        finally:
            loop.set_exception_handler(original_handler)


if __name__ == "__main__":
    unittest.main()
