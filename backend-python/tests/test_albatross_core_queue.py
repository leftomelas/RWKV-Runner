import asyncio
import unittest

from albatross_engine.core import EventLoopQueueDispatcher, ThreadSafeAsyncQueue


class FakeEventLoop:
    def __init__(self):
        self.callbacks = []

    def is_closed(self):
        return False

    def call_soon_threadsafe(self, callback):
        self.callbacks.append(callback)


class ThreadSafeAsyncQueueDispatcherTests(unittest.TestCase):
    def test_shared_dispatcher_coalesces_items_across_result_queues(self):
        loop = FakeEventLoop()
        dispatcher = EventLoopQueueDispatcher(loop)
        first_queue = asyncio.Queue()
        second_queue = asyncio.Queue()
        first_channel = ThreadSafeAsyncQueue(loop, first_queue, dispatcher=dispatcher)
        second_channel = ThreadSafeAsyncQueue(loop, second_queue, dispatcher=dispatcher)

        first_channel.put_nowait("first")
        second_channel.put_nowait("second")

        self.assertEqual(len(loop.callbacks), 1)
        loop.callbacks[0]()
        self.assertEqual(first_queue.get_nowait(), "first")
        self.assertEqual(second_queue.get_nowait(), "second")


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
