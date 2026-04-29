import asyncio
import pathlib
import queue
import threading
from typing import Dict, List, Optional, Tuple, Union

from utils.rwkv import AbstractRWKV, ModelConfigBody, RWKVType


class AlbatrossCompletion:
    def __init__(self, generator, abort_callback):
        self._generator = generator
        self._abort_callback = abort_callback
        self._aborted = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._aborted:
            raise StopIteration
        return next(self._generator)

    def abort(self):
        if not self._aborted:
            self._aborted = True
            if self._abort_callback:
                self._abort_callback()


class AlbatrossRWKV(AbstractRWKV):
    def __init__(
        self,
        model_path: str,
        worker_num: int = 1,
        batch_size: int = 32,
        tokenizer: Optional[str] = None,
    ):
        self.EOS_ID = 0
        self.name = "albatross"
        self.model_path = model_path
        self.version = 7
        self.model = None
        self.pipeline = None
        self.model_state = None
        self.model_tokens = []
        self.rwkv_type = RWKVType.World
        self.tokenizer_len = 65536

        self.max_tokens_per_generation = 500
        self.temperature = 1.0
        self.top_p = 0.3
        self.top_k = 0
        self.penalty_alpha_presence = 0
        self.penalty_alpha_frequency = 1
        self.penalty_decay = 0.996
        self.global_penalty = False
        self.state_path = ""
        self.state_tuned = None

        self.interface = ":"
        self.user = "User"
        self.bot = "Assistant"

        self._worker_num = worker_num
        self._batch_size = batch_size
        self._vocab_path = tokenizer or self._get_default_vocab_path()

        self._engine_core = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._engine_thread: Optional[threading.Thread] = None
        self._is_initialized = False
        self._init_lock = threading.Lock()

        self._init_engine()

    def _get_default_vocab_path(self) -> str:
        possible_paths = [
            pathlib.Path(__file__).parent.parent
            / "albatross"
            / "rwkv_vocab_v20230424.txt",
            pathlib.Path(__file__).parent.parent
            / "rwkv_pip"
            / "rwkv_vocab_v20230424.txt",
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
        return str(possible_paths[0])

    def _init_engine(self):
        with self._init_lock:
            if self._is_initialized:
                return

            from albatross_engine.core import AsyncEngineCore
            from albatross_engine.task import ModelLoadConfig

            model_config = ModelLoadConfig(
                model_path=self.model_path,
                vocab_path=self._vocab_path,
                vocab_size=65536,
                head_size=64,
            )

            init_event = threading.Event()
            init_error = [None]

            def run_event_loop():
                try:
                    self._event_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._event_loop)
                    self._engine_core = AsyncEngineCore()

                    async def init_workers():
                        try:
                            init_task = self._engine_core.init(
                                worker_num=self._worker_num,
                                model_config=model_config,
                                batch_size=self._batch_size + 1,
                            )
                            await init_task
                            print(
                                "Albatross engine initialized: "
                                f"workers={self._worker_num}, batch_size={self._batch_size}"
                            )
                        except Exception as e:
                            init_error[0] = e
                            raise
                        finally:
                            init_event.set()

                    self._event_loop.run_until_complete(init_workers())
                    self._event_loop.run_forever()
                except Exception as e:
                    init_error[0] = e
                    init_event.set()

            self._engine_thread = threading.Thread(
                target=run_event_loop,
                daemon=True,
                name="albatross-engine",
            )
            self._engine_thread.start()

            if not init_event.wait(timeout=300):
                raise TimeoutError("Albatross engine initialization timed out")
            if init_error[0] is not None:
                raise RuntimeError(
                    f"Albatross engine initialization failed: {init_error[0]}"
                )

            self._is_initialized = True

    def adjust_occurrence(self, occurrence: Dict, token: int):
        pass

    def adjust_forward_logits(self, logits: List[float], occurrence: Dict, i: int):
        pass

    def fix_tokens(self, tokens) -> List[int]:
        return tokens

    def run_rnn(self, _tokens: List[str], newline_adj: int = 0):
        raise NotImplementedError(
            "AlbatrossRWKV uses batch inference. Use generate() instead of run_rnn()."
        )

    def delta_postprocess(self, delta: str) -> str:
        return delta

    def _generation_config(
        self,
        body: ModelConfigBody,
        prompt: str,
        stop_token_ids: Union[List[int], None] = None,
    ):
        temperature = body.temperature if body.temperature is not None else self.temperature
        if temperature < 0.1:
            temperature = 0.1
        top_p = body.top_p if body.top_p is not None else self.top_p
        top_k = body.top_k if body.top_k is not None else self.top_k
        presence_penalty = (
            body.presence_penalty
            if body.presence_penalty is not None
            else self.penalty_alpha_presence
        )
        frequency_penalty = (
            body.frequency_penalty
            if body.frequency_penalty is not None
            else self.penalty_alpha_frequency
        )
        penalty_decay = (
            body.penalty_decay
            if body.penalty_decay is not None
            else self.penalty_decay
        )
        max_tokens = (
            body.max_tokens
            if body.max_tokens is not None
            else self.max_tokens_per_generation
        )

        effective_stop_tokens = [self.EOS_ID]
        if stop_token_ids:
            effective_stop_tokens.extend(stop_token_ids)
        effective_stop_tokens = list(set(effective_stop_tokens))

        return {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "penalty_decay": penalty_decay,
            "max_tokens": max_tokens,
            "stop_tokens": effective_stop_tokens,
            "prompt_tokens": len(self._engine_core.tokenizer.encode(prompt)),
        }

    def generate(
        self,
        body: ModelConfigBody,
        prompt: str,
        stop: Union[str, List[str], None] = None,
        stop_token_ids: Union[List[int], None] = None,
    ) -> AlbatrossCompletion:
        config = self._generation_config(body, prompt, stop_token_ids)
        result_queue: queue.Queue = queue.Queue()
        abort_event = threading.Event()
        current_completion_holder = [None]

        async def async_completion():
            try:
                completion = self._engine_core.completion(
                    prompt_str=prompt,
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    top_k=config["top_k"],
                    presence_penalty=config["presence_penalty"],
                    frequency_penalty=config["frequency_penalty"],
                    penalty_decay=config["penalty_decay"],
                    max_tokens=config["max_tokens"],
                    stop_tokens=config["stop_tokens"],
                )
                current_completion_holder[0] = completion
                async for event in completion:
                    if abort_event.is_set():
                        completion.abort()
                        break
                    result_queue.put(event)
            except Exception as e:
                result_queue.put(("error", str(e)))
            finally:
                current_completion_holder[0] = None
                result_queue.put(None)

        asyncio.run_coroutine_threadsafe(async_completion(), self._event_loop)

        def abort_this_completion():
            abort_event.set()
            if current_completion_holder[0]:
                try:
                    current_completion_holder[0].abort()
                except Exception:
                    pass

        def generate_tokens():
            response = ""
            completion_tokens = 0
            try:
                while True:
                    try:
                        event = result_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if event is None:
                        break
                    if event[0] == "error":
                        raise RuntimeError(f"Albatross generation error: {event[1]}")
                    if event[0] != "token":
                        continue

                    token_id, delta = event[1], event[2]
                    completion_tokens += 1
                    response += delta

                    should_stop = False
                    if stop:
                        stops = [stop] if isinstance(stop, str) else stop
                        for stop_text in stops:
                            if stop_text in response:
                                response = response.split(stop_text)[0]
                                should_stop = True
                                break

                    yield (
                        "text",
                        response,
                        delta,
                        config["prompt_tokens"],
                        completion_tokens,
                    )

                    if should_stop:
                        abort_event.set()
                        break
            except GeneratorExit:
                abort_this_completion()
                raise

        return AlbatrossCompletion(generate_tokens(), abort_this_completion)

    async def async_generate(
        self,
        body: ModelConfigBody,
        prompt: str,
        stop: Union[str, List[str], None] = None,
        stop_token_ids: Union[List[int], None] = None,
    ):
        config = self._generation_config(body, prompt, stop_token_ids)
        result_queue: asyncio.Queue = asyncio.Queue()
        caller_loop = asyncio.get_running_loop()
        abort_event = threading.Event()
        current_completion_holder = [None]

        def put_result(event):
            try:
                result_queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

        async def async_completion():
            try:
                completion = self._engine_core.completion(
                    prompt_str=prompt,
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    top_k=config["top_k"],
                    presence_penalty=config["presence_penalty"],
                    frequency_penalty=config["frequency_penalty"],
                    penalty_decay=config["penalty_decay"],
                    max_tokens=config["max_tokens"],
                    stop_tokens=config["stop_tokens"],
                )
                current_completion_holder[0] = completion
                async for event in completion:
                    if abort_event.is_set():
                        completion.abort()
                        break
                    caller_loop.call_soon_threadsafe(put_result, event)
            except Exception as e:
                caller_loop.call_soon_threadsafe(put_result, ("error", str(e)))
            finally:
                current_completion_holder[0] = None
                caller_loop.call_soon_threadsafe(put_result, None)

        future = asyncio.run_coroutine_threadsafe(async_completion(), self._event_loop)

        def abort_this_completion():
            abort_event.set()
            if current_completion_holder[0]:
                try:
                    current_completion_holder[0].abort()
                except Exception:
                    pass

        response = ""
        completion_tokens = 0
        try:
            while True:
                event = await result_queue.get()
                if event is None:
                    break
                if event[0] == "error":
                    raise RuntimeError(f"Albatross generation error: {event[1]}")
                if event[0] != "token":
                    continue

                token_id, delta = event[1], event[2]
                completion_tokens += 1
                response += delta

                should_stop = False
                if stop:
                    stops = [stop] if isinstance(stop, str) else stop
                    for stop_text in stops:
                        if stop_text in response:
                            response = response.split(stop_text)[0]
                            should_stop = True
                            break

                yield (
                    "text",
                    response,
                    delta,
                    config["prompt_tokens"],
                    completion_tokens,
                )

                if should_stop:
                    abort_this_completion()
                    break
        finally:
            if not future.done():
                abort_this_completion()

    def get_embedding(self, input: str, fast_mode: bool) -> Tuple[List[float], int]:
        raise NotImplementedError(
            "AlbatrossRWKV does not support embeddings. "
            "Use the standard RWKV backend for embedding tasks."
        )

    def shutdown(self):
        if self._engine_core:
            try:
                self._engine_core.shutdown()
            except Exception as e:
                print(f"Error shutting down Albatross engine: {e}")

        if self._event_loop:
            try:
                self._event_loop.call_soon_threadsafe(self._event_loop.stop)
            except Exception:
                pass

        self._is_initialized = False
        self._engine_core = None
        self._event_loop = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
