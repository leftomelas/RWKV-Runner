# Albatross Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal stable Albatross high-performance RWKV-7 CUDA backend to RWKV Runner without regressing existing backends.

**Architecture:** The implementation introduces a separate Albatross model package and engine package, then exposes it through a Runner adapter. Existing FastAPI routes keep their public API; they only branch when the Albatross adapter must bypass the single-request generation lock.

**Tech Stack:** Python, FastAPI, Pydantic, PyTorch, Albatross CUDA extensions, existing RWKV Runner backend route structure, pytest-compatible unit tests.

---

## File Structure

- Create `backend-python/albatross/`: copied and lightly namespaced Albatross model/kernel code.
- Create `backend-python/albatross_engine/`: continuous batching engine and Runner adapter.
- Create `backend-python/albatross/kernels/manifest.json`: maps precompiled CUDA extensions to runtime compatibility metadata.
- Create `backend-python/scripts/build_albatross_kernel.py`: builds and records precompiled CUDA extension artifacts.
- Modify `backend-python/routes/config.py`: route Albatross strategy to the new adapter.
- Modify `backend-python/routes/completion.py`: bypass `completion_lock` only for Albatross adapter and return explicit unsupported errors where needed.
- Modify `backend-python/requirements.txt`: add optional Albatross dependency comments without making normal installs fail.
- Modify frontend config files after the backend is stable:
  - `frontend/src/types/configs.ts`
  - `frontend/src/utils/index.tsx`
  - `frontend/src/pages/Configs.tsx`
- Create backend tests:
  - `backend-python/tests/test_albatross_strategy.py`
  - `backend-python/tests/test_albatross_completion_contract.py`
  - `backend-python/tests/test_albatross_unsupported.py`
  - `backend-python/tests/test_albatross_abort.py`
- Create manual tools:
  - `backend-python/bench/albatross_api_benchmark.py`
  - `backend-python/tests/manual_albatross_smoke.py`

## Task 1: Add Strategy Parser And Backend Config

**Files:**
- Create: `backend-python/albatross_engine/__init__.py`
- Create: `backend-python/albatross_engine/config.py`
- Test: `backend-python/tests/test_albatross_strategy.py`

- [ ] **Step 1: Write the failing parser tests**

```python
from albatross_engine.config import AlbatrossBackendConfig, is_albatross_strategy, parse_albatross_strategy


def test_is_albatross_strategy_accepts_new_and_legacy_names():
    assert is_albatross_strategy("albatross")
    assert is_albatross_strategy("albatross workers=2 batch=64")
    assert is_albatross_strategy("chirrup workers=1 batch=32")
    assert not is_albatross_strategy("cuda fp16")
    assert not is_albatross_strategy("")


def test_parse_albatross_strategy_defaults():
    assert parse_albatross_strategy("albatross") == AlbatrossBackendConfig(
        worker_num=1,
        batch_size=32,
    )


def test_parse_albatross_strategy_custom_values():
    assert parse_albatross_strategy("albatross workers=2 batch=64") == AlbatrossBackendConfig(
        worker_num=2,
        batch_size=64,
    )


def test_parse_albatross_strategy_ignores_invalid_values():
    assert parse_albatross_strategy("albatross workers=nope batch=-1") == AlbatrossBackendConfig(
        worker_num=1,
        batch_size=32,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend-python
python -m pytest tests/test_albatross_strategy.py -v
```

Expected: FAIL because `albatross_engine.config` does not exist.

- [ ] **Step 3: Add minimal parser implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class AlbatrossBackendConfig:
    worker_num: int = 1
    batch_size: int = 32


def is_albatross_strategy(strategy: str | None) -> bool:
    if not strategy:
        return False
    first = strategy.strip().lower().split(maxsplit=1)[0]
    return first in {"albatross", "chirrup"}


def parse_albatross_strategy(strategy: str | None) -> AlbatrossBackendConfig:
    worker_num = 1
    batch_size = 32
    if not strategy:
        return AlbatrossBackendConfig(worker_num=worker_num, batch_size=batch_size)

    for part in strategy.lower().split()[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            parsed = int(value)
        except ValueError:
            continue
        if parsed < 1:
            continue
        if key in {"workers", "worker", "worker_num"}:
            worker_num = parsed
        elif key in {"batch", "batch_size"}:
            batch_size = parsed

    return AlbatrossBackendConfig(worker_num=worker_num, batch_size=batch_size)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd backend-python
python -m pytest tests/test_albatross_strategy.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend-python/albatross_engine/__init__.py backend-python/albatross_engine/config.py backend-python/tests/test_albatross_strategy.py
git commit -m "feat: add albatross backend strategy parser"
```

## Task 2: Move Temporary Engine Into A Namespaced Package

**Files:**
- Create/modify: `backend-python/albatross/`
- Create/modify: `backend-python/albatross_engine/core.py`
- Create/modify: `backend-python/albatross_engine/task.py`
- Create/modify: `backend-python/albatross_engine/worker.py`
- Create/modify: `backend-python/albatross_engine/sampling.py`
- Create/modify: `backend-python/albatross_engine/state_cache.py`

- [ ] **Step 1: Copy only required runtime files from `temp/backend-python`**

Copy these sources into the new package layout:

```text
temp/backend-python/Albatross/rwkv7.py -> backend-python/albatross/rwkv7.py
temp/backend-python/Albatross/utils.py -> backend-python/albatross/utils.py
temp/backend-python/Albatross/rwkv_vocab_v20230424.txt -> backend-python/albatross/rwkv_vocab_v20230424.txt
temp/backend-python/Albatross/cuda/ -> backend-python/albatross/cuda/
temp/backend-python/Albatross/hip/ -> backend-python/albatross/hip/
temp/backend-python/chirrup/core_structure.py -> backend-python/albatross_engine/task.py
temp/backend-python/chirrup/engine_core.py -> backend-python/albatross_engine/core.py
temp/backend-python/chirrup/worker.py -> backend-python/albatross_engine/worker.py
temp/backend-python/chirrup/utils/samplers.py -> backend-python/albatross_engine/sampling.py
temp/backend-python/chirrup/utils/state_cache.py -> backend-python/albatross_engine/state_cache.py
```

- [ ] **Step 2: Update imports**

Replace imports:

```python
from chirrup.core_structure import Task, ModelLoadConfig, DEFAULT_SAMPLING_CONFIG, DEFAULT_STOP_TOKENS
from chirrup.engine_core import AsyncEngineCore
from chirrup.worker import Worker, TRIE_TOKENIZER
from chirrup.utils.samplers import sample_logits_real_batch
from Albatross.rwkv7 import RWKV_x070 as RWKV_x070_ORIGINAL
from Albatross.utils import TRIE_TOKENIZER
```

with:

```python
from albatross_engine.task import Task, ModelLoadConfig, DEFAULT_SAMPLING_CONFIG, DEFAULT_STOP_TOKENS
from albatross_engine.core import AsyncEngineCore
from albatross_engine.worker import Worker, TRIE_TOKENIZER
from albatross_engine.sampling import sample_logits_real_batch
from albatross.rwkv7 import RWKV_x070 as RWKV_x070_ORIGINAL
from albatross.utils import TRIE_TOKENIZER
```

- [ ] **Step 3: Remove standalone web service from the first merge**

Do not copy:

```text
temp/backend-python/chirrup/web_service/
```

Runner's FastAPI routes remain the API owner.

- [ ] **Step 4: Run import checks**

Run:

```bash
cd backend-python
python - <<'PY'
import albatross_engine.config
import albatross_engine.task
import albatross_engine.core
print("imports ok")
PY
```

Expected: prints `imports ok`. It must not compile CUDA just by importing `albatross_engine.config`, `task`, or `core`.

- [ ] **Step 5: Commit**

```bash
git add backend-python/albatross backend-python/albatross_engine
git commit -m "feat: add albatross engine package"
```

## Task 2.5: Add CUDA Extension Precompile Loader

**Files:**
- Create: `backend-python/albatross/kernels/manifest.json`
- Create: `backend-python/albatross/kernel_loader.py`
- Create: `backend-python/scripts/build_albatross_kernel.py`
- Modify: `backend-python/albatross/rwkv7.py`
- Test: `backend-python/tests/test_albatross_kernel_loader.py`

- [ ] **Step 1: Write no-CUDA-safe loader tests**

```python
import json
import tempfile
import unittest
from pathlib import Path

from albatross.kernel_loader import RuntimeKernelContext, find_precompiled_kernel


class AlbatrossKernelLoaderTests(unittest.TestCase):
    def test_find_precompiled_kernel_matches_runtime_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "torch-2.7.1+cu128" / "win_amd64" / "cp310" / "rwkv7_state_fwd_fp16.pyd"
            library.parent.mkdir(parents=True)
            library.write_bytes(b"fake")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "kernels": [
                            {
                                "name": "rwkv7_state_fwd_fp16",
                                "torch": "2.7.1+cu128",
                                "cuda": "12.8",
                                "python_abi": "cp310",
                                "platform": "win_amd64",
                                "arch": ["sm80", "sm86", "sm89"],
                                "path": "torch-2.7.1+cu128/win_amd64/cp310/rwkv7_state_fwd_fp16.pyd",
                                "source": "local-build",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = find_precompiled_kernel(
                manifest,
                RuntimeKernelContext(
                    torch_version="2.7.1+cu128",
                    cuda_version="12.8",
                    python_abi="cp310",
                    platform_tag="win_amd64",
                    cuda_arch="sm89",
                ),
            )

            self.assertEqual(result, library)

    def test_find_precompiled_kernel_returns_none_for_arch_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "kernels": [
                            {
                                "name": "rwkv7_state_fwd_fp16",
                                "torch": "2.7.1+cu128",
                                "cuda": "12.8",
                                "python_abi": "cp310",
                                "platform": "win_amd64",
                                "arch": ["sm80"],
                                "path": "missing.pyd",
                                "source": "local-build",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = find_precompiled_kernel(
                manifest,
                RuntimeKernelContext(
                    torch_version="2.7.1+cu128",
                    cuda_version="12.8",
                    python_abi="cp310",
                    platform_tag="win_amd64",
                    cuda_arch="sm89",
                ),
            )

            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend-python
python -m unittest tests/test_albatross_kernel_loader.py -v
```

Expected: FAIL because `albatross.kernel_loader` does not exist.

- [ ] **Step 3: Implement manifest lookup**

Create `backend-python/albatross/kernel_loader.py`:

```python
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class RuntimeKernelContext:
    torch_version: str
    cuda_version: str | None
    python_abi: str
    platform_tag: str
    cuda_arch: str


def find_precompiled_kernel(manifest_path: Path, context: RuntimeKernelContext) -> Path | None:
    if not manifest_path.is_file():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest.get("kernels", []):
        if entry.get("name") != "rwkv7_state_fwd_fp16":
            continue
        if entry.get("torch") != context.torch_version:
            continue
        if entry.get("cuda") != context.cuda_version:
            continue
        if entry.get("python_abi") != context.python_abi:
            continue
        if entry.get("platform") != context.platform_tag:
            continue
        if context.cuda_arch not in entry.get("arch", []):
            continue
        candidate = manifest_path.parent / entry["path"]
        if candidate.is_file():
            return candidate
    return None
```

- [ ] **Step 4: Add runtime context helpers and loader**

Extend `kernel_loader.py` with:

```python
import platform
import sys
import torch


def current_python_abi() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def current_platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows" and machine in {"amd64", "x86_64"}:
        return "win_amd64"
    if system == "linux" and machine in {"amd64", "x86_64"}:
        return "linux_x86_64"
    return f"{system}_{machine}"


def current_cuda_arch() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def current_runtime_context() -> RuntimeKernelContext:
    return RuntimeKernelContext(
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        python_abi=current_python_abi(),
        platform_tag=current_platform_tag(),
        cuda_arch=current_cuda_arch(),
    )


def load_precompiled_kernel_if_available(manifest_path: Path) -> bool:
    if not torch.cuda.is_available():
        return False
    kernel_path = find_precompiled_kernel(manifest_path, current_runtime_context())
    if kernel_path is None:
        return False
    torch.ops.load_library(str(kernel_path))
    return True
```

- [ ] **Step 5: Wire `rwkv7.py` to prefer precompiled kernels**

In `backend-python/albatross/rwkv7.py`, before calling `torch.utils.cpp_extension.load(...)`, add:

```python
from pathlib import Path
from albatross.kernel_loader import load_precompiled_kernel_if_available

_KERNEL_MANIFEST = Path(__file__).parent / "kernels" / "manifest.json"
_PRECOMPILED_LOADED = load_precompiled_kernel_if_available(_KERNEL_MANIFEST)
```

Wrap the existing extension compile block:

```python
if not _PRECOMPILED_LOADED:
    # existing torch.utils.cpp_extension.load(...) branches
```

- [ ] **Step 6: Add empty manifest**

Create `backend-python/albatross/kernels/manifest.json`:

```json
{
  "kernels": []
}
```

- [ ] **Step 7: Add build script skeleton**

Create `backend-python/scripts/build_albatross_kernel.py` with argparse options:

```python
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="sm80,sm86,sm89,sm90")
    parser.add_argument("--output-root", default="albatross/kernels")
    args = parser.parse_args()
    print(f"Requested Albatross kernel build for arch={args.arch} output_root={args.output_root}")
    print("Build implementation should compile rwkv7_state_fwd_fp16 and update manifest.json.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run loader tests**

Run:

```bash
cd backend-python
python -m unittest tests/test_albatross_kernel_loader.py -v
```

Expected: PASS.

- [ ] **Step 9: Run import check**

Run:

```bash
cd backend-python
python - <<'PY'
from albatross.kernel_loader import RuntimeKernelContext, find_precompiled_kernel
print(RuntimeKernelContext)
print(find_precompiled_kernel)
PY
```

Expected: prints the class and function without compiling CUDA.

- [ ] **Step 10: Commit**

```bash
git add backend-python/albatross/kernel_loader.py backend-python/albatross/kernels/manifest.json backend-python/scripts/build_albatross_kernel.py backend-python/albatross/rwkv7.py backend-python/tests/test_albatross_kernel_loader.py
git commit -m "feat: add albatross kernel precompile loader"
```

## Task 3: Add Runner Adapter

**Files:**
- Create: `backend-python/albatross_engine/adapter.py`
- Test: `backend-python/tests/test_albatross_unsupported.py`

- [ ] **Step 1: Write unsupported feature tests**

```python
import pytest

from albatross_engine.adapter import AlbatrossRWKV


def test_albatross_embedding_is_explicitly_unsupported(monkeypatch):
    monkeypatch.setattr(AlbatrossRWKV, "_init_engine", lambda self: None)
    model = AlbatrossRWKV("models/example-rwkv7.pth")

    with pytest.raises(NotImplementedError, match="does not support embeddings"):
        model.get_embedding("hello", fast_mode=False)


def test_albatross_run_rnn_is_explicitly_unsupported(monkeypatch):
    monkeypatch.setattr(AlbatrossRWKV, "_init_engine", lambda self: None)
    model = AlbatrossRWKV("models/example-rwkv7.pth")

    with pytest.raises(NotImplementedError, match="batch inference"):
        model.run_rnn([1, 2, 3])
```

- [ ] **Step 2: Add adapter by porting `temp/backend-python/utils/chirrup_adapter.py`**

Use class names:

```python
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
    def run_rnn(self, _tokens, newline_adj=0):
        raise NotImplementedError("AlbatrossRWKV uses batch inference. Use generate() instead of run_rnn().")

    def get_embedding(self, input: str, fast_mode: bool):
        raise NotImplementedError("AlbatrossRWKV does not support embeddings. Use the standard RWKV backend for embedding tasks.")
```

The adapter must:

- set `self.name = "albatross"`
- not call `super().__init__()` because the engine owns the model
- start an event loop thread only in `_init_engine()`
- expose `generate(body, prompt, stop, stop_token_ids)`
- expose `shutdown()`
- raise `NotImplementedError` for `run_rnn()` and `get_embedding()`

- [ ] **Step 3: Run unsupported feature tests**

Run:

```bash
cd backend-python
python -m pytest tests/test_albatross_unsupported.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add backend-python/albatross_engine/adapter.py backend-python/tests/test_albatross_unsupported.py
git commit -m "feat: add albatross runner adapter"
```

## Task 4: Wire `/switch-model`

**Files:**
- Modify: `backend-python/routes/config.py`
- Test: `backend-python/tests/test_albatross_strategy.py`

- [ ] **Step 1: Add route selection tests**

Add a test that monkeypatches `routes.config.AlbatrossRWKV` and verifies `.pth + albatross` selects it, while `.gguf + albatross` still selects llama.cpp or fails clearly according to current Runner behavior.

- [ ] **Step 2: Update `config.py` imports**

Add:

```python
from albatross_engine.adapter import AlbatrossRWKV
from albatross_engine.config import is_albatross_strategy, parse_albatross_strategy
```

- [ ] **Step 3: Update model selection**

Use this ordering:

```python
if body.model.endswith(".gguf"):
    global_var.set(global_var.Model, Llama(model_path=body.model, strategy=body.strategy))
elif is_albatross_strategy(body.strategy):
    backend_config = parse_albatross_strategy(body.strategy)
    global_var.set(
        global_var.Model,
        AlbatrossRWKV(
            model_path=body.model,
            worker_num=backend_config.worker_num,
            batch_size=backend_config.batch_size,
            tokenizer=body.tokenizer,
        ),
    )
else:
    global_var.set(
        global_var.Model,
        RWKV(model=body.model, strategy=body.strategy, tokenizer=body.tokenizer),
    )
```

- [ ] **Step 4: Run route selection tests**

Run:

```bash
cd backend-python
python -m pytest tests/test_albatross_strategy.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend-python/routes/config.py backend-python/tests/test_albatross_strategy.py
git commit -m "feat: wire albatross backend selection"
```

## Task 5: Wire Completion Routes

**Files:**
- Modify: `backend-python/routes/completion.py`
- Test: `backend-python/tests/test_albatross_completion_contract.py`
- Test: `backend-python/tests/test_albatross_abort.py`

- [ ] **Step 1: Add fake adapter tests for non-streaming response**

Create a fake model whose `generate()` yields:

```python
("text", "Hello", "Hello", 3, 1)
("text", "Hello world", " world", 3, 2)
```

Assert the non-streaming response object is `chat.completion`, has assistant content `Hello world`, and usage totals are correct.

- [ ] **Step 2: Add fake adapter tests for streaming response**

Assert streaming emits JSON chunks with `choices[0].delta.content` and then emits `[DONE]`.

- [ ] **Step 3: Add abort test**

Use a fake request where `is_disconnected()` returns `True` after the first token. Assert the fake completion's `abort()` method is called exactly once.

- [ ] **Step 4: Update completion route imports**

Add:

```python
from albatross_engine.adapter import AlbatrossRWKV
```

- [ ] **Step 5: Rename and port the temp `eval_chirrup()` logic**

Use:

```python
async def eval_albatross(model, request, body, prompt, stream, stop, stop_token_ids, chat_mode):
    completion = model.generate(body, prompt, stop=stop, stop_token_ids=stop_token_ids)
    response_type, response, prompt_tokens, completion_tokens = "text", "", 0, 0

    try:
        for response_type, response, delta, prompt_tokens, completion_tokens in completion:
            if await request.is_disconnected():
                completion.abort()
                break
            if stream:
                yield json.dumps({
                    "object": "chat.completion.chunk" if chat_mode else "text_completion",
                    "model": model.name,
                    "choices": [{
                        "delta": {"content": delta} if chat_mode else None,
                        "text": delta if not chat_mode else None,
                        "index": 0,
                        "finish_reason": None,
                    }],
                })
    finally:
        if await request.is_disconnected():
            completion.abort()
```

Keep the behavior:

- do not use `completion_lock`
- call per-request `completion.abort()` on disconnect
- compute prompt/completion usage
- emit current Runner-compatible stream chunks
- emit current Runner-compatible non-stream response

- [ ] **Step 6: Dispatch Albatross before the existing lock path**

At the top of `eval()`:

```python
if isinstance(model, AlbatrossRWKV):
    async for result in eval_albatross(model, request, body, prompt, stream, stop, stop_token_ids, chat_mode):
        yield result
    return
```

- [ ] **Step 7: Run completion tests**

Run:

```bash
cd backend-python
python -m pytest tests/test_albatross_completion_contract.py tests/test_albatross_abort.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add backend-python/routes/completion.py backend-python/tests/test_albatross_completion_contract.py backend-python/tests/test_albatross_abort.py
git commit -m "feat: support albatross completion routes"
```

## Task 6: Frontend High-Performance Mode

**Files:**
- Modify: `frontend/src/types/configs.ts`
- Modify: `frontend/src/utils/index.tsx`
- Modify: `frontend/src/pages/Configs.tsx`

- [ ] **Step 1: Add device type**

Extend `Device`:

```ts
export type Device =
  | 'CPU'
  | 'CPU (rwkv.cpp)'
  | 'CUDA'
  | 'CUDA-Beta'
  | 'CUDA High Performance'
  | 'WebGPU'
  | 'WebGPU (Python)'
  | 'MPS'
  | 'Custom'
```

- [ ] **Step 2: Generate Albatross strategy**

In `getStrategy()`, add:

```ts
case 'CUDA High Performance':
  strategy = 'albatross workers=1 batch=32'
  break
```

- [ ] **Step 3: Add conservative UI copy**

Display the option as RWKV-7 `.pth` CUDA-only. Do not make it the default config in this task.

- [ ] **Step 4: Run frontend typecheck/build**

Run:

```bash
cd frontend
npm run build
```

Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/types/configs.ts frontend/src/utils/index.tsx frontend/src/pages/Configs.tsx
git commit -m "feat: add albatross frontend mode"
```

## Task 7: Manual CUDA Smoke And Benchmark

**Files:**
- Create: `backend-python/tests/manual_albatross_smoke.py`
- Create: `backend-python/bench/albatross_api_benchmark.py`

- [ ] **Step 1: Add manual smoke script**

The script should:

- POST `/switch-model` with `strategy: "albatross workers=1 batch=16"`
- POST one non-streaming `/v1/chat/completions`
- POST one streaming `/v1/chat/completions`
- print response status codes and first generated text

- [ ] **Step 2: Add benchmark script**

Measure:

- first token latency
- total generated tokens
- total wall time
- tokens per second
- concurrent request count

- [ ] **Step 3: Run no-GPU-safe tests**

Run:

```bash
cd backend-python
python -m pytest tests/test_albatross_strategy.py tests/test_albatross_completion_contract.py tests/test_albatross_unsupported.py tests/test_albatross_abort.py -v
```

Expected: PASS without requiring CUDA.

- [ ] **Step 4: Run manual CUDA smoke on a CUDA machine**

Run:

```bash
cd backend-python
python main.py --port 8000
python tests/manual_albatross_smoke.py --model models/YOUR_RWKV7_MODEL.pth --port 8000
```

Expected: model loads, both completion modes return text, server logs show Albatross backend.

- [ ] **Step 5: Commit**

```bash
git add backend-python/tests/manual_albatross_smoke.py backend-python/bench/albatross_api_benchmark.py
git commit -m "test: add albatross smoke and benchmark tools"
```

## Self-Review

- Spec coverage: Tasks cover backend package layout, adapter, strategy parsing, route wiring, unsupported API behavior, frontend entry, and validation tools.
- Deferred scope is explicit: batch translation, SQLite state pool, CUDAGraph, int8, ROCm, and multi-GPU are not included in this first plan.
- Type consistency: the plan consistently uses `AlbatrossRWKV`, `AlbatrossCompletion`, `AlbatrossBackendConfig`, `is_albatross_strategy`, and `parse_albatross_strategy`.
- No placeholders: every task has concrete file paths, commands, and expected outcomes. Manual CUDA model path is intentionally user-provided because the repository does not contain a valid RWKV-7 model file.
