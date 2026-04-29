# Albatross Backend Design

## Goal

Integrate the high-performance RWKV-7 inference path based on Albatross into RWKV Runner as a first-class optional backend, while preserving the current `rwkv_pip`, `rwkv.cpp`, WebGPU, GGUF, MIDI, and embedding behavior.

The first deliverable is a minimal stable backend for RWKV-7 `.pth` models on CUDA fp16. It must support the existing OpenAI-compatible chat and completion routes in streaming and non-streaming mode. Batch translation, SQLite state pools, CUDAGraph, int8, ROCm, and multi-GPU support are deferred until the backend boundary is stable.

## Current Context

RWKV Runner currently uses a Wails/Go desktop shell and a FastAPI Python backend. Model loading is centered on `backend-python/routes/config.py`, generation on `backend-python/routes/completion.py`, and RWKV model behavior on `backend-python/utils/rwkv.py`.

The partial backend in `temp/backend-python` already includes:

- `utils/chirrup_adapter.py`: adapts an async continuous batching engine to Runner's `AbstractRWKV` style.
- `chirrup/`: worker, task, state cache, and engine core logic.
- `Albatross/`: copied RWKV-7 fp16 kernels and model wrapper.
- route changes that allow `strategy = "chirrup workers=1 batch=32"`.

Albatross provides the model and kernels. `rwkv_lightning` provides useful reference designs for OpenAI-compatible routes, state cache, continuous batching, batch APIs, and benchmark scripts.

## Scope

### In Scope For The First Implementation

- Add a clearly named Albatross backend package under `backend-python`.
- Add a Runner adapter for the backend.
- Add backend strategy parsing for a high-performance RWKV-7 mode.
- Preserve existing `/v1/chat/completions`, `/chat/completions`, `/v1/completions`, and `/completions` contracts.
- Make unsupported APIs fail explicitly and readably.
- Add unit tests for strategy parsing, route contract behavior, unsupported feature handling, abort behavior, and no-GPU error paths.
- Add a manual CUDA smoke test and benchmark script.

### Out Of Scope For The First Implementation

- Replacing the existing `rwkv_pip` backend.
- Batch translation APIs.
- Stateful session APIs.
- SQLite state persistence.
- CUDAGraph decode path.
- INT8 quantized Albatross weights.
- ROCm support.
- Multi-GPU routing.
- Embeddings on Albatross.
- MIDI model support on Albatross.

## Architecture

The backend will be integrated as a separate engine behind the existing FastAPI API surface.

```text
frontend config
  -> getStrategy()
  -> /switch-model
  -> BackendFactory / strategy parser
  -> AlbatrossRWKV adapter
  -> AsyncEngineCore
  -> Worker
  -> Albatross RWKV-7 model + kernels
```

The adapter is the only object that route handlers should need to know about. It exposes Runner-compatible generation behavior and hides the async worker/event-loop implementation. Route code should only branch on the adapter type where the existing single-request `completion_lock` must be bypassed.

## Backend Boundaries

### Albatross Model Package

The copied Albatross code should live in a clearly isolated package, for example:

```text
backend-python/albatross/
  __init__.py
  rwkv7.py
  utils.py
  rwkv_vocab_v20230424.txt
  cuda/
  hip/
```

The first implementation may keep HIP files for source parity, but the runtime should report that ROCm is unsupported by Runner's Albatross mode until tested.

### Engine Package

The continuous batching engine should live separately from model kernels:

```text
backend-python/albatross_engine/
  __init__.py
  adapter.py
  config.py
  core.py
  task.py
  worker.py
  sampling.py
  state_cache.py
```

The temporary `chirrup/web_service` should not be merged into the first implementation. Runner already owns the API surface.

### CUDA Extension Loading

Runner should support both precompiled and source-built Albatross CUDA extensions.

The preferred runtime path is:

1. Resolve the current Python, platform, Torch, CUDA, and GPU architecture context.
2. Look up a matching precompiled extension in an Albatross kernel manifest.
3. Load the extension with `torch.ops.load_library(...)` when a match exists.
4. Fall back to `torch.utils.cpp_extension.load(...)` when no compatible prebuilt artifact is available.

The precompiled artifact layout should follow the existing Runner kernel style where practical:

```text
backend-python/albatross/kernels/
  manifest.json
  torch-2.7.1+cu128/
    win_amd64/
      cp310/
        rwkv7_state_fwd_fp16.pyd
```

The manifest must record at least:

- torch version
- CUDA version
- Python ABI
- platform tag
- supported CUDA arch list
- library path
- source hash or build source description

The first implementation should target the local release environment first: Windows, embedded Python 3.10, and the Torch/CUDA version used by RWKV Runner. Broader Torch/CUDA/ROCm coverage can be added after the backend is stable.

### Strategy And Frontend Configuration

Internal strategy format may start as:

```text
albatross workers=1 batch=32
```

For backward compatibility with local experiments, the parser can also accept:

```text
chirrup workers=1 batch=32
```

The frontend should expose this as a device or mode, not as raw strategy text. Suggested label: `CUDA High Performance`.

## API Behavior

When an Albatross backend is loaded:

- Chat/completion streaming and non-streaming routes use the existing OpenAI-compatible response shapes.
- Generation requests do not acquire `completion_lock`; the Albatross engine owns concurrency.
- Each request owns an abort handle. If the client disconnects, only that request is aborted.
- `/v1/embeddings` returns HTTP 400 with a clear message that this backend does not support embeddings.
- State-tuned model loading returns HTTP 400 with a clear message until Albatross state format support is designed.
- GGUF models continue to use llama.cpp and must not be routed to Albatross.

## Error Handling

The backend should reject unsupported environments early:

- no CUDA device available
- model path is not `.pth`
- model is not RWKV-7 compatible
- tokenizer path is missing
- Albatross extension compile/load failure
- precompiled extension exists but does not match the current Torch/CUDA/Python runtime

Errors must include the backend name and a short remediation hint. Example:

```text
Albatross backend requires a CUDA-capable GPU and an RWKV-7 .pth model.
```

## Testing

Most integration tests must run without a real GPU by using a fake Albatross adapter or fake engine core.

Required tests:

- strategy parser accepts `albatross`, `albatross workers=2 batch=64`, and legacy `chirrup`.
- `/switch-model` selects Albatross only for `.pth` models and the Albatross strategy.
- chat/completion non-streaming response shape matches current Runner behavior.
- chat/completion streaming emits data chunks and `[DONE]`.
- embeddings returns HTTP 400 when Albatross is active.
- request abort calls only the current completion abort handle.
- missing CUDA produces a clear load error.

Manual GPU smoke tests should verify:

- one RWKV-7 `.pth` model loads successfully.
- one streaming chat request works.
- two concurrent streaming chat requests progress without `completion_lock` serialization.
- memory is released after `/switch-model` unload or process exit.
- precompiled extension loading works when a matching artifact exists.
- source compilation fallback works when no matching artifact exists.

## Release Strategy

The first release should be hidden behind explicit backend selection. It should not change default configs for existing users. After manual CUDA validation and user feedback, add a default preset for high-performance RWKV-7 CUDA.

## Follow-Up Phases

1. Prefix state cache: start with in-memory LRU, then evaluate SQLite persistence.
2. Operational metrics: expose active slots, decode count, prefill count, loop time, queue size, and VRAM peak.
3. Batch APIs: add rollout and translation endpoints only after normal chat is stable.
4. Performance paths: CUDAGraph bsz=1, int8 weights, ROCm, and multi-GPU.
