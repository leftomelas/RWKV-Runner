# Albatross Python Performance Design

## Context

RWKV-Runner now has a Python Albatross backend with precompiled CUDA kernels and a local C++ `rwkv_lightning` comparison target. Recent measurements on the same RWKV7 1.5B model show a large throughput gap:

- Python Albatross internal, `batch=960`, `max_tokens=300`, no stop: about 2875 tok/s.
- Python HTTP non-streaming long output: about 2618 tok/s.
- C++ HTTP `/v1/chat/completions`, `contents=960`, `max_tokens=300`, no stop: about 6084 tok/s.
- C++ native `benchmark.exe`, `batch=960`, `steps=300`: about 8168 tok/s full.

The Python backend also shows lower observed GPU utilization, around 50%, while the C++ backend can reach 80% or higher. Local code inspection shows the Python backend does not currently use CUDA Graph capture/replay. The upstream Albatross benchmark does use `torch.cuda.CUDAGraph()` for fixed-shape decode paths, and its README calls out CUDAGraph improvements. Local inspection also shows the Python sampler is implemented as PyTorch composite operations with full-vocabulary sort and multinomial, while the C++ backend has fused CUDA sampling kernels.

## Goals

The goal is to close the Python backend performance gap while preserving the current service behavior and retaining safe fallbacks.

Primary targets:

- Identify the exact decode-time bottlenecks in Python instead of relying on GPU utilization alone.
- Replace the current Python composite sampler with a fused CUDA sampler.
- Add a CUDA Graph decode fast path for fixed-shape, high-throughput batches.
- Keep existing continuous batching behavior working when the fast path is not applicable.
- Preserve the ability to switch features on and off through environment variables for benchmarking and rollback.

Performance targets:

- Short term: improve Python internal `batch=960,max_tokens=300` throughput from about 2875 tok/s to at least 4000 tok/s.
- Medium term: approach C++ HTTP batch throughput, about 6000 tok/s, for the same high-throughput non-streaming workload.
- Long term: determine whether Python can approach C++ native throughput after deeper state layout and kernel alignment.

## Non-Goals

This work will not initially replace the entire Albatross worker scheduler. It will not remove the current Python sampler until the CUDA sampler is validated. It will not make CUDA Graph mandatory for all workloads. It will not attempt to optimize unrelated RWKV backends or Go frontend code.

Streaming output is not the first optimization target. The long-output non-streaming path already isolates engine throughput better, and the current C++ streaming path in the tested binary returned an empty body, so it is not a reliable comparison baseline.

## Architecture

The work is split into four stages.

### Stage 1: Profiling and Baseline Lockdown

Add a dedicated profiling path for the Python Albatross worker loop. This stage records per-token and per-phase timings without changing model behavior.

The profile should break down:

- task intake and slot management
- one-token decode forward
- sequence prefill forward
- penalty update
- sampling
- output queue delivery
- HTTP response overhead where applicable

The benchmark harness should always print:

- model path
- selected kernel architecture
- batch slots
- request concurrency
- max tokens
- stop-token mode
- total generated tokens
- wall time
- aggregate tokens/s
- optional phase timing table

This creates a repeatable baseline for every later change.

### Stage 2: Fused CUDA Sampling

Replace the current Python sampler with a fused CUDA sampler adapted from `rwkv_lightning_libtorch` or upstream `faster2_251201`.

The current Python sampler performs:

- temperature scaling
- softmax
- full-vocabulary `torch.sort`
- top-k filtering
- another full-vocabulary `torch.sort`
- top-p filtering
- `torch.multinomial`

This is expensive at `batch=960,vocab=65536` and launches multiple PyTorch kernels per token. The C++ backend instead has CUDA sampling kernels that combine penalty, top-k/top-p, and token selection. The Python backend should expose the fused sampler through the existing PyTorch extension and call it from `albatross_engine.sampling`.

Feature flag:

- `ALBATROSS_SAMPLER=python` keeps the current sampler.
- `ALBATROSS_SAMPLER=cuda` enables the fused CUDA sampler.
- Default should remain `python` until validation passes, then can change to `cuda`.

Fallback behavior:

- If the fused kernel is unavailable, log a warning and use the Python sampler.
- If unsupported parameters are requested, use the Python sampler for that request.

### Stage 3: CUDA Graph Decode Fast Path

Add CUDA Graph capture/replay for steady-state decode where tensor shapes are fixed.

The graph path should be narrowly scoped:

- fixed active batch size
- decode-only one-token step
- CUDA sampler path enabled
- no dynamic Python control inside the captured region
- no `.item()` or CPU synchronization inside capture
- no shape-changing active batch shrink inside capture

The graph cache should be keyed by at least:

- batch size
- model identity
- sampler mode
- dtype
- device
- kernel architecture

Feature flag:

- `ALBATROSS_CUDA_GRAPH=0` disables graph.
- `ALBATROSS_CUDA_GRAPH=1` enables graph when supported.

Fallback behavior:

- If a request pattern does not match the graph constraints, use the existing decode path.
- If graph capture fails, disable the graph path for that worker and continue with the normal path.

The first graph target should be the high-throughput `batch=960,max_tokens=300` benchmark. Smaller graph cache entries can be added later for common batch sizes such as 128, 256, and 512.

### Stage 4: Faster2 and State Layout Alignment

If stages 2 and 3 do not close most of the gap, compare the local Python `rwkv7.py` and CUDA kernels against upstream `faster2_251201` and the C++ implementation.

This stage evaluates:

- updated RWKV7 CUDA kernel variants
- coalesced state read/write layout
- fused sampling API differences
- whether Python state layout prevents optimal memory access
- whether a dedicated fixed-batch engine should exist alongside continuous batching

The likely result is a separate high-throughput batch engine rather than forcing every continuous-batching code path into the same shape.

## Components

### `backend-python/bench/albatross_internal_benchmark.py`

Keep this as the stable internal benchmark. Extend it only if needed to report sampler and graph mode.

### `backend-python/bench/albatross_decode_profile.py`

Create a profiling benchmark that runs the same workload as the internal benchmark but enables worker phase timing and optional PyTorch profiler output.

### `backend-python/albatross_engine/profiling.py`

Create a small profiling helper for counters, timers, and summary formatting. It should be optional and low overhead when disabled.

### `backend-python/albatross_engine/worker.py`

Add phase timing around decode and prefill operations. Later, route sampling through the sampler abstraction and add CUDA Graph fast-path checks.

### `backend-python/albatross_engine/sampling.py`

Keep the Python sampler as a fallback. Add a dispatch function that chooses Python or CUDA sampler based on environment, availability, and request parameters.

### `backend-python/albatross/cuda/rwkv7_state_fwd_fp16.cu`

Add fused sampling kernels or integrate the upstream/C++ kernels into the existing extension source.

### `backend-python/albatross/cuda/rwkv7_state_fwd_fp16.cpp`

Register the new fused sampling operations with PyTorch.

### `backend-python/albatross/kernel_loader.py`

Extend the kernel manifest and precompiled artifact validation so sampler-capable kernels can be detected explicitly.

### `backend-python/albatross_engine/cuda_graph.py`

Create only when implementing stage 3. It should own graph capture state, graph replay, and fallback decisions.

## Data Flow

Normal path:

1. API or benchmark creates completion tasks.
2. `AsyncEngineCore` places tasks in the worker queue.
3. `Worker` groups active slots by state category.
4. Decode tasks call `model.forward_seq_batch`.
5. Sampling selects next tokens.
6. Penalty state and task state are updated.
7. Generated token events are pushed to output queues.

CUDA sampler path:

1. Steps 1 to 4 remain unchanged.
2. `sampling.py` dispatches to fused CUDA sampling.
3. The CUDA op returns a GPU tensor of token ids.
4. Worker consumes token ids and updates task state.

CUDA Graph path:

1. Worker reaches a fixed-shape decode batch.
2. Static graph input tensors are updated.
3. Graph replay runs model decode and fused sampler.
4. Worker reads the resulting token tensor and updates task state outside the graph.
5. Unsupported cases fall back to the normal path.

## Error Handling

Feature flags must make every optimization reversible.

If fused sampler initialization fails, the backend logs the failure once and uses the Python sampler. If a CUDA Graph capture or replay fails, the worker disables graph mode for the current process and continues with the non-graph path. Benchmark scripts should print active sampler and graph modes so failed fallbacks are visible in results.

Unsupported sampler configurations should not crash the service. The dispatch layer should choose the Python sampler when parameters are not supported by the fused kernel.

## Testing

Testing should focus on correctness first, then performance.

Correctness tests:

- Python sampler fallback still returns one token per active batch row.
- CUDA sampler returns a tensor with the expected shape, dtype, and device.
- CUDA sampler handles `top_k=0`, `top_p=0.3`, `temperature=1.0`.
- Stop token handling remains in worker logic and still finishes requests.
- Feature flags select the expected sampler mode.
- CUDA Graph disabled mode behaves exactly like the existing path.

Benchmark tests:

- Internal benchmark with Python sampler.
- Internal benchmark with CUDA sampler.
- Internal benchmark with CUDA sampler plus CUDA Graph.
- HTTP non-streaming benchmark after each optimization stage.

Manual validation:

- Run `batch=960,concurrency=960,max_tokens=300,stop_tokens=[]`.
- Compare against previous Python and C++ results.
- Observe GPU utilization with `nvidia-smi`.
- Confirm no CUDA memory leak across repeated benchmark runs.

## Rollout

Roll out in this order:

1. Merge profiling with no behavior change.
2. Add fused CUDA sampler behind `ALBATROSS_SAMPLER=cuda`.
3. Precompile sampler-capable sm80 and sm120 kernels.
4. Change default sampler only after validation.
5. Add CUDA Graph behind `ALBATROSS_CUDA_GRAPH=1`.
6. Keep graph disabled by default until memory usage and fallback behavior are stable.
7. Evaluate faster2 state layout only after stages 2 and 3 are measured.

## Risks

CUDA Graph requires fixed shapes and stable memory addresses. The continuous batching worker is dynamic, so graph use must be conservative.

Fused sampling can change sampling distribution if top-p, top-k, penalty, or random-state handling differs from the Python implementation. Exact token-by-token equivalence is not required for stochastic sampling, but output distribution and API semantics must remain reasonable.

CUDA Graph can increase VRAM usage. Upstream Albatross explicitly notes this, so graph must be optional and benchmarked under realistic batch sizes.

Kernel registration changes can break precompiled artifact compatibility. The manifest must distinguish kernels that include fused sampling from older forward-only kernels.

## Open Decisions

The implementation plan should decide whether to first port the C++ fused sampling kernel directly or to import the latest upstream `faster2_251201` kernel wholesale. The lower-risk first step is direct sampling-kernel porting while leaving existing RWKV forward kernels unchanged.

The implementation plan should also decide the first CUDA Graph target. The recommended target is `batch=960` decode-only, because it matches the current performance investigation and has a stable comparison baseline.
