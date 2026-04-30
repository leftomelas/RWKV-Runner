# Albatross Dual Kernel and HTTP Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Albatross keep both native `sm120` and compatible `sm80,compute80` precompiled kernels, then measure and reduce HTTP streaming overhead under high concurrency.

**Architecture:** Kernel artifacts are selected from manifest entries by runtime context, exact architecture preference, and optional environment override. HTTP optimization is evidence-led: add route/benchmark diagnostics first, then reduce per-token work in the Albatross streaming path while preserving OpenAI-compatible responses.

**Tech Stack:** Python 3.10, PyTorch CUDA extension loader, FastAPI, `sse-starlette`, asyncio, unittest, local PowerShell benchmark scripts.

---

## Current Real HTTP Benchmark Baseline

Measured on 2026-05-01 with `ALBATROSS_KERNEL_ARCH=sm80_compute80`, 1 Albatross worker, batch/concurrency/requests 960, `max_tokens=300`, `temperature=1.0`, `top_p=0.3`, `top_k=0`, and the long GPU batch inference prompt.

| Path | Sampler | Result |
| --- | --- | ---: |
| Real HTTP non-stream | Python | 2201.94 tok/s |
| Real HTTP non-stream | CUDA | 2784.08 tok/s |
| Real HTTP stream | Python | 1813.44 tok/s |
| Real HTTP stream | CUDA | 2324.10 tok/s |

Observed uplift:

- Non-stream CUDA vs Python: +26.44%.
- Stream CUDA vs Python: +28.16%.
- Non-stream CUDA vs previous 2344 tok/s ad-hoc result: +18.77%.

The stream benchmark can hit transient TCP connection refused errors when 960 long-lived connections are opened at once from the same Python process. This happens before requests enter model generation; short `max_tokens=20` probes reproduce it. Treat connection refused counts separately from generation throughput, and use `--connect-retries` when measuring steady-state real HTTP generation throughput.

After adding shared event-loop result dispatch, measured CUDA sampler results improved to:

| Path | Result | Change vs previous CUDA |
| --- | ---: | ---: |
| Real HTTP non-stream | 3897.33 tok/s | +39.99% |
| Real HTTP stream | 3298.82 tok/s | +41.94% |

The internal profile showed why: `decode_output_enqueue_total_ms` dropped from roughly 4010 ms to 283 ms on a 960 concurrency, 60 token CUDA run. The optimization batches cross-thread `asyncio.Queue` puts across all completion result channels sharing the same event loop.

After caching CUDA sampler RNG states, the same internal CUDA profile showed `sampling_total_ms` drop from roughly 5683 ms to 5296 ms. Total wall time stayed within run-to-run noise, so this is a small sampler-path cleanup rather than the next major HTTP throughput win.

Canonical real HTTP commands:

```powershell
cd C:\Users\josSt\_S\RWKV-Runner

py310\python.exe backend-python\bench\albatross_real_http_benchmark.py `
  --sampler python `
  --sampler-fallback `
  --kernel-arch sm80_compute80 `
  --non-stream `
  --concurrency 960 `
  --requests 960 `
  --max-tokens 300 `
  --timeout 900

py310\python.exe backend-python\bench\albatross_real_http_benchmark.py `
  --sampler cuda `
  --kernel-arch sm80_compute80 `
  --non-stream `
  --concurrency 960 `
  --requests 960 `
  --max-tokens 300 `
  --timeout 900

py310\python.exe backend-python\bench\albatross_real_http_benchmark.py `
  --sampler python `
  --sampler-fallback `
  --kernel-arch sm80_compute80 `
  --connect-retries 8 `
  --concurrency 960 `
  --requests 960 `
  --max-tokens 300 `
  --timeout 900

py310\python.exe backend-python\bench\albatross_real_http_benchmark.py `
  --sampler cuda `
  --kernel-arch sm80_compute80 `
  --connect-retries 8 `
  --concurrency 960 `
  --requests 960 `
  --max-tokens 300 `
  --timeout 900
```

### Task 1: Dual Kernel Artifact Selection

**Files:**
- Modify: `backend-python/albatross/kernel_loader.py`
- Modify: `backend-python/tests/test_albatross_kernel_loader.py`

- [ ] **Step 1: Write failing tests for exact-arch priority and forced arch selection**

Add tests that create two manifest entries with the same runtime context:

```python
def test_find_precompiled_kernel_prefers_exact_arch_over_ptx_fallback(self):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fallback = root / "torch-2.7.1+cu128/win_amd64/cp310/sm80_compute80/rwkv7_state_fwd_fp16.pyd"
        exact = root / "torch-2.7.1+cu128/win_amd64/cp310/sm120/rwkv7_state_fwd_fp16.pyd"
        fallback.parent.mkdir(parents=True)
        exact.parent.mkdir(parents=True)
        fallback.write_bytes(b"fallback")
        exact.write_bytes(b"exact")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"kernels": [
            {"name": "rwkv7_state_fwd_fp16", "torch": "2.7.1+cu128", "cuda": "12.8", "python_abi": "cp310", "platform": "win_amd64", "arch": ["sm80", "compute80"], "path": fallback.relative_to(root).as_posix(), "source": "local-build"},
            {"name": "rwkv7_state_fwd_fp16", "torch": "2.7.1+cu128", "cuda": "12.8", "python_abi": "cp310", "platform": "win_amd64", "arch": ["sm120"], "path": exact.relative_to(root).as_posix(), "source": "local-build"},
        ]}), encoding="utf-8")
        result = find_precompiled_kernel(manifest, RuntimeKernelContext("2.7.1+cu128", "12.8", "cp310", "win_amd64", "sm120"))
        self.assertEqual(result, exact)
```

Add a second test for environment override:

```python
def test_find_precompiled_kernel_honors_forced_kernel_arch(self):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fallback = root / "torch-2.7.1+cu128/win_amd64/cp310/sm80_compute80/rwkv7_state_fwd_fp16.pyd"
        exact = root / "torch-2.7.1+cu128/win_amd64/cp310/sm120/rwkv7_state_fwd_fp16.pyd"
        fallback.parent.mkdir(parents=True)
        exact.parent.mkdir(parents=True)
        fallback.write_bytes(b"fallback")
        exact.write_bytes(b"exact")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"kernels": [
            {"name": "rwkv7_state_fwd_fp16", "torch": "2.7.1+cu128", "cuda": "12.8", "python_abi": "cp310", "platform": "win_amd64", "arch": ["sm120"], "path": exact.relative_to(root).as_posix(), "source": "local-build"},
            {"name": "rwkv7_state_fwd_fp16", "torch": "2.7.1+cu128", "cuda": "12.8", "python_abi": "cp310", "platform": "win_amd64", "arch": ["sm80", "compute80"], "path": fallback.relative_to(root).as_posix(), "source": "local-build"},
        ]}), encoding="utf-8")
        result = find_precompiled_kernel(
            manifest,
            RuntimeKernelContext("2.7.1+cu128", "12.8", "cp310", "win_amd64", "sm120"),
            forced_arch="sm80_compute80",
        )
        self.assertEqual(result, fallback)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\py310\python.exe -m unittest backend-python.tests.test_albatross_kernel_loader -v
```

Expected: tests fail because `find_precompiled_kernel()` has no `forced_arch` parameter and does not rank exact matches.

- [ ] **Step 3: Implement ranked manifest selection**

Update `find_precompiled_kernel()` to:

```python
def _arch_key(entry_arches: list[str]) -> str:
    return "_".join(entry_arches)

def _arch_score(entry_arches: list[str], runtime_arch: str) -> int:
    if runtime_arch in entry_arches:
        return 0
    if _supports_runtime_arch(entry_arches, runtime_arch):
        return 1
    return 2
```

Then collect compatible candidates, filter by `forced_arch` when provided, and return the lowest score. `load_precompiled_kernel_if_available()` should pass `os.environ.get("ALBATROSS_KERNEL_ARCH")`.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
.\py310\python.exe -m unittest backend-python.tests.test_albatross_kernel_loader -v
```

Expected: all loader tests pass.

### Task 2: Build Script Produces Coexisting Artifact Paths

**Files:**
- Modify: `backend-python/scripts/build_albatross_kernel.py`
- Modify: `backend-python/tests/test_albatross_build_script.py`
- Modify: `backend-python/albatross/kernels/manifest.json`
- Add: `backend-python/albatross/kernels/torch-2.7.1+cu128/win_amd64/cp310/sm120/rwkv7_state_fwd_fp16.pyd`
- Add later after local build: `backend-python/albatross/kernels/torch-2.7.1+cu128/win_amd64/cp310/sm80_compute80/rwkv7_state_fwd_fp16.pyd`
- Delete after migration: `backend-python/albatross/kernels/torch-2.7.1+cu128/win_amd64/cp310/rwkv7_state_fwd_fp16.pyd`

- [ ] **Step 1: Write failing tests for artifact arch directory and manifest replacement scope**

Add tests:

```python
def test_arch_directory_name_joins_arches(self):
    module = load_build_script()
    self.assertEqual(module.arch_directory_name(["sm80", "compute80"]), "sm80_compute80")
    self.assertEqual(module.arch_directory_name(["sm120"]), "sm120")

def test_update_manifest_replaces_matching_arch_context_only(self):
    module = load_build_script()
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "manifest.json"
        sm80 = {"name": "rwkv7_state_fwd_fp16", "torch": "2.7.1+cu128", "cuda": "12.8", "python_abi": "cp310", "platform": "win_amd64", "arch": ["sm80", "compute80"], "path": "sm80/kernel.pyd", "source": "local-build"}
        sm120 = dict(sm80)
        sm120["arch"] = ["sm120"]
        sm120["path"] = "old-sm120/kernel.pyd"
        replacement = dict(sm120)
        replacement["path"] = "sm120/kernel.pyd"
        manifest_path.write_text(json.dumps({"kernels": [sm80, sm120]}), encoding="utf-8")
        module.update_manifest(manifest_path, replacement)
        kernels = json.loads(manifest_path.read_text(encoding="utf-8"))["kernels"]
        self.assertIn(sm80, kernels)
        self.assertIn(replacement, kernels)
        self.assertEqual(len(kernels), 2)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\py310\python.exe -m unittest backend-python.tests.test_albatross_build_script -v
```

Expected: tests fail because `arch_directory_name()` does not exist and manifest replacement currently ignores `arch`.

- [ ] **Step 3: Implement arch-specific output paths**

Add:

```python
def arch_directory_name(arches: list[str]) -> str:
    return "_".join(arches)
```

Change `relative_library` to include the arch directory:

```python
relative_library = (
    Path(f"torch-{torch.__version__}")
    / current_platform_tag()
    / current_python_abi()
    / arch_directory_name(arches)
    / f"{KERNEL_NAME}{library_suffix()}"
)
```

Change `update_manifest.same_context()` keys to include `"arch"`.

- [ ] **Step 4: Move committed sm120 artifact into the new path**

Use PowerShell move/copy commands only after verifying paths:

```powershell
New-Item -ItemType Directory -Force backend-python\albatross\kernels\torch-2.7.1+cu128\win_amd64\cp310\sm120
Move-Item backend-python\albatross\kernels\torch-2.7.1+cu128\win_amd64\cp310\rwkv7_state_fwd_fp16.pyd backend-python\albatross\kernels\torch-2.7.1+cu128\win_amd64\cp310\sm120\rwkv7_state_fwd_fp16.pyd
```

Update `manifest.json` to point at the `sm120/` path.

- [ ] **Step 5: Build and register sm80_compute80**

Run:

```powershell
$env:ALBATROSS_ARCH='sm80,compute80'
cmd /c backend-python\scripts\build_albatross_kernel_vs2022.cmd
```

Expected: build succeeds and creates the `sm80_compute80/` artifact without overwriting `sm120/`.

- [ ] **Step 6: Run build and loader tests**

Run:

```powershell
.\py310\python.exe -m unittest backend-python.tests.test_albatross_kernel_loader backend-python.tests.test_albatross_build_script -v
```

Expected: all tests pass.

### Task 3: HTTP Throughput Diagnostics

**Files:**
- Modify: `backend-python/bench/albatross_api_benchmark.py`
- Add: `backend-python/bench/albatross_internal_benchmark.py`

- [ ] **Step 1: Add benchmark options for stop behavior and non-streaming**

Extend `albatross_api_benchmark.py` with:

```python
parser.add_argument("--no-stop", action="store_true")
parser.add_argument("--non-stream", action="store_true")
parser.add_argument("--count-chunks", action="store_true")
```

When `--no-stop` is set, send `"stop_token_ids": []` and `"stop": []` if accepted by the schema. When `--non-stream` is set, post `"stream": False` and count usage `completion_tokens` if returned.

- [ ] **Step 2: Add internal benchmark file matching the previous ad-hoc script**

Create `backend-python/bench/albatross_internal_benchmark.py` with CLI args:

```powershell
--model
--workers
--batch
--concurrency
--max-tokens
--prompt
--no-stop
```

It should initialize `AsyncEngineCore`, create `concurrency` completions, consume them concurrently, and print total tokens, wall time, aggregate tokens/s, first token p50/p95, and request wall p50/p95.

- [ ] **Step 3: Run diagnostics on sm120**

Run:

```powershell
.\py310\python.exe backend-python\bench\albatross_internal_benchmark.py --model D:\RWKV_Runner\models\RWKV7-G1-1.5B-16%25trained-20250308-ctx4k.pth --batch 960 --concurrency 960 --max-tokens 300 --no-stop
```

Then run API streaming and non-streaming benchmarks with the same `max_tokens`. Compare token counts before optimizing HTTP.

### Task 4: Optimize Albatross HTTP Streaming Path

**Files:**
- Modify: `backend-python/routes/completion.py`
- Modify: `backend-python/tests/test_albatross_completion_contract.py`

- [ ] **Step 1: Write tests for reduced disconnect polling**

Add a fake request test where 10 streamed tokens produce fewer than 12 `is_disconnected()` calls when the poll interval is 8. This test should call `eval_albatross()` and assert the stream still ends with `[DONE]`.

- [ ] **Step 2: Implement Albatross disconnect polling throttle**

In `eval_albatross()`, replace per-token `await request.is_disconnected()` with a helper:

```python
async def should_abort_after_token(token_index: int):
    return token_index == 1 or token_index % 8 == 0
```

Check disconnect only at those intervals and in `finally`.

- [ ] **Step 3: Write tests for pre-encoded streaming chunks**

Verify streaming output is still JSON-compatible and final `[DONE]` is present.

- [ ] **Step 4: Implement compact JSON serialization**

Use `json.dumps(payload, separators=(",", ":"))` in the Albatross streaming path to reduce payload size and CPU work. Keep non-streaming response dictionaries unchanged for compatibility.

- [ ] **Step 5: Run contract tests**

Run:

```powershell
.\py310\python.exe -m unittest backend-python.tests.test_albatross_completion_contract -v
```

Expected: all contract tests pass.

### Task 5: Verification and Benchmarks

**Files:**
- No new source files beyond previous tasks.

- [ ] **Step 1: Run focused unit tests**

Run:

```powershell
.\py310\python.exe -m unittest backend-python.tests.test_albatross_kernel_loader backend-python.tests.test_albatross_build_script backend-python.tests.test_albatross_completion_contract backend-python.tests.test_albatross_core_queue backend-python.tests.test_server_args -v
```

Expected: all tests pass.

- [ ] **Step 2: Import selected kernel**

Run:

```powershell
.\py310\python.exe -c "import sys; sys.path.insert(0,'backend-python'); import albatross.rwkv7; print('rwkv7 import ok')"
```

Expected: `rwkv7 import ok`.

- [ ] **Step 3: Benchmark sm120 HTTP path after optimization**

Start backend with:

```powershell
.\py310\python.exe backend-python\main.py --host 127.0.0.1 --port 8001 --no-access-log
```

Switch model with `manual_albatross_smoke.py`, then run:

```powershell
.\py310\python.exe backend-python\bench\albatross_api_benchmark.py --port 8001 --concurrency 960 --requests 960 --max-tokens 300 --timeout 900
```

Expected: `960 ok, 0 failed`. Compare aggregate tokens/s with the earlier `1604.65 tok/s` sm120 API result and report whether the bottleneck moved.

- [ ] **Step 4: Check git status**

Run:

```powershell
git status --short
```

Expected: only intended Albatross files changed, plus the pre-existing unrelated `go.mod`, `go.sum`, and `temp/`.
