import argparse
import asyncio
import pathlib
import statistics
import sys
import time


BACKEND_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from albatross_engine.core import AsyncEngineCore
from albatross_engine.task import ModelLoadConfig


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, int(round((len(values) - 1) * pct)))
    return values[index]


async def consume_completion(completion) -> dict:
    started = time.perf_counter()
    first_token_at = None
    tokens = 0
    async for event in completion:
        if event[0] != "token":
            continue
        if first_token_at is None:
            first_token_at = time.perf_counter()
        tokens += 1
    ended = time.perf_counter()
    return {
        "tokens": tokens,
        "wall_time": ended - started,
        "first_token_latency": (
            None if first_token_at is None else first_token_at - started
        ),
    }


async def run_benchmark(args) -> int:
    engine = AsyncEngineCore()
    vocab = (
        pathlib.Path(args.vocab)
        if args.vocab
        else BACKEND_ROOT / "albatross" / "rwkv_vocab_v20230424.txt"
    )
    model_config = ModelLoadConfig(
        model_path=args.model,
        vocab_path=str(vocab.resolve()),
        vocab_size=args.vocab_size,
        head_size=args.head_size,
    )

    init_started = time.perf_counter()
    await engine.init(
        worker_num=args.workers,
        model_config=model_config,
        batch_size=args.batch + 1,
    )
    print(f"init wall: {time.perf_counter() - init_started:.3f}s", flush=True)

    stop_tokens = [] if args.no_stop else [0]
    completions = [
        engine.completion(
            prompt_str=args.prompt,
            max_tokens=args.max_tokens,
            stop_tokens=stop_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        for _ in range(args.concurrency)
    ]

    started = time.perf_counter()
    results = await asyncio.gather(
        *(consume_completion(completion) for completion in completions)
    )
    ended = time.perf_counter()
    engine.shutdown()

    ok_results = [result for result in results if result["tokens"] >= 0]
    total_tokens = sum(result["tokens"] for result in ok_results)
    total_wall = ended - started
    first_latencies = [
        result["first_token_latency"]
        for result in ok_results
        if result["first_token_latency"] is not None
    ]
    wall_times = [result["wall_time"] for result in ok_results]

    print("Summary")
    print(f"requests: {len(ok_results)} ok, {len(results) - len(ok_results)} failed")
    print(f"concurrency: {args.concurrency}")
    print(f"batch slots: {args.batch}")
    print(f"max tokens: {args.max_tokens}")
    print(f"total tokens: {total_tokens}")
    print(f"expected tokens: {args.concurrency * args.max_tokens}")
    print(f"total wall time: {total_wall:.3f}s")
    print(f"aggregate tokens/s: {total_tokens / total_wall if total_wall else 0:.2f}")
    if first_latencies:
        print(f"first token latency avg: {statistics.mean(first_latencies):.3f}s")
        print(f"first token latency p50: {percentile(first_latencies, 0.50):.3f}s")
        print(f"first token latency p95: {percentile(first_latencies, 0.95):.3f}s")
    if wall_times:
        print(f"request wall time avg: {statistics.mean(wall_times):.3f}s")
        print(f"request wall time p50: {percentile(wall_times, 0.50):.3f}s")
        print(f"request wall time p95: {percentile(wall_times, 0.95):.3f}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Albatross AsyncEngineCore without HTTP overhead."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocab", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="Say hello briefly.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument(
        "--no-stop",
        action="store_true",
        help="do not stop on EOS, so total generated tokens should match max_tokens",
    )
    args = parser.parse_args()
    return asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    raise SystemExit(main())
