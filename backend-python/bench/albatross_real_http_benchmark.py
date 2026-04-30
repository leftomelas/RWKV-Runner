import argparse
import json
import os
import pathlib
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


BACKEND_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from bench.albatross_api_benchmark import (
    NO_PROXY_OPENER,
    percentile,
    post_non_stream_chat,
    post_stream_chat,
    run_chat_request,
)


def request_json(
    url: str,
    method: str = "GET",
    payload: dict | None = None,
    timeout: float = 30,
):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with NO_PROXY_OPENER.open(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        try:
            return response.status, json.loads(body)
        except json.JSONDecodeError:
            return response.status, body


def wait_for_server(port: int, timeout: float = 30):
    deadline = time.perf_counter() + timeout
    last_error = None
    while time.perf_counter() < deadline:
        try:
            return request_json(f"http://127.0.0.1:{port}/", timeout=2)
        except Exception as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"server did not start: {last_error}")


def load_uvicorn_app(args):
    os.environ.setdefault("ALBATROSS_PROFILE", "1")
    os.environ["ALBATROSS_KERNEL_ARCH"] = args.kernel_arch
    os.environ["ALBATROSS_SAMPLER"] = args.sampler
    os.environ["ALBATROSS_SAMPLER_FALLBACK"] = "1" if args.sampler_fallback else "0"
    os.environ.setdefault(
        "RWKV_RUNNER_PARAMS",
        f"--host 127.0.0.1 --port {args.port} --no-access-log",
    )
    os.chdir(BACKEND_ROOT)

    import uvicorn
    from main import app

    return uvicorn, app


def print_results(args, results: list[dict], total_wall: float) -> int:
    ok_results = [result for result in results if result["ok"]]
    failed_results = [result for result in results if not result["ok"]]
    total_tokens = sum(result["tokens"] for result in ok_results)
    latencies = [
        result["first_token_latency"]
        for result in ok_results
        if result["first_token_latency"] is not None
    ]
    walls = [result["wall_time"] for result in ok_results]

    print("\nSummary")
    print(f"requests: {len(ok_results)} ok, {len(failed_results)} failed")
    print(f"concurrency: {args.concurrency}")
    print(f"total tokens: {total_tokens}")
    print(f"total wall time: {total_wall:.3f}s")
    print(f"aggregate tokens/s: {total_tokens / total_wall if total_wall else 0:.2f}")
    retry_results = [result for result in results if result.get("attempts", 1) > 1]
    if retry_results:
        print(f"requests retried after connection refused: {len(retry_results)}")
        print(f"max attempts: {max(result.get('attempts', 1) for result in results)}")
    if failed_results:
        error_counts = {}
        for result in failed_results:
            error_counts[result["error"]] = error_counts.get(result["error"], 0) + 1
        print("errors:")
        for error, count in sorted(
            error_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"  {count}x {error}")
    if latencies:
        print(f"first token latency avg: {statistics.mean(latencies):.3f}s")
        print(f"first token latency p50: {percentile(latencies, 0.50):.3f}s")
        print(f"first token latency p95: {percentile(latencies, 0.95):.3f}s")
    if walls:
        print(f"request wall time avg: {statistics.mean(walls):.3f}s")
        print(f"request wall time p95: {percentile(walls, 0.95):.3f}s")
    return 0 if not failed_results else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start RWKV Runner and benchmark real Albatross HTTP traffic."
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=960)
    parser.add_argument("--requests", type=int, default=960)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--non-stream", action="store_true")
    parser.add_argument("--connect-retries", type=int, default=0)
    parser.add_argument("--connect-retry-delay", type=float, default=0.05)
    parser.add_argument("--kernel-arch", default="sm80_compute80")
    parser.add_argument(
        "--sampler",
        choices=("python", "greedy", "gumbel", "cuda"),
        default=os.environ.get("ALBATROSS_SAMPLER", "python"),
    )
    parser.add_argument("--sampler-fallback", action="store_true")
    parser.add_argument(
        "--model",
        default=r"D:\RWKV_Runner\models\RWKV7-G1-1.5B-16%25trained-20250308-ctx4k.pth",
    )
    parser.add_argument(
        "--tokenizer",
        default=str(BACKEND_ROOT / "albatross" / "rwkv_vocab_v20230424.txt"),
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Write a very long detailed technical article about GPU batch inference, "
            "with many paragraphs, examples, and no conclusion until the end."
        ),
    )
    args = parser.parse_args()

    uvicorn, app = load_uvicorn_app(args)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=args.port,
            workers=1,
            access_log=False,
            timeout_keep_alive=120,
            backlog=2048,
            log_level="info",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_for_server(args.port)

    switch_payload = {
        "model": args.model,
        "strategy": "albatross workers=1 batch=960",
        "tokenizer": args.tokenizer,
        "customCuda": False,
        "deploy": False,
    }
    status, body = request_json(
        f"http://127.0.0.1:{args.port}/switch-model",
        method="POST",
        payload=switch_payload,
        timeout=300,
    )
    print(f"switch-model status: {status} body: {body}", flush=True)
    request_json(f"http://127.0.0.1:{args.port}/albatross/profile/reset", method="POST")

    print(f"ALBATROSS_PROFILE: {os.environ.get('ALBATROSS_PROFILE', '0')}")
    print(f"ALBATROSS_KERNEL_ARCH: {os.environ.get('ALBATROSS_KERNEL_ARCH', '<auto>')}")
    print(f"ALBATROSS_SAMPLER: {os.environ.get('ALBATROSS_SAMPLER', 'python')}")
    print(f"ALBATROSS_SAMPLER_FALLBACK: {os.environ.get('ALBATROSS_SAMPLER_FALLBACK', '1')}")
    print(f"stream: {not args.non_stream}")
    print(f"connect retries: {args.connect_retries}")

    payload = {
        "messages": [{"role": "user", "content": args.prompt}],
        "model": "rwkv",
        "stream": not args.non_stream,
        "max_tokens": args.max_tokens,
        "temperature": 1,
        "top_p": 0.3,
        "top_k": 0,
        "stop": None,
        "stop_token_ids": [],
    }
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    post_chat = post_non_stream_chat if args.non_stream else post_stream_chat
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_chat_request,
                post_chat,
                url,
                payload,
                args.timeout,
                args.connect_retries,
                args.connect_retry_delay,
            )
            for _ in range(args.requests)
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if not result["ok"]:
                print(
                    f"request {idx}/{args.requests}: FAILED error={result.get('error')}",
                    flush=True,
                )
                continue
            print(
                f"request {idx}/{args.requests}: status={result['status']} "
                f"tokens={result['tokens']} wall={result['wall_time']:.3f}s",
                flush=True,
            )

    total_wall = time.perf_counter() - started
    exit_code = print_results(args, results, total_wall)

    _, profile = request_json(f"http://127.0.0.1:{args.port}/albatross/profile")
    print("\nServer profile")
    for key, value in profile.items():
        print(f"{key}: {value}")
    server.should_exit = True
    thread.join(timeout=10)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
