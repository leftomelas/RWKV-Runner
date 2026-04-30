import argparse
import json
import os
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


BACKEND_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from bench.albatross_http_client import run_with_connect_retries


NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request_json(url: str, method: str = "GET") -> dict:
    request = urllib.request.Request(url, method=method)
    with NO_PROXY_OPENER.open(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post_stream_chat(url: str, payload: dict, timeout: float) -> dict:
    started = time.perf_counter()
    first_token_at = None
    chunks = 0
    text_parts = []
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with NO_PROXY_OPENER.open(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_line = line[6:]
                if data_line == "[DONE]":
                    break
                event = json.loads(data_line)
                delta = event["choices"][0].get("delta", {})
                content = delta.get("content")
                if not content:
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                chunks += 1
                text_parts.append(content)
        ended = time.perf_counter()
        return {
            "ok": True,
            "status": response.status,
            "first_token_latency": (
                None if first_token_at is None else first_token_at - started
            ),
            "wall_time": ended - started,
            "tokens": chunks,
            "text": "".join(text_parts),
        }
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": error.code,
            "error": body[:500],
            "first_token_latency": None,
            "wall_time": time.perf_counter() - started,
            "tokens": 0,
        }

def post_non_stream_chat(url: str, payload: dict, timeout: float) -> dict:
    started = time.perf_counter()
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with NO_PROXY_OPENER.open(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        ended = time.perf_counter()
        event = json.loads(body)
        usage = event.get("usage") or {}
        message = event["choices"][0].get("message", {})
        content = message.get("content") or event["choices"][0].get("text") or ""
        return {
            "ok": True,
            "status": response.status,
            "first_token_latency": None,
            "wall_time": ended - started,
            "tokens": usage.get("completion_tokens", 0),
            "text": content,
        }
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": error.code,
            "error": body[:500],
            "first_token_latency": None,
            "wall_time": time.perf_counter() - started,
            "tokens": 0,
        }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, int(round((len(values) - 1) * pct)))
    return values[index]


def run_chat_request(
    post_chat,
    url: str,
    payload: dict,
    timeout: float,
    connect_retries: int,
    connect_retry_delay: float,
) -> dict:
    started = time.perf_counter()
    try:
        return run_with_connect_retries(
            lambda: post_chat(url, payload, timeout),
            connect_retries=connect_retries,
            connect_retry_delay=connect_retry_delay,
        )
    except Exception as error:
        return {
            "ok": False,
            "status": None,
            "error": str(error),
            "first_token_latency": None,
            "wall_time": time.perf_counter() - started,
            "tokens": 0,
            "attempts": getattr(error, "attempts", 1),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark RWKV Runner's Albatross chat completion endpoint."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--connect-retries",
        type=int,
        default=0,
        help="retry only TCP connection refused errors before counting a request failed",
    )
    parser.add_argument(
        "--connect-retry-delay",
        type=float,
        default=0.05,
        help="base retry delay in seconds; attempt N sleeps delay*N",
    )
    parser.add_argument(
        "--non-stream",
        action="store_true",
        help="use non-streaming chat completions and count usage completion_tokens",
    )
    parser.add_argument(
        "--no-stop",
        action="store_true",
        help="disable text stop strings; EOS stop token may still end generation",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="reset and print /albatross/profile around this benchmark run",
    )
    parser.add_argument(
        "--prompt",
        default="Write one concise paragraph about parallel language model inference.",
    )
    args = parser.parse_args()

    print(f"ALBATROSS_PROFILE: {os.environ.get('ALBATROSS_PROFILE', '0')}")
    print(f"ALBATROSS_SAMPLER: {os.environ.get('ALBATROSS_SAMPLER', 'python')}")
    print(f"ALBATROSS_SCHEDULER: {os.environ.get('ALBATROSS_SCHEDULER', 'legacy')}")

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    profile_url = f"http://{args.host}:{args.port}/albatross/profile"
    if args.profile:
        request_json(f"{profile_url}/reset", method="POST")

    payload = {
        "messages": [{"role": "user", "content": args.prompt}],
        "model": "rwkv",
        "stream": not args.non_stream,
        "max_tokens": args.max_tokens,
        "temperature": 1,
        "top_p": 0.3,
    }
    if args.no_stop:
        payload["stop"] = None
        payload["stop_token_ids"] = []

    started = time.perf_counter()
    results = []
    post_chat = post_non_stream_chat if args.non_stream else post_stream_chat
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
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = result["status"] if result["status"] is not None else "error"
            print(
                f"request {len(results)}/{args.requests}: "
                f"status={status} tokens={result['tokens']} "
                f"wall={result['wall_time']:.3f}s"
            )
    ended = time.perf_counter()

    ok_results = [result for result in results if result["ok"]]
    failed_results = [result for result in results if not result["ok"]]
    latencies = [
        result["first_token_latency"]
        for result in ok_results
        if result["first_token_latency"] is not None
    ]
    wall_times = [result["wall_time"] for result in ok_results]
    total_tokens = sum(result["tokens"] for result in ok_results)
    total_wall = ended - started

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
    if latencies:
        print(f"first token latency avg: {statistics.mean(latencies):.3f}s")
        print(f"first token latency p50: {percentile(latencies, 0.50):.3f}s")
        print(f"first token latency p95: {percentile(latencies, 0.95):.3f}s")
    if wall_times:
        print(f"request wall time avg: {statistics.mean(wall_times):.3f}s")
        print(f"request wall time p95: {percentile(wall_times, 0.95):.3f}s")
    if failed_results:
        print("\nFirst failure:")
        print(failed_results[0].get("error", "unknown error"))
        return 1
    if args.profile:
        profile = request_json(profile_url)
        print("\nServer profile")
        for key in (
            "requests",
            "stream_requests",
            "tokens",
            "stream_chunks",
            "bytes",
            "completion_wait_ms",
            "disconnect_check_ms",
            "json_dump_ms",
            "yield_resume_ms",
            "request_wall_ms",
        ):
            print(f"{key}: {profile.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
