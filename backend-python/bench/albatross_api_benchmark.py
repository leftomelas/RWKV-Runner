import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        with urllib.request.urlopen(request, timeout=timeout) as response:
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
    except Exception as error:
        return {
            "ok": False,
            "status": None,
            "error": str(error),
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
        "--prompt",
        default="Write one concise paragraph about parallel language model inference.",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": args.prompt}],
        "model": "rwkv",
        "stream": True,
        "max_tokens": args.max_tokens,
        "temperature": 1,
        "top_p": 0.3,
    }

    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(post_stream_chat, url, payload, args.timeout)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
