import argparse
import json
import sys
import urllib.error
import urllib.request


def api_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def post_json(url: str, payload: dict, timeout: float):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return error.code, body


def stream_chat(url: str, payload: dict, timeout: float) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    first_text = ""
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
                if content:
                    first_text += content
                    if len(first_text) >= 80:
                        break
            return response.status, first_text
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return error.code, body


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a manual Albatross CUDA smoke test against RWKV Runner."
    )
    parser.add_argument("--model", required=True, help="Path to a RWKV-7 .pth model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    root = f"http://{args.host}:{args.port}"
    strategy = f"albatross workers={args.workers} batch={args.batch}"

    switch_status, switch_body = post_json(
        api_url(args.host, args.port, "/switch-model"),
        {
            "model": args.model,
            "strategy": strategy,
            "tokenizer": "",
            "customCuda": False,
        },
        args.timeout,
    )
    print(f"switch-model status: {switch_status}")
    print(switch_body[:500])
    if switch_status != 200:
        return 1

    chat_payload = {
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "model": "rwkv",
        "stream": False,
        "max_tokens": args.max_tokens,
        "temperature": 1,
        "top_p": 0.3,
    }
    chat_status, chat_body = post_json(
        api_url(args.host, args.port, "/v1/chat/completions"),
        chat_payload,
        args.timeout,
    )
    print(f"non-stream chat status: {chat_status}")
    try:
        chat_json = json.loads(chat_body)
        print("non-stream text:", chat_json["choices"][0]["message"]["content"][:200])
    except Exception:
        print(chat_body[:500])
    if chat_status != 200:
        return 1

    stream_payload = dict(chat_payload)
    stream_payload["stream"] = True
    stream_status, first_stream_text = stream_chat(
        api_url(args.host, args.port, "/v1/chat/completions"),
        stream_payload,
        args.timeout,
    )
    print(f"stream chat status: {stream_status}")
    print("stream first text:", first_stream_text[:200])
    print("server:", root)
    return 0 if stream_status == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
