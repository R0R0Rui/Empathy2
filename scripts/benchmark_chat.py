from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def post_chat(url: str, message: str, session_id: str | None) -> dict[str, object]:
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the classifier plus Qwen response path.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/chat")
    parser.add_argument("--message", default="I've been overwhelmed lately and I don't know what to do.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reuse-session", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_id: str | None = None

    for index in range(args.warmup):
        result = post_chat(args.url, args.message, session_id)
        session_id = str(result["session_id"]) if args.reuse_session else None
        print(f"warmup {index + 1}: {result['performance']}")

    classifier_ms: list[float] = []
    generation_ms: list[float] = []
    total_ms: list[float] = []
    wall_ms: list[float] = []

    for index in range(args.runs):
        started = time.perf_counter()
        result = post_chat(args.url, args.message, session_id)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        session_id = str(result["session_id"]) if args.reuse_session else None
        performance = result["performance"]
        classifier_ms.append(float(performance["classifier_ms"]))
        generation_ms.append(float(performance["generation_ms"]))
        total_ms.append(float(performance["total_ms"]))
        wall_ms.append(elapsed_ms)
        print(f"run {index + 1}: {performance} wall_ms={elapsed_ms:.2f}")

    print("\nmeans")
    print(f"classifier_ms={mean(classifier_ms):.2f}")
    print(f"generation_ms={mean(generation_ms):.2f}")
    print(f"total_ms={mean(total_ms):.2f}")
    print(f"wall_ms={mean(wall_ms):.2f}")


if __name__ == "__main__":
    main()
