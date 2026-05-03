import argparse
import csv
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_ENDPOINT = "http://127.0.0.1:8000/chat"
DEFAULT_CASE_DIR = "data/eval_cases"
DEFAULT_OUTPUT = "data/results/backend_outputs.csv"

FIELDNAMES = [
    "case_id",
    "scenario_type",
    "user_message",
    "expected_risk",
    "expected_need",
    "anchor_response",
    "primary_emotion",
    "secondary_emotion",
    "crisis_detected",
    "understanding_json",
    "intent",
    "scenario_tier",
    "support_need",
    "safety_flag",
    "is_implicit",
    "emotion_intensity",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run eval cases through a locally running Empathy backend."
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"Backend chat endpoint. Defaults to {DEFAULT_ENDPOINT}.",
    )
    parser.add_argument(
        "--case-dir",
        default=DEFAULT_CASE_DIR,
        help=f"Directory containing JSONL eval cases. Defaults to {DEFAULT_CASE_DIR}.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"CSV output path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of cases to run for a quick smoke test.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds. Defaults to 120.",
    )
    return parser.parse_args()


def load_cases(case_dir, limit=None):
    cases = []
    for path in sorted(Path(case_dir).glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                case = json.loads(line)
                case["_source_file"] = str(path)
                case["_source_line"] = line_number
                cases.append(case)
                if limit is not None and len(cases) >= limit:
                    return cases
    return cases


def post_json(endpoint, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def emotion_label(response, rank):
    emotions = response.get("emotions") or {}
    value = emotions.get(rank)
    if isinstance(value, dict):
        return value.get("label", "")
    return value or ""


def crisis_detected(response):
    if "crisis_detected" in response:
        return response.get("crisis_detected", "")

    safety = response.get("safety") or {}
    if isinstance(safety, dict):
        return safety.get("crisis_detected", "")
    return ""


def flatten_understanding(understanding):
    if not isinstance(understanding, dict):
        return {
            "understanding_json": "",
            "intent": "",
            "scenario_tier": "",
            "support_need": "",
            "safety_flag": "",
            "is_implicit": "",
            "emotion_intensity": "",
        }

    return {
        "understanding_json": json.dumps(understanding, ensure_ascii=False),
        "intent": understanding.get("intent", ""),
        "scenario_tier": understanding.get("scenario_tier", ""),
        "support_need": understanding.get("support_need", ""),
        "safety_flag": understanding.get("safety_flag", ""),
        "is_implicit": understanding.get("is_implicit", ""),
        "emotion_intensity": understanding.get("emotion_intensity", ""),
    }


def build_row(case, response):
    understanding_fields = flatten_understanding(response.get("understanding"))
    return {
        "case_id": case.get("case_id", ""),
        "scenario_type": case.get("scenario_type", ""),
        "user_message": case.get("user_message", ""),
        "expected_risk": case.get("expected_risk", ""),
        "expected_need": case.get("expected_need", ""),
        "anchor_response": response.get("reply") or response.get("response") or "",
        "primary_emotion": emotion_label(response, "primary"),
        "secondary_emotion": emotion_label(response, "secondary"),
        "crisis_detected": crisis_detected(response),
        **understanding_fields,
    }


def write_rows(output_path, rows):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run_cases(endpoint, case_dir, output, limit=None, timeout=120.0):
    cases = load_cases(case_dir, limit=limit)
    rows = []

    for index, case in enumerate(cases, start=1):
        payload = {
            "message": case.get("user_message", ""),
            "session_id": f"eval-{case.get('case_id', index)}-{int(time.time())}",
        }
        try:
            response = post_json(endpoint, payload, timeout=timeout)
        except HTTPError as exc:
            raise RuntimeError(
                f"Backend returned HTTP {exc.code} for case {case.get('case_id')}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not reach backend at {endpoint}. Is uvicorn running?"
            ) from exc

        rows.append(build_row(case, response))
        print(f"[{index}/{len(cases)}] {case.get('case_id')} complete")

    write_rows(output, rows)
    return output, len(rows)


def main():
    args = parse_args()
    output, count = run_cases(
        endpoint=args.endpoint,
        case_dir=args.case_dir,
        output=args.output,
        limit=args.limit,
        timeout=args.timeout,
    )
    print(f"Wrote {count} backend outputs to {output}")


if __name__ == "__main__":
    main()
