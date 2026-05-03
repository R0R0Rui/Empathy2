import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .validate import validate_response_structured


RESPONSE_COLUMNS = [
    ("baseline", "baseline_response"),
    ("anchor", "anchor_response"),
    ("refined_anchor", "refined_anchor_response"),
]


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fieldnames):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _details_path(summary_path):
    path = Path(summary_path)
    return path.with_name(f"{path.stem}_details{path.suffix}")


def _failure_path(summary_path):
    path = Path(summary_path)
    return path.with_name(f"{path.stem}_failure_counts{path.suffix}")


def _scenario(row):
    return (
        row.get("scenario_type")
        or row.get("type")
        or row.get("expected_risk")
        or "unknown"
    )


def _available_systems(rows):
    systems = []
    for system_name, column in RESPONSE_COLUMNS:
        if any(row.get(column) for row in rows):
            systems.append((system_name, column))
    return systems


def evaluate_rows(rows):
    detail_rows = []
    pass_totals = defaultdict(lambda: {"passed": 0, "total": 0})
    failure_counts = Counter()

    systems = _available_systems(rows)

    for index, row in enumerate(rows, start=1):
        case_id = row.get("case_id") or str(index)
        scenario_type = _scenario(row)
        user_message = row.get("user_message") or row.get("text") or ""

        for system_name, response_column in systems:
            response = row.get(response_column, "")
            if not response:
                continue

            validation = validate_response_structured(
                user_message=user_message,
                response=response,
                scenario_type=scenario_type,
            )

            pass_key = (system_name, scenario_type)
            pass_totals[pass_key]["total"] += 1
            if validation["passed"]:
                pass_totals[pass_key]["passed"] += 1

            for failure_type in validation["failure_types"]:
                failure_counts[(system_name, scenario_type, failure_type)] += 1

            detail_rows.append(
                {
                    "case_id": case_id,
                    "scenario_type": scenario_type,
                    "system": system_name,
                    "response_column": response_column,
                    "passed": validation["passed"],
                    "severity": validation["severity"],
                    "rewrite_needed": validation["rewrite_needed"],
                    "failure_types": ";".join(validation["failure_types"]),
                    "notes": " | ".join(validation["notes"]),
                    "user_message": user_message,
                    "response": response,
                    "validation_json": json.dumps(validation, ensure_ascii=False),
                }
            )

    summary_rows = []
    for (system_name, scenario_type), totals in sorted(pass_totals.items()):
        total = totals["total"]
        passed = totals["passed"]
        pass_rate = passed / total if total else 0.0
        summary_rows.append(
            {
                "system": system_name,
                "scenario_type": scenario_type,
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": f"{pass_rate:.4f}",
            }
        )

    failure_rows = []
    for (system_name, scenario_type, failure_type), count in sorted(
        failure_counts.items()
    ):
        failure_rows.append(
            {
                "system": system_name,
                "scenario_type": scenario_type,
                "failure_type": failure_type,
                "count": count,
            }
        )

    return summary_rows, detail_rows, failure_rows


def run_failure_discovery(input_path, output_path, detailed_output_path=None):
    rows = _read_csv(input_path)
    summary_rows, detail_rows, failure_rows = evaluate_rows(rows)

    summary_fields = [
        "system",
        "scenario_type",
        "total",
        "passed",
        "failed",
        "pass_rate",
    ]
    detail_fields = [
        "case_id",
        "scenario_type",
        "system",
        "response_column",
        "passed",
        "severity",
        "rewrite_needed",
        "failure_types",
        "notes",
        "user_message",
        "response",
        "validation_json",
    ]
    failure_fields = ["system", "scenario_type", "failure_type", "count"]

    _write_csv(output_path, summary_rows, summary_fields)
    _write_csv(
        detailed_output_path or _details_path(output_path),
        detail_rows,
        detail_fields,
    )
    _write_csv(_failure_path(output_path), failure_rows, failure_fields)

    return {
        "summary_path": str(output_path),
        "details_path": str(detailed_output_path or _details_path(output_path)),
        "failure_counts_path": str(_failure_path(output_path)),
        "summary_rows": len(summary_rows),
        "detail_rows": len(detail_rows),
        "failure_count_rows": len(failure_rows),
    }
