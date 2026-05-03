import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from evaluation.refine import refine_response
from evaluation.support_plan import (
    build_support_plan,
    parse_understanding_json,
    summarize_support_plan,
)
from evaluation.validate import validate_response_structured
from scripts.run_backend_cases import run_cases


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000/chat"
DEFAULT_CASE_DIR = "data/eval_cases"
DEFAULT_RAW_OUTPUT = "data/results/backend_outputs.csv"
DEFAULT_FINAL_OUTPUT = "data/results/final_with_refinement.csv"
DEFAULT_SUMMARY_OUTPUT = "data/results/eval_summary.csv"
DEFAULT_DETAILS_OUTPUT = "data/results/eval_summary_details.csv"
DEFAULT_FAILURE_COUNTS_OUTPUT = "data/results/eval_summary_failure_counts.csv"

FINAL_COLUMNS = [
    "case_id",
    "scenario_type",
    "user_message",
    "expected_risk",
    "expected_need",
    "primary_emotion",
    "secondary_emotion",
    "crisis_detected",
    "intent",
    "scenario_tier",
    "support_need",
    "safety_flag",
    "is_implicit",
    "emotion_intensity",
    "support_plan_json",
    "support_plan_summary",
    "anchor_response",
    "anchor_passed",
    "anchor_failure_types",
    "refined_anchor_response",
    "refined_passed",
    "refined_failure_types",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full offline backend evaluation and refinement pipeline."
    )
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--case-dir", default=DEFAULT_CASE_DIR)
    parser.add_argument("--raw-output", default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-backend",
        action="store_true",
        help="Use an existing raw-output CSV instead of calling the backend.",
    )
    return parser.parse_args()


def _first_present(row, columns):
    for column in columns:
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            return row[column]
    return ""


def _failure_text(validation):
    return ", ".join(validation.get("failure_types", []))


def _load_raw_outputs(path):
    return pd.read_csv(path).fillna("")


def normalize_columns(df):
    df = df.copy()

    if "anchor_response" not in df.columns:
        if "reply" in df.columns:
            df["anchor_response"] = df["reply"]
        else:
            df["anchor_response"] = ""
    else:
        df["anchor_response"] = df.apply(
            lambda row: _first_present(row, ["anchor_response", "reply"]),
            axis=1,
        )

    if "scenario_type" not in df.columns:
        if "type" in df.columns:
            df["scenario_type"] = df["type"]
        elif "scenario_tier" in df.columns:
            df["scenario_type"] = df["scenario_tier"]
        else:
            df["scenario_type"] = "unknown"
    else:
        df["scenario_type"] = df.apply(
            lambda row: _first_present(row, ["scenario_type", "type", "scenario_tier"])
            or "unknown",
            axis=1,
        )

    if "case_id" not in df.columns:
        df["case_id"] = [f"case_{index + 1}" for index in range(len(df))]

    for column in FINAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    return df


def row_understanding(row):
    understanding = {}
    if "understanding_json" in row.index:
        understanding.update(parse_understanding_json(row.get("understanding_json")))

    for column in [
        "intent",
        "scenario_tier",
        "support_need",
        "safety_flag",
        "is_implicit",
        "emotion_intensity",
    ]:
        value = row.get(column, "")
        if pd.notna(value) and str(value).strip():
            understanding[column] = value

    return understanding


def add_support_plan_columns(df):
    df = df.copy()
    plan_json = []
    plan_summary = []

    for _, row in df.iterrows():
        understanding = row_understanding(row)
        if not understanding:
            plan_json.append("")
            plan_summary.append("")
            continue

        plan = build_support_plan(understanding, user_message=row.get("user_message", ""))
        plan_json.append(json.dumps(plan, ensure_ascii=False))
        plan_summary.append(summarize_support_plan(plan))

    df["support_plan_json"] = plan_json
    df["support_plan_summary"] = plan_summary
    return df


def validate_anchor_responses(df):
    df = df.copy()
    validations = []

    for _, row in df.iterrows():
        validation = validate_response_structured(
            user_message=row.get("user_message", ""),
            response=row.get("anchor_response", ""),
            scenario_type=row.get("scenario_type", None),
            understanding=row_understanding(row),
        )
        validations.append(validation)

    df["anchor_validation"] = [json.dumps(item, ensure_ascii=False) for item in validations]
    df["anchor_passed"] = [item["passed"] for item in validations]
    df["anchor_failure_types"] = [_failure_text(item) for item in validations]
    df["anchor_severity"] = [item["severity"] for item in validations]
    return df


def refine_responses(df):
    df = df.copy()
    refined = []
    for _, row in df.iterrows():
        if bool(row.get("anchor_passed", False)):
            refined.append(row.get("anchor_response", ""))
        else:
            refined.append(
                refine_response(
                    user_message=row.get("user_message", ""),
                    original_response=row.get("anchor_response", ""),
                    validation_result=json.loads(row.get("anchor_validation", "{}")),
                    support_plan=parse_understanding_json(row.get("support_plan_json")),
                    understanding=row_understanding(row),
                    scenario_type=row.get("scenario_type", None),
                )
            )
    df["refined_anchor_response"] = refined
    return df


def validate_refined_responses(df):
    df = df.copy()
    validations = []

    for _, row in df.iterrows():
        validation = validate_response_structured(
            user_message=row.get("user_message", ""),
            response=row.get("refined_anchor_response", ""),
            scenario_type=row.get("scenario_type", None),
            understanding=row_understanding(row),
        )
        validations.append(validation)

    df["refined_validation"] = [json.dumps(item, ensure_ascii=False) for item in validations]
    df["refined_passed"] = [item["passed"] for item in validations]
    df["refined_failure_types"] = [_failure_text(item) for item in validations]
    df["refined_severity"] = [item["severity"] for item in validations]
    return df


def _pass_rate(series):
    if len(series) == 0:
        return 0.0
    return float(series.astype(bool).sum()) / float(len(series))


def _failure_counts(df, column, stage):
    counts = Counter()
    for value in df[column].fillna(""):
        for failure_type in [item.strip() for item in str(value).split(",") if item.strip()]:
            counts[(stage, failure_type)] += 1
    return counts


def write_summary_outputs(
    df,
    summary_path=DEFAULT_SUMMARY_OUTPUT,
    details_path=DEFAULT_DETAILS_OUTPUT,
    failure_counts_path=DEFAULT_FAILURE_COUNTS_OUTPUT,
):
    failure_counts = Counter()
    failure_counts.update(_failure_counts(df, "anchor_failure_types", "anchor"))
    failure_counts.update(_failure_counts(df, "refined_failure_types", "refined"))

    summary_rows = [
        {
            "section": "overall",
            "system": "all",
            "scenario_type": "all",
            "metric": "number_of_cases",
            "value": len(df),
        },
        {
            "section": "overall",
            "system": "anchor",
            "scenario_type": "all",
            "metric": "pass_rate",
            "value": f"{_pass_rate(df['anchor_passed']):.4f}",
        },
        {
            "section": "overall",
            "system": "refined",
            "scenario_type": "all",
            "metric": "pass_rate",
            "value": f"{_pass_rate(df['refined_passed']):.4f}",
        },
    ]

    for scenario_type, group in sorted(df.groupby("scenario_type")):
        summary_rows.append(
            {
                "section": "by_scenario",
                "system": "anchor",
                "scenario_type": scenario_type,
                "metric": "pass_rate",
                "value": f"{_pass_rate(group['anchor_passed']):.4f}",
            }
        )
        summary_rows.append(
            {
                "section": "by_scenario",
                "system": "refined",
                "scenario_type": scenario_type,
                "metric": "pass_rate",
                "value": f"{_pass_rate(group['refined_passed']):.4f}",
            }
        )

    for (system, failure_type), count in sorted(failure_counts.items()):
        summary_rows.append(
            {
                "section": "failure_counts",
                "system": system,
                "scenario_type": "all",
                "metric": failure_type,
                "value": count,
            }
        )

    detail_rows = []
    for _, row in df.iterrows():
        for system, passed_col, severity_col, failure_col, response_col in [
            (
                "anchor",
                "anchor_passed",
                "anchor_severity",
                "anchor_failure_types",
                "anchor_response",
            ),
            (
                "refined",
                "refined_passed",
                "refined_severity",
                "refined_failure_types",
                "refined_anchor_response",
            ),
        ]:
            detail_rows.append(
                {
                    "case_id": row.get("case_id", ""),
                    "scenario_type": row.get("scenario_type", ""),
                    "system": system,
                    "passed": row.get(passed_col, ""),
                    "severity": row.get(severity_col, ""),
                    "failure_types": row.get(failure_col, ""),
                    "user_message": row.get("user_message", ""),
                    "response": row.get(response_col, ""),
                }
            )

    failure_rows = [
        {"system": system, "failure_type": failure_type, "count": count}
        for (system, failure_type), count in sorted(failure_counts.items())
    ]

    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(detail_rows).to_csv(details_path, index=False)
    pd.DataFrame(failure_rows).to_csv(failure_counts_path, index=False)


def run_full_pipeline(
    backend_url=DEFAULT_BACKEND_URL,
    case_dir=DEFAULT_CASE_DIR,
    raw_output=DEFAULT_RAW_OUTPUT,
    limit=None,
    skip_backend=False,
    final_output=DEFAULT_FINAL_OUTPUT,
    summary_output=DEFAULT_SUMMARY_OUTPUT,
    details_output=DEFAULT_DETAILS_OUTPUT,
    failure_counts_output=DEFAULT_FAILURE_COUNTS_OUTPUT,
):
    if not skip_backend:
        run_cases(
            endpoint=backend_url,
            case_dir=case_dir,
            output=raw_output,
            limit=limit,
        )

    raw_path = Path(raw_output)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw output CSV not found: {raw_output}")

    df = _load_raw_outputs(raw_path)
    df = normalize_columns(df)
    df = add_support_plan_columns(df)
    df = validate_anchor_responses(df)
    df = refine_responses(df)
    df = validate_refined_responses(df)

    Path(final_output).parent.mkdir(parents=True, exist_ok=True)
    df[FINAL_COLUMNS].to_csv(final_output, index=False)
    write_summary_outputs(
        df,
        summary_path=summary_output,
        details_path=details_output,
        failure_counts_path=failure_counts_output,
    )

    return {
        "cases": len(df),
        "final_output": final_output,
        "summary_output": summary_output,
        "details_output": details_output,
        "failure_counts_output": failure_counts_output,
        "anchor_pass_rate": _pass_rate(df["anchor_passed"]),
        "refined_pass_rate": _pass_rate(df["refined_passed"]),
    }


def main():
    args = parse_args()
    result = run_full_pipeline(
        backend_url=args.backend_url,
        case_dir=args.case_dir,
        raw_output=args.raw_output,
        limit=args.limit,
        skip_backend=args.skip_backend,
    )
    print(f"Processed cases: {result['cases']}")
    print(f"Anchor pass rate: {result['anchor_pass_rate']:.4f}")
    print(f"Refined pass rate: {result['refined_pass_rate']:.4f}")
    print(f"Wrote final output: {result['final_output']}")
    print(f"Wrote summary: {result['summary_output']}")
    print(f"Wrote details: {result['details_output']}")
    print(f"Wrote failure counts: {result['failure_counts_output']}")


if __name__ == "__main__":
    main()
