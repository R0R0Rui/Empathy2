import argparse

from .failure_discovery import run_failure_discovery


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run offline empathy response validation over comparison CSVs."
    )
    parser.add_argument("--input", required=True, help="Input CSV with generated outputs.")
    parser.add_argument("--output", required=True, help="Summary CSV output path.")
    parser.add_argument(
        "--details-output",
        default=None,
        help="Optional detailed CSV output path. Defaults to <output>_details.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_failure_discovery(
        input_path=args.input,
        output_path=args.output,
        detailed_output_path=args.details_output,
    )

    print(f"Wrote summary: {result['summary_path']}")
    print(f"Wrote details: {result['details_path']}")
    print(f"Wrote failure counts: {result['failure_counts_path']}")


if __name__ == "__main__":
    main()
