
from .validate import validate_response_structured
from .refine import refine_if_needed


def run_refinement_pipeline(df, generate_rewrite_response):
    df = df.copy()

    df["anchor_validation"] = df.apply(
        lambda row: validate_response_structured(
            row["user_message"],
            row["anchor_response"],
        ),
        axis=1,
    )

    df["anchor_passed"] = df["anchor_validation"].apply(lambda x: x["passed"])
    df["anchor_failure_types"] = df["anchor_validation"].apply(
        lambda x: ", ".join(x["failure_types"])
    )

    df["refined_anchor_response"] = df.apply(
        lambda row: refine_if_needed(row, generate_rewrite_response),
        axis=1,
    )

    df["refined_validation"] = df.apply(
        lambda row: validate_response_structured(
            row["user_message"],
            row["refined_anchor_response"],
        ),
        axis=1,
    )

    df["refined_passed"] = df["refined_validation"].apply(lambda x: x["passed"])
    df["refined_failure_types"] = df["refined_validation"].apply(
        lambda x: ", ".join(x["failure_types"])
    )

    return df
