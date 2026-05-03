from pathlib import Path

import pandas as pd
import streamlit as st


CSV_CANDIDATES = [
    Path("data/results/final_with_refinement.csv"),
    Path("data/results/backend_outputs.csv"),
    Path("data/presentation_refinement_table.csv"),
]


st.set_page_config(
    page_title="Empathy2 Quiet Companion",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def read_csv(path):
    return pd.read_csv(path).fillna("")


def choose_source():
    for path in CSV_CANDIDATES:
        if path.exists():
            return path
    st.error(
        "No demo CSV found. Expected data/results/final_with_refinement.csv, "
        "data/results/backend_outputs.csv, or data/presentation_refinement_table.csv."
    )
    st.stop()


def normalize_dataframe(df):
    df = df.copy()

    if "scenario_type" not in df.columns:
        if "type" in df.columns:
            df["scenario_type"] = df["type"]
        elif "scenario_tier" in df.columns:
            df["scenario_type"] = df["scenario_tier"]
        else:
            df["scenario_type"] = "unknown"

    if "case_id" not in df.columns:
        df["case_id"] = [f"case_{index + 1}" for index in range(len(df))]

    if "anchor_response" not in df.columns:
        df["anchor_response"] = df["reply"] if "reply" in df.columns else ""

    for column in [
        "user_message",
        "baseline_response",
        "refined_anchor_response",
        "anchor_failure_types",
        "refined_failure_types",
        "refined_passed",
        "primary_emotion",
        "secondary_emotion",
        "intent",
        "scenario_tier",
        "support_need",
        "safety_flag",
        "is_implicit",
        "emotion_intensity",
        "crisis_detected",
    ]:
        if column not in df.columns:
            df[column] = ""

    return df


def value(row, column, default=""):
    if column not in row.index:
        return default
    item = row.get(column, default)
    if pd.isna(item) or str(item).strip() == "":
        return default
    return item


def pretty(value_text):
    text = str(value_text or "").strip()
    if not text:
        return "not available"
    return text.replace("_", " ")


def snippet(text, limit=76):
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def split_failures(text):
    if not text:
        return []
    normalized = str(text).replace(";", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def companion_response(row):
    refined = value(row, "refined_anchor_response")
    if refined:
        return refined
    return value(row, "anchor_response", "No response available for this run.")


def show_sidebar(df, source_path):
    st.sidebar.title("Quiet sessions")
    st.sidebar.caption(f"Source: {source_path}")

    scenario_values = sorted(
        scenario for scenario in df["scenario_type"].astype(str).unique() if scenario
    )
    scenario_options = ["All"] + scenario_values
    selected_scenario = st.sidebar.radio(
        "Choose a path",
        scenario_options,
        index=0,
    )

    filtered = df
    if selected_scenario != "All":
        filtered = df[df["scenario_type"].astype(str) == selected_scenario]

    if filtered.empty:
        st.warning("No rows match the selected path.")
        st.stop()

    labels = []
    label_to_index = {}
    for index, row in filtered.reset_index().iterrows():
        label = f"{row['case_id']} · {snippet(row.get('user_message', ''))}"
        labels.append(label)
        label_to_index[label] = row["index"]

    selected_label = st.sidebar.radio(
        "Choose a moment",
        labels,
        index=0,
    )
    st.sidebar.divider()
    st.sidebar.caption("This viewer is offline. It reads saved CSV outputs only.")

    return df.loc[label_to_index[selected_label]]


def show_sensed_context(row):
    fields = [
        ("Emotion", value(row, "primary_emotion")),
        ("Second layer", value(row, "secondary_emotion")),
        ("Need", value(row, "support_need")),
        ("Path", value(row, "scenario_tier") or value(row, "scenario_type")),
        ("Safety", value(row, "safety_flag") or value(row, "crisis_detected")),
        ("Implicit", value(row, "is_implicit")),
        ("Intensity", value(row, "emotion_intensity")),
    ]
    fields = [(label, item) for label, item in fields if str(item).strip()]

    if not fields:
        st.info("Understanding output not available for this run.")
        return

    cols = st.columns(4)
    for index, (label, item) in enumerate(fields):
        with cols[index % 4]:
            with st.container(border=True):
                st.caption(label)
                st.write(pretty(item))


def show_response_history(row):
    tabs = st.tabs(["Final companion reply", "Earlier drafts", "Reliability check"])

    with tabs[0]:
        st.chat_message("assistant").write(companion_response(row))

    with tabs[1]:
        baseline = value(row, "baseline_response")
        anchor = value(row, "anchor_response")
        refined = value(row, "refined_anchor_response")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.caption("Baseline")
            st.info(baseline or "Not available for this run.")
        with col_b:
            st.caption("Anchor draft")
            st.warning(anchor or "Not available for this run.")
        with col_c:
            st.caption("Refined final")
            st.success(refined or "Not available for this run.")

    with tabs[2]:
        anchor_failures = split_failures(value(row, "anchor_failure_types"))
        refined_failures = split_failures(value(row, "refined_failure_types"))
        refined_passed = str(value(row, "refined_passed")).lower() in {
            "true",
            "1",
            "yes",
        }

        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Before refinement")
            if anchor_failures:
                for failure in anchor_failures:
                    st.error(pretty(failure), icon="!")
            else:
                st.success("No failures listed.", icon="✓")
        with col_b:
            st.caption("After refinement")
            if refined_failures:
                for failure in refined_failures:
                    st.error(pretty(failure), icon="!")
            elif refined_passed:
                st.success("Passed validation.", icon="✓")
            else:
                st.success("No failures listed.", icon="✓")


source_path = choose_source()
df = normalize_dataframe(read_csv(source_path))
row = show_sidebar(df, source_path)

st.title("Empathy2 Quiet Companion")
st.caption("An offline glimpse of a companion that listens first, then responds with care.")

if source_path.name == "backend_outputs.csv":
    st.warning(
        "You are viewing raw backend output. Run `python -m scripts.run_full_eval_pipeline "
        "--skip-backend --raw-output data/results/backend_outputs.csv` to generate the "
        "full refined companion view."
    )

st.divider()

left_col, right_col = st.columns([1.05, 1.35], gap="large")

with left_col:
    st.subheader("The moment")
    with st.container(border=True):
        st.caption("What the person brought in")
        st.chat_message("user").write(value(row, "user_message", "No user message available."))

    st.subheader("What was sensed")
    show_sensed_context(row)

with right_col:
    st.subheader("A steadier reply")
    with st.container(border=True):
        st.caption("Designed to feel like a calm guide, not a diagnostic panel")
        st.chat_message("assistant").write(companion_response(row))

    st.caption(
        "From expressive but unstable empathy to a response that is calmer, safer, "
        "and more emotionally attuned."
    )

st.divider()

with st.expander("Open the evaluation view"):
    show_response_history(row)

raw = value(row, "understanding_json")
if raw:
    with st.expander("Raw understanding JSON"):
        st.code(raw, language="json")
