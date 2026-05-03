from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000/chat"


st.set_page_config(
    page_title="Empathy2 Chat",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def post_chat(backend_url, message, session_id=None):
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id

    request = Request(
        backend_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def support_plan_summary(support_plan):
    if not support_plan:
        return "Not available"
    goal = support_plan.get("support_goal", "")
    acts = ", ".join(support_plan.get("response_acts", [])[:3])
    return f"{goal}: {acts}".strip(": ")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

with st.sidebar:
    st.title("Empathy2")
    backend_url = st.text_input("Backend URL", DEFAULT_BACKEND_URL)
    show_debug = st.toggle("Show debug panels", value=False)
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

st.title("Empathy2")
st.caption("A real-time empathy companion with validation and refinement running quietly behind the reply.")

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.write(item["content"])
        if item["role"] == "assistant" and show_debug and item.get("debug"):
            debug = item["debug"]
            with st.expander("Debug: control layer"):
                st.write("Refined:", debug.get("refined"))
                st.write("Failure types:", debug.get("failure_types") or [])
                st.write("Anchor/raw response:")
                st.code(debug.get("anchor_reply") or "Not available")
                st.write("Support plan:")
                st.code(support_plan_summary(debug.get("support_plan")))
                st.write("Validation:")
                st.json(debug.get("validation") or {})
                st.write("Refined validation:")
                st.json(debug.get("refined_validation") or {})

prompt = st.chat_input("Tell me what is weighing on you...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Listening..."):
            try:
                result = post_chat(
                    backend_url=backend_url,
                    message=prompt,
                    session_id=st.session_state.session_id,
                )
            except HTTPError as exc:
                st.error(f"Backend returned HTTP {exc.code}.")
                st.stop()
            except URLError:
                st.error("Could not reach the backend. Start it with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.")
                st.stop()

        reply = result.get("reply", "")
        st.write(reply)
        if show_debug:
            with st.expander("Debug: control layer"):
                st.write("Refined:", result.get("refined"))
                st.write("Failure types:", result.get("failure_types") or [])
                st.write("Anchor/raw response:")
                st.code(result.get("anchor_reply") or "Not available")
                st.write("Support plan:")
                st.code(support_plan_summary(result.get("support_plan")))
                st.write("Validation:")
                st.json(result.get("validation") or {})
                st.write("Refined validation:")
                st.json(result.get("refined_validation") or {})

    st.session_state.session_id = result.get("session_id")
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "debug": {
                "anchor_reply": result.get("anchor_reply") or result.get("raw_reply"),
                "support_plan": result.get("support_plan"),
                "validation": result.get("validation"),
                "refined_validation": result.get("refined_validation"),
                "refined": result.get("refined"),
                "failure_types": result.get("failure_types"),
            },
        }
    )
