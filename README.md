# Empathy AI Backend

FastAPI backend for the two-stage empathy agent:

1. Classify emotion with the GoEmotions classifier at `models/goemotions-production` (downloaded from Hugging Face — see Setup below).
2. Build an empathy prompt with recent in-memory session history.
3. Generate a response with `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` plus the Hugging Face PEFT LoRA adapter `JamieYCR/qwen25-7b-empathy`.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` pins CUDA 12.8 PyTorch `2.10.0+cu128`, because current Unsloth requires `torch<2.11`. If CUDA disappears after installing packages, rerun:

```powershell
pip install --force-reinstall `
  torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128
```

### The emotion classifier (auto-downloaded)

The trained GoEmotions classifier is hosted on Hugging Face at
[`JamieYCR/goemotions-chatbot-emotion-classifier`](https://huggingface.co/JamieYCR/goemotions-chatbot-emotion-classifier)
and is **not** committed to this repo (~1.7 GB). The app auto-downloads it
into `models/goemotions-production/` on first startup if the files are
missing — no manual step needed.

To pre-warm the download (e.g. for offline dev or CI), or to force a refresh:

```powershell
python scripts/download_models.py            # idempotent
python scripts/download_models.py --force    # re-download
python scripts/download_models.py --repo-id user/repo   # different mirror
```

To disable auto-download (e.g. air-gapped environments), set
`EMPATHY_AUTO_DOWNLOAD_MODELS=false` in `.env`. Startup will then fail
loudly if the files are missing, instead of fetching them.

The server validates those classifier files during startup. Check the local classifier path without importing the ML stack:

```powershell
python scripts/check_real_models.py
```

Then run:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open the chat page:

```text
http://127.0.0.1:8000/
```

If `/chat` says Unsloth cannot find a torch accelerator, run this inside the same activated `.venv`:

```powershell
python scripts/check_cuda.py
```

## API

`POST /chat`

```json
{
  "session_id": "optional-existing-session",
  "message": "What is generalized anxiety disorder?"
}
```

Returns:

```json
{
  "session_id": "abc-123",
  "reply": "That is a thoughtful question...",
  "emotions": {
    "primary": {"label": "neutral", "confidence": 0.41},
    "secondary": {"label": "curiosity", "confidence": 0.32},
    "tertiary": {"label": "confusion", "confidence": 0.14}
  },
  "performance": {
    "classifier_ms": 38.72,
    "generation_ms": 842.31,
    "understanding_ms": 12.94,
    "control_ms": 2.11,
    "total_ms": 881.44
  },
  "safety": {"crisis_detected": false},
  "anchor_reply": "Original model response before validation/refinement.",
  "support_plan": {"support_goal": "validate and respond to fear"},
  "validation": {"passed": true, "failure_types": [], "severity": "none", "rewrite_needed": false, "notes": []},
  "refined": false,
  "failure_types": []
}
```

Session memory is a Python dictionary. Restarting the process clears it. You can also clear it manually:

```powershell
curl -X POST http://127.0.0.1:8000/sessions/reset-all
```

The `performance` object reports request-local timings in milliseconds. The first `/chat` call includes model load time because models are loaded lazily; send one warm-up message before comparing steady-state latency.

You can run a quick local benchmark after starting the server:

```powershell
python scripts/benchmark_chat.py --runs 10 --warmup 1
```

## Notes

The backend loads the base model from `EMPATHY_LLM_BASE_MODEL`, currently `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`, and applies the adapter from `EMPATHY_LLM_ADAPTER_MODEL`, currently `JamieYCR/qwen25-7b-empathy`.

The default `.env` is tuned for your RTX 4090 path: Unsloth 4-bit Qwen2.5-7B, `bfloat16`, and your Hugging Face LoRA adapter. There is no retrieval path and no stub inference path in this project right now.

## Evaluation

We implement a failure-aware refinement pipeline:

1. Detect failure types (validation rules)
2. Map failures to repair strategies
3. Trigger conditional rewrite
4. Re-validate improved responses

Example outputs are shown in `data/presentation_refinement_table.csv`.

## Evaluation and Reliability Layer

The evaluation layer is an offline reliability and refinement layer. It does
not replace the Qwen2.5 + LoRA generator, the emotion classifier, the
understanding layer, or the FastAPI backend.

Expanded evaluation cases live in `data/eval_cases/`, and the validator in
`evaluation/validate.py` detects response failure types such as generic
empathy, AI self-experience claims, missed self-dismissal, unsafe crisis
handling, over-advice, and ignored user boundaries. This supports iterative,
failure-driven improvement while keeping teammate-owned model and backend
modules unchanged.

To evaluate generated comparison outputs:

```powershell
python -m evaluation.run_evaluation --input data/results/final_comparison.csv --output data/results/eval_summary.csv
```

The command writes a summary CSV, a detailed per-response CSV, and failure-type
counts. It works whether or not `refined_anchor_response` is present in the
input file.

To generate fresh backend outputs from the offline eval cases, start the local
backend first:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then run the batch case runner:

```powershell
python -m scripts.run_backend_cases
```

For a quick smoke test, limit the number of cases:

```powershell
python -m scripts.run_backend_cases --limit 3
```

Finally, evaluate the captured backend outputs:

```powershell
python -m evaluation.run_evaluation --input data/results/backend_outputs.csv --output data/results/eval_summary.csv
```

## Runtime Control Architecture

The live `/chat` path now runs the same reliability controls used by the
offline evaluation pipeline:

```text
user message
-> emotion classifier
-> understanding layer
-> support plan
-> initial generator response
-> validation
-> conditional refinement
-> final reply
```

The user-facing `reply` is always the final natural response. Debug metadata
such as `anchor_reply`, `support_plan`, `validation`, `refined`, and
`failure_types` is returned separately so clients can inspect the control layer
without leaking internal notes into the conversation.

## Support Planning Layer

`app/planning/` contains a CoCoMo-inspired support planning layer that converts
understanding JSON into a structured `SupportPlan`. The plan describes the
support goal, response acts, constraints, safety notes, and repair priorities
used by runtime validation/refinement.

To inspect example plans:

```powershell
python -m scripts.demo_support_plan
```

## Offline Demo Viewer

`ui/offline_demo.py` is a lightweight Streamlit viewer for previously generated
CSV outputs. It does not load ML models, call the backend, or modify runtime
application code. By default it reads `data/results/backend_outputs.csv`; if
that file is missing, it falls back to `data/presentation_refinement_table.csv`.

Run it with:

```powershell
streamlit run ui/offline_demo.py
```

## Chatbot UI

Start the backend:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then launch the live chatbot UI:

```powershell
streamlit run ui/chat_demo.py
```

The UI shows a clean chat surface by default. Optional debug panels can display
the raw anchor response, support plan summary, validation result, and whether
refinement was triggered.

## One-command Offline Evaluation Pipeline

`scripts/run_full_eval_pipeline.py` runs the offline workflow end to end. It
can call an already-running local backend over the eval cases, normalize the
raw CSV, derive an offline support plan when understanding fields are present,
validate the anchor response, apply deterministic rule-based refinement when
needed, re-validate, and write final plus summary CSV outputs.

Start the backend in one terminal:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then run a small end-to-end sample:

```powershell
python -m scripts.run_full_eval_pipeline --limit 10
```

If backend outputs already exist, skip the backend call:

```powershell
python -m scripts.run_full_eval_pipeline --skip-backend --raw-output data/results/backend_outputs.csv
```

Outputs are written to:

```text
data/results/final_with_refinement.csv
data/results/eval_summary.csv
data/results/eval_summary_details.csv
data/results/eval_summary_failure_counts.csv
```
