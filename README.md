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
    "total_ms": 881.44
  },
  "safety": {"crisis_detected": false}
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
