from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings


def main() -> int:
    print(f"emotion classifier path: {settings.emotion_model_dir}")
    print(f"qwen base model: {settings.llm_base_model}")
    print(f"qwen lora adapter model: {settings.llm_adapter_model}")

    missing = settings.missing_local_model_files()
    if missing:
        print("\nMissing required local files:")
        for path in missing:
            print(f"- {path}")
        return 1

    print("\nOK: backend is configured for your local classifier and Hugging Face Qwen2.5 adapter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
