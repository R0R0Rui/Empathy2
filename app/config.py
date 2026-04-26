from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")
_load_dotenv(Path(".env"))


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _path_from_env(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default)).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT

    emotion_model_dir: Path = _path_from_env(
        "EMPATHY_EMOTION_MODEL_DIR",
        "models/goemotions-production",
    )
    emotion_max_length: int = _int_from_env("EMPATHY_EMOTION_MAX_LENGTH", 512)
    emotion_model_repo: str = os.getenv(
        "EMPATHY_EMOTION_MODEL_REPO",
        "JamieYCR/goemotions-chatbot-emotion-classifier",
    )
    emotion_model_revision: str = os.getenv("EMPATHY_EMOTION_MODEL_REVISION", "main")
    auto_download_models: bool = _bool_from_env("EMPATHY_AUTO_DOWNLOAD_MODELS", True)

    llm_base_model: str = os.getenv(
        "EMPATHY_LLM_BASE_MODEL",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    )
    llm_adapter_model: str = os.getenv(
        "EMPATHY_LLM_ADAPTER_MODEL",
        "JamieYCR/qwen25-7b-empathy",
    )
    max_new_tokens: int = _int_from_env("EMPATHY_MAX_NEW_TOKENS", 260)
    temperature: float = _float_from_env("EMPATHY_TEMPERATURE", 0.7)
    top_p: float = _float_from_env("EMPATHY_TOP_P", 0.9)

    history_limit: int = _int_from_env("EMPATHY_HISTORY_LIMIT", 10)
    max_turns_per_session: int = _int_from_env("EMPATHY_MAX_TURNS_PER_SESSION", 80)

    def missing_local_model_files(self) -> list[str]:
        missing: list[str] = []
        emotion_files = [
            self.emotion_model_dir / "encoder",
            self.emotion_model_dir / "tokenizer",
            self.emotion_model_dir / "head_a.pt",
            self.emotion_model_dir / "metadata.json",
        ]
        for path in emotion_files:
            if not path.exists():
                missing.append(str(path))
        return missing

    def validate_local_model_files(self) -> None:
        missing = self.missing_local_model_files()
        if missing:
            formatted = "; ".join(missing)
            raise RuntimeError(f"Required local model files are missing: {formatted}")


settings = Settings()
