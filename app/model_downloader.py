"""Helpers for fetching model artifacts that are too large to commit to git.

Currently used to pull the GoEmotions production classifier from Hugging Face
on first run. The download is idempotent: if every required file already
exists locally, ``ensure_emotion_classifier`` is a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


DEFAULT_EMOTION_REPO = "JamieYCR/goemotions-chatbot-emotion-classifier"
EMOTION_REQUIRED_PATHS: tuple[str, ...] = (
    "encoder",
    "tokenizer",
    "head_a.pt",
    "metadata.json",
)


def files_present(target_dir: Path, required: Iterable[str] = EMOTION_REQUIRED_PATHS) -> bool:
    return all((target_dir / name).exists() for name in required)


def download_emotion_classifier(
    target_dir: Path,
    repo_id: str = DEFAULT_EMOTION_REPO,
    revision: str = "main",
    force: bool = False,
) -> Path:
    """Mirror ``repo_id`` into ``target_dir``.

    Returns the resolved target directory. Raises ``ImportError`` if
    ``huggingface_hub`` is not installed, or ``RuntimeError`` if the download
    finishes but expected files are still missing.
    """
    target_dir = target_dir.expanduser().resolve()
    if not force and files_present(target_dir):
        return target_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for model download. Install with: pip install huggingface_hub"
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )

    missing = [name for name in EMOTION_REQUIRED_PATHS if not (target_dir / name).exists()]
    if missing:
        raise RuntimeError(
            f"Download from {repo_id} finished but required items are missing: {missing}"
        )
    return target_dir


def ensure_emotion_classifier(
    target_dir: Path,
    repo_id: str = DEFAULT_EMOTION_REPO,
    revision: str = "main",
    log: bool = True,
) -> Path:
    """Idempotent variant: download only if files are missing.

    Designed for app startup. If everything is already there, returns
    immediately without touching the network.
    """
    target_dir = target_dir.expanduser().resolve()
    if files_present(target_dir):
        return target_dir

    if log:
        print(
            f"[model_downloader] Emotion classifier missing at {target_dir}; "
            f"downloading {repo_id}@{revision} ...",
            file=sys.stderr,
        )
    return download_emotion_classifier(target_dir, repo_id=repo_id, revision=revision, force=False)
