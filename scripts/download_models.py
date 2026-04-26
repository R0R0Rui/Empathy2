"""Download the GoEmotions production classifier from Hugging Face.

Pulls the classifier repo (default: ``JamieYCR/goemotions-chatbot-emotion-classifier``)
into the local ``models/goemotions-production`` directory expected by
``app.config.settings.emotion_model_dir``.

The app also auto-downloads on startup if files are missing
(``EMPATHY_AUTO_DOWNLOAD_MODELS=true`` by default), so this script is mainly
useful for pre-warming on a fresh clone or for forcing a refresh.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --repo-id user/repo --revision main
    python scripts/download_models.py --force  # re-download even if files exist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.model_downloader import (
    DEFAULT_EMOTION_REPO,
    download_emotion_classifier,
    files_present,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the GoEmotions classifier from Hugging Face.")
    parser.add_argument(
        "--repo-id",
        default=settings.emotion_model_repo or DEFAULT_EMOTION_REPO,
        help=f"Hugging Face repo id (default: {settings.emotion_model_repo or DEFAULT_EMOTION_REPO}).",
    )
    parser.add_argument(
        "--revision",
        default=settings.emotion_model_revision,
        help="Git revision (branch, tag, or commit) to download (default: main).",
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="Override the local target directory (default: settings.emotion_model_dir).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if all required files are already present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_dir = Path(args.target_dir).expanduser() if args.target_dir else settings.emotion_model_dir
    target_dir = target_dir if target_dir.is_absolute() else (ROOT / target_dir)

    if files_present(target_dir) and not args.force:
        print(f"Classifier already present at {target_dir}. Use --force to re-download.")
        return 0

    print(f"Downloading {args.repo_id}@{args.revision} into {target_dir} ...")
    try:
        download_emotion_classifier(
            target_dir,
            repo_id=args.repo_id,
            revision=args.revision,
            force=args.force,
        )
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"OK: classifier ready at {target_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
