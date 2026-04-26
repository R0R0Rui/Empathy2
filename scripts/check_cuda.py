from __future__ import annotations

import importlib.util
import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch is not installed in this Python environment.")
        return 1

    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"gpu count: {torch.cuda.device_count()}")
        print(f"gpu 0: {torch.cuda.get_device_name(0)}")
    else:
        print("gpu 0: unavailable")

    unsloth_spec = importlib.util.find_spec("unsloth")
    print(f"unsloth installed: {unsloth_spec is not None}")
    return 0 if torch.cuda.is_available() and unsloth_spec is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
