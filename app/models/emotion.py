from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any


GOEMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

DEAD_CLASS_MERGES = {
    "grief": "sadness",
    "pride": "admiration",
    "relief": "joy",
}


def _confidence_level(probability: float) -> str:
    if probability >= 0.50:
        return "high"
    if probability >= 0.35:
        return "medium"
    return "low"


def _merge_dead_class_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    merged = dict(probabilities)
    for source, target in DEAD_CLASS_MERGES.items():
        merged[target] += merged[source]
        merged[source] = 0.0
    return {label: probability for label, probability in merged.items() if label not in DEAD_CLASS_MERGES}


def _top_k(probabilities: dict[str, float], k: int = 3) -> list[dict[str, float | str]]:
    return [
        {"label": label, "confidence": float(probability)}
        for label, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:k]
    ]


def _as_ranked_payload(top: list[dict[str, float | str]]) -> dict[str, dict[str, float | str]]:
    padded = [*top]
    while len(padded) < 3:
        padded.append({"label": "neutral", "confidence": 0.0})
    return {
        "primary": padded[0],
        "secondary": padded[1],
        "tertiary": padded[2],
    }


class EmotionClassifier:
    def __init__(
        self,
        model_dir: Path,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.max_length = max_length
        self.device_name = device
        self._lock = RLock()
        self._loaded = False
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._device: Any = None

    @property
    def source(self) -> str:
        return f"local:{self.model_dir}"

    def _load(self) -> None:
        if self._loaded:
            return

        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer

        class GoEmotionsProductionClassifier(nn.Module):
            def __init__(self, model_root: Path, dropout: float) -> None:
                super().__init__()
                self.encoder = AutoModel.from_pretrained(model_root / "encoder")
                hidden_size = self.encoder.config.hidden_size
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(hidden_size, len(GOEMOTIONS_LABELS))

            def forward(self, **inputs: Any) -> Any:
                outputs = self.encoder(**inputs)
                pooled = self.dropout(outputs.last_hidden_state[:, 0])
                return self.classifier(pooled)

        self._validate_files()
        metadata = self._load_metadata()
        dropout = float(metadata.get("dropout", 0.2))
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "tokenizer", use_fast=True)
        state = torch.load(self.model_dir / "head_a.pt", map_location="cpu")
        model = GoEmotionsProductionClassifier(self.model_dir, dropout=dropout)
        model.classifier.load_state_dict(state["classifier"])

        self._device = torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model = model.to(self._device)
        self._model.eval()
        self._tokenizer = tokenizer
        self._torch = torch
        self._loaded = True

    def _validate_files(self) -> None:
        required = [
            self.model_dir / "encoder",
            self.model_dir / "tokenizer",
            self.model_dir / "head_a.pt",
            self.model_dir / "metadata.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Emotion classifier files are missing: {'; '.join(missing)}")

    def _load_metadata(self) -> dict[str, Any]:
        with (self.model_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def predict(self, text: str, speaker: str = "User", top_k: int = 3) -> dict[str, Any]:
        with self._lock:
            self._load()
            encoded = self._tokenizer(
                f"[TARGET] {speaker}: {text}",
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.no_grad():
                logits = self._model(**encoded)[0].detach().cpu().float()
            probabilities = self._torch.softmax(logits, dim=-1).tolist()

        probability_by_label = {
            label: float(probability)
            for label, probability in zip(GOEMOTIONS_LABELS, probabilities)
        }
        merged = _merge_dead_class_probabilities(probability_by_label)
        top = _top_k(merged, top_k)
        ranked = _as_ranked_payload(top)
        return {
            **ranked,
            "top_emotions": top,
            "confidence_level": _confidence_level(float(top[0]["confidence"]) if top else 0.0),
        }
