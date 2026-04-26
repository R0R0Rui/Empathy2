from __future__ import annotations

from threading import RLock
from typing import Any

import torch
from unsloth import FastLanguageModel


class UnslothGenerator:
    def __init__(
        self,
        base_model: str,
        adapter_model: str,
        max_new_tokens: int = 260,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        self.base_model = base_model
        self.adapter_model = adapter_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._lock = RLock()
        self._loaded = False
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None

    @property
    def adapter_source(self) -> str:
        return f"huggingface:{self.adapter_model}"

    def _load(self) -> None:
        if self._loaded:
            return

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        model.load_adapter(self.adapter_model)
        FastLanguageModel.for_inference(model)

        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        self._loaded = True

    def generate(self, messages: list[dict[str, str]]) -> str:
        with self._lock:
            self._load()
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)
            with self._torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            generated = outputs[0][inputs["input_ids"].shape[-1] :]
            response = self._tokenizer.decode(generated, skip_special_tokens=True)
            return response.strip()
