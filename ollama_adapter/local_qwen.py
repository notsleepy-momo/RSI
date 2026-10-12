"""In-process Hugging Face backend for Qwen3.5 (no Ollama/server needed)."""

from __future__ import annotations

import time
from typing import Any


class LocalQwen:
    def __init__(self, model_name_or_path: str, device_map: str = "auto", dtype: str = "auto"):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3.5 local inference requires a CUDA GPU, but torch.cuda.is_available() is false")
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        kwargs: dict[str, Any] = {"device_map": device_map}
        if dtype != "auto":
            kwargs["dtype"] = getattr(torch, dtype)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name_or_path, low_cpu_mem_usage=True, **kwargs
        ).eval()

    def chat(self, prompt: str, max_new_tokens: int, temperature: float, seed: int) -> tuple[str, float]:
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=False,
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation.update(temperature=temperature, top_p=0.95)
        started = time.time()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        elapsed = time.time() - started
        new_tokens = output[0, inputs["input_ids"].shape[-1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip(), elapsed
