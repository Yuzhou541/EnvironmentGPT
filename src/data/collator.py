from __future__ import annotations
from typing import List, Dict, Any, Optional
import torch

class SFTCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in batch]
        answers = [x.get("answer", "") for x in batch]

        texts = []
        for p, a in zip(prompts, answers):
            if a:
                texts.append(p.rstrip() + "\nA: " + a.strip())
            else:
                texts.append(p.rstrip())

        enc = self.tok(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        # If no answer (anchors), keep labels = input_ids (KD/geom uses forward logits/hidden).
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}
