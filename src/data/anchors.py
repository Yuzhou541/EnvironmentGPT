from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
import random

DEFAULT_ANCHORS = [
    "You are a helpful assistant.\nQ: Explain the difference between correlation and causation.\nA:",
    "You are a helpful assistant.\nQ: Summarize the key idea of gradient descent.\nA:",
    "You are a helpful assistant.\nQ: What is the purpose of a unit test in software engineering?\nA:",
    "You are a helpful assistant.\nQ: Give three tips for writing clear technical documentation.\nA:",
    "You are a helpful assistant.\nQ: Explain what a queue is in data structures.\nA:",
    "You are a helpful assistant.\nQ: What is the central limit theorem (high-level)?\nA:",
]

def build_general_anchors(out_path: str, n: int = 200, seed: int = 42) -> None:
    random.seed(seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    for i in range(n):
        p = random.choice(DEFAULT_ANCHORS)
        items.append({"id": f"gen-{i:05d}", "prompt": p, "answer": ""})

    with out.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
