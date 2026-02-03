from __future__ import annotations
import os
import json
import re
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers.utils import logging as hf_logging

from .utils.config import load_yaml, parse_overrides, deep_update
from .utils.seed import set_seed
from .utils.textnorm import normalize_text
from .data.loader import JsonlDataset

hf_logging.set_verbosity_error()

NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))

def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = len(set(p) & set(g))
    if common == 0:
        return 0.0
    precision = common / len(set(p))
    recall = common / len(set(g))
    return 2 * precision * recall / (precision + recall)

def numeric_score(pred: str, gold: str) -> float:
    """
    Simple numeric matching: if both have numbers, compare the first number relative error.
    """
    pn = NUM_RE.findall(pred)
    gn = NUM_RE.findall(gold)
    if not pn or not gn:
        return 0.0
    try:
        p = float(pn[0]); g = float(gn[0])
    except ValueError:
        return 0.0
    denom = max(1e-6, abs(g))
    rel = abs(p - g) / denom
    return float(rel <= 0.2)  # within 20% counts as correct (tunable)

def generate(model, tok, prompts: List[str], max_new_tokens: int, temperature: float, top_p: float) -> List[str]:
    model.eval()
    outs = []
    for p in prompts:
        inputs = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
            )
        text = tok.decode(gen[0], skip_special_tokens=True)
        ans = text.split("A:", 1)[-1].strip() if "A:" in text else text.strip()
        outs.append(ans)
    return outs

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("overrides", nargs="*", help="key=value overrides")
    args = ap.parse_args()

    cfg = deep_update(load_yaml(args.config), parse_overrides(args.overrides))
    set_seed(int(cfg.get("seed", 42)))

    base_name = cfg["model"]["name_or_path"]
    ckpt = cfg["eval"]["checkpoint_path"]
    if not ckpt or ckpt == "REPLACE_ME":
        raise ValueError("Set eval.checkpoint_path to a valid checkpoint directory.")

    tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if cfg["model"].get("torch_dtype", "float16") == "bfloat16" else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=str(cfg["model"].get("attn_implementation", "sdpa")),
    )
    model = PeftModel.from_pretrained(base, ckpt)

    domain = JsonlDataset(cfg["eval"]["domain_path"])
    general = JsonlDataset(cfg["eval"]["general_path"])

    def eval_dataset(ds: JsonlDataset, name: str) -> Dict[str, float]:
        prompts = [x["prompt"] for x in ds]
        golds = [x.get("answer", "") for x in ds]
        preds = generate(
            model, tok, prompts,
            max_new_tokens=int(cfg["eval"]["max_new_tokens"]),
            temperature=float(cfg["eval"]["temperature"]),
            top_p=float(cfg["eval"]["top_p"]),
        )
        em = sum(exact_match(p, g) for p, g in zip(preds, golds)) / max(1, len(ds))
        f1 = sum(token_f1(p, g) for p, g in zip(preds, golds)) / max(1, len(ds))
        num = sum(numeric_score(p, g) for p, g in zip(preds, golds)) / max(1, len(ds))
        return {"exact_match": em, "token_f1": f1, "numeric": num}

    res_domain = eval_dataset(domain, "domain")
    res_general = eval_dataset(general, "general")

    outdir = os.path.join(os.path.dirname(ckpt), "eval")
    os.makedirs(outdir, exist_ok=True)
    out = {"domain": res_domain, "general": res_general}
    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Evaluation saved to:", os.path.join(outdir, "results.json"))
    print("Domain:", res_domain)
    print("General:", res_general)

if __name__ == "__main__":
    main()
