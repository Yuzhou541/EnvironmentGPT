import os, json, argparse, time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_prompts(path, max_n):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "prompt" in obj:
                prompts.append(obj["prompt"])
            if len(prompts) >= max_n:
                break
    return prompts

def build_inputs(tokenizer, prompts, max_len):
    # chat template -> single text per prompt
    texts = []
    for p in prompts:
        t = tokenizer.apply_chat_template(
            [{"role":"user","content":p}],
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(t)

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )
    return enc

@torch.no_grad()
def forward_logits(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    # shift for next-token distribution
    logits = out.logits[:, :-1, :].float()          # (B, T-1, V)
    am = attention_mask[:, 1:].float()             # align with shifted tokens (exclude first)
    return logits, am

@torch.no_grad()
def compute_kl_token_avg(logits0, logits1, mask):
    # KL(base||adapt) = sum p0*(logp0-logp1)
    logp0 = F.log_softmax(logits0, dim=-1)
    logp1 = F.log_softmax(logits1, dim=-1)
    p0 = logp0.exp()
    kl_tok = (p0 * (logp0 - logp1)).sum(dim=-1)    # (B, T-1)
    # masked token-average
    denom = mask.sum().clamp_min(1.0)
    return (kl_tok * mask).sum() / denom, int(denom.item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_prompts", type=int, default=300)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=2)
    args = ap.parse_args()

    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, args.adapter, device_map="auto")
    model.eval()

    prompts = load_prompts(args.prompts_jsonl, args.max_prompts)
    if not prompts:
        raise SystemExit("[ERR] no prompts loaded")

    # adapter disable context (robust)
    disable_ctx = model.disable_adapter if hasattr(model, "disable_adapter") else None

    total_kl = 0.0
    total_tok = 0

    # for sanity: keep first few per-prompt KL (computed in batch but we can approximate by splitting)
    preview = []

    bs = max(1, args.batch_size)
    for start in range(0, len(prompts), bs):
        batch_prompts = prompts[start:start+bs]
        enc = build_inputs(tok, batch_prompts, args.max_seq_len)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        # base logits (adapter OFF)
        ctx = disable_ctx() if disable_ctx is not None else nullcontext()
        with ctx:
            logits0, mask = forward_logits(model, input_ids, attention_mask)

        # adapt logits (adapter ON)
        logits1, mask2 = forward_logits(model, input_ids, attention_mask)

        # mask2 should equal mask
        kl_avg, n_tok = compute_kl_token_avg(logits0, logits1, mask)
        total_kl += float(kl_avg.item()) * n_tok
        total_tok += n_tok

        # lightweight preview: per batch KL (not per single prompt)
        if len(preview) < 10:
            preview.append({
                "batch_start": start,
                "batch_size": len(batch_prompts),
                "kl_token_avg_batch": float(kl_avg.item()),
                "tokens_batch": n_tok
            })

    summary = {
        "N": len(prompts),
        "max_seq_len": args.max_seq_len,
        "batch_size": bs,
        "kl_token_avg": total_kl / max(total_tok, 1),
        "total_tokens": int(total_tok),
        "adapter": args.adapter,
        "elapsed_sec": round(time.time() - t0, 3),
    }

    out = {"summary": summary, "preview": preview}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[OK]", args.out_json)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
