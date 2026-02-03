import os, json, argparse, math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def pick_device(model):
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.inference_mode()
def forward_logits(model, inputs):
    return model(**inputs, use_cache=False).logits

@torch.inference_mode()
def kl_topk(teacher_logits, student_logits, attn_mask, k=50):
    """
    KL( softmax(t_topk) || softmax(s_on_same_topk_support) ) averaged over valid tokens.
    teacher_logits/student_logits: [B, T, V]
    attn_mask: [B, T]
    """
    # shift: next-token prediction positions
    t = teacher_logits[:, :-1, :]
    s = student_logits[:, :-1, :]
    m = attn_mask[:, 1:]  # align with next-token positions

    # topk on teacher
    vals, idx = torch.topk(t, k=min(k, t.size(-1)), dim=-1)  # [B,T-1,K]
    t_logp = F.log_softmax(vals, dim=-1)                     # [B,T-1,K]
    t_p = t_logp.exp()

    # gather student logits on teacher topk support
    s_sel = torch.gather(s, dim=-1, index=idx)               # [B,T-1,K]
    s_logp = F.log_softmax(s_sel, dim=-1)

    kl = (t_p * (t_logp - s_logp)).sum(dim=-1)               # [B,T-1]

    # mask and average
    kl = kl * m
    denom = m.sum().clamp_min(1)
    return (kl.sum() / denom).item()

@torch.inference_mode()
def argmax_agree_rate(teacher_logits, student_logits, attn_mask):
    t = teacher_logits[:, :-1, :].argmax(dim=-1)
    s = student_logits[:, :-1, :].argmax(dim=-1)
    m = attn_mask[:, 1:].bool()
    agree = ((t == s) & m).sum().item()
    total = m.sum().item()
    return (agree / total) if total > 0 else 0.0

def load_prompts(path, n=None):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"] if "prompt" in obj else obj.get("text", ""))
            if n and len(prompts) >= n:
                break
    return prompts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base, args.adapter, device_map="auto")
    model.eval()

    dev = pick_device(model)

    prompts = load_prompts(args.anchors)
    kl_list = []
    agree_list = []

    # try using disable_adapter for teacher logits; fallback: not available -> load separate teacher
    has_disable = hasattr(model, "disable_adapter")

    teacher_model = None
    if not has_disable:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        teacher_model.eval()

    for p in prompts:
        chat = tok.apply_chat_template(
            [{"role":"user","content":p}],
            tokenize=False,
            add_generation_prompt=True
        )
        enc = tok(
            chat,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_len,
            padding=False
        )
        enc = {k: v.to(dev) for k, v in enc.items()}

        if has_disable:
            with model.disable_adapter():
                t_logits = forward_logits(model, enc)
        else:
            # teacher_model might be sharded; inputs should be on first non-meta device
            tdev = pick_device(teacher_model)
            enc_t = {k: v.to(tdev) for k, v in enc.items()}
            t_logits = forward_logits(teacher_model, enc_t).to(dev)

        s_logits = forward_logits(model, enc)

        attn = enc.get("attention_mask", torch.ones_like(enc["input_ids"], device=dev))
        klv = kl_topk(t_logits, s_logits, attn, k=args.topk)
        agr = argmax_agree_rate(t_logits, s_logits, attn)

        kl_list.append(klv)
        agree_list.append(agr)

    import statistics as st
    out = {
        "N": len(kl_list),
        "kl_topk": args.topk,
        "kl_topk_mean": float(st.mean(kl_list)) if kl_list else None,
        "kl_topk_std": float(st.pstdev(kl_list)) if len(kl_list) > 1 else 0.0,
        "argmax_agree_mean": float(st.mean(agree_list)) if agree_list else None,
        "argmax_agree_std": float(st.pstdev(agree_list)) if len(agree_list) > 1 else 0.0,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
