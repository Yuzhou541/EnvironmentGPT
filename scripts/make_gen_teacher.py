
import json, random, argparse, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--infile", type=str, default="data/train/gen_anchor.jsonl")
parser.add_argument("--outfile", type=str, default="data/train/gen_anchor_teacher.jsonl")
parser.add_argument("--num", type=int, default=20000)
parser.add_argument("--max_new_tokens", type=int, default=96)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--every", type=int, default=50)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

random.seed(args.seed)

inp = Path(args.infile)
out = Path(args.outfile)
out.parent.mkdir(parents=True, exist_ok=True)

# 读 prompts
prompts = []
with inp.open("r", encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        p = (o.get("prompt") or "").strip()
        if p:
            prompts.append(p)

random.shuffle(prompts)
prompts = prompts[: min(args.num, len(prompts))]
print(f"[load] prompts={len(prompts)} from {inp}", flush=True)

# resume：跳过已有行数
start = 0
if args.resume and out.exists():
    with out.open("r", encoding="utf-8") as f:
        for _ in f:
            start += 1
    print(f"[resume] existing_lines={start}", flush=True)

prompts = prompts[start:]

# 性能设置
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

print("[init] loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

print("[init] loading model (this can take a while)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    device_map="auto",
)
model.eval()
print("[init] model ready.", flush=True)

def gen_one(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful general-purpose assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    gen = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()

ok = 0
t0 = time.time()

with out.open("a", encoding="utf-8") as w:
    for i, prompt in enumerate(prompts, 1):
        try:
            resp = gen_one(prompt)
            w.write(json.dumps({"type": "gen", "prompt": prompt, "response": resp}, ensure_ascii=False) + "\n")
        except Exception as e:
            w.write(json.dumps({"type": "gen", "prompt": prompt, "response": "", "error": repr(e)}, ensure_ascii=False) + "\n")

        w.flush()
        ok += 1

        if i % args.every == 0:
            dt = time.time() - t0
            speed = ok / max(dt, 1e-6)
            print(f"[progress] wrote_total≈{start+ok} (+{ok}), speed={speed:.2f} samples/s", flush=True)

print(f"[done] wrote_now={ok}, wrote_total≈{start+ok} -> {out}", flush=True)
