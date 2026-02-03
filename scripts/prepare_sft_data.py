import json, random, os
from pathlib import Path

random.seed(7)

env_fp = Path("data/train/env_sft.jsonl")
teacher_fp = Path("data/train/gen_anchor_teacher.jsonl")  # 可选
out_dir = Path("data/train")
out_dir.mkdir(parents=True, exist_ok=True)

def load_jsonl(fp: Path):
    rows = []
    if not fp.exists():
        return rows
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            p = (o.get("prompt") or "").strip()
            r = (o.get("response") or "").strip()
            if not p or not r:
                continue
            rows.append({"prompt": p, "response": r})
    return rows

env_rows = load_jsonl(env_fp)
teacher_rows = load_jsonl(teacher_fp)

mix = env_rows + teacher_rows
random.shuffle(mix)

print(f"[prepare] env_sft={len(env_rows)}")
print(f"[prepare] gen_anchor_teacher={len(teacher_rows)} (optional)")
print(f"[prepare] total_usable_for_sft={len(mix)}")

# split: 2% dev, 2% test
n = len(mix)
n_dev = int(n * 0.02)
n_test = int(n * 0.02)
dev = mix[:n_dev]
test = mix[n_dev:n_dev+n_test]
train = mix[n_dev+n_test:]

for name, arr in [("sft_train.jsonl", train), ("sft_dev.jsonl", dev), ("sft_test.jsonl", test)]:
    fp = out_dir / name
    with fp.open("w", encoding="utf-8") as w:
        for o in arr:
            w.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"[write] {name} = {len(arr)} -> {fp}")
