import json, random
from pathlib import Path

random.seed(7)
env = Path("data/train/env_sft.jsonl")
gen = Path("data/train/gen_anchor.jsonl")

mix = []
for fp in [env, gen]:
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            mix.append(json.loads(line))

random.shuffle(mix)

# 留出 dev/test（domain 评测一定要有）
def split(items, p=0.02):
    n = int(len(items)*p)
    return items[:n], items[n:2*n], items[2*n:]

dev, test, train = split(mix, p=0.02)

Path("data/train").mkdir(parents=True, exist_ok=True)
for name, arr in [("mix_train.jsonl", train), ("mix_dev.jsonl", dev), ("mix_test.jsonl", test)]:
    out = Path("data/train")/name
    with out.open("w", encoding="utf-8") as w:
        for o in arr:
            w.write(json.dumps(o, ensure_ascii=False)+"\n")
    print(name, len(arr))
