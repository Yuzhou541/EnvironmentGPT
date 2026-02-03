import json, random
from datasets import load_dataset
from pathlib import Path

random.seed(7)
out = Path("data/train/gen_anchor.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

# 选一个常用指令数据集；AutoDL 有网一般可直接拉
# 若你们更偏“通用推理”，可换成 openbookqa/arc/mmlu 的问题部分做 anchor
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train")

N = 80000  # 8 万条 anchor，足够支撑 TopoGuard
cnt = 0
with out.open("w", encoding="utf-8") as w:
    for ex in ds:
        # ultrachat 字段：messages (list)
        msgs = ex.get("messages", [])
        if not msgs:
            continue
        # 取 user 首轮作为 prompt
        user = None
        for m in msgs:
            if m.get("role") == "user":
                user = m.get("content","").strip()
                break
        if not user or len(user) < 20:
            continue
        w.write(json.dumps({"type":"gen","prompt":user,"response":""}, ensure_ascii=False) + "\n")
        cnt += 1
        if cnt >= N:
            break

print(f"[gen_anchor] samples={cnt} out={out}")
