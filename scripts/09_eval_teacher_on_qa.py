import json, re, subprocess, sys
from pathlib import Path

QA_PATH = Path("data/processed/qa_env_q1_all.jsonl")

def parse_range(s: str):
    # grabs first "ab" or "a-b"
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:|-)\s*([0-9]+(?:\.[0-9]+)?)", s)
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    if a > b: a, b = b, a
    return a, b

qas = [json.loads(x) for x in QA_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
if not qas:
    print("QA empty"); sys.exit(1)

hit = 0
tot = 0

for qa in qas:
    q = qa["question"]
    gt_lo = float(qa["meta"]["value_min"])
    gt_hi = float(qa["meta"]["value_max"])
    typ = qa["meta"]["type"]

    # call teacher script and read stdout json file
    out = Path("runs/_tmp_teacher.json")
    cmd = ["python", "scripts/08_teacher_answer.py", "--query", q, "--topk", "30", "--out", str(out)]
    subprocess.check_call(cmd)

    pred = json.loads(out.read_text(encoding="utf-8"))
    ans = pred["answer"]
    rr = parse_range(ans)

    tot += 1
    if rr is None:
        continue
    pr_lo, pr_hi = rr

    # simple overlap criterion
    inter = max(0.0, min(gt_hi, pr_hi) - max(gt_lo, pr_lo))
    union = max(gt_hi, pr_hi) - min(gt_lo, pr_lo)
    iou = inter / union if union > 0 else 0.0

    if iou >= 0.25:  # loose sanity threshold
        hit += 1

print("teacher_hit =", hit, "/", tot, "hit_rate =", (hit / tot if tot else 0.0))
