import json
from pathlib import Path

def read_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

SYS = (
    "You are EnvironmentGPT, a closed-book environmental chemistry expert. "
    "Follow the required output format strictly."
)

def make_prompt(q: str, typ: str) -> str:
    if typ == "pH":
        fmt = 'Output exactly ONE line: "pH: <lo>-<hi>"'
    else:
        fmt = 'Output exactly ONE line: "HRT: <lo>-<hi> hours"'
    return (
        f"{SYS}\n\n"
        f"{fmt}\n"
        f"Do not add extra words.\n\n"
        f"Question: {q}\n"
        f"Answer:"
    )

def convert(split: str):
    src = Path(f"data/processed/teacher_trainset_q1_{split}.jsonl")
    dst = Path(f"data/processed/student_sft_q1_{split}.jsonl")
    rows = read_jsonl(src)
    out = []
    for r in rows:
        q = r["question"]
        typ = r["type"]  # "pH" or "HRT"
        ans = r["teacher_answer"].strip()

        # 训练时默认不喂 evidence，保证闭卷写入参数
        out.append({
            "id": r.get("id", ""),
            "type": typ,
            "pdf_file": r.get("pdf_file",""),
            "prompt": make_prompt(q, typ),
            "response": ans,
            "evidence": r.get("evidence", []),  # 仅用于审计/可解释性，不进 prompt
        })
    write_jsonl(dst, out)
    print(f"Wrote: {dst}  lines={len(out)}")

def main():
    for s in ["train","dev","test"]:
        convert(s)

if __name__ == "__main__":
    main()
