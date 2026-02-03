import json
from pathlib import Path
from tqdm import tqdm

inp = Path("data/processed/corpus_q1.jsonl")
outp = Path("data/processed/chunks_q1.jsonl")
outp.parent.mkdir(parents=True, exist_ok=True)

# 简单字符分块（本地跑通优先；后续你上 autodl 再换 token-based 更精细）
CHUNK = 1800
OVERLAP = 200

def chunk_text(s: str):
    s = s.strip()
    i = 0
    n = len(s)
    while i < n:
        j = min(n, i + CHUNK)
        yield s[i:j]
        if j == n:
            break
        i = max(0, j - OVERLAP)

with inp.open("r", encoding="utf-8") as r, outp.open("w", encoding="utf-8") as w:
    for line in tqdm(r, desc="Chunk"):
        rec = json.loads(line)
        doc_id = rec["doc_id"]
        pdf_file = rec["pdf_file"]
        text = rec["text"]
        k = 0
        for ch in chunk_text(text):
            if len(ch) < 300:
                continue
            w.write(json.dumps({
                "chunk_id": f"{doc_id}_{k}",
                "doc_id": doc_id,
                "pdf_file": pdf_file,
                "text": ch
            }, ensure_ascii=False) + "\n")
            k += 1

print("Wrote:", outp)
