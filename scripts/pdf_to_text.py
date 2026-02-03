import os, json
from pathlib import Path
import fitz  # pymupdf

pdf_roots = [Path("data/pdfs")]
out = Path("data/corpus/openalex_corpus.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

pdfs = []
for r in pdf_roots:
    if r.exists():
        pdfs += list(r.rglob("*.pdf"))

def extract_text(pdf_path: Path, max_pages=50):
    doc = fitz.open(pdf_path)
    n = min(len(doc), max_pages)
    parts = []
    for i in range(n):
        try:
            parts.append(doc.load_page(i).get_text("text"))
        except Exception:
            continue
    doc.close()
    text = "\n".join(parts)
    text = text.replace("\x00", " ").strip()
    return text

with out.open("w", encoding="utf-8") as w:
    ok, bad = 0, 0
    for p in pdfs:
        try:
            t = extract_text(p)
            if len(t) < 500:
                bad += 1
                continue
            w.write(json.dumps({"pdf_path": str(p), "text": t}, ensure_ascii=False) + "\n")
            ok += 1
        except Exception:
            bad += 1
    print(f"[corpus] pdf_total={len(pdfs)} ok={ok} bad_or_empty={bad} out={out}")
