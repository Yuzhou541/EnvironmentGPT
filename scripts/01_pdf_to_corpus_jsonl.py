from pathlib import Path
import fitz
import json
import hashlib
from tqdm import tqdm

pdf_dir = Path("data/pdfs")
out_path = Path("data/processed/corpus_q1.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

pdfs = sorted(pdf_dir.glob("*.pdf"))
with out_path.open("w", encoding="utf-8") as w:
    for p in tqdm(pdfs, desc="Extract text"):
        try:
            doc = fitz.open(p)
            texts = []
            for i in range(doc.page_count):
                texts.append(doc.load_page(i).get_text("text"))
            doc.close()
            text = "\n".join(texts).strip()
            if len(text) < 500:
                continue
            rec = {
                "doc_id": sha1(p.name),
                "pdf_file": p.name,
                "text": text
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            continue

print("Wrote:", out_path)
