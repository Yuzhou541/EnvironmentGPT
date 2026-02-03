from __future__ import annotations
import argparse
import os
import json
from pathlib import Path
import fitz  # pymupdf

def extract_text_from_pdf(pdf_path: Path, max_pages: int | None = None) -> str:
    doc = fitz.open(pdf_path.as_posix())
    texts = []
    n_pages = min(len(doc), max_pages) if max_pages else len(doc)
    for i in range(n_pages):
        page = doc[i]
        t = page.get_text("text")
        t = t.replace("\x00", " ").strip()
        if t:
            texts.append(t)
    doc.close()
    return "\n\n".join(texts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--max_pages", type=int, default=0, help="0 means all pages.")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdfs = sorted([p for p in pdf_dir.glob("*.pdf")])
    if not pdfs:
        raise RuntimeError(f"No PDFs found under: {pdf_dir}")

    with out_path.open("w", encoding="utf-8") as f:
        for p in pdfs:
            text = extract_text_from_pdf(p, max_pages=None if args.max_pages == 0 else args.max_pages)
            obj = {
                "doc_id": p.stem,
                "filename": p.name,
                "text": text,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved {len(pdfs)} docs to {out_path}")

if __name__ == "__main__":
    main()
