from pathlib import Path
import fitz  # pymupdf
from tqdm import tqdm

pdf_dir = Path("data/pdfs")
bad_dir = Path("data/pdfs_bad")
bad_dir.mkdir(parents=True, exist_ok=True)

pdfs = sorted(pdf_dir.glob("*.pdf"))
bad = 0
for p in tqdm(pdfs, desc="Validate PDFs"):
    try:
        doc = fitz.open(p)
        n = doc.page_count
        doc.close()
        if n <= 0:
            raise RuntimeError("0 pages")
    except Exception:
        bad += 1
        p.rename(bad_dir / p.name)

print(f"Total={len(pdfs)} bad_moved={bad} good_left={len(list(pdf_dir.glob('*.pdf')))}")
