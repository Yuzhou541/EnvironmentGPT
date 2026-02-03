import json
import re
from pathlib import Path

openalex_path = Path("data/processed/openalex_q1_core.jsonl")
pdf_dir = Path("data/pdfs")
out_path = Path("data/processed/docmeta_q1.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

def norm_doi(x: str) -> str:
    if not x:
        return ""
    x = x.strip().lower()
    x = x.replace("https://doi.org/", "").replace("http://doi.org/", "")
    x = x.replace("https://dx.doi.org/", "").replace("http://dx.doi.org/", "")
    return x

def doi_from_pdf_filename(pdf_file: str) -> str:
    # e.g. "10.1007s11274-023-03845-4_xxx.pdf" -> "10.1007/s11274-023-03845-4"
    stem = Path(pdf_file).stem
    left = stem.split("_", 1)[0]  # take part before first underscore
    m = re.match(r"^(10\.\d+)(.+)$", left)
    if not m:
        return ""
    return f"{m.group(1)}/{m.group(2)}".lower()

def pick_title(rec: dict) -> str:
    return (rec.get("title")
            or rec.get("display_name")
            or rec.get("paper_title")
            or "")

def pick_year(rec: dict):
    return (rec.get("publication_year")
            or rec.get("year")
            or rec.get("published_year")
            or None)

def pick_venue(rec: dict) -> str:
    hv = rec.get("host_venue") or {}
    if isinstance(hv, dict):
        v = hv.get("display_name") or hv.get("name")
        if v:
            return v
    pl = rec.get("primary_location") or {}
    if isinstance(pl, dict):
        src = pl.get("source") or {}
        if isinstance(src, dict):
            v = src.get("display_name") or src.get("name")
            if v:
                return v
    return ""

def pick_authors(rec: dict) -> str:
    auths = rec.get("authorships")
    if isinstance(auths, list) and len(auths) > 0:
        names = []
        for a in auths:
            if not isinstance(a, dict): 
                continue
            au = a.get("author") or {}
            if isinstance(au, dict):
                n = au.get("display_name") or au.get("name")
                if n:
                    names.append(n)
        if names:
            return ", ".join(names[:8]) + (", et al." if len(names) > 8 else "")
    # fallback
    a2 = rec.get("authors")
    if isinstance(a2, list) and a2:
        return ", ".join([str(x) for x in a2[:8]]) + (", et al." if len(a2) > 8 else "")
    return ""

# 1) build doi->meta from OpenAlex
doi2meta = {}
bad = 0
total = 0
with openalex_path.open("r", encoding="utf-8") as r:
    for line in r:
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            rec = json.loads(line)
        except Exception:
            bad += 1
            continue
        doi = norm_doi(rec.get("doi") or rec.get("ids", {}).get("doi") or "")
        if not doi:
            continue
        doi2meta[doi] = {
            "doi": doi,
            "title": pick_title(rec),
            "year": pick_year(rec),
            "venue": pick_venue(rec),
            "authors": pick_authors(rec),
        }

print("OpenAlex lines =", total, "bad_json =", bad, "doi_indexed =", len(doi2meta))

# 2) map pdf_file -> meta
pdf_files = sorted([p.name for p in pdf_dir.glob("*.pdf")])
docmeta = {}
hit = 0
miss = 0
no_doi = 0

for f in pdf_files:
    doi = doi_from_pdf_filename(f)
    if not doi:
        no_doi += 1
        docmeta[f] = {"pdf_file": f, "doi": "", "title": "", "year": None, "venue": "", "authors": ""}
        continue
    m = doi2meta.get(doi, None)
    if m is None:
        miss += 1
        docmeta[f] = {"pdf_file": f, "doi": doi, "title": "", "year": None, "venue": "", "authors": ""}
    else:
        hit += 1
        docmeta[f] = {"pdf_file": f, **m}

with out_path.open("w", encoding="utf-8") as w:
    json.dump(docmeta, w, ensure_ascii=False, indent=2)

print("PDFs =", len(pdf_files), "hit =", hit, "miss =", miss, "no_doi =", no_doi)
print("Wrote:", out_path)
