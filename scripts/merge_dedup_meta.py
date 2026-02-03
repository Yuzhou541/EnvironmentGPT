import glob, json, re, hashlib
from pathlib import Path

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def key_of(o):
    doi = norm(o.get("doi") or o.get("DOI"))
    if doi:
        return "doi:" + doi
    oid = o.get("id") or o.get("openalex_id") or o.get("work_id")
    if oid:
        return "oa:" + str(oid).strip()
    title = norm(o.get("title"))
    year = str(o.get("publication_year") or o.get("year") or "")
    return "ty:" + title + "::" + year

files = sorted(glob.glob("data/processed/openalex_q*.jsonl"))
out = Path("data/curated/openalex_all_dedup.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

seen = set()
kept = 0
total = 0
with out.open("w", encoding="utf-8") as w:
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: 
                    continue
                total += 1
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                k = key_of(o)
                if k in seen:
                    continue
                seen.add(k)
                w.write(json.dumps(o, ensure_ascii=False) + "\n")
                kept += 1

print(f"[meta] input_lines={total}, kept_unique={kept}, out={out}")
