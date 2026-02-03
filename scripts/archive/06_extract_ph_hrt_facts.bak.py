import json
import re
from pathlib import Path
from tqdm import tqdm

chunks_path = Path("data/processed/chunks_q1.jsonl")
docmeta_path = Path("data/processed/docmeta_q1.json")
out_path = Path("data/processed/facts_ph_hrt_q1.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

# load docmeta (pdf_file -> meta)
docmeta = {}
if docmeta_path.exists():
    with docmeta_path.open("r", encoding="utf-8") as r:
        docmeta = json.load(r)

# --- regex ---
# pH range: "pH 5.5-6.5", "pH 5.56.5", "pH 5.5 to 6.5"
RE_PH_RANGE = re.compile(
    r"\bpH\b[^0-9]{0,10}([0-9](?:\.[0-9])?)\s*(?:-||to)\s*([0-9](?:\.[0-9])?)",
    flags=re.IGNORECASE
)
# pH single: "pH 5.5"
RE_PH_SINGLE = re.compile(
    r"\bpH\b[^0-9]{0,10}([0-9](?:\.[0-9])?)",
    flags=re.IGNORECASE
)

# HRT range: "HRT 6-12 h", "HRT 6 to 12 h"
RE_HRT_RANGE = re.compile(
    r"\bHRT\b[^0-9]{0,15}([0-9]+(?:\.[0-9])?)\s*(?:-||to)\s*([0-9]+(?:\.[0-9])?)\s*(h|hr|hrs|hour|hours|d|day|days)\b",
    flags=re.IGNORECASE
)
# HRT single: "HRT 8 h"
RE_HRT_SINGLE = re.compile(
    r"\bHRT\b[^0-9]{0,15}([0-9]+(?:\.[0-9])?)\s*(h|hr|hrs|hour|hours|d|day|days)\b",
    flags=re.IGNORECASE
)

def norm_unit(u: str) -> str:
    u = u.lower()
    if u in {"h","hr","hrs","hour","hours"}:
        return "h"
    if u in {"d","day","days"}:
        return "d"
    return u

def near_optimal(text: str, span_start: int, span_end: int, window: int = 80) -> bool:
    # Prefer facts that appear near "optimal/optimum" signals
    a = max(0, span_start - window)
    b = min(len(text), span_end + window)
    ctx = text[a:b].lower()
    return ("optimal" in ctx) or ("optimum" in ctx)

facts = 0
ph_facts = 0
hrt_facts = 0

with chunks_path.open("r", encoding="utf-8") as r, out_path.open("w", encoding="utf-8") as w:
    for line in tqdm(r, desc="Extract pH/HRT facts"):
        rec = json.loads(line)
        text = rec["text"]
        chunk_id = rec["chunk_id"]
        pdf_file = rec["pdf_file"]

        dm = docmeta.get(pdf_file, {})
        base_meta = {
            "chunk_id": chunk_id,
            "pdf_file": pdf_file,
            "doi": dm.get("doi",""),
            "title": dm.get("title",""),
            "year": dm.get("year", None),
            "venue": dm.get("venue",""),
            "authors": dm.get("authors",""),
        }

        # --- pH ranges ---
        for m in RE_PH_RANGE.finditer(text):
            lo = float(m.group(1)); hi = float(m.group(2))
            if lo > hi: lo, hi = hi, lo
            if not (0.0 < lo < 14.5 and 0.0 < hi < 14.5):
                continue
            if not near_optimal(text, m.start(), m.end()):
                continue
            w.write(json.dumps({
                **base_meta,
                "type": "pH_range",
                "value_min": lo,
                "value_max": hi,
                "unit": ""
            }, ensure_ascii=False) + "\n")
            facts += 1; ph_facts += 1

        # --- HRT ranges ---
        for m in RE_HRT_RANGE.finditer(text):
            lo = float(m.group(1)); hi = float(m.group(2))
            if lo > hi: lo, hi = hi, lo
            unit = norm_unit(m.group(3))
            if not (0.0 < lo < 1e4 and 0.0 < hi < 1e4):
                continue
            if not near_optimal(text, m.start(), m.end()):
                continue
            w.write(json.dumps({
                **base_meta,
                "type": "HRT_range",
                "value_min": lo,
                "value_max": hi,
                "unit": unit
            }, ensure_ascii=False) + "\n")
            facts += 1; hrt_facts += 1

print("Wrote:", out_path)
print("facts_total =", facts, "pH_range =", ph_facts, "HRT_range =", hrt_facts)
