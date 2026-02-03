import json
import re
from pathlib import Path
from tqdm import tqdm

chunks_path = Path("data/processed/chunks_q1.jsonl")
docmeta_path = Path("data/processed/docmeta_q1.json")
out_path = Path("data/processed/facts_ph_hrt_q1.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

docmeta = {}
if docmeta_path.exists():
    with docmeta_path.open("r", encoding="utf-8") as r:
        docmeta = json.load(r)

# IMPORTANT: allow hyphen / en-dash / em-dash / "to"
DASH = r"(?:-|\u2013|\u2014|\bto\b)"

RE_PH_RANGE = re.compile(
    r"\bpH\b[^0-9]{0,15}([0-9](?:\.[0-9]+)?)\s*" + DASH + r"\s*([0-9](?:\.[0-9]+)?)",
    re.IGNORECASE
)
RE_PH_SINGLE = re.compile(r"\bpH\b[^0-9]{0,15}([0-9](?:\.[0-9]+)?)", re.IGNORECASE)

# include synonyms for HRT
HRT_KEY = r"(?:HRT|hydraulic\s+retention\s+time|retention\s+time)"
RE_HRT_RANGE = re.compile(
    r"\b" + HRT_KEY + r"\b[^0-9]{0,25}([0-9]+(?:\.[0-9]+)?)\s*" + DASH +
    r"\s*([0-9]+(?:\.[0-9]+)?)\s*(h|hr|hrs|hour|hours|d|day|days)\b",
    re.IGNORECASE
)
RE_HRT_SINGLE = re.compile(
    r"\b" + HRT_KEY + r"\b[^0-9]{0,25}([0-9]+(?:\.[0-9]+)?)\s*(h|hr|hrs|hour|hours|d|day|days)\b",
    re.IGNORECASE
)

KEYWORDS = ("optimal","optimum","maximum","highest","best","favorable","preferred","enhanced","improved","significant")

def norm_unit(u: str) -> str:
    u = u.lower()
    if u in {"h","hr","hrs","hour","hours"}: return "h"
    if u in {"d","day","days"}: return "d"
    return u

def evidence_window(text: str, a: int, b: int, win: int = 180) -> str:
    s = text[max(0, a-win):min(len(text), b+win)]
    s = re.sub(r"\s+", " ", s).strip()
    return s[:800]

def has_signal(text: str, a: int, b: int, win: int = 120) -> bool:
    ctx = text[max(0, a-win):min(len(text), b+win)].lower()
    return any(k in ctx for k in KEYWORDS)

facts_total = 0
ph_total = 0
hrt_total = 0

with chunks_path.open("r", encoding="utf-8") as r, out_path.open("w", encoding="utf-8") as w:
    for line in tqdm(r, desc="Extract pH/HRT facts v2 (fixed)"):
        rec = json.loads(line)
        text = rec["text"]
        chunk_id = rec["chunk_id"]
        pdf_file = rec["pdf_file"]

        dm = docmeta.get(pdf_file, {})
        base = {
            "chunk_id": chunk_id,
            "pdf_file": pdf_file,
            "doi": dm.get("doi",""),
            "title": dm.get("title",""),
            "year": dm.get("year", None),
            "venue": dm.get("venue",""),
            "authors": dm.get("authors",""),
        }

        seen = set()

        for m in RE_PH_RANGE.finditer(text):
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo > hi: lo, hi = hi, lo
            if not (0.0 < lo < 14.5 and 0.0 < hi < 14.5):
                continue
            key = ("pH_range", lo, hi, "")
            if key in seen: continue
            seen.add(key)
            sig = has_signal(text, m.start(), m.end())
            w.write(json.dumps({
                **base, "type":"pH_range",
                "value_min": lo, "value_max": hi, "unit":"",
                "signal": sig,
                "evidence": evidence_window(text, m.start(), m.end())
            }, ensure_ascii=False) + "\n")
            facts_total += 1; ph_total += 1

        for m in RE_PH_SINGLE.finditer(text):
            v = float(m.group(1))
            if not (0.0 < v < 14.5): continue
            sig = has_signal(text, m.start(), m.end())
            if not sig: continue
            key = ("pH_single", v, v, "")
            if key in seen: continue
            seen.add(key)
            w.write(json.dumps({
                **base, "type":"pH_single",
                "value_min": v, "value_max": v, "unit":"",
                "signal": True,
                "evidence": evidence_window(text, m.start(), m.end())
            }, ensure_ascii=False) + "\n")
            facts_total += 1; ph_total += 1

        for m in RE_HRT_RANGE.finditer(text):
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo > hi: lo, hi = hi, lo
            unit = norm_unit(m.group(3))
            if not (0.0 < lo < 1e4 and 0.0 < hi < 1e4):
                continue
            key = ("HRT_range", lo, hi, unit)
            if key in seen: continue
            seen.add(key)
            sig = has_signal(text, m.start(), m.end())
            w.write(json.dumps({
                **base, "type":"HRT_range",
                "value_min": lo, "value_max": hi, "unit": unit,
                "signal": sig,
                "evidence": evidence_window(text, m.start(), m.end())
            }, ensure_ascii=False) + "\n")
            facts_total += 1; hrt_total += 1

        for m in RE_HRT_SINGLE.finditer(text):
            v = float(m.group(1))
            unit = norm_unit(m.group(2))
            sig = has_signal(text, m.start(), m.end())
            if not sig: continue
            if not (0.0 < v < 1e4): continue
            key = ("HRT_single", v, v, unit)
            if key in seen: continue
            seen.add(key)
            w.write(json.dumps({
                **base, "type":"HRT_single",
                "value_min": v, "value_max": v, "unit": unit,
                "signal": True,
                "evidence": evidence_window(text, m.start(), m.end())
            }, ensure_ascii=False) + "\n")
            facts_total += 1; hrt_total += 1

print("Wrote:", out_path)
print("facts_total =", facts_total, "ph_total =", ph_total, "hrt_total =", hrt_total)
