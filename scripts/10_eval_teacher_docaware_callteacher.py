import json, re, sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

# ---- must match teacher ----
DASH = "(?:-|\u2013|\u2014|\\bto\\b)"
HRT_KEY = r"(?:HRT|hydraulic\s+retention\s+time|retention\s+time)"

RE_PH_RANGE  = re.compile(r"\bpH\b[^0-9]{0,15}([0-9](?:\.[0-9]+)?)\s*" + DASH + r"\s*([0-9](?:\.[0-9]+)?)", re.IGNORECASE)
RE_PH_SINGLE = re.compile(r"\bpH\b[^0-9]{0,15}([0-9](?:\.[0-9]+)?)", re.IGNORECASE)

RE_HRT_RANGE  = re.compile(r"\b" + HRT_KEY + r"\b[^0-9]{0,25}([0-9]+(?:\.[0-9]+)?)\s*" + DASH + r"\s*([0-9]+(?:\.[0-9]+)?)\s*(h|hr|hrs|hour|hours|d|day|days)\b", re.IGNORECASE)
RE_HRT_SINGLE = re.compile(r"\b" + HRT_KEY + r"\b[^0-9]{0,25}([0-9]+(?:\.[0-9]+)?)\s*(h|hr|hrs|hour|hours|d|day|days)\b", re.IGNORECASE)

KEYWORDS = ("optimal","optimum","maximum","highest","best","favorable","preferred","enhanced","improved","significant")

QA_PATH   = Path("data/processed/qa_env_q1_all.jsonl")
INDEX_DIR = Path("data/index_bruteforce_q1")
CHUNKS    = Path("data/processed/chunks_q1.jsonl")

TOPK_IN_DOC = 200

def norm_unit(u: str) -> str:
    u = u.lower()
    if u in {"h","hr","hrs","hour","hours"}: return "h"
    if u in {"d","day","days"}: return "d"
    return u

def to_hours(v: float, unit: str) -> float:
    return v * 24.0 if unit == "d" else v

def has_signal(text: str, a: int, b: int, win: int = 120) -> bool:
    ctx = text[max(0,a-win):min(len(text), b+win)].lower()
    return any(k in ctx for k in KEYWORDS)

def pH_prior_penalty(lo: float, hi: float) -> float:
    mid = 0.5 * (lo + hi)
    pen = 0.015 * abs(mid - 5.5)
    if mid < 3.0 or mid > 9.0:
        pen += 0.20
    return pen

def iou_interval(gt_lo, gt_hi, pr_lo, pr_hi):
    inter = max(0.0, min(gt_hi, pr_hi) - max(gt_lo, pr_lo))
    union = max(gt_hi, pr_hi) - min(gt_lo, pr_lo)
    return inter / union if union > 0 else 0.0

# ---- load QA ----
qas = [json.loads(x) for x in QA_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]

# ---- load index/model ----
emb = np.load(INDEX_DIR / "emb.npy", mmap_mode="r")
meta_blob = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
meta_list = meta_blob["meta"]
model_name = meta_blob.get("model", "sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer(model_name)

# ---- load chunks ----
chunks_dict = {}
with CHUNKS.open("r", encoding="utf-8") as r:
    for line in r:
        rec = json.loads(line)
        chunks_dict[rec["chunk_id"]] = rec

# ---- pdf -> rows ----
pdf2rows = {}
for row_i, m in enumerate(meta_list):
    pdf2rows.setdefault(m["pdf_file"], []).append(row_i)

def best_in_doc(q: str, pdf_file: str, want: str):
    rows = pdf2rows.get(pdf_file, [])
    if not rows:
        return None

    qemb = model.encode([q], normalize_embeddings=True).astype(np.float32)[0]
    sims = np.asarray(emb @ qemb)

    rows = np.asarray(rows, dtype=np.int64)
    if rows.size > TOPK_IN_DOC:
        s = sims[rows]
        kk = min(TOPK_IN_DOC, s.size)
        sel = np.argpartition(-s, kth=kk-1)[:kk]
        rows = rows[sel]
    rows = rows[np.argsort(-sims[rows])]

    best = None
    best_score = -1e18

    for r_i in rows:
        m = meta_list[int(r_i)]
        text = chunks_dict.get(m["chunk_id"], {}).get("text","")
        sim = float(sims[int(r_i)])

        if want == "pH":
            for mm in RE_PH_RANGE.finditer(text):
                lo, hi = float(mm.group(1)), float(mm.group(2))
                if lo > hi: lo, hi = hi, lo
                if not (0.0 < lo < 14.5 and 0.0 < hi < 14.5):
                    continue
                sig = has_signal(text, mm.start(), mm.end())
                width = hi - lo
                score = sim + (0.08 if sig else -0.02) - 0.01*width - pH_prior_penalty(lo, hi)
                if score > best_score:
                    best_score = score
                    best = (lo, hi)

            for mm in RE_PH_SINGLE.finditer(text):
                v = float(mm.group(1))
                if not (0.0 < v < 14.5):
                    continue
                sig = has_signal(text, mm.start(), mm.end())
                if not sig:
                    continue
                score = sim + 0.08 - pH_prior_penalty(v, v)
                if score > best_score:
                    best_score = score
                    best = (v, v)

        else:
            for mm in RE_HRT_RANGE.finditer(text):
                lo, hi = float(mm.group(1)), float(mm.group(2))
                if lo > hi: lo, hi = hi, lo
                unit = norm_unit(mm.group(3))
                if not (0.0 < lo < 1e4 and 0.0 < hi < 1e4):
                    continue
                lo_h = to_hours(lo, unit)
                hi_h = to_hours(hi, unit)
                sig = has_signal(text, mm.start(), mm.end())
                width = hi_h - lo_h
                score = sim + (0.08 if sig else -0.02) - 0.0005*width
                if score > best_score:
                    best_score = score
                    best = (lo_h, hi_h)

            for mm in RE_HRT_SINGLE.finditer(text):
                v = float(mm.group(1))
                unit = norm_unit(mm.group(2))
                if not (0.0 < v < 1e4):
                    continue
                sig = has_signal(text, mm.start(), mm.end())
                if not sig:
                    continue
                v_h = to_hours(v, unit)
                score = sim + 0.08
                if score > best_score:
                    best_score = score
                    best = (v_h, v_h)

    return best

hit = tot = 0
hit_ph = tot_ph = 0
hit_hrt = tot_hrt = 0

for idx, qa in enumerate(tqdm(qas, desc="docaware_eval_fast"), 1):
    typ = qa["meta"]["type"]
    pdf = qa["citations"][0]["pdf_file"]
    q = qa["question"]

    gt_lo = float(qa["meta"]["value_min"])
    gt_hi = float(qa["meta"]["value_max"])
    gt_unit = (qa["meta"].get("unit","") or "").strip()

    if typ.startswith("HRT") and gt_unit == "d":
        gt_lo *= 24.0
        gt_hi *= 24.0

    if typ.startswith("pH"):
        tot += 1; tot_ph += 1
        pr = best_in_doc(q, pdf, "pH")
        if pr is None:
            continue
        pr_lo, pr_hi = pr
        if iou_interval(gt_lo, gt_hi, pr_lo, pr_hi) >= 0.25:
            hit += 1; hit_ph += 1
    else:
        tot += 1; tot_hrt += 1
        pr = best_in_doc(q, pdf, "HRT")
        if pr is None:
            continue
        pr_lo, pr_hi = pr
        if iou_interval(gt_lo, gt_hi, pr_lo, pr_hi) >= 0.25:
            hit += 1; hit_hrt += 1

    if idx % 25 == 0:
        print(f"[{idx}/{len(qas)}] running_hit={hit}/{tot}={hit/tot:.4f}")

print(f"docaware_hit = {hit} / {tot}  hit_rate = {hit/tot:.4f}")
print(f"pH_hit       = {hit_ph} / {tot_ph}  hit_rate = {hit_ph/tot_ph:.4f}")
print(f"HRT_hit      = {hit_hrt} / {tot_hrt}  hit_rate = {hit_hrt/tot_hrt:.4f}")
