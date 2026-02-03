import json, re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

QA_PATH = Path("data/processed/qa_env_q1_all.jsonl")
INDEX_DIR = Path("data/index_bruteforce_q1")
CHUNKS = Path("data/processed/chunks_q1.jsonl")
DOCMETA = Path("data/processed/docmeta_q1.json")

DASH = "(?:-|\u2013|\u2014|\\bto\\b)"
HRT_KEY = r"(?:HRT|hydraulic\s+retention\s+time|retention\s+time)"

RE_PH_RANGE = re.compile(r"\bpH\b[^0-9]{0,15}([0-9](?:\.[0-9]+)?)\s*" + DASH + r"\s*([0-9](?:\.[0-9]+)?)", re.IGNORECASE)
RE_HRT_RANGE = re.compile(r"\b" + HRT_KEY + r"\b[^0-9]{0,25}([0-9]+(?:\.[0-9]+)?)\s*" + DASH + r"\s*([0-9]+(?:\.[0-9]+)?)\s*(h|hr|hrs|hour|hours|d|day|days)\b", re.IGNORECASE)

KEYWORDS = ("optimal","optimum","maximum","highest","best","favorable","preferred","enhanced","improved","significant")

def norm_unit(u: str) -> str:
    u = u.lower()
    if u in {"h","hr","hrs","hour","hours"}: return "h"
    if u in {"d","day","days"}: return "d"
    return u

def has_signal(text: str, a: int, b: int, win: int = 120) -> bool:
    ctx = text[max(0, a-win):min(len(text), b+win)].lower()
    return any(k in ctx for k in KEYWORDS)

def load_chunks_dict(chunks_path: Path):
    d = {}
    with chunks_path.open("r", encoding="utf-8") as r:
        for line in r:
            rec = json.loads(line)
            d[rec["chunk_id"]] = rec
    return d

# ---- load once ----
qas = [json.loads(x) for x in QA_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
emb = np.load(INDEX_DIR / "emb.npy", mmap_mode="r")
meta_blob = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
meta_list = meta_blob["meta"]
model_name = meta_blob.get("model", "sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer(model_name)
chunks_dict = load_chunks_dict(CHUNKS)

topk = 80

def predict_range(q: str, want: str):
    qemb = model.encode([q], normalize_embeddings=True).astype(np.float32)[0]
    sims = np.asarray(emb @ qemb)
    k = min(topk, sims.shape[0])
    idx = np.argpartition(-sims, kth=k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    best = None
    best_score = -1e9

    for i in idx:
        m = meta_list[int(i)]
        text = chunks_dict.get(m["chunk_id"], {}).get("text","")
        sim = float(sims[int(i)])

        if want == "pH":
            for mm in RE_PH_RANGE.finditer(text):
                lo, hi = float(mm.group(1)), float(mm.group(2))
                if lo > hi: lo, hi = hi, lo
                if not (0.0 < lo < 14.5 and 0.0 < hi < 14.5):
                    continue
                sig = has_signal(text, mm.start(), mm.end())
                score = sim + (0.03 if sig else 0.0)
                if score > best_score:
                    best_score = score
                    best = (lo, hi)
        else:
            for mm in RE_HRT_RANGE.finditer(text):
                lo, hi = float(mm.group(1)), float(mm.group(2))
                if lo > hi: lo, hi = hi, lo
                unit = norm_unit(mm.group(3))
                if not (0.0 < lo < 1e4 and 0.0 < hi < 1e4):
                    continue
                sig = has_signal(text, mm.start(), mm.end())
                score = sim + (0.03 if sig else 0.0)
                if score > best_score:
                    best_score = score
                    best = (lo, hi, unit)

    return best

def iou_interval(gt_lo, gt_hi, pr_lo, pr_hi):
    inter = max(0.0, min(gt_hi, pr_hi) - max(gt_lo, pr_lo))
    union = max(gt_hi, pr_hi) - min(gt_lo, pr_lo)
    return inter / union if union > 0 else 0.0

hit = 0
tot = 0
hit_ph = tot_ph = 0
hit_hrt = tot_hrt = 0

for qa in qas:
    typ = qa["meta"]["type"]
    gt_lo = float(qa["meta"]["value_min"])
    gt_hi = float(qa["meta"]["value_max"])
    q = qa["question"]

    if typ.startswith("pH"):
        tot_ph += 1
        pr = predict_range(q, "pH")
        if pr is None:
            tot += 1
            continue
        pr_lo, pr_hi = pr
        score = iou_interval(gt_lo, gt_hi, pr_lo, pr_hi)
        if score >= 0.25:
            hit += 1; hit_ph += 1
        tot += 1

    elif typ.startswith("HRT"):
        tot_hrt += 1
        pr = predict_range(q, "HRT")
        if pr is None:
            tot += 1
            continue
        pr_lo, pr_hi, _unit = pr
        score = iou_interval(gt_lo, gt_hi, pr_lo, pr_hi)
        if score >= 0.25:
            hit += 1; hit_hrt += 1
        tot += 1

print(f"teacher_hit = {hit} / {tot}  hit_rate = {hit/tot if tot else 0.0:.4f}")
print(f"pH_hit      = {hit_ph} / {tot_ph}  hit_rate = {hit_ph/tot_ph if tot_ph else 0.0:.4f}")
print(f"HRT_hit     = {hit_hrt} / {tot_hrt}  hit_rate = {hit_hrt/tot_hrt if tot_hrt else 0.0:.4f}")
