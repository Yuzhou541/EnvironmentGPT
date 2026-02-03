import json, re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# IMPORTANT: include real unicode dashes
DASH = "(?:-|\u2013|\u2014|\\bto\\b)"
HRT_KEY = r"(?:HRT|hydraulic\s+retention\s+time|retention\s+time)"

RE_PH_RANGE  = re.compile(r"\bpH\b[^0-9]{0,15}([0-9](?:\.[0-9]+)?)\s*" + DASH + r"\s*([0-9](?:\.[0-9]+)?)", re.IGNORECASE)
RE_PH_SINGLE = re.compile(r"\bpH\b[^0-9]{0,15}([0-9](?:\.[0-9]+)?)", re.IGNORECASE)

RE_HRT_RANGE  = re.compile(r"\b" + HRT_KEY + r"\b[^0-9]{0,25}([0-9]+(?:\.[0-9]+)?)\s*" + DASH + r"\s*([0-9]+(?:\.[0-9]+)?)\s*(h|hr|hrs|hour|hours|d|day|days)\b", re.IGNORECASE)
RE_HRT_SINGLE = re.compile(r"\b" + HRT_KEY + r"\b[^0-9]{0,25}([0-9]+(?:\.[0-9]+)?)\s*(h|hr|hrs|hour|hours|d|day|days)\b", re.IGNORECASE)

KEYWORDS = ("optimal","optimum","maximum","highest","best","favorable","preferred","enhanced","improved","significant")

def norm_unit(u: str) -> str:
    u = u.lower()
    if u in {"h","hr","hrs","hour","hours"}: return "h"
    if u in {"d","day","days"}: return "d"
    return u

def to_hours(v: float, unit: str) -> float:
    return v * 24.0 if unit == "d" else v

def evidence_window(text: str, a: int, b: int, win: int = 180) -> str:
    s = text[max(0, a-win):min(len(text), b+win)]
    s = re.sub(r"\s+", " ", s).strip()
    return s[:800]

def has_signal(text: str, a: int, b: int, win: int = 120) -> bool:
    ctx = text[max(0,a-win):min(len(text), b+win)].lower()
    return any(k in ctx for k in KEYWORDS)

def pH_prior_penalty(lo: float, hi: float) -> float:
    # soft prior: typical dark-fermentation pH around 5-6 (weak)
    mid = 0.5*(lo+hi)
    pen = 0.015 * abs(mid - 5.5)
    if mid < 3.0 or mid > 9.0:
        pen += 0.20
    return pen

class TeacherEngine:
    def __init__(
        self,
        index_dir="data/index_bruteforce_q1",
        chunks_path="data/processed/chunks_q1.jsonl",
        docmeta_path="data/processed/docmeta_q1.json",
        topk_in_doc=200,
    ):
        self.index_dir = Path(index_dir)
        self.emb = np.load(self.index_dir / "emb.npy", mmap_mode="r")
        meta_blob = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
        self.meta_list = meta_blob["meta"]
        self.model_name = meta_blob.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)

        # chunks
        self.chunks_dict = {}
        with Path(chunks_path).open("r", encoding="utf-8") as r:
            for line in r:
                rec = json.loads(line)
                self.chunks_dict[rec["chunk_id"]] = rec

        # docmeta
        self.docmeta = {}
        dp = Path(docmeta_path)
        if dp.exists():
            self.docmeta = json.loads(dp.read_text(encoding="utf-8"))

        # pdf -> rows
        self.pdf2rows = {}
        for row_i, m in enumerate(self.meta_list):
            self.pdf2rows.setdefault(m["pdf_file"], []).append(row_i)

        self.topk_in_doc = int(topk_in_doc)

    def encode_query(self, q: str) -> np.ndarray:
        return self.model.encode([q], normalize_embeddings=True).astype(np.float32)[0]

    def _select_rows_in_doc(self, qemb: np.ndarray, pdf_file: str, topk: int):
        rows = self.pdf2rows.get(pdf_file, [])
        if not rows:
            return np.array([], dtype=np.int64), None
        rows = np.asarray(rows, dtype=np.int64)

        # compute sims ONLY inside doc
        doc_emb = self.emb[rows]
        sims = np.asarray(doc_emb @ qemb, dtype=np.float32)

        k = min(topk, sims.shape[0])
        if k <= 0:
            return np.array([], dtype=np.int64), sims
        idx = np.argpartition(-sims, kth=k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return rows[idx], sims  # rows subset, sims aligned to all doc rows

    def _add_ph_candidates(self, cands, seen, text, sim, pdf_file, chunk_id):
        # range
        for mm in RE_PH_RANGE.finditer(text):
            lo, hi = float(mm.group(1)), float(mm.group(2))
            if lo > hi: lo, hi = hi, lo
            if not (0.0 < lo < 14.5 and 0.0 < hi < 14.5):
                continue
            key = ("pH", lo, hi, pdf_file, chunk_id)
            if key in seen: 
                continue
            seen.add(key)
            sig = has_signal(text, mm.start(), mm.end())
            width = hi - lo
            score = sim + (0.08 if sig else -0.02) - 0.01*width - pH_prior_penalty(lo, hi)
            cands.append({
                "type":"pH",
                "score": float(score),
                "signal": bool(sig),
                "value_min": lo, "value_max": hi, "unit":"",
                "pdf_file": pdf_file, "chunk_id": chunk_id,
                "evidence": evidence_window(text, mm.start(), mm.end()),
            })

        # single (must have signal)
        for mm in RE_PH_SINGLE.finditer(text):
            v = float(mm.group(1))
            if not (0.0 < v < 14.5):
                continue
            sig = has_signal(text, mm.start(), mm.end())
            if not sig:
                continue
            lo = hi = v
            key = ("pH", lo, hi, pdf_file, chunk_id, "single")
            if key in seen:
                continue
            seen.add(key)
            score = sim + 0.08 - pH_prior_penalty(v, v)
            cands.append({
                "type":"pH",
                "score": float(score),
                "signal": True,
                "value_min": lo, "value_max": hi, "unit":"",
                "pdf_file": pdf_file, "chunk_id": chunk_id,
                "evidence": evidence_window(text, mm.start(), mm.end()),
            })

    def _add_hrt_candidates(self, cands, seen, text, sim, pdf_file, chunk_id):
        # range
        for mm in RE_HRT_RANGE.finditer(text):
            lo, hi = float(mm.group(1)), float(mm.group(2))
            if lo > hi: lo, hi = hi, lo
            unit = norm_unit(mm.group(3))
            if not (0.0 < lo < 1e4 and 0.0 < hi < 1e4):
                continue
            lo_h = to_hours(lo, unit)
            hi_h = to_hours(hi, unit)
            key = ("HRT", lo_h, hi_h, "h", pdf_file, chunk_id)
            if key in seen:
                continue
            seen.add(key)
            sig = has_signal(text, mm.start(), mm.end())
            width = hi_h - lo_h
            score = sim + (0.08 if sig else -0.02) - 0.0005*width
            cands.append({
                "type":"HRT",
                "score": float(score),
                "signal": bool(sig),
                "value_min": lo_h, "value_max": hi_h, "unit":"h",
                "pdf_file": pdf_file, "chunk_id": chunk_id,
                "evidence": evidence_window(text, mm.start(), mm.end()),
            })

        # single (must have signal)
        for mm in RE_HRT_SINGLE.finditer(text):
            v = float(mm.group(1))
            unit = norm_unit(mm.group(2))
            if not (0.0 < v < 1e4):
                continue
            sig = has_signal(text, mm.start(), mm.end())
            if not sig:
                continue
            v_h = to_hours(v, unit)
            lo_h = hi_h = v_h
            key = ("HRT", lo_h, hi_h, "h", pdf_file, chunk_id, "single")
            if key in seen:
                continue
            seen.add(key)
            score = sim + 0.08
            cands.append({
                "type":"HRT",
                "score": float(score),
                "signal": True,
                "value_min": lo_h, "value_max": hi_h, "unit":"h",
                "pdf_file": pdf_file, "chunk_id": chunk_id,
                "evidence": evidence_window(text, mm.start(), mm.end()),
            })

    def answer_doc(self, query: str, pdf_file: str, topk_in_doc=None, topn=10):
        topk_in_doc = int(topk_in_doc or self.topk_in_doc)
        qemb = self.encode_query(query)
        rows_sel, sims_doc = self._select_rows_in_doc(qemb, pdf_file, topk_in_doc)

        cands_ph, cands_hrt = [], []
        seen = set()

        for r_i in rows_sel:
            m = self.meta_list[int(r_i)]
            chunk_id = m["chunk_id"]
            rec = self.chunks_dict.get(chunk_id, {})
            text = rec.get("text","")
            # recover sim for this row within doc: recompute cheap
            sim = float((self.emb[int(r_i)] @ qemb).item())

            self._add_ph_candidates(cands_ph, seen, text, sim, pdf_file, chunk_id)
            self._add_hrt_candidates(cands_hrt, seen, text, sim, pdf_file, chunk_id)

        cands_ph.sort(key=lambda x: (x["signal"], x["score"]), reverse=True)
        cands_hrt.sort(key=lambda x: (x["signal"], x["score"]), reverse=True)
        cands_ph = cands_ph[:topn]
        cands_hrt = cands_hrt[:topn]

        best_ph = cands_ph[0] if cands_ph else None
        best_hrt = cands_hrt[0] if cands_hrt else None

        lines = []
        lines.append(f"pH: {best_ph['value_min']:g}-{best_ph['value_max']:g}" if best_ph else "pH: not found")
        lines.append(f"HRT: {best_hrt['value_min']:g}-{best_hrt['value_max']:g} hours" if best_hrt else "HRT: not found")

        dm = self.docmeta.get(pdf_file, {})
        out = {
            "query": query,
            "pdf_constraint": pdf_file,
            "docmeta": {"doi": dm.get("doi",""), "title": dm.get("title",""), "year": dm.get("year", None)},
            "answer": "\n".join(lines),
            "best": {"pH": best_ph, "HRT": best_hrt},
            "candidates": {"pH": cands_ph, "HRT": cands_hrt},
        }
        return out
