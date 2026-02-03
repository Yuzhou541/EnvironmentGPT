import json
from pathlib import Path
from teacher_engine import TeacherEngine

QA_PATH = Path("data/processed/qa_env_q1_all.jsonl")
EPS_PH = 0.25
EPS_HRT_H = 1.0

def expand(lo, hi, eps):
    lo = float(lo); hi = float(hi)
    if lo > hi: lo, hi = hi, lo
    if lo == hi:
        return lo - eps, hi + eps
    return lo, hi

def iou_interval_eps(gt_lo, gt_hi, pr_lo, pr_hi, eps):
    gt_lo, gt_hi = expand(gt_lo, gt_hi, eps)
    pr_lo, pr_hi = expand(pr_lo, pr_hi, eps)
    inter = max(0.0, min(gt_hi, pr_hi) - max(gt_lo, pr_lo))
    union = max(gt_hi, pr_hi) - min(gt_lo, pr_lo)
    return inter / union if union > 0 else 0.0

def main():
    qas = [json.loads(x) for x in QA_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    eng = TeacherEngine(topk_in_doc=200)

    hit = tot = 0
    hit_ph = tot_ph = 0
    hit_hrt = tot_hrt = 0

    for qa in qas:
        typ = qa["meta"]["type"]
        pdf = qa["citations"][0]["pdf_file"]
        q = qa["question"]

        gt_lo = float(qa["meta"]["value_min"])
        gt_hi = float(qa["meta"]["value_max"])
        gt_unit = (qa["meta"].get("unit","") or "").strip()

        if typ.startswith("HRT") and gt_unit == "d":
            gt_lo *= 24.0
            gt_hi *= 24.0

        pred = eng.answer_doc(q, pdf, topk_in_doc=200, topn=10)

        if typ.startswith("pH"):
            tot += 1; tot_ph += 1
            best = pred["best"]["pH"]
            if not best: 
                continue
            pr_lo, pr_hi = best["value_min"], best["value_max"]
            if iou_interval_eps(gt_lo, gt_hi, pr_lo, pr_hi, EPS_PH) >= 0.25:
                hit += 1; hit_ph += 1

        else:
            tot += 1; tot_hrt += 1
            best = pred["best"]["HRT"]
            if not best:
                continue
            pr_lo, pr_hi = best["value_min"], best["value_max"]
            if iou_interval_eps(gt_lo, gt_hi, pr_lo, pr_hi, EPS_HRT_H) >= 0.25:
                hit += 1; hit_hrt += 1

    print(f"docaware_hit = {hit} / {tot}  hit_rate = {hit/tot:.4f}")
    print(f"pH_hit       = {hit_ph} / {tot_ph}  hit_rate = {hit_ph/tot_ph:.4f}")
    print(f"HRT_hit      = {hit_hrt} / {tot_hrt}  hit_rate = {hit_hrt/tot_hrt:.4f}")

if __name__ == "__main__":
    main()
