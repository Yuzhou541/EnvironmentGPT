import argparse, json
from pathlib import Path
from teacher_engine import TeacherEngine

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train", choices=["train","dev","test","all"])
    ap.add_argument("--qa_dir", type=str, default="data/processed")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--topk_in_doc", type=int, default=200)
    ap.add_argument("--drop_notfound", action="store_true")
    args = ap.parse_args()

    qa_map = {
        "train": "qa_env_q1_train.jsonl",
        "dev":   "qa_env_q1_dev.jsonl",
        "test":  "qa_env_q1_test.jsonl",
        "all":   "qa_env_q1_all.jsonl",
    }
    qa_path = Path(args.qa_dir) / qa_map[args.split]
    out_path = Path(args.out) if args.out else Path(f"data/processed/teacher_trainset_q1_{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    qas = [json.loads(x) for x in qa_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    eng = TeacherEngine(topk_in_doc=args.topk_in_doc)

    kept = 0
    with out_path.open("w", encoding="utf-8") as w:
        for qa in qas:
            q = qa["question"]
            pdf = qa["citations"][0]["pdf_file"]
            typ = qa["meta"]["type"]
            want = "pH" if typ.startswith("pH") else "HRT"

            pred = eng.answer_doc(q, pdf, topk_in_doc=args.topk_in_doc, topn=args.topn)
            best = pred["best"][want]
            if best is None and args.drop_notfound:
                continue

            if want == "pH":
                teacher_answer = f"pH: {best['value_min']:g}-{best['value_max']:g}" if best else "pH: not found"
                evidence = pred["candidates"]["pH"]
            else:
                teacher_answer = f"HRT: {best['value_min']:g}-{best['value_max']:g} hours" if best else "HRT: not found"
                evidence = pred["candidates"]["HRT"]

            rec = {
                "question": q,
                "pdf_file": pdf,
                "type": want,
                "teacher_answer": teacher_answer,
                "evidence": evidence,  # 每条含 chunk_id + evidence snippet + value range
                "gt_meta": qa.get("meta", {}),
                "gt_citation": qa.get("citations", []),
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote: {out_path}  lines={kept}  (drop_notfound={args.drop_notfound})")

if __name__ == "__main__":
    main()
