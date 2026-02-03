import argparse, json
from pathlib import Path
from teacher_engine import TeacherEngine

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--pdf", type=str, required=True)          # 强制 doc-aware
    ap.add_argument("--topk_in_doc", type=int, default=200)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    eng = TeacherEngine(topk_in_doc=args.topk_in_doc)
    out = eng.answer_doc(args.query, args.pdf, topk_in_doc=args.topk_in_doc, topn=args.topn)

    out_path = Path(args.out) if args.out else (Path("runs") / "teacher_answer.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.quiet:
        print(f"Wrote: {out_path}")
        print(out["answer"])

if __name__ == "__main__":
    main()
