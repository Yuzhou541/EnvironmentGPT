from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
from .anchors import build_general_anchors

def qa_templates(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build parametric QA pairs from a single structured record.
    This is intentionally conservative and relies on extracted numeric fields.
    """
    out: List[Dict[str, Any]] = []
    doc = rec["doc_id"]

    ph = rec.get("ph")
    temp = rec.get("temp_c")
    hrt = rec.get("hrt")
    olr = rec.get("olr")
    cn = rec.get("cn")
    yld = rec.get("h2_yield")

    cond_parts = []
    if ph is not None: cond_parts.append(f"pH≈{ph}")
    if temp is not None: cond_parts.append(f"T≈{temp}°C")
    if hrt is not None: cond_parts.append(f"HRT≈{hrt}")
    if olr is not None: cond_parts.append(f"OLR≈{olr}")
    if cn is not None: cond_parts.append(f"C/N≈{cn}")
    cond = ", ".join(cond_parts) if cond_parts else "partially specified conditions"

    # T1: Condition -> yield interval (we keep it as point or approximate interval)
    if yld is not None:
        prompt = (
            "You are an environmental chemistry and bioprocess assistant specialized in dark fermentation.\n"
            f"Q: Under {cond}, what hydrogen yield (rough scale) would you expect? Answer with a numeric value and unit.\nA:"
        )
        ans = f"{yld} (as reported)"
        out.append({"prompt": prompt, "answer": ans, "meta": {"task": "cond_to_yield", "doc_id": doc}})

    # T2: Condition -> key parameters
    prompt2 = (
        "You are an environmental chemistry and bioprocess assistant specialized in dark fermentation.\n"
        "Q: List key operating parameters that should be reported for a dark fermentation experiment and why they matter.\nA:"
    )
    ans2 = "Key parameters include pH, temperature, HRT, OLR, substrate type, inoculum, and C/N ratio, because they control microbial pathways, inhibition, and productivity."
    out.append({"prompt": prompt2, "answer": ans2, "meta": {"task": "concept", "doc_id": doc}})

    # T3: Counterfactual comparison question (template, answer derived from general principles)
    prompt3 = (
        "You are an environmental chemistry and bioprocess assistant specialized in dark fermentation.\n"
        "Q: If OLR increases too much while other factors stay similar, what failure mode can occur and why?\nA:"
    )
    ans3 = "Excessively high OLR can cause rapid acidification and accumulation of volatile fatty acids, inhibiting hydrogen-producing microbes and reducing yield."
    out.append({"prompt": prompt3, "answer": ans3, "meta": {"task": "counterfactual", "doc_id": doc}})

    return out

def split_write(items: List[Dict[str, Any]], out_dir: Path, seed: int = 42) -> None:
    random.seed(seed)
    random.shuffle(items)
    n = len(items)
    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)
    train = items[:n_train]
    dev = items[n_train:n_train+n_dev]
    test = items[n_train+n_dev:]

    def dump(path: Path, arr: List[Dict[str, Any]]):
        with path.open("w", encoding="utf-8") as f:
            for i, x in enumerate(arr):
                x2 = dict(x)
                x2["id"] = x2.get("id", f"{path.stem}-{i:06d}")
                f.write(json.dumps(x2, ensure_ascii=False) + "\n")

    dump(out_dir / "envqa_train.jsonl", train)
    dump(out_dir / "envqa_dev.jsonl", dev)
    dump(out_dir / "envqa_test.jsonl", test)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_qas", type=int, default=500, help="If too few QAs are produced, you should add more PDFs.")
    args = ap.parse_args()

    in_path = Path(args.records_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_qas: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            all_qas.extend(qa_templates(rec))

    if len(all_qas) < args.min_qas:
        print(f"[Warning] Only produced {len(all_qas)} QAs. Add more PDFs or enrich extraction/templates.")
    split_write(all_qas, out_dir, seed=args.seed)

    # Build general anchors
    build_general_anchors(str(out_dir / "general_anchors.jsonl"), n=300, seed=args.seed)

    print(f"Saved datasets to: {out_dir}")
    print(f"Train/Dev/Test: envqa_train.jsonl / envqa_dev.jsonl / envqa_test.jsonl")
    print(f"General anchors: general_anchors.jsonl")

if __name__ == "__main__":
    main()
