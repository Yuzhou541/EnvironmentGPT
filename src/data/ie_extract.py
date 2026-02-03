from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple

# Minimal, robust regexes for common environmental fermentation parameters.
RE_PH = re.compile(r"\bpH\s*(?:=|:)?\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)
RE_TEMP = re.compile(r"\b(?:temperature|temp\.?)\s*(?:=|:)?\s*([0-9]{2,3}(?:\.[0-9]+)?)\s*(?:°\s*C|deg\.?\s*C|C)\b", re.IGNORECASE)
RE_HRT = re.compile(r"\bHRT\s*(?:=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:h|hr|hours|d|day|days)\b", re.IGNORECASE)
RE_OLR = re.compile(r"\bOLR\s*(?:=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:g\s*VS\s*/\s*L\s*/\s*d|g\s*COD\s*/\s*L\s*/\s*d|kg\s*COD\s*/\s*m3\s*/\s*d)\b", re.IGNORECASE)
RE_CN = re.compile(r"\bC\s*/\s*N\s*(?:=|:)?\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)

# Hydrogen yield patterns (very diverse; we keep conservative extraction)
RE_YIELD = re.compile(r"\b(?:H2\s*yield|hydrogen\s*yield|biohydrogen\s*yield)\s*(?:=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:mL\s*H2\s*/\s*g\s*VS|mL\s*/\s*g\s*VS|mol\s*H2\s*/\s*mol)\b", re.IGNORECASE)

def find_all(pattern: re.Pattern, text: str) -> List[str]:
    return [m.group(1) for m in pattern.finditer(text)]

def summarize_value(vals: List[str]) -> Optional[float]:
    if not vals:
        return None
    try:
        nums = [float(v) for v in vals]
    except ValueError:
        return None
    # Use median to be robust.
    nums.sort()
    return nums[len(nums)//2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers_path", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--snippet_chars", type=int, default=240, help="Store short snippet only (avoid saving full text).")
    args = ap.parse_args()

    in_path = Path(args.papers_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            doc = json.loads(line)
            text = doc.get("text", "")
            if not text:
                continue

            ph = summarize_value(find_all(RE_PH, text))
            temp = summarize_value(find_all(RE_TEMP, text))
            hrt = summarize_value(find_all(RE_HRT, text))
            olr = summarize_value(find_all(RE_OLR, text))
            cn = summarize_value(find_all(RE_CN, text))
            yld = summarize_value(find_all(RE_YIELD, text))

            # Keep only docs that contain at least two key params or yield.
            present = sum(v is not None for v in [ph, temp, hrt, olr, cn, yld])
            if present < 2:
                continue

            snippet = text[: args.snippet_chars].replace("\n", " ")
            rec = {
                "doc_id": doc["doc_id"],
                "filename": doc["filename"],
                "ph": ph,
                "temp_c": temp,
                "hrt": hrt,
                "olr": olr,
                "cn": cn,
                "h2_yield": yld,
                "snippet": snippet,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Processed docs: {n_in}, kept records: {n_out}")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()
