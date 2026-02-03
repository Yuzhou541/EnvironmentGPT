import os, re, json, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REQ_KEYS = [
    "pH", "Temperature", "HRT", "OLR", "Substrate", "COD",
    "Inoculum", "ORP", "VFA", "Alkalinity", "Nutrients", "Trace",
    "Mixing", "Inhibitors", "NH3", "Sulfide", "Heavy", "Salinity", "O2"
]

def normalize_ws(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s.strip())

def extract_table(text: str) -> str:
    # keep only table + END
    t = text.strip()
    if "Answer:" in t:
        t = t.split("Answer:", 1)[0].strip()
    # take until END if present
    if "END" in t:
        head = t.split("END", 1)[0].rstrip()
        return head + "\nEND"
    return t

def is_pipe_table(s: str) -> bool:
    lines = [x for x in s.splitlines() if x.strip()]
    # must contain END last
    if not lines or lines[-1].strip() != "END":
        return False
    # table must have at least header+sep+1 row
    tbl = lines[:-1]
    pipe_lines = [ln for ln in tbl if ln.strip().startswith("|") and ln.strip().endswith("|")]
    return len(pipe_lines) >= 3

def parse_rows(s: str):
    # return list of rows (list of cells)
    lines = [x for x in s.splitlines() if x.strip()]
    # drop END
    if lines and lines[-1].strip() == "END":
        lines = lines[:-1]
    # keep only pipe rows
    rows = []
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("|") and ln.endswith("|"):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            rows.append(cells)
    return rows

def words_count(cell: str) -> int:
    # count words roughly; treat '/' and '·' as separators too
    c = re.sub(r"[\u00b7/|]", " ", cell)
    toks = [t for t in re.split(r"\s+", c.strip()) if t]
    return len(toks)

def schema_valid(s: str) -> bool:
    if not is_pipe_table(s):
        return False
    rows = parse_rows(s)
    if len(rows) < 3:
        return False
    header = [normalize_ws(x).lower() for x in rows[0]]
    # minimal header check
    must = ["parameter", "typical target range", "monitoring", "high/low effects", "practical control actions"]
    ok = sum(any(m in h for h in header) for m in must)
    return ok >= 4

def cell_len_ok(s: str, max_words=18) -> bool:
    rows = parse_rows(s)
    if len(rows) < 3:
        return False
    # skip header + sep
    body = rows[2:] if len(rows) >= 3 else []
    for r in body:
        for c in r:
            if words_count(c) > max_words:
                return False
    return True

def no_varies(s: str) -> bool:
    return ("varies by" not in s.lower()) and ("varies" not in s.lower())

def coverage_score(s: str) -> float:
    low = s.lower()
    hit = 0
    for k in REQ_KEYS:
        if k.lower() in low:
            hit += 1
    return hit / len(REQ_KEYS)

def range_plausible(s: str) -> bool:
    # light sanity checks on numeric ranges if present
    low = s.lower()
    # pH sanity: any "ph" row should not be 0-14 extreme; allow broad
    # temperature sanity: if has °c, should be between 10 and 80
    # ORP: if mV, should be between +200 and -800
    nums = [float(x) for x in re.findall(r"(-?\d+(?:\.\d+)?)", low)]
    if not nums:
        return True
    # overly huge numbers usually indicate formatting failure
    if any(abs(x) > 1e6 for x in nums):
        return False
    return True

def gen(model, tok, messages, max_new_tokens=900):
    # avoid warnings when do_sample=False
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.temperature = None
        gc.top_p = None
        gc.top_k = None

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = model.device
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return ans.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out_json", default="results/env_table_eval.json")
    ap.add_argument("--max_new_tokens", type=int, default=900)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, local_files_only=True, dtype=dtype, device_map="auto"
    ).eval()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, device_map="auto").eval()

    prompt = """You are advising an engineer operating an anaerobic dark fermentation H2 reactor.

Return ONLY one Markdown pipe table with columns:
| Parameter | Typical target range (with units) | Monitoring | High/low effects | Practical control actions |

Rules:
- Each cell <= 18 words.
- No HTML tags.
- Do NOT use “Varies by ...”. If uncertain, give best-practice typical ranges for dark fermentation H2.
- Must include at least: pH, temperature, HRT, OLR, substrate/COD, inoculum pretreatment, ORP, VFA/alkalinity,
  nutrients (N,P,trace metals), mixing, key inhibitors (NH3, sulfide, heavy metals, salinity, O2 intrusion).
- No text before/after the table.
- End with a single line: END
"""
    messages = [
        {"role":"system","content":"You are an environmental engineering expert."},
        {"role":"user","content":prompt},
    ]
    raw = gen(model, tok, messages, max_new_tokens=args.max_new_tokens)
    out = extract_table(raw)

    metrics = {
        "schema_valid": schema_valid(out),
        "cell_len_ok": cell_len_ok(out),
        "no_varies": no_varies(out),
        "coverage": coverage_score(out),
        "range_plausible": range_plausible(out),
        "text": out,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: metrics[k] for k in metrics if k!="text"}, indent=2))

if __name__ == "__main__":
    main()
