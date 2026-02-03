import os, re, json, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REQ_CATS = {
  "ph": ["ph"],
  "temperature": ["temperature","temp"],
  "hrt": ["hrt","hydraulic retention"],
  "olr": ["olr","organic loading"],
  "substrate_cod": ["substrate","cod"],
  "inoculum_pretreat": ["inoculum","pretreatment","pre-treatment","heat shock","acid"],
  "orp": ["orp","oxidation-reduction"],
  "vfa_alk": ["vfa","alkalinity"],
  "nutrients": ["nutrient","n/p","nitrogen","phosphorus","trace","fe","ni","co"],
  "mixing": ["mixing","agitation"],
  "inhibitors": ["inhibitor","nh3","ammonia","sulfide","h2s","heavy metal","salinity","oxygen","o2"],
}

def clean_gc(m):
    gc = getattr(m, "generation_config", None)
    if gc is not None:
        gc.temperature=None; gc.top_p=None; gc.top_k=None

def extract_table(text: str) -> str:
    t = text.strip()
    if "Answer:" in t:
        t = t.split("Answer:", 1)[0].strip()
    if "END" in t:
        t = t.split("END", 1)[0].rstrip() + "\nEND"
    return t

def is_strict_pipe_table(t: str) -> bool:
    lines = [x for x in t.splitlines() if x.strip()]
    if not lines or lines[-1].strip() != "END":
        return False
    tbl = lines[:-1]
    pipe = [ln for ln in tbl if ln.strip().startswith("|") and ln.strip().endswith("|")]
    return len(pipe) >= 3

def parse_pipe_rows(t: str):
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    if lines and lines[-1] == "END":
        lines = lines[:-1]
    rows=[]
    for ln in lines:
        if ln.startswith("|") and ln.endswith("|"):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            rows.append(cells)
    return rows

def words_count(cell: str) -> int:
    cell = re.sub(r"[\u00b7/]", " ", cell)
    toks = [x for x in re.split(r"\s+", cell.strip()) if x]
    return len(toks)

def cell_len_ok(t: str, max_words=18) -> bool:
    rows = parse_pipe_rows(t)
    if len(rows) < 3:
        return False
    body = rows[2:]
    for r in body:
        for c in r:
            if words_count(c) > max_words:
                return False
    return True

def no_html(t: str) -> bool:
    return ("<sub" not in t.lower()) and ("<sup" not in t.lower()) and ("<" not in t and ">" not in t)

def no_varies(t: str) -> bool:
    return "varies by" not in t.lower()

def coverage(t: str) -> float:
    rows = parse_pipe_rows(t)
    if len(rows) < 3:
        return 0.0
    first_col = " ".join([r[0].lower() for r in rows[2:] if r])
    hit=0
    for _, syns in REQ_CATS.items():
        if any(s in first_col for s in syns):
            hit += 1
    return hit / len(REQ_CATS)

def gen(model, tok, system, prompt, max_new_tokens=900):
    clean_gc(model)
    messages = [{"role":"system","content":system},{"role":"user","content":prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return ans.strip()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--eval_jsonl", default="data/eval/env_table_eval.jsonl")
    ap.add_argument("--out_json", default="results/env_table_eval_out.json")
    ap.add_argument("--max_new_tokens", type=int, default=900)
    args=ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, local_files_only=True, dtype=dtype, device_map="auto"
    ).eval()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, device_map="auto").eval()

    items=[]
    with open(args.eval_jsonl,"r",encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    per=[]
    for it in items:
        raw = gen(model, tok, it["system"], it["prompt"], max_new_tokens=args.max_new_tokens)
        t = extract_table(raw)
        m = {
            "id": it["id"],
            "strict_pipe": is_strict_pipe_table(t),
            "cell_len_ok": cell_len_ok(t),
            "no_varies": no_varies(t),
            "no_html": no_html(t),
            "coverage": coverage(t),
        }
        per.append(m)

    def avg(key): return sum(x[key] for x in per)/len(per)
    def rate(key): return sum(1 for x in per if x[key])/len(per)

    summary = {
        "N": len(per),
        "strict_pipe_rate": rate("strict_pipe"),
        "cell_len_ok_rate": rate("cell_len_ok"),
        "no_varies_rate": rate("no_varies"),
        "no_html_rate": rate("no_html"),
        "coverage_mean": avg("coverage"),
        "coverage_ge_0p8_rate": sum(1 for x in per if x["coverage"]>=0.8)/len(per),
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json,"w",encoding="utf-8") as f:
        json.dump({"summary": summary, "per_sample": per}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))

if __name__=="__main__":
    main()
