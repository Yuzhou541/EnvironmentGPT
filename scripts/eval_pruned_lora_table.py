import os, re, json, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from safetensors.torch import load_file

# reuse a light metric set
REQ = ["ph","temperature","hrt","olr","orp","vfa","alkalinity","nutrients","mixing","inhibitors","nh3","sulfide","heavy","salinity","o2"]

def extract(text: str) -> str:
    t = text.strip()
    if "Answer:" in t: t = t.split("Answer:",1)[0].strip()
    if "END" in t: t = t.split("END",1)[0].rstrip() + "\nEND"
    return t

def schema_ok(t: str) -> bool:
    ls=[x for x in t.splitlines() if x.strip()]
    return (ls and ls[-1].strip()=="END" and sum(ln.strip().startswith("|") and ln.strip().endswith("|") for ln in ls[:-1])>=3)

def coverage(t: str) -> float:
    low=t.lower()
    return sum(k in low for k in REQ)/len(REQ)

def gen(model, tok, messages, max_new_tokens=900):
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.temperature=None; gc.top_p=None; gc.top_k=None
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--rank_csv", required=True)
    ap.add_argument("--keep_frac", type=float, required=True)
    ap.add_argument("--out_json", default="results/pruned_eval.json")
    args=ap.parse_args()

    # load rank list
    modules=[]
    with open(args.rank_csv,"r",encoding="utf-8") as f:
        next(f)
        for line in f:
            m=line.split(",",1)[0]
            modules.append(m)
    keep_n=max(1, int(len(modules)*args.keep_frac))
    keep=set(modules[:keep_n])

    # load adapter tensors and zero out those not kept
    st_path=os.path.join(args.adapter_dir,"adapter_model.safetensors")
    sd=load_file(st_path)
    new_sd={}
    for k,v in sd.items():
        # module prefix before .lora_A/.lora_B
        if ".lora_A." in k:
            base=k.replace(".lora_A.weight","")
            if base not in keep: v=torch.zeros_like(v)
        elif ".lora_B." in k:
            base=k.replace(".lora_B.weight","")
            if base not in keep: v=torch.zeros_like(v)
        new_sd[k]=v
    tmp=os.path.join(args.adapter_dir, f"_pruned_keep{args.keep_frac:.3f}.safetensors")
    from safetensors.torch import save_file
    save_file(new_sd, tmp)

    tok=AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0]>=8) else torch.float16
    base=AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True, dtype=dtype, device_map="auto").eval()

    # load adapter config but override weights path by temporary safetensors
    model=PeftModel.from_pretrained(base, args.adapter_dir, device_map="auto").eval()
    # overwrite adapter weights in-place
    model.load_adapter(args.adapter_dir, adapter_name="default", is_trainable=False)
    model.peft_config["default"].base_model_name_or_path = args.base_model
    model.load_state_dict(new_sd, strict=False)

    prompt = """You are advising an engineer operating an anaerobic dark fermentation H2 reactor.
Return ONLY one Markdown pipe table with columns:
| Parameter | Typical target range (with units) | Monitoring | High/low effects | Practical control actions |
Rules: each cell<=18 words; no HTML; no "Varies"; include pH,T,HRT,OLR,COD,ORP,VFA/Alk,nutrients,trace,mixing,inhibitors; end with END.
"""
    messages=[{"role":"system","content":"You are an environmental engineering expert."},{"role":"user","content":prompt}]
    raw=gen(model,tok,messages,max_new_tokens=900)
    out=extract(raw)

    res={
        "keep_frac": args.keep_frac,
        "kept_modules": keep_n,
        "schema_ok": schema_ok(out),
        "coverage": coverage(out),
        "text": out[:2000],  # truncate in json
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json,"w",encoding="utf-8") as f:
        json.dump(res,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:res[k] for k in res if k!="text"}, indent=2))

if __name__=="__main__":
    main()
