import os, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = os.environ["BASE_MODEL"]
adapter = os.environ["ADAPTER_DIR"]

def clean_generation_config(m):
    # avoid warnings when do_sample=False
    gc = getattr(m, "generation_config", None)
    if gc is not None:
        gc.temperature = None
        gc.top_p = None
        gc.top_k = None

def postprocess(text: str) -> str:
    text = text.strip()
    # drop duplicated "Answer:" blocks if any
    if "Answer:" in text:
        text = text.split("Answer:", 1)[0].rstrip()
    # cut at END
    if "END" in text:
        text = text.split("END", 1)[0].rstrip() + "\nEND"
    # remove trailing empty lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

# tokenizer
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"

prompt = """You are advising an engineer operating an anaerobic dark fermentation H2 reactor.

Return ONLY one Markdown table with columns:
Parameter | Typical target range (with units) | Monitoring | High/low effects | Practical control actions

Rules:
- Each cell <= 18 words.
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
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(device)

def gen(m, max_new_tokens=900):
    clean_generation_config(m)
    with torch.no_grad():
        out = m.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return postprocess(ans)

# base model (one load, reused)
base_model = AutoModelForCausalLM.from_pretrained(
    base,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=dtype,
    device_map="auto",
).eval()

print("\n===== BASE ONLY (TABLE) =====\n")
print(gen(base_model))

lora_model = PeftModel.from_pretrained(base_model, adapter, device_map="auto").eval()
print("\n===== BASE + LORA (TABLE) =====\n")
print(gen(lora_model))
