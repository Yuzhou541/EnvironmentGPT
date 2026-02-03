import os, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = os.environ["BASE_MODEL"]
adapter = os.environ["ADAPTER_DIR"]

def log(msg):
    print(msg, flush=True)

t0 = time.time()
log(f"BASE={base}")
log(f"ADAPTER={adapter}")

log("[1] loading tokenizer...")
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
log(f"[1] done. elapsed={time.time()-t0:.1f}s")

dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
log(f"[cfg] dtype={dtype}")

log("[2] loading base model...")
t1 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    base,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=dtype,
    device_map="auto",
)
log(f"[2] done. elapsed={time.time()-t1:.1f}s")

log("[3] loading LoRA adapter...")
t2 = time.time()
model = PeftModel.from_pretrained(model, adapter, device_map="auto")
model.eval()
log(f"[3] done. elapsed={time.time()-t2:.1f}s")

prompt = "List key operating parameters for anaerobic dark fermentation hydrogen production."
messages = [
    {"role":"system","content":"You are an environmental engineering expert."},
    {"role":"user","content":prompt},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)

log("[4] generating...")
t3 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

log(f"[4] done. elapsed={time.time()-t3:.1f}s")
log("\n=== OUTPUT ===\n" + ans.strip())
