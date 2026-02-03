import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROMPTS = [
  "Explain why low pH suppresses hydrogen production in dark fermentation and propose corrective actions.",
  "Why does high ammonia inhibit hydrogen-producing bacteria in anaerobic fermentation?",
  "Explain the role of ORP in maintaining hydrogen-producing metabolic pathways.",
  "Why does long HRT promote methanogenesis and how can it be suppressed?",
  "Explain how VFA accumulation destabilizes dark fermentation systems."
]

BASE_MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
FULL_LORA="/root/EnvironmentGPT/outputs/envgpt_qwen2p5_7b_lora"
SEED2_LORA="/root/EnvironmentGPT/outputs/envgpt_qwen2p5_7b_lora_seed2"

def load_model(base, adapter=None):
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True, device_map="auto")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, device_map="auto")
    model.eval()
    return tok, model

def run(name, base, adapter=None):
    tok, model = load_model(base, adapter)
    print(f"\n===== {name} =====")
    for i, p in enumerate(PROMPTS):
        text = tok.apply_chat_template(
            [{"role":"user","content":p}],
            tokenize=False,
            add_generation_prompt=True
        )
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=220, do_sample=False)
        ans = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n--- Q{i+1} ---\n{ans.strip()}")

if __name__ == "__main__":
    run("BASE", BASE_MODEL)
    run("FULL_LORA", BASE_MODEL, FULL_LORA)
    run("SEED2_LORA", BASE_MODEL, SEED2_LORA)
