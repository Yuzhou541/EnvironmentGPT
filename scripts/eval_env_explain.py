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

def load_model(base, adapter=None):
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base,
        trust_remote_code=True,
        device_map="auto"
    )

    if adapter is not None:
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
        ans = tok.decode(
            out[0][inp["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        print(f"\n--- Q{i+1} ---")
        print(ans.strip())

if __name__ == "__main__":
    BASE_MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/" + \
                 sorted(__import__("os").listdir(
                     "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
                 ))[0]

    run("BASE", BASE_MODEL)
    run("FULL_LORA", BASE_MODEL, "outputs/envgpt_qwen2p5_7b_lora")
    run("PRUNED_LORA_0.05", BASE_MODEL, "outputs/envgpt_qwen2p5_7b_lora_pruned_0.05")
