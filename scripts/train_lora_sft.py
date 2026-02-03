import os, inspect
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

def _filter_kwargs(callable_obj, kwargs: dict):
    sig = inspect.signature(callable_obj).parameters
    return {k: v for k, v in kwargs.items() if k in sig}

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
train_file = os.environ.get("TRAIN_FILE", "data/train/sft_train.jsonl")
eval_file  = os.environ.get("EVAL_FILE",  "data/train/sft_dev.jsonl")
out_dir    = os.environ.get("OUT_DIR",    "outputs/envgpt_qwen2p5_7b_lora")

max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "2048"))
lr = float(os.environ.get("LR", "2e-4"))
epochs = float(os.environ.get("EPOCHS", "1"))
bsz = int(os.environ.get("BSZ", "1"))
gas = int(os.environ.get("GAS", "8"))
save_steps = int(os.environ.get("SAVE_STEPS", "500"))
eval_steps = int(os.environ.get("EVAL_STEPS", "500"))
logging_steps = int(os.environ.get("LOG_STEPS", "20"))

os.makedirs(out_dir, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

print(f"[cfg] model={MODEL}", flush=True)
print(f"[cfg] train_file={train_file}", flush=True)
print(f"[cfg] eval_file={eval_file}", flush=True)
print(f"[cfg] out_dir={out_dir}", flush=True)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

print("[init] loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
print(f"[cfg] torch_dtype={dtype}", flush=True)

print("[init] loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto",
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

print("[data] loading dataset jsonl...", flush=True)
ds = load_dataset("json", data_files={"train": train_file, "eval": eval_file})
print(f"[data] train={len(ds['train'])} eval={len(ds['eval'])}", flush=True)

# 关键：把 (prompt,response) 映射成单列 text，避免 TRL 去找 completion
def to_text(ex):
    p = (ex.get("prompt") or "").strip()
    r = (ex.get("response") or "").strip()
    if not p or not r:
        return {"text": ""}
    messages = [
        {"role": "user", "content": p},
        {"role": "assistant", "content": r},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

remove_cols = ds["train"].column_names
ds = ds.map(to_text, remove_columns=remove_cols)
# 过滤空样本（保险）
ds["train"] = ds["train"].filter(lambda x: bool((x.get("text") or "").strip()))
ds["eval"]  = ds["eval"].filter(lambda x: bool((x.get("text") or "").strip()))
print(f"[data] after_map train={len(ds['train'])} eval={len(ds['eval'])}", flush=True)

# transformers 4.57+ 把 evaluation_strategy 改名为 eval_strategy
ta_sig = inspect.signature(TrainingArguments.__init__).parameters
eval_key = "evaluation_strategy" if "evaluation_strategy" in ta_sig else "eval_strategy"

ta_kwargs = dict(
    output_dir=out_dir,
    per_device_train_batch_size=bsz,
    per_device_eval_batch_size=bsz,
    gradient_accumulation_steps=gas,
    learning_rate=lr,
    num_train_epochs=epochs,
    logging_steps=logging_steps,
    save_steps=save_steps,
    eval_steps=eval_steps,
    save_strategy="steps",
    logging_strategy="steps",
    report_to="none",
    remove_unused_columns=False,
    bf16=(dtype == torch.bfloat16),
    fp16=(dtype == torch.float16),
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    weight_decay=0.0,
    max_grad_norm=1.0,
)
ta_kwargs[eval_key] = "steps" if eval_steps > 0 else "no"
training_args = TrainingArguments(**_filter_kwargs(TrainingArguments.__init__, ta_kwargs))
print(f"[train] TrainingArguments OK ({eval_key}={ta_kwargs[eval_key]})", flush=True)

# 兼容不同 TRL 版本参数
sft_sig = inspect.signature(SFTTrainer.__init__).parameters
trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["eval"],
    peft_config=lora_config,
)

# tokenizer/processing_class
if "tokenizer" in sft_sig:
    trainer_kwargs["tokenizer"] = tokenizer
elif "processing_class" in sft_sig:
    trainer_kwargs["processing_class"] = tokenizer

# 用 text 字段
if "dataset_text_field" in sft_sig:
    trainer_kwargs["dataset_text_field"] = "text"

# max length
if "max_seq_length" in sft_sig:
    trainer_kwargs["max_seq_length"] = max_seq_len
elif "max_length" in sft_sig:
    trainer_kwargs["max_length"] = max_seq_len

if "packing" in sft_sig:
    trainer_kwargs["packing"] = False

print("[train] building trainer...", flush=True)
trainer = SFTTrainer(**trainer_kwargs)

print("[train] start training...", flush=True)
trainer.train()

print("[save] saving adapter + tokenizer...", flush=True)
trainer.save_model(out_dir)
tokenizer.save_pretrained(out_dir)
print("[done] finished ->", out_dir, flush=True)
