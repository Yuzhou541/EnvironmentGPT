import os, sys, time, inspect, logging
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("train_lora_sft_v3")

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

log.info(f"[cfg] BASE_MODEL={MODEL}")
log.info(f"[cfg] train_file={train_file}")
log.info(f"[cfg] eval_file={eval_file}")
log.info(f"[cfg] out_dir={out_dir}")
log.info(f"[cfg] max_seq_len={max_seq_len} lr={lr} epochs={epochs} bsz={bsz} gas={gas}")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    log.info("[cfg] cuda available, TF32 enabled")

# --- tokenizer ---
log.info("[init] loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
log.info(f"[cfg] torch_dtype={dtype}")

# --- model ---
log.info("[init] loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto",
    local_files_only=True,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# --- LoRA ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

# --- dataset ---
log.info("[data] loading dataset jsonl...")
ds = load_dataset(
    "json",
    data_files={"train": train_file, "eval": eval_file},
)

log.info(f"[data] train={len(ds['train'])} eval={len(ds['eval'])}")

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a helpful environmental chemistry assistant.")

def to_text(example):
    p = (example.get("prompt") or "").strip()
    r = (example.get("response") or "").strip()
    if not p or not r:
        return {"text": ""}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": p},
        {"role": "assistant", "content": r},
    ]
    txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": txt}

log.info("[data] building text field (no disk cache)...")
ds = ds.map(
    to_text,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=False,
)

# 过滤空行
ds = ds.filter(lambda x: x["text"] is not None and len(x["text"]) > 0)

log.info(f"[data] after filter: train={len(ds['train'])} eval={len(ds['eval'])}")

# --- TrainingArguments (兼容不同 transformers 版本参数名) ---
ta = dict(
    output_dir=out_dir,
    per_device_train_batch_size=bsz,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=gas,
    num_train_epochs=epochs,
    learning_rate=lr,
    logging_steps=logging_steps,
    save_steps=save_steps,
    save_total_limit=2,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    weight_decay=0.0,
    fp16=(dtype == torch.float16),
    bf16=(dtype == torch.bfloat16),
    report_to="none",
)

sig = inspect.signature(TrainingArguments).parameters
if "evaluation_strategy" in sig:
    ta["evaluation_strategy"] = "steps"
elif "eval_strategy" in sig:
    ta["eval_strategy"] = "steps"

# 有的版本需要显式 do_eval
if "do_eval" in sig:
    ta["do_eval"] = True

# steps 参数名基本不变
ta["eval_steps"] = eval_steps

training_args = TrainingArguments(**_filter_kwargs(TrainingArguments, ta))
log.info("[train] TrainingArguments created.")

# --- Trainer ---
trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["eval"],
    peft_config=lora_config,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=max_seq_len,
    packing=False,
)

trainer = SFTTrainer(**_filter_kwargs(SFTTrainer, trainer_kwargs))
log.info("[train] trainer built, starting training...")

t0 = time.time()
trainer.train()
dt = time.time() - t0
log.info(f"[done] train finished in {dt/60:.1f} min")

log.info("[save] saving adapter + tokenizer...")
trainer.save_model(out_dir)
tokenizer.save_pretrained(out_dir)
log.info(f"[done] saved to: {out_dir}")
