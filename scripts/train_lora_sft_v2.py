#!/usr/bin/env python
import os


# [patch] deterministic seed + eval strategy knobs
from transformers import set_seed
SEED = int(os.environ.get('SEED', '0'))
set_seed(SEED)
EVAL_STRATEGY = os.environ.get('EVAL_STRATEGY', 'no')

# [patch] ensure global max_seq_len for tokenize_fn()
max_seq_len = int(os.environ.get('MAX_SEQ_LEN', os.environ.get('MAX_SEQ_LENGTH', '2048')))

import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s train_lora_sft_v2: %(message)s"
)
logger = logging.getLogger(__name__)

# =========================
# Configuration (STRICT)
# =========================
MODEL = os.environ["MODEL"]              # 必须是本地路径
TRAIN_FILE = os.environ["TRAIN_FILE"]
EVAL_FILE = os.environ["EVAL_FILE"]
OUT_DIR = os.environ["OUT_DIR"]

EPOCHS = float(os.environ.get("EPOCHS", 1))
BSZ = int(os.environ.get("BSZ", 1))
GAS = int(os.environ.get("GAS", 8))
LR = float(os.environ.get("LR", 2e-4))
MAX_LEN = int(os.environ.get("MAX_LEN", 2048))

logger.info(f"[cfg] model={MODEL}")
logger.info(f"[cfg] train_file={TRAIN_FILE}")
logger.info(f"[cfg] eval_file={EVAL_FILE}")
logger.info(f"[cfg] out_dir={OUT_DIR}")
logger.info(f"[cfg] max_seq_len={MAX_LEN} lr={LR} epochs={EPOCHS} bsz={BSZ} gas={GAS}")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    logger.info("[cfg] cuda available, TF32 enabled")

# =========================
# Tokenizer (OFFLINE SAFE)
# =========================
logger.info("[init] loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True,
    local_files_only=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# Model (OFFLINE SAFE)
# =========================
logger.info("[init] loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True
)

# =========================
# LoRA
# =========================
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# =========================
# Dataset
# =========================
logger.info("[data] loading datasets...")
dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE, "eval": EVAL_FILE}
)


def _build_texts(examples, tokenizer):
    # batched=True: examples is dict[str, list]
    if "text" in examples:
        return examples["text"]

    # chat-style
    if "messages" in examples:
        return [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in examples["messages"]
        ]
    if "conversations" in examples:
        # some datasets use "conversations" instead of "messages"
        return [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in examples["conversations"]
        ]

    # prompt/response style
    if "prompt" in examples and "response" in examples:
        return [
            tokenizer.apply_chat_template(
                [{"role":"user","content":p},{"role":"assistant","content":r}],
                tokenize=False,
                add_generation_prompt=False
            )
            for p, r in zip(examples["prompt"], examples["response"])
        ]

    # alpaca-style
    if "instruction" in examples and "output" in examples:
        inputs = examples.get("input", [""] * len(examples["instruction"]))
        texts = []
        for inst, inp, out in zip(examples["instruction"], inputs, examples["output"]):
            u = inst if not inp else (inst + "\n" + inp)
            texts.append(
                tokenizer.apply_chat_template(
                    [{"role":"user","content":u},{"role":"assistant","content":out}],
                    tokenize=False,
                    add_generation_prompt=False
                )
            )
        return texts

    raise KeyError(f"Unsupported dataset columns: {list(examples.keys())}")

def tokenize_fn(examples):
    texts = _build_texts(examples, tokenizer)
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
    )

dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# =========================
# Trainer
# =========================
args = TrainingArguments(
    
        prediction_loss_only=True,
        per_device_eval_batch_size=1,
output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BSZ,
    gradient_accumulation_steps=GAS,
    learning_rate=LR,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy=EVAL_STRATEGY,
    bf16=torch.cuda.is_available(),
    fp16=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

logger.info("[train] start training")
trainer.train()
trainer.save_model(OUT_DIR)
logger.info("[done] training finished")
