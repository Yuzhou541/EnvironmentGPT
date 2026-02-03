import argparse, json, math, os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

class JsonlSFT(Dataset):
    def __init__(self, path, tok, max_len=768):
        self.rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        self.tok = tok
        self.max_len = int(max_len)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = r["prompt"].strip()
        resp = r["response"].strip()

        # 拼接：prompt + response + eos
        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        resp_ids = self.tok(resp, add_special_tokens=False).input_ids + [self.tok.eos_token_id]

        # 右侧优先保留 response，必要时截断 prompt
        max_len = self.max_len
        if len(prompt_ids) + len(resp_ids) > max_len:
            keep_prompt = max(0, max_len - len(resp_ids))
            prompt_ids = prompt_ids[-keep_prompt:]

        input_ids = prompt_ids + resp_ids
        labels = [-100]*len(prompt_ids) + resp_ids[:]  # 只监督回答部分

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--train_file", type=str, default="data/processed/student_sft_q1_train.jsonl")
    ap.add_argument("--eval_file", type=str, default="data/processed/student_sft_q1_dev.jsonl")
    ap.add_argument("--output_dir", type=str, default="runs/student_lora_q1")
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = JsonlSFT(args.train_file, tok, max_len=args.max_len)
    eval_ds  = JsonlSFT(args.eval_file,  tok, max_len=args.max_len)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch * args.grad_accum))
    max_steps = steps_per_epoch * args.epochs

    targs = TrainingArguments(
        output_dir=str(outdir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=max(50, steps_per_epoch//2),
        save_steps=max(50, steps_per_epoch//2),
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(outdir))
    tok.save_pretrained(str(outdir))
    print(f"Saved: {outdir}")

if __name__ == "__main__":
    main()
