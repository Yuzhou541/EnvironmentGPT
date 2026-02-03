from __future__ import annotations
import os
import math
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model

from .utils.config import load_yaml, parse_overrides, deep_update
from .utils.seed import set_seed
from .data.loader import JsonlDataset
from .data.collator import SFTCollator

from .models.gated_lora import LoRAArgs
from .models.topoguard import TopoGuardModel, TopoGuardArgs
from .losses.kd import kd_kl_loss
from .losses.geometry import sentence_embeddings, relational_geometry_loss

hf_logging.set_verbosity_error()

def _maybe_bnb_kwargs(load_in_4bit: bool, dtype: torch.dtype) -> Dict[str, Any]:
    if not load_in_4bit:
        return {}
    return dict(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

def load_base_model(model_name: str, load_in_4bit: bool, torch_dtype: str, attn_impl: str):
    dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map="auto",
        **_maybe_bnb_kwargs(load_in_4bit, dtype),
    )
    return model, tok, dtype

def build_lora_model(base_model, cfg: Dict[str, Any]):
    m = cfg["method"]
    lcfg = LoraConfig(
        r=m["lora_r"],
        lora_alpha=m["lora_alpha"],
        lora_dropout=m["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=m["target_modules"],
    )
    return get_peft_model(base_model, lcfg)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("overrides", nargs="*", help="key=value overrides")
    args = ap.parse_args()

    cfg = deep_update(load_yaml(args.config), parse_overrides(args.overrides))
    set_seed(int(cfg.get("seed", 42)))

    model_name = cfg["model"]["name_or_path"]
    if not model_name or model_name == "REPLACE_ME":
        raise ValueError("Set model.name_or_path via config or CLI override.")

    base, tok, dtype = load_base_model(
        model_name=model_name,
        load_in_4bit=bool(cfg["model"].get("load_in_4bit", False)),
        torch_dtype=str(cfg["model"].get("torch_dtype", "bfloat16")),
        attn_impl=str(cfg["model"].get("attn_implementation", "sdpa")),
    )

    train_ds = JsonlDataset(cfg["data"]["train_path"])
    dev_ds = JsonlDataset(cfg["data"]["dev_path"])
    gen_ds = JsonlDataset(cfg["data"]["general_path"])

    collator = SFTCollator(tok, int(cfg["data"]["max_length"]))
    bs = int(cfg["train"]["per_device_train_batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_ds, batch_size=1, shuffle=False, collate_fn=collator)
    gen_loader = DataLoader(gen_ds, batch_size=bs, shuffle=True, collate_fn=collator)
    gen_iter = iter(gen_loader)

    method = cfg["method"]["name"]
    use_topoguard = (method == "topoguard")

    if use_topoguard:
        lora_args = LoRAArgs(
            r=int(cfg["method"]["lora_r"]),
            alpha=int(cfg["method"]["lora_alpha"]),
            dropout=float(cfg["method"]["lora_dropout"]),
            target_modules=cfg["method"]["target_modules"],
        )
        tg_cfg = cfg.get("topoguard", {})
        tg_args = TopoGuardArgs(
            l0_lambda=float(tg_cfg.get("l0_lambda", 1e-4)),
            route_lambda=float(tg_cfg.get("route_lambda", 1e-3)),
            kd_lambda=float(tg_cfg.get("kd_lambda", 1.0)),
            kd_temperature=float(tg_cfg.get("kd_temperature", 1.0)),
            geom_lambda=float(tg_cfg.get("geom_lambda", 0.2)),
            geom_layer=int(tg_cfg.get("geom_layer", -1)),
            geom_metric=str(tg_cfg.get("geom_metric", "cosine")),
        )
        model = TopoGuardModel(base, lora_args, tg_args)
        model.train()
    else:
        model = build_lora_model(base, cfg)
        model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.6f}%)")

    lr = float(cfg["train"]["learning_rate"])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    num_epochs = int(cfg["train"]["num_train_epochs"])
    grad_acc = int(cfg["train"]["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(train_loader) / grad_acc)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * float(cfg["train"].get("warmup_ratio", 0.03)))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    outdir = cfg["train"]["output_dir"]
    os.makedirs(outdir, exist_ok=True)

    best_dev = float("inf")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    def run_dev_loss() -> float:
        model.eval()
        loss_sum = 0.0
        with torch.no_grad():
            for b in dev_loader:
                b = {k: v.to(next(model.parameters()).device) for k, v in b.items()}
                if use_topoguard:
                    out = model.forward_student(**b)
                    loss = out.loss
                else:
                    out = model(**b)
                    loss = out.loss
                loss_sum += float(loss.item())
        model.train()
        return loss_sum / max(1, len(dev_loader))

    for epoch in range(num_epochs):
        running = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}

            if not use_topoguard:
                out = model(**batch)
                loss = out.loss
            else:
                # Domain SFT
                out_s = model.forward_student(**batch)
                loss_domain = out_s.loss

                # Sparsity (global L0)
                l0 = model.student.l0_penalty() * model.tg.l0_lambda

                # General anchors batch
                try:
                    g = next(gen_iter)
                except StopIteration:
                    gen_iter = iter(gen_loader)
                    g = next(gen_iter)
                g = {k: v.to(next(model.parameters()).device) for k, v in g.items()}

                # Encourage router to close gates on general anchors
                model.set_gates(g["input_ids"], g["attention_mask"], training=True)
                z = model.student._active_z  # [B, K]
                route_loss = z.abs().mean() * model.tg.route_lambda

                # KD on general anchors (teacher vs student)
                t_out = model.forward_teacher(**g)
                s_out = model.forward_student(**g)
                kd = kd_kl_loss(s_out.logits, t_out.logits, temperature=model.tg.kd_temperature) * model.tg.kd_lambda

                # Geometry consistency on general anchors: compare sentence embeddings from a chosen layer
                # We request hidden states on both models; keep it small (anchors batch size is small).
                with torch.no_grad():
                    t_h = model.teacher(**g, output_hidden_states=True).hidden_states[model.tg.geom_layer]
                s_h = model.student.model(**g, output_hidden_states=True).hidden_states[model.tg.geom_layer]
                t_sent = sentence_embeddings(t_h, g["attention_mask"])
                s_sent = sentence_embeddings(s_h, g["attention_mask"])
                geom = relational_geometry_loss(s_sent, t_sent, metric=model.tg.geom_metric) * model.tg.geom_lambda

                loss = loss_domain + l0 + route_loss + kd + geom

            (loss / grad_acc).backward()
            running += float(loss.item())

            if (step + 1) % grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % int(cfg["train"].get("logging_steps", 10)) == 0:
                    if use_topoguard:
                        stats = model.gate_open_stats()
                        print(f"epoch={epoch} step={global_step} loss={running:.4f} gates={stats}")
                    else:
                        print(f"epoch={epoch} step={global_step} loss={running:.4f}")
                    running = 0.0

                if global_step % int(cfg["train"].get("eval_steps", 200)) == 0:
                    dev_loss = run_dev_loss()
                    print(f"[dev] step={global_step} dev_loss={dev_loss:.4f}")
                    if dev_loss < best_dev:
                        best_dev = dev_loss
                        ckpt = os.path.join(outdir, "checkpoint-best")
                        os.makedirs(ckpt, exist_ok=True)
                        # Save only PEFT adapter when possible
                        if use_topoguard:
                            model.student.model.save_pretrained(ckpt)
                            tok.save_pretrained(ckpt)
                            torch.save(model.router.state_dict(), os.path.join(ckpt, "router.pt"))
                            torch.save([p.detach().cpu() for p in model.student.log_alpha], os.path.join(ckpt, "log_alpha.pt"))
                        else:
                            model.save_pretrained(ckpt)
                            tok.save_pretrained(ckpt)

                if global_step % int(cfg["train"].get("save_steps", 200)) == 0:
                    ckpt = os.path.join(outdir, f"checkpoint-step-{global_step}")
                    os.makedirs(ckpt, exist_ok=True)
                    if use_topoguard:
                        model.student.model.save_pretrained(ckpt)
                        tok.save_pretrained(ckpt)
                        torch.save(model.router.state_dict(), os.path.join(ckpt, "router.pt"))
                        torch.save([p.detach().cpu() for p in model.student.log_alpha], os.path.join(ckpt, "log_alpha.pt"))
                    else:
                        model.save_pretrained(ckpt)
                        tok.save_pretrained(ckpt)

    print("Training done.")
    print(f"Best dev checkpoint in: {os.path.join(outdir, 'checkpoint-best')} (dev_loss={best_dev:.4f})")

if __name__ == "__main__":
    main()
