from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .gated_lora import LoRAArgs, build_peft_lora, GatedLoRAController
from .router import PromptRouter

@dataclass
class TopoGuardArgs:
    l0_lambda: float = 1e-4
    route_lambda: float = 1e-3
    kd_lambda: float = 1.0
    kd_temperature: float = 1.0
    geom_lambda: float = 0.2
    geom_layer: int = -1
    geom_metric: str = "cosine"

class TopoGuardModel(nn.Module):
    """
    Wrapper: base model (teacher) + gated LoRA student + router.
    Training uses:
      - domain SFT loss
      - sparsity (L0 expected)
      - routing loss on general anchors
      - KD loss on general anchors (teacher=base)
      - geometry consistency loss on general anchors (teacher=base)
    """
    def __init__(self, base_model: AutoModelForCausalLM, lora_args: LoRAArgs, tg_args: TopoGuardArgs):
        super().__init__()
        self.teacher = base_model.eval()  # frozen reference
        for p in self.teacher.parameters():
            p.requires_grad = False

        student_peft = build_peft_lora(base_model, lora_args)
        self.student = GatedLoRAController(student_peft)

        # Router dimension from hidden size
        d_model = base_model.config.hidden_size
        self.router = PromptRouter(d_model=d_model, num_gates=self.student.num_gates, hidden=min(1024, d_model), dropout=0.0)

        self.tg = tg_args

    def pool_prompt_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool token embeddings as prompt representation for routing.
        """
        emb_layer = self.teacher.get_input_embeddings()
        x = emb_layer(input_ids)  # [B, T, H]
        mask = attention_mask.unsqueeze(-1).to(x.dtype)  # [B, T, 1]
        pooled = (x * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))
        return pooled  # [B, H]

    def set_gates(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, training: bool) -> torch.Tensor:
        pooled = self.pool_prompt_embedding(input_ids, attention_mask)
        logits = self.router(pooled)  # [B, K]
        self.student.set_router_logits(logits, training=training)
        return logits

    def forward_student(self, **batch):
        self.set_gates(batch["input_ids"], batch["attention_mask"], training=self.training)
        return self.student.model(**batch)

    def forward_teacher(self, **batch):
        with torch.no_grad():
            return self.teacher(**batch)

    def gate_open_stats(self) -> Dict[str, float]:
        return self.student.gate_open_rate()
