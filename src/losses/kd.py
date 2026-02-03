from __future__ import annotations
import torch
import torch.nn.functional as F

def kd_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    KL( teacher || student ) on token distributions.
    logits: [B, T, V]
    """
    T = float(temperature)
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)
