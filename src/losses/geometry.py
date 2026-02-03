from __future__ import annotations
import torch
import torch.nn.functional as F

def sentence_embeddings(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    hidden_states: [B, T, H]
    returns: [B, H]
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    pooled = (hidden_states * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))
    return pooled

def pairwise_dist(x: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    """
    x: [B, H]
    returns D: [B, B]
    """
    if metric == "cosine":
        x = F.normalize(x, dim=-1)
        # cosine distance = 1 - cosine similarity
        sim = x @ x.t()
        return 1.0 - sim
    if metric == "l2":
        return torch.cdist(x, x, p=2)
    raise ValueError(f"Unknown metric: {metric}")

def relational_geometry_loss(student_h: torch.Tensor, teacher_h: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    """
    Topology/geometry-inspired structure preservation:
    match pairwise distance matrices after normalization.
    """
    Ds = pairwise_dist(student_h, metric=metric)
    Dt = pairwise_dist(teacher_h, metric=metric)

    # normalize to reduce scale sensitivity
    Ds = Ds / (Ds.mean().clamp_min(1e-6))
    Dt = Dt / (Dt.mean().clamp_min(1e-6))

    return F.mse_loss(Ds, Dt)
