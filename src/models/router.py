from __future__ import annotations
import torch
import torch.nn as nn

class PromptRouter(nn.Module):
    """
    A lightweight router that outputs per-module gate logits from prompt embeddings.
    """
    def __init__(self, d_model: int, num_gates: int, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_gates),
        )

    def forward(self, prompt_emb: torch.Tensor) -> torch.Tensor:
        """
        prompt_emb: [B, d_model] pooled embedding
        returns logits: [B, num_gates]
        """
        return self.net(prompt_emb)
