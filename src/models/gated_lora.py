from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

@dataclass
class LoRAArgs:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None

def _expected_l0(log_alpha: torch.Tensor, beta: float, gamma: float, zeta: float) -> torch.Tensor:
    t = math.log(-gamma / zeta)
    return torch.sigmoid(log_alpha - beta * t)

def _sample_hard_concrete(log_alpha: torch.Tensor, beta: float, gamma: float, zeta: float) -> torch.Tensor:
    u = torch.rand_like(log_alpha)
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
    s_bar = s * (zeta - gamma) + gamma
    return torch.clamp(s_bar, 0.0, 1.0)

def _deterministic_hard_concrete(log_alpha: torch.Tensor, gamma: float, zeta: float) -> torch.Tensor:
    s = torch.sigmoid(log_alpha)
    s_bar = s * (zeta - gamma) + gamma
    return torch.clamp(s_bar, 0.0, 1.0)

class GatedLoRAController(nn.Module):
    """
    Adds module-level gates to LoRA modules. Gates can be input-dependent via router logits.
    """
    def __init__(self, peft_model: nn.Module, beta: float = 2.0, gamma: float = -0.1, zeta: float = 1.1):
        super().__init__()
        self.model = peft_model
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta

        self.module_names: List[str] = []
        self._idx_of: Dict[str, int] = {}

        # Register a learnable global log_alpha per gated module (interpretable circuit strength).
        log_alphas: List[nn.Parameter] = []

        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # One gate per module
                idx = len(self.module_names)
                self.module_names.append(name)
                self._idx_of[name] = idx
                log_alphas.append(nn.Parameter(torch.zeros(())))
                # Patch forward to read active z from controller
                self._patch_module_forward(module, idx)

        if not self.module_names:
            raise RuntimeError("No LoRA modules found to gate. Check target_modules and model architecture.")

        self.log_alpha = nn.ParameterList(log_alphas)

        # Runtime cache: set before forward
        self._active_router_logits: torch.Tensor | None = None  # [B, K]
        self._active_z: torch.Tensor | None = None              # [B, K]

    @property
    def num_gates(self) -> int:
        return len(self.module_names)

    def set_router_logits(self, router_logits: torch.Tensor, training: bool) -> None:
        """
        router_logits: [B, K]
        """
        self._active_router_logits = router_logits
        # Compute z per example per gate
        # log_alpha is scalar per gate; broadcast to [B]
        zs = []
        for k in range(self.num_gates):
            la = self.log_alpha[k].expand(router_logits.size(0))
            # input-conditioned: la + router_logits[:,k]
            logit = la + router_logits[:, k]
            if training:
                z = _sample_hard_concrete(logit, self.beta, self.gamma, self.zeta)
            else:
                z = _deterministic_hard_concrete(logit, self.gamma, self.zeta)
            zs.append(z)
        self._active_z = torch.stack(zs, dim=1)  # [B, K]

    def l0_penalty(self) -> torch.Tensor:
        """
        Expected number of open gates (global part). Router-dependent sparsity is encouraged by route loss separately.
        """
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        for k in range(self.num_gates):
            total = total + _expected_l0(self.log_alpha[k], self.beta, self.gamma, self.zeta)
        return total

    def gate_open_rate(self, threshold: float = 0.5) -> Dict[str, float]:
        if self._active_z is None:
            return {"open_fraction": 0.0}
        z = self._active_z.detach()
        open_frac = float((z >= threshold).float().mean().item())
        return {"open_fraction": open_frac}

    def _patch_module_forward(self, module: nn.Module, gate_idx: int) -> None:
        """
        Patch PEFT LoRA module forward: out = base(x) + z * delta(x).
        """
        base_layer = getattr(module, "base_layer", None) or getattr(module, "linear", None)
        if base_layer is None:
            raise RuntimeError("Cannot locate base_layer for a LoRA module (PEFT internals changed).")

        def patched_forward(x, *args, **kwargs):
            out = base_layer(x, *args, **kwargs)

            # active adapters
            adapters = getattr(module, "active_adapters", None)
            if adapters is None:
                a = getattr(module, "active_adapter", "default")
                adapters = [a] if isinstance(a, str) else list(a)

            # fetch z: [B]
            ctrl: GatedLoRAController = self  # closure
            if ctrl._active_z is None:
                # Fallback: deterministic global gate only
                la = ctrl.log_alpha[gate_idx]
                z = _deterministic_hard_concrete(la, ctrl.gamma, ctrl.zeta).view(1, 1, 1)
            else:
                z_b = ctrl._active_z[:, gate_idx]  # [B]
                # broadcast to [B, 1, 1] for linear output shape [B, T, H]
                z = z_b.view(-1, 1, 1).to(out.dtype)

            for ad in adapters:
                if ad not in module.lora_A or ad not in module.lora_B:
                    continue
                x_d = module.lora_dropout[ad](x) if hasattr(module, "lora_dropout") else x
                delta = module.lora_B[ad](module.lora_A[ad](x_d))
                scale = module.scaling[ad] if isinstance(module.scaling, dict) else module.scaling
                out = out + z * delta * scale
            return out

        module.forward = patched_forward

def build_peft_lora(base_model: nn.Module, lora: LoRAArgs) -> nn.Module:
    cfg = LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora.target_modules,
    )
    return get_peft_model(base_model, cfg)
