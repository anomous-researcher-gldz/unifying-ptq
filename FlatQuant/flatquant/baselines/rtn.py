"""Round-to-nearest weight-only quantization, with optional DBAF folding.

Intentionally a 'weak' baseline: no rotation, no calibration loop, no GPTQ
block solve. Purpose: expose how much DBAF helps when the baseline doesn't
already address outliers.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _dbaf_fold(w: torch.Tensor, alpha: float = 0.75, T_sigma: float = 3.0):
    sigma = w.std()
    T = T_sigma * sigma
    sgn = torch.sign(w)
    mask = w.abs() > T
    out = w.clone()
    out[mask] = sgn[mask] * T + alpha * (w[mask] - sgn[mask] * T)
    return out, T, alpha


def _dbaf_unfold(w_q: torch.Tensor, T: torch.Tensor, alpha: float) -> torch.Tensor:
    sgn = torch.sign(w_q)
    mask = w_q.abs() > T
    out = w_q.clone()
    out[mask] = sgn[mask] * T + (1.0 / alpha) * (w_q[mask] - sgn[mask] * T)
    return out


def _quantize_tensor_uniform(w: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max() / qmax
    if scale.item() == 0:
        return w.clone()
    q = torch.round(w / scale).clamp(-qmax, qmax)
    return (q * scale).to(w.dtype)


def quantize_model(model: nn.Module, bits: int = 4, use_dbaf: bool = False, alpha: float = 0.75, **_unused) -> nn.Module:
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if "lm_head" in name:
            continue
        w = mod.weight.data
        if use_dbaf:
            w_fold, T, a = _dbaf_fold(w, alpha=alpha)
            w_q = _quantize_tensor_uniform(w_fold, bits)
            w_out = _dbaf_unfold(w_q, T, a)
        else:
            w_out = _quantize_tensor_uniform(w, bits)
        mod.weight.data = w_out.to(mod.weight.dtype)
    return model
