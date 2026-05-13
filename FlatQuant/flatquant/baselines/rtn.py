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


def _quantize_tensor_uniform(w: torch.Tensor, bits: int, per_channel: bool = True) -> torch.Tensor:
    """Uniform symmetric quantization.

    For 2D Linear weights [out, in], per_channel=True uses one scale per
    output row (dim=0). This is the standard LLM-friendly granularity —
    per-tensor scales collapse dynamic range and destroy accuracy.
    """
    qmax = 2 ** (bits - 1) - 1
    if per_channel and w.dim() == 2:
        scale = w.abs().amax(dim=1, keepdim=True) / qmax
        scale = scale.clamp(min=1e-9)
        q = torch.round(w / scale).clamp(-qmax, qmax)
        return (q * scale).to(w.dtype)
    scale = w.abs().max() / qmax
    if scale.item() == 0:
        return w.clone()
    q = torch.round(w / scale).clamp(-qmax, qmax)
    return (q * scale).to(w.dtype)


def _quantize_per_channel_with_dbaf(w: torch.Tensor, bits: int, alpha: float, T_sigma: float = 3.0) -> torch.Tensor:
    """Per-row DBAF fold + quant + unfold, computing T per row from row sigma."""
    qmax = 2 ** (bits - 1) - 1
    # Per-row T
    sigma = w.std(dim=1, keepdim=True)
    T = T_sigma * sigma  # [out, 1]
    sgn = torch.sign(w)
    mask = w.abs() > T
    w_fold = torch.where(mask, sgn * T + alpha * (w - sgn * T), w)
    # Per-row scale on the folded magnitudes
    scale = w_fold.abs().amax(dim=1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-9)
    q = torch.round(w_fold / scale).clamp(-qmax, qmax)
    w_q = q * scale
    # Unfold: only the regions originally outside ±T get expanded back
    sgn_q = torch.sign(w_q)
    mask_q = w_q.abs() > T
    w_out = torch.where(mask_q, sgn_q * T + (1.0 / alpha) * (w_q - sgn_q * T), w_q)
    return w_out.to(w.dtype)


def quantize_model(model: nn.Module, bits: int = 4, use_dbaf: bool = False, alpha: float = 0.75, **_unused) -> nn.Module:
    """Per-channel (per-output-row) RTN with optional DBAF folding."""
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if "lm_head" in name:
            continue
        w = mod.weight.data
        if use_dbaf and w.dim() == 2:
            w_out = _quantize_per_channel_with_dbaf(w, bits, alpha=alpha)
        else:
            w_out = _quantize_tensor_uniform(w, bits, per_channel=True)
        mod.weight.data = w_out.to(mod.weight.dtype)
    return model
