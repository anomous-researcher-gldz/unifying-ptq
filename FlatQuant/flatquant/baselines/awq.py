"""Lightweight AWQ baseline: per-channel activation-magnitude scaling
followed by RTN quantization, with optional DBAF folding pre-quantization.

Simplified port — not the full AWQ paper algorithm — sufficient as a
'weak rotation-free baseline'.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from flatquant.baselines.rtn import _dbaf_fold, _dbaf_unfold, _quantize_tensor_uniform


def _activation_scales(model: nn.Module, calibration_data: torch.Tensor) -> dict:
    scales: dict = {}
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            def make_hook(n):
                def hook(_module, inp, _out):
                    x = inp[0].detach().reshape(-1, inp[0].shape[-1]).abs().mean(dim=0)
                    if n not in scales:
                        scales[n] = x
                    else:
                        scales[n] = (scales[n] + x) / 2
                return hook
            handles.append(mod.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(calibration_data)
    for h in handles:
        h.remove()
    return scales


def quantize_model(
    model: nn.Module,
    bits: int = 4,
    calibration_data: torch.Tensor | None = None,
    use_dbaf: bool = False,
    alpha_dbaf: float = 0.75,
    **_unused,
) -> nn.Module:
    if calibration_data is None:
        raise ValueError("AWQ requires calibration_data")
    scales = _activation_scales(model, calibration_data)
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or "lm_head" in name:
            continue
        s = scales.get(name)
        if s is None:
            continue
        s_clip = s.clamp(min=1e-5).pow(0.5).to(mod.weight.dtype)
        w = mod.weight.data * s_clip.view(1, -1)
        if use_dbaf:
            w_fold, T, a = _dbaf_fold(w, alpha=alpha_dbaf)
            w_q = _quantize_tensor_uniform(w_fold, bits)
            w_out = _dbaf_unfold(w_q, T, a)
        else:
            w_out = _quantize_tensor_uniform(w, bits)
        mod.weight.data = (w_out / s_clip.view(1, -1)).to(mod.weight.dtype)
    return model
