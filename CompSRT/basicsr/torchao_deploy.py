"""Apply torchao real INT4 to a CompSRT-calibrated SwinIR.

CompSRT applies Hadamard transformations and DBAF before quantization. We
bake those transforms into FP weights and then hand to torchao.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torchao.quantization import quantize_, Int4WeightOnlyConfig
try:
    from torchao.quantization import Int4DynamicActivationInt4WeightConfig
except ImportError:
    Int4DynamicActivationInt4WeightConfig = None  # type: ignore


def _bake_compsrt_state(model: nn.Module, state):
    if state is None:
        return model
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name in state:
            entry = state[name]
            with torch.no_grad():
                w = mod.weight.data
                if "hadamard" in entry:
                    H = entry["hadamard"].to(w.device, w.dtype)
                    w = H @ w
                if "dbaf_alpha" in entry and "dbaf_T" in entry:
                    alpha = entry["dbaf_alpha"]
                    T = entry["dbaf_T"]
                    sgn = torch.sign(w)
                    mask = w.abs() > T
                    w[mask] = sgn[mask] * T + alpha * (w[mask] - sgn[mask] * T)
                mod.weight.data = w
    return model


def _filter_swinir(mod, fqn):
    return isinstance(mod, nn.Linear) and "conv_" not in fqn


def apply_torchao_swinir(model: nn.Module, config_name: str = "w4a4", calibration_state=None) -> nn.Module:
    model = _bake_compsrt_state(model, calibration_state)
    if config_name == "w4a4" and Int4DynamicActivationInt4WeightConfig is not None:
        try:
            quantize_(model, Int4DynamicActivationInt4WeightConfig(), filter_fn=_filter_swinir)
            return model
        except Exception as e:
            print(f"[torchao SwinIR] W4A4 failed ({e}); W4A16 fallback")
    quantize_(model, Int4WeightOnlyConfig(group_size=128), filter_fn=_filter_swinir)
    return model
