"""Apply torchao real INT4 quantization to a FlatQuant-calibrated model.

Supports W4A4 (preferred) and W4A16 (fallback). Loads FlatQuant rotation
matrices + learnable diagonals into the FP weights before applying torchao,
so the rotated/normalized weights are what torchao packs to INT4.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torchao.quantization import quantize_, Int4WeightOnlyConfig
try:
    from torchao.quantization import Int4DynamicActivationInt4WeightConfig
except ImportError:
    Int4DynamicActivationInt4WeightConfig = None  # type: ignore


def _load_calibration(model: nn.Module, state):
    """Apply FlatQuant rotations + learnable params from saved state. No-op if state is None.

    Expects state keyed by module-path -> dict(rotation_R, learnable_scale, ...).
    """
    if state is None:
        return model
    for name, mod in model.named_modules():
        if name in state:
            for k, v in state[name].items():
                if hasattr(mod, k):
                    if torch.is_tensor(v) and v.dim() > 0:
                        setattr(mod, k, nn.Parameter(v.to(mod.weight.device)))
                    else:
                        setattr(mod, k, v)
    return model


def _filter_linear(mod, fqn):
    """torchao filter fn: only quantize nn.Linear, skip lm_head."""
    return isinstance(mod, nn.Linear) and "lm_head" not in fqn


def apply_torchao(model: nn.Module, config_name: str = "w4a4", calibration_state=None) -> nn.Module:
    """Quantize `model` in-place using torchao.
    config_name: 'w4a4' (preferred) or 'w4a16' (fallback).
    Returns the same model (mutated).
    """
    model = _load_calibration(model, calibration_state)
    if config_name == "w4a4":
        if Int4DynamicActivationInt4WeightConfig is None:
            print("[torchao] W4A4 config unavailable; falling back to W4A16")
            config_name = "w4a16"
        else:
            try:
                quantize_(model, Int4DynamicActivationInt4WeightConfig(), filter_fn=_filter_linear)
                return model
            except Exception as e:
                print(f"[torchao] W4A4 failed ({type(e).__name__}: {e}); falling back to W4A16")
                config_name = "w4a16"
    if config_name == "w4a16":
        quantize_(model, Int4WeightOnlyConfig(group_size=128), filter_fn=_filter_linear)
        return model
    raise ValueError(f"Unknown config_name: {config_name}")
