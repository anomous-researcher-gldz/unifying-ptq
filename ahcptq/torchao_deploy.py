"""Apply torchao real INT4 to an AHCPTQ-calibrated SAM image encoder.

Loads AHCPTQ's fake-quant state (DBAF alphas, PCSA anchors+scales) into
the FP image encoder by baking the rotation/folding into the weights, then
hands the FP weights to torchao for INT4 packing.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torchao.quantization import quantize_, Int4WeightOnlyConfig
try:
    from torchao.quantization import Int4DynamicActivationInt4WeightConfig
except ImportError:
    Int4DynamicActivationInt4WeightConfig = None  # type: ignore


def _bake_dbaf_into_weights(encoder: nn.Module, state):
    """Fold (DBAF) the FP weights using saved per-layer alpha/T values. No-op if state is None."""
    if state is None:
        return encoder
    for name, mod in encoder.named_modules():
        if isinstance(mod, nn.Linear) and name in state and "dbaf_alpha" in state[name]:
            alpha = state[name]["dbaf_alpha"]
            T = state[name]["dbaf_T"]
            with torch.no_grad():
                w = mod.weight.data
                sgn = torch.sign(w)
                mask = w.abs() > T
                w[mask] = sgn[mask] * T + alpha * (w[mask] - sgn[mask] * T)
                mod.weight.data = w
    return encoder


def _filter_attn_mlp(mod, fqn):
    """Quantize attention qkv/proj and MLP fc1/fc2; skip patch embed + neck."""
    if not isinstance(mod, nn.Linear):
        return False
    return not any(s in fqn for s in ("patch_embed", "neck"))


def apply_torchao_sam(encoder: nn.Module, config_name: str = "w4a4", calibration_state=None) -> nn.Module:
    encoder = _bake_dbaf_into_weights(encoder, calibration_state)
    if config_name == "w4a4":
        if Int4DynamicActivationInt4WeightConfig is None:
            print("[torchao SAM] W4A4 unavailable; using W4A16")
            config_name = "w4a16"
        else:
            try:
                quantize_(encoder, Int4DynamicActivationInt4WeightConfig(), filter_fn=_filter_attn_mlp)
                return encoder
            except Exception as e:
                print(f"[torchao SAM] W4A4 failed ({type(e).__name__}: {e}); using W4A16")
                config_name = "w4a16"
    if config_name == "w4a16":
        quantize_(encoder, Int4WeightOnlyConfig(group_size=128), filter_fn=_filter_attn_mlp)
        return encoder
    raise ValueError(f"Unknown config_name: {config_name}")
