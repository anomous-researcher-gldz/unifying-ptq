"""SmoothQuant W4A4 baseline with optional DBAF folding.

SmoothQuant migrates per-channel activation outliers into the weight matrix
via an offline diagonal scale:
    s_c = max(|x_c|)^alpha / max(|w_c|)^(1-alpha)
    x' = x / diag(s)
    W' = diag(s) @ W
After migration, the activation distribution is flatter and the weight
distribution slightly steeper. Each is then quantized with per-channel /
per-token RTN. No gradient training, no rotation. Reference:
  Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
  Quantization for Large Language Models", ICML 2023.

DBAF flag: applies dense-with-outliers folding on the migrated weight (which
still has some residual outliers when alpha < 1). Activation quant uses
per-token RTN regardless of DBAF.

Designed to slot into run_training_free_full_table.py alongside RTN/GPTQ/AWQ.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf


def _collect_act_scales(model, calibration_data, alpha: float = 0.5):
    """Offline pass: collect per-channel max abs activation per Linear layer.

    Returns: dict[layer_name -> tensor(d_in,)] of channel-wise max abs.
    """
    model.eval()
    act_scales: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            # Flatten leading dims, take per-channel (last dim) max abs
            x = x.detach().abs()
            x = x.flatten(end_dim=-2)  # [N, d_in]
            cur = x.amax(dim=0)
            if name in act_scales:
                act_scales[name] = torch.maximum(act_scales[name], cur)
            else:
                act_scales[name] = cur.clone()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for batch in calibration_data:
            if isinstance(batch, torch.Tensor):
                ids = batch.to(device)
            elif isinstance(batch, (list, tuple)):
                ids = batch[0].to(device)
            elif isinstance(batch, dict):
                ids = batch["input_ids"].to(device)
            else:
                continue
            # Ensure (B, T) — _calib_batch_llm yields 1D rows when iterated.
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            _ = model(ids)

    for h in hooks:
        h.remove()

    return act_scales


def _smooth_per_layer(weight: torch.Tensor, act_scale: torch.Tensor,
                     alpha: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute and apply the diagonal migration to a single Linear weight.

    weight:    [out, in]
    act_scale: [in]  (per-channel max abs activation seen during calib)

    Returns: (new_weight, scale_vec) where scale_vec[i] is what x is divided by.
    """
    w_max = weight.abs().amax(dim=0).clamp(min=1e-5)            # [in]
    a_max = act_scale.to(weight.dtype).clamp(min=1e-5)          # [in]
    s = (a_max.pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)
    # Migrate: x' = x / s, W' = W * s (column-wise)
    new_w = weight * s.view(1, -1)
    return new_w, s


def _quantize_act_per_token(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Per-token symmetric act quant (same granularity used by all our baselines)."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return (q * scale).to(x.dtype)


class _ActDivideWrapper(nn.Module):
    """Wraps a Linear so the act is divided by `s` before the matmul.

    s is fixed (registered as buffer) from the SmoothQuant offline pass.
    Activation quant happens AFTER division (so per-token scale sees flatter act).
    """
    def __init__(self, linear: nn.Linear, s: torch.Tensor, act_bits: int):
        super().__init__()
        self.linear = linear
        self.register_buffer("smooth_scale", s.detach())
        self.act_bits = act_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_div = x / self.smooth_scale
        x_q = _quantize_act_per_token(x_div, bits=self.act_bits)
        return self.linear(x_q)


def quantize_model(model, bits: int = 4, calibration_data=None,
                  alpha: float = 0.5,
                  use_dbaf: bool = False,
                  act_bits: int | None = None):
    """SmoothQuant W{bits}A{act_bits} migration + RTN per-channel weights.

    Args:
        bits: weight bits.
        calibration_data: iterable of token-id batches.
        alpha: smoothing strength (0=no migration; 1=full migration to weights).
                ICML paper recommends 0.5 for general LLMs.
        use_dbaf: if True, apply DBAF folding gate on the migrated weight per
                  Linear layer (matches the RTN baseline's DBAF integration).
        act_bits: defaults to `bits` if None.
    """
    if calibration_data is None:
        raise ValueError("SmoothQuant requires calibration_data to compute "
                         "activation channel max statistics.")
    if act_bits is None:
        act_bits = bits

    # 1) Collect per-layer activation channel-max scales
    act_scales = _collect_act_scales(model, calibration_data, alpha=alpha)

    # 2) Migrate + quantize per Linear; wrap with _ActDivideWrapper to divide
    #    activations at runtime by the smooth scale.
    name_to_module = dict(model.named_modules())
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name not in act_scales:
            continue
        s = act_scales[name].to(module.weight.device)
        new_w, smooth_s = _smooth_per_layer(module.weight.data, s, alpha=alpha)

        if use_dbaf:
            qw = _quantize_per_channel_with_dbaf(new_w, bits=bits, alpha=0.75)
        else:
            qw = _quantize_tensor_uniform(new_w, bits=bits, per_channel=True)
        module.weight.data.copy_(qw)

        # Replace this module with a wrapped version that divides x by smooth_s
        # before the matmul. Walk the model to find the parent and slot index.
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = name_to_module[parent_name] if parent_name else model
        setattr(parent, child_name,
                _ActDivideWrapper(module, smooth_s, act_bits=act_bits))

    return model
