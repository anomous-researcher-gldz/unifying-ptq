import torch
from typing import Tuple, Dict, Any

@torch.no_grad()
def build_outlier_mask(x: torch.Tensor, k: float = 3.0) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Create a boolean mask for elements outside k standard deviations from the mean.
    Returns:
      mask: bool tensor, True where element is an outlier
      stats: dict with mean, std, low, high (all broadcastable scalars)
    """
    # Compute per-tensor mean/std on the same device/dtype
    mean = x.mean()
    # Use unbiased=False for numerical stability with small tensors
    std = x.std(unbiased=False)

    # If std is zero (constant tensor), no outliers
    if torch.isnan(std) or (std == 0):
        mask = torch.zeros_like(x, dtype=torch.bool)
        stats = {"mean": mean, "std": std, "low": mean, "high": mean}
        return mask, stats

    low = mean - k * std
    high = mean + k * std
    mask = (x < low) | (x > high)
    stats = {"mean": mean, "std": std, "low": low, "high": high}
    return mask, stats


@torch.no_grad()
def extract_outliers(x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Grab outlier indices and full-precision values for later restoration.
    Returns:
      idx: int64 tensor of shape [N, ndim] with coordinates
      vals: float32 tensor of shape [N]
    """
    if mask.numel() == 0:
        return torch.empty((0, x.ndim), dtype=torch.long, device=x.device), torch.empty((0,), dtype=torch.float32, device=x.device)

    # Coordinates as [N, ndim]
    coords = torch.nonzero(mask, as_tuple=False)  # (N, D)
    vals = x[mask].to(torch.float32)             # store in FP32 regardless of original dtype
    return coords, vals


@torch.no_grad()
def restore_outliers(tgt: torch.Tensor, idx: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    """
    In-place restore: scatter the full-precision outlier values back into 'tgt'.
    Args:
      tgt: tensor to be modified (typically dequantized tensor)
      idx: [N, ndim] long coordinates
      vals: [N] float32 values to write
    Returns:
      tgt (same object), for convenience
    """
    if idx.numel() == 0:
        return tgt

    # Convert coords to tuple for advanced indexing
    # idx: (N, D) -> tuple of D index tensors of shape (N,)
    index_tuple = tuple(idx[:, d] for d in range(idx.shape[1]))
    tgt[index_tuple] = vals.to(tgt.dtype)
    return tgt
def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def fake_quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max):
    # k = 3.0
    # mask, stats = build_outlier_mask(x, k=k)
    # idx, vals = extract_outliers(x, mask)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    # x_dequant = restore_outliers(x_dequant, idx, vals)
    return x_dequant

def fake_logquantize_per_tensor_affine(x, scale, quant_min, quant_max, tau=2):
    levels = quant_max - quant_min + 1
    # k = 3.0
    # mask, stats = build_outlier_mask(x, k=k)
    # idx, vals = extract_outliers(x, mask)
    x = torch.clamp(x,1e-20,None)
    x_int = round_ste(-1 * (x/scale).log2() * tau)
    softmax_mask = ((x_int >= levels))
    x_q = torch.clamp(x_int, 0, levels - 1)
    X = scale * 2 ** (-1 * x_q / tau )
    X[softmax_mask] = torch.Tensor([0.0])
    # X = restore_outliers(X, idx, vals)

    return X

def fake_quantize_per_channel_affine(x, scale, zero_point, ch_axis, quant_min, quant_max):
    # k = 3.0
    # mask, stats = build_outlier_mask(x, k=k)
    # idx, vals = extract_outliers(x, mask)
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = scale.reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    # x_dequant = restore_outliers(x_dequant, idx, vals)
    return x_dequant


def fake_quantize_learnable_per_tensor_affine_training(x, scale, zero_point, quant_min, quant_max, grad_factor):
    # k = 3.0
    # mask, stats = build_outlier_mask(x, k=k)
    # idx, vals = extract_outliers(x, mask)
    if (x.device!= scale.device): 
        x = x.to("cuda")
        scale = scale.to("cuda")
        # zero_point = zero_point.to("cuda")
    scale = grad_scale(scale, grad_factor)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    # x_dequant = restore_outliers(x_dequant, idx, vals)
    return x_dequant


def fake_quantize_learnable_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
    # k = 3.0
    # mask, stats = build_outlier_mask(x, k=k)
    # idx, vals = extract_outliers(x, mask)
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    # x_dequant = restore_outliers(x_dequant, idx, vals)
    return x_dequant


def fake_quantize_learnableplus_per_tensor_affine_training(x, scale, zero_point, quant_min, quant_max, grad_factor):
    # k = 3.0
    # mask, stats = build_outlier_mask(x, k=k)
    # idx, vals = extract_outliers(x, mask)
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    scale = grad_scale(scale, grad_factor)
    zero_point = grad_scale(zero_point, grad_factor)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    # x_dequant = restore_outliers(x_dequant, idx, vals)
    return x_dequant


def fake_quantize_learnableplus_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
    # k = 3.0
    # mask, stats = build_outlier_mask(x, k=k)
    # idx, vals = extract_outliers(x, mask)
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = grad_scale(zero_point, grad_factor).reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    # x_dequant = restore_outliers(x_dequant, idx, vals)
    return x_dequant


def fake_hybrid_quantize_per_tensor_affine(x, fp_min, quant_min, quant_max, scale_log, scale_uni, grid_rate):
    # k = 3.0
    # mask, stats = build_outlier_mask(x, k=k)
    # idx, vals = extract_outliers(x, mask)
    levels = quant_max - quant_min + 1
    levels_log = levels * grid_rate
    levels_uni = levels - levels_log
    xq = x.clone()
    xq = xq - fp_min
    mask_log = (xq <= scale_log)
    mask_uni = ~mask_log

    xq[mask_log] = torch.clamp(xq[mask_log], 1e-20, None)
    xq[mask_log] = round_ste(-1 * (xq[mask_log] / scale_log).log2())
    softmax_mask = (xq >= levels_log)
    xq[mask_log] = torch.clamp(xq[mask_log], 0, levels_log - 1)
    xq[mask_log] = scale_log * 2 ** (-1 * xq[mask_log])
    xq[softmax_mask] = torch.Tensor([0.0])

    xq[mask_uni] = round_ste((xq[mask_uni] - scale_log) / scale_uni)
    xq[mask_uni] = torch.clamp(xq[mask_uni], quant_min, levels_uni - 1)
    xq[mask_uni] = xq[mask_uni] * scale_uni + scale_log

    xq = xq + fp_min
    # xq = restore_outliers(xq, idx, vals)
    return xq



def grad_scale(t, scale):
    return (t - (t * scale)).detach() + (t * scale)
