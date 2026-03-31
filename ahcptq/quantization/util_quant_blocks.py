
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any

__all__ = [
    "round_ste",
    "grad_scale",
    "fake_quantize_per_block_affine",
    "flatten_into_blocks", 
    "reconstruct_from_blocks",
    "_expand_param_for_blocks",
    "_sanitize_scale",
    "compute_block_min_max", 
    "flatten_into_channels", 
    "reconstruct_from_channels",
    "build_outlier_mask", 
    "extract_outliers",
    "restore_outliers" 



]
_MIN_SCALE = 1e-8
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

def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator for rounding: forward uses round, backward passes gradients."""
    return (x.round() - x).detach() + x

def grad_scale(x, scale: float):
    # forward: y == x; backward: dy/dx scaled by `scale`
    return (x - x.detach()) * scale + x.detach()

def flatten_into_blocks(x: torch.Tensor, block_size: int, pad_value: int = 0):
    """
    Fully flatten `x` to 1D, split into blocks of length `block_size`.
    If total length isn't divisible by `block_size`, pad the tail with `pad_value`.

    Returns:
      blocks:   [num_blocks, block_size] tensor
      meta:     dict with info for reconstruction:
                {
                  'original_shape': tuple,
                  'valid_length': int,   # number of real (unpadded) elements
                  'block_size': int,
                  'pad_value': float,
                  'dtype': torch.dtype,
                  'device': torch.device,
                }
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    original_shape = tuple(x.shape)
    device, dtype = x.device, x.dtype

    flat = x.reshape(-1)  # 1D
    valid_length = flat.numel()
    remainder = valid_length % block_size
    if remainder != 0:
        # print("hello",pad_len)
        pad_len = int(block_size - remainder)
        pad = torch.zeros(pad_len, dtype=flat.dtype, device=flat.device)
        flat = torch.cat([flat, pad], dim=0)
    else:
        pad_len = 0

    num_blocks = flat.numel() // block_size
    # print("hello", num_blocks, block_size)
    blocks = flat.view(num_blocks, block_size)

    meta = {
        'original_shape': original_shape,
        'valid_length': valid_length,
        'block_size': block_size,
        'pad_value': pad_value,
        'dtype': dtype,
        'device': device,
    }
    return blocks, meta


def reconstruct_from_blocks(blocks: torch.Tensor, meta: dict) -> torch.Tensor:
    """
    Inverse of `flatten_into_blocks`.
    Concatenate blocks, trim padding, and reshape to the original shape.

    Args:
      blocks: [num_blocks, block_size]
      meta:   dict returned by `flatten_into_blocks`

    Returns:
      x_reconstructed: Tensor with shape == meta['original_shape']
    """
    if blocks.dim() != 2:
        raise ValueError("`blocks` must be 2D, shape [num_blocks, block_size]")

    block_size = meta['block_size']
    if blocks.size(1) != block_size:
        raise ValueError(f"blocks.size(1) != block_size ({blocks.size(1)} vs {block_size})")

    flat = blocks.reshape(-1)
    valid_length = meta['valid_length']
    flat = flat[:valid_length]  # drop padding
    return flat.view(*meta['original_shape'])

def _sanitize_scale(scale_b: torch.Tensor) -> torch.Tensor:
    scale_b = torch.nan_to_num(scale_b, nan=_MIN_SCALE, posinf=1.0, neginf=1.0)
    return scale_b.abs().clamp(min=_MIN_SCALE)
# --------------------------


def _expand_param_for_blocks(p: torch.Tensor,
                            blocks: torch.Tensor,
                            *,
                            materialize: bool = False) -> torch.Tensor:
    """
    Expand/broadcast parameter `p` to match blocks of shape [N, B].

    Target behavior examples (blocks: [N, B]):
      - scalar            -> [N, B]
      - [N]               -> [N, B]
      - [B]               -> [N, B]
      - [N, 1]            -> [N, B]   <-- your requested case
      - [1, B]            -> [N, B]
      - [N, B]            -> [N, B] (as-is)
      - other shapes that broadcast to [N, B] are attempted

    Args:
      p:        tensor-like (will be moved to blocks' dtype/device)
      blocks:   tensor with shape [N, B]
      materialize:
          If True, returns a *dense* tensor with real memory (using .repeat / .expand(...).clone()).
          If False (default), returns a broadcasted view when possible (zero-stride expansions).

    Returns:
      Tensor shaped [N, B] on the same device/dtype as `blocks`.
    """
    if not torch.is_tensor(p):
        p = torch.tensor(p, dtype=blocks.dtype, device=blocks.device)
    else:
        p = p.to(dtype=blocks.dtype, device=blocks.device)

    N, B = blocks.shape

    # Fast paths for common shapes
    if p.ndim == 0:  # scalar
        out = p.view(1, 1).expand(N, B)
    elif p.ndim == 1:
        if p.numel() == N:
            # [N] -> [N, B]
            out = p.view(N, 1).expand(N, B)
        elif p.numel() == B:
            # [B] -> [N, B]
            out = p.view(1, B).expand(N, B)
        else:
            # Try to broadcast directly to [N, B]
            out = torch.broadcast_to(p, (N, B))
    elif p.ndim == 2:
        if p.shape == (N, B):
            out = p
        elif p.shape == (N, 1):
            out = p.expand(N, B)       # <-- your case: [N,1] -> [N,B]
        elif p.shape == (1, B):
            out = p.expand(N, B)
        elif p.numel() == 1:
            out = p.view(1, 1).expand(N, B)
        else:
            # Fallback: attempt broadcast
            out = torch.broadcast_to(p, (N, B))
    else:
        # Any higher-dim shape: attempt broadcast (will raise if impossible)
        out = torch.broadcast_to(p, (N, B))

    if materialize:
        # Turn zero-stride broadcast into a real dense tensor.
        # Use .repeat for clear semantics (costs memory),
        # or .clone() if already (N,B) but with zero strides.
        if out.shape != (N, B):
            out = torch.broadcast_to(out, (N, B))
        # If it's an expanded view (has any stride 0), clone to materialize.
        if any(s == 0 for s in out.stride()):
            out = out.clone()
        # Alternatively, if you specifically want repetition from a lower-rank p:
        # - You could detect original shape and use .repeat accordingly.
    return out

def compute_block_min_max(blocks: torch.Tensor):
    """
    Per-block min/max over the *last* dimension.
    blocks: [Nblocks, B] where B = block_size
    Returns:
      min_b: [Nblocks, 1]
      max_b: [Nblocks, 1]
    """
    # In recent PyTorch, .min/.max with dim= return an object with .values/.indices
    min_b = blocks.min(dim=-1, keepdim=True).values  # [Nblocks, 1]
    max_b = blocks.max(dim=-1, keepdim=True).values  # [Nblocks, 1]
    return min_b, max_b

# per_channel_flatten_reconstruct.py
@torch.no_grad()
def flatten_into_channels(x: torch.Tensor, ch_axis: int = 0) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Bring the 'channel' dimension to front and flatten the rest so you can
    do per-channel quantization (e.g., one scale/zero-point per output channel).

    Args:
      x:        input tensor, e.g., Conv2d weights [OC, IC, KH, KW] or Linear [OC, IC]
      ch_axis:  which dimension is the 'channel' you want to treat as output channels
                (0 for conv/linear weights; 1 for NCHW activations)

    Returns:
      ch_flat:  Tensor of shape [C, L] where C = size of channel dim, L = prod(other dims)
      meta:     Dict with info needed to reconstruct:
                {
                  'original_shape': tuple,
                  'ch_axis': int,
                  'perm': list,          # permutation applied to move ch_axis -> 0
                  'rest_shape': tuple,   # shape of remaining dims after channel
                  'dtype': torch.dtype,
                  'device': torch.device,
                }
    """
    if ch_axis < 0:
        ch_axis = x.ndim + ch_axis
    assert 0 <= ch_axis < x.ndim, "ch_axis out of range"

    # Build permutation that moves ch_axis to front
    perm = list(range(x.ndim))
    perm[0], perm[ch_axis] = perm[ch_axis], perm[0]

    x_perm = x.permute(perm).contiguous()
    C = x_perm.shape[0]
    rest_shape = x_perm.shape[1:]  # tuple
    ch_flat = x_perm.reshape(C, -1)

    meta = {
        "original_shape": tuple(x.shape),
        "ch_axis": ch_axis,
        "perm": perm,
        "rest_shape": tuple(rest_shape),
        "dtype": x.dtype,
        "device": x.device,
    }
    return ch_flat, meta


@torch.no_grad()
def reconstruct_from_channels(ch_flat: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
    """
    Invert flatten_into_channels: take [C, L] back to original shape.

    Args:
      ch_flat: [C, L]
      meta:    dict returned by flatten_into_channels

    Returns:
      x_recon: tensor with shape meta['original_shape']
    """
    C = ch_flat.shape[0]
    rest_shape = meta["rest_shape"]
    perm = meta["perm"]

    # Rebuild the permuted tensor
    x_perm = ch_flat.reshape((C, *rest_shape))

    # Invert permutation
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i

    x_recon = x_perm.permute(inv_perm).contiguous()
    return x_recon

def fake_quantize_per_block_affine(
    x: torch.Tensor,
    scale_b: torch.Tensor,
    zp_b: torch.Tensor,
    qmin: int,
    qmax: int,
    block_size: int,
    *,
    pad_value: int = 0,
    grad_factor: float = 1.0 
):
    """
    Affine fake-quantization per *1D* block using flatten_into_blocks/reconstruct_from_blocks.

    How it works:
      1) Fully flatten x -> 1D.
      2) Split into blocks of size (block_h * block_w).  (Pads tail if needed.)
      3) Apply per-block affine fake-quant.
      4) Concatenate & trim padding, then reshape back to x's original shape.

    Per-block params:
      - scale_b, zp_b can be:
         * scalar
         * shape [Nblocks]
         * shape [block_size]
         * shape [Nblocks, block_size]
         * any shape broadcastable to [Nblocks, block_size]

    Notes:
      - This treats blocks purely as *1D chunks*. If your original logic relied on 2D tiling
        across H×W specifically (per-channel/block grids), reshape your scale/zp accordingly
        to match the 1D block indexing.
    """
    from typing import Tuple
    # You must have these helpers defined somewhere:
    #   flatten_into_blocks(x, block_size: int, pad_value: float) -> (blocks [N,B], meta dict)
    #   reconstruct_from_blocks(blocks: Tensor, meta: dict) -> Tensor
    # If they live in another module, import them instead.
    try:
        flatten_into_blocks
        reconstruct_from_blocks
    except NameError:
        raise RuntimeError("Please define `flatten_into_blocks` and `reconstruct_from_blocks` "
                           "before calling this function.")

    # 1) Flatten to 1D blocks
    blocks, meta = flatten_into_blocks(x, block_size=block_size, pad_value=pad_value)  # [Nblocks, B]
    # blocks,meta = flatten_into_channels(x) 
    scale_b = grad_scale(scale_b, grad_factor)
    scale_b_exp = scale_b
    zp_b_exp = zp_b 
    k = 3.0
    mask, stats = build_outlier_mask(x, k=k)
    idx, vals = extract_outliers(x, mask)
    # # 2) Make per-block params broadcastable to [Nblocks, B]
    # scale_b_exp = _expand_param_for_blocks(scale_b.to(blocks.dtype).to(blocks.device), blocks)
    # zp_b_exp    = _expand_param_for_blocks(zp_b.to(blocks.dtype).to(blocks.device), blocks)

    # 3) Fake-quantize per block (affine)
    x_int = round_ste(blocks / scale_b_exp) + zp_b_exp
    x_q   =  torch.clamp(x_int, qmin, qmax) 
    x_deq = (x_q - zp_b_exp) * scale_b_exp  # still [Nblocks, B]

    # 4) Reconstruct original shape
    x_out = reconstruct_from_blocks(x_deq, meta)  # same shape as input x
    # print(torch.max(x_out))
    x_out = restore_outliers(x_out, idx, vals)
    # print(torch.max(x_out))
    return x_out

# def fake_quantize_learnable_per_block_affine_training(
#     x: torch.Tensor,         # [Nblocks, B]
#     scale_b: torch.Tensor,        # [Nblocks, 1] or [Nblocks, B]
#     zero_point_b: torch.Tensor,   # [Nblocks, 1] or [Nblocks, B] (int)
#     quant_min: int,
#     quant_max: int,
#     grad_factor: float,
# ):
#     # LSQ: apply grad scaling to the *learnable* scales
#     blocks, meta = flatten_into_blocks(x, block_size=block_size, pad_value=pad_value)  # [Nblocks, B]
#     scale_b = grad_scale(scale_b, grad_factor)
#     x_int = round_ste(blocks / scale_b) + zero_point_b
#     x_q = torch.clamp(x_int, quant_min, quant_max)
#     x_deq = (x_q - zero_point_b) * scale_b
#     x_out = reconstruct_from_blocks(x_deq, meta)  # same shape as input x
#     return x_out