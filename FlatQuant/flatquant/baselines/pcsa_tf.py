"""Training-free PCSA: k-means on calibration prompt descriptors + per-anchor
max-abs activation scales. No gradient training. Composes on any host method
that has activation tensors.

API:
  fit_pcsa_tf(descs, acts, K) -> state dict {anchors, scales}
  route_pcsa_tf(desc, state) -> anchor_id tensor
  apply_pcsa_tf_to_activation(x, desc, state, bits) -> fake-quantized x
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def _kmeans(x: torch.Tensor, k: int, n_iter: int = 25) -> torch.Tensor:
    """k-means on [N, D] -> [k, D] centroids. CPU-friendly, no gradients."""
    x = x.detach()
    N, D = x.shape
    # init from random rows; clamp k if we have fewer descriptors than clusters
    k = min(k, N)
    idx = torch.randperm(N)[:k]
    centroids = x[idx].clone()
    for _ in range(n_iter):
        # assign each row to nearest centroid (cosine on L2-normalized vectors)
        xn = F.normalize(x, dim=-1)
        cn = F.normalize(centroids, dim=-1)
        sims = xn @ cn.T
        assign = sims.argmax(dim=-1)
        # update centroids = mean of assigned rows
        new_cents = torch.zeros_like(centroids)
        for j in range(k):
            mask = (assign == j)
            if mask.any():
                new_cents[j] = x[mask].mean(dim=0)
            else:
                new_cents[j] = centroids[j]
        if torch.allclose(new_cents, centroids, atol=1e-6):
            break
        centroids = new_cents
    return centroids


@torch.no_grad()
def fit_pcsa_tf(
    descs: torch.Tensor,
    acts: torch.Tensor,
    K: int = 8,
) -> dict:
    """Fit K anchors via k-means on `descs`, then per-anchor max-abs scale on `acts`.

    Args:
      descs: [N, D_desc] prompt-level descriptors (e.g., mean-pooled hidden states)
      acts:  [N, T, D_act] or [N, D_act] activations per prompt; if 3D, max over T
      K: number of anchors

    Returns dict {"anchors": [K, D_desc], "scales": [K]} (both no grad).
    """
    descs = descs.detach().float()
    acts = acts.detach().float()
    if acts.dim() == 3:
        per_prompt_max = acts.abs().amax(dim=(1, 2))  # [N]
    elif acts.dim() == 2:
        per_prompt_max = acts.abs().amax(dim=1)  # [N]
    else:
        raise ValueError(f"acts must be [N,T,D] or [N,D], got {acts.shape}")
    anchors = _kmeans(descs, K)
    K_actual = anchors.shape[0]  # may be < K if N < K (clamped in _kmeans)
    # route each prompt to its nearest anchor and take max-abs activation
    sims = F.normalize(descs, dim=-1) @ F.normalize(anchors, dim=-1).T
    assign = sims.argmax(dim=-1)  # [N]
    scales = torch.zeros(K_actual)
    for j in range(K_actual):
        mask = (assign == j)
        scales[j] = per_prompt_max[mask].max() if mask.any() else per_prompt_max.max()
    return {"anchors": anchors, "scales": scales}


@torch.no_grad()
def route_pcsa_tf(desc: torch.Tensor, state: dict) -> torch.Tensor:
    """desc: [B, D]; returns [B] anchor indices."""
    sims = F.normalize(desc, dim=-1) @ F.normalize(state["anchors"], dim=-1).T
    return sims.argmax(dim=-1)


@torch.no_grad()
def apply_pcsa_tf_to_activation(
    x: torch.Tensor,
    desc: torch.Tensor,
    state: dict,
    bits: int = 4,
) -> torch.Tensor:
    """Per-prompt symmetric INT[bits] fake-quantization using anchor-routed scale.

    x: [B, ...] activation tensor; desc: [B, D] prompt descriptors.
    Returns: same shape as x, fake-quantized.
    """
    qmax = 2 ** bits - 1
    anchor_ids = route_pcsa_tf(desc, state)  # [B]
    scale_per_prompt = state["scales"][anchor_ids]  # [B]
    # Broadcast scale over the trailing dims of x
    extra_dims = x.dim() - 1
    scale = scale_per_prompt.view(-1, *([1] * extra_dims)) / qmax
    scale = scale.clamp(min=1e-9)
    q = torch.round(x / scale).clamp(-qmax // 2, qmax // 2)
    return (q * scale).to(x.dtype)
