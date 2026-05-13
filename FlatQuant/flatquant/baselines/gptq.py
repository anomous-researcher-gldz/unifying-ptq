"""GPTQ as a baseline, with optional DBAF folding pre-quantization."""
from __future__ import annotations
import torch
import torch.nn as nn

from flatquant.baselines.rtn import _dbaf_fold, _dbaf_unfold, _quantize_tensor_uniform


class _GPTQ:
    """Minimal GPTQ per-layer Hessian solver.

    Tracks an inverse-Hessian-style sum of input^T input over calibration
    batches and uses it to greedily round weights with error compensation.
    Simpler than the official GPTQ paper but sufficient to test DBAF's
    composability vs RTN.
    """

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.W = layer.weight.data.clone()
        self.dev = self.W.device
        rows, cols = self.W.shape
        self.H = torch.zeros((cols, cols), device=self.dev)
        self.n = 0

    def add_batch(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        x = x.to(self.dev).float()
        new_n = self.n + x.shape[0]
        self.H = (self.H * self.n + x.t() @ x) / max(new_n, 1)
        self.n = new_n

    def quantize(self, bits: int, dbaf_alpha: float | None = None):
        """GPTQ-style column sweep with per-row (per-output-channel) scales.

        Per-row scale is set ONCE from the original weight magnitude per row,
        used for all column-wise updates. This is the standard LLM quantization
        granularity and avoids the per-tensor scale collapse seen on large matrices.
        """
        W = self.W.clone()
        if dbaf_alpha is not None:
            W, T, a = _dbaf_fold(W, alpha=dbaf_alpha)
        # Per-row scales [out, 1] from folded weights
        qmax = 2 ** (bits - 1) - 1
        row_scale = W.abs().amax(dim=1, keepdim=True) / qmax
        row_scale = row_scale.clamp(min=1e-9)
        # Damp Hessian for stability
        damp = 0.01 * torch.mean(torch.diag(self.H))
        diag = torch.arange(self.H.shape[0], device=self.dev)
        H = self.H.clone()
        H[diag, diag] += damp
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
        except RuntimeError:
            Hinv = torch.linalg.pinv(H)
        Q = torch.zeros_like(W)
        for col in range(W.shape[1]):
            d = Hinv[col, col]
            w_col = W[:, col]  # [out]
            # Per-row scale already computed
            q = torch.round(w_col / row_scale.squeeze(1)).clamp(-qmax, qmax)
            q_dq = q * row_scale.squeeze(1)
            err = (w_col - q_dq) / d  # [out]
            Q[:, col] = q_dq
            if col + 1 < W.shape[1]:
                W[:, col + 1:] -= err.unsqueeze(1) * Hinv[col, col + 1:].unsqueeze(0)
        if dbaf_alpha is not None:
            Q = _dbaf_unfold(Q, T, a)
        self.layer.weight.data = Q.to(self.layer.weight.dtype)


def quantize_model(
    model: nn.Module,
    bits: int = 4,
    calibration_data: torch.Tensor | None = None,
    use_dbaf: bool = False,
    alpha: float = 0.75,
    **_unused,
) -> nn.Module:
    """Capture per-layer inputs, then GPTQ-quantize each Linear layer.

    `calibration_data` is a tensor of token ids of shape [B, T].
    """
    if calibration_data is None:
        raise ValueError("GPTQ requires calibration_data")
    captured: dict = {}
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            def make_hook(n):
                def hook(_module, inp, _out):
                    captured.setdefault(n, []).append(inp[0].detach())
                return hook
            handles.append(mod.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(calibration_data)
    for h in handles:
        h.remove()

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or "lm_head" in name:
            continue
        if name not in captured:
            continue
        gptq = _GPTQ(mod)
        for batch in captured[name]:
            gptq.add_batch(batch)
        gptq.quantize(bits=bits, dbaf_alpha=alpha if use_dbaf else None)
    return model
