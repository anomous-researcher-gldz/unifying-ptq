# Ported from ahcptq/model/prompt_anchor.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptBank(nn.Module):
    """
    Maintains K prompt anchors in descriptor space.
    Each anchor corresponds to one set of quantization scales.

    During training:
        desc = hidden_states.mean(dim=1)
        desc = desc / (desc.norm(dim=-1, keepdim=True) + 1e-6)
        anchor_ids = bank.assign(desc, update=True)

    During inference:
        anchor_ids = bank.assign(desc, update=False)
    """

    def __init__(
        self,
        num_anchors: int,
        descriptor_dim: int,
        ema_momentum: float = 0.9,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.descriptor_dim = descriptor_dim
        self.ema_momentum = ema_momentum

        anchors = torch.randn(num_anchors, descriptor_dim)
        anchors = F.normalize(anchors, dim=-1)
        self.register_buffer("anchors", anchors)

        counts = torch.zeros(num_anchors)
        self.register_buffer("counts", counts)

    @torch.no_grad()
    def _cosine_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x: [B, D], y: [K, D] -> cosine distance [B, K]"""
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1).to(device=x.device, dtype=x.dtype)
        return 1.0 - (x @ y.t())

    @torch.no_grad()
    def assign(self, desc: torch.Tensor, update: bool = False) -> torch.Tensor:
        """
        desc: [B, D]  (should already be L2-normalized by caller)
        returns anchor_ids: [B] in [0, K-1]
        """
        dist = self._cosine_distance(desc, self.anchors)  # [B, K]
        anchor_ids = dist.argmin(dim=-1)                   # [B]
        if update:
            self._update_anchors(desc, anchor_ids)
        return anchor_ids

    @torch.no_grad()
    def _update_anchors(self, desc: torch.Tensor, anchor_ids: torch.Tensor):
        """EMA update: a_k <- m * a_k + (1 - m) * desc[assigned_to_k]"""
        # NOTE: O(B) loop — update=True is intended for calibration only (small batch sizes).
        for b in range(desc.size(0)):
            k = anchor_ids[b].item()
            d = desc[b].to(device=self.anchors.device, dtype=self.anchors.dtype)
            c = self.counts[k].item()
            m = 0.0 if c < 1 else self.ema_momentum
            self.anchors[k] = F.normalize(
                m * self.anchors[k] + (1.0 - m) * d,
                dim=-1,
            )
            self.counts[k] += 1.0
