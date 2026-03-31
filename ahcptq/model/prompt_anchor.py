# prompt_anchor.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptAnchorBank(nn.Module):
    """
    Maintains K "prompt anchors" in descriptor space.
    Each anchor corresponds to one set of quantization scales.

    Typical use:
      - During calibration:
          desc = bank.compute_descriptor(prompt_tokens)   # [B, D]
          anchor_ids = bank.assign_and_update(desc)       # [B]
      - During inference:
          desc = bank.compute_descriptor(prompt_tokens)
          anchor_ids = bank.assign(desc, update=False)

    You usually only need 4–8 anchors in practice.
    """

    def __init__(
        self,
        num_anchors: int,
        descriptor_dim: int,
        ema_momentum: float = 0.9,
        normalize: bool = True,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        # print(self.num_anchors)
        self.descriptor_dim = descriptor_dim
        self.ema_momentum = ema_momentum
        self.normalize = normalize

        # anchors: K x D
        anchors = torch.randn(num_anchors, descriptor_dim)
        anchors = F.normalize(anchors, dim=-1)
        self.register_buffer("anchors", anchors)

        # how many times each anchor has been updated (for warmup)
        counts = torch.zeros(num_anchors)
        self.register_buffer("counts", counts)

    @torch.no_grad()
    def compute_descriptor(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        """
        prompt_tokens: [B, N, C] (e.g., queries + query_pe BEFORE attention).
        Returns a [B, C] descriptor (mean-pooled + normalized).
        """
        # mean pool over tokens
        desc = prompt_tokens.mean(dim=1)  # [B, C]

        if self.normalize:
            desc = F.normalize(desc, dim=-1)

        return desc

    @torch.no_grad()
    def _cosine_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D], y: [K, D]
        returns cosine *distance* [B, K] = 1 - cos_sim
        """
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        if x.device!= y.device: 
            y = y.to("cuda")
        sim = x @ y.t()  # [B, K]
        return 1.0 - sim

    @torch.no_grad()
    def assign(self, desc: torch.Tensor, update: bool = False) -> torch.Tensor:
        """
        desc: [B, D]
        returns anchor_ids: [B] in [0, K-1]
        If update=True, also EMA-update anchor vectors.
        """
        # cosine distance to all anchors
        dist = self._cosine_distance(desc, self.anchors)  # [B, K]
        anchor_ids = dist.argmin(dim=-1)                  # [B]

        if update:
            self._update_anchors(desc, anchor_ids)
        # cold = self.counts[anchor_ids] < 2
        # if cold.any():
        #     anchor_ids = anchor_ids.clone()
        #     anchor_ids[cold] = 0
        return anchor_ids

    @torch.no_grad()
    def assign_and_update(self, desc: torch.Tensor) -> torch.Tensor:
        return self.assign(desc, update=True)

    @torch.no_grad()
    def _update_anchors(self, desc: torch.Tensor, anchor_ids: torch.Tensor):
        """
        Simple EMA update:
          a_k <- m * a_k + (1 - m) * mean(desc[assigned_to_k])
        """
        B = desc.size(0)
        for b in range(B):
            k = anchor_ids[b].item()
            d = desc[b]

            # effective momentum: warmup anchors in early steps
            c = self.counts[k].item()
            if c < 1:
                m = 0.0
            else:
                m = self.ema_momentum

            self.anchors[k] = F.normalize(
                m* self.anchors[k].to("cuda") + (1.0 - m) * d.to("cuda"),
                dim=-1,
            )
            self.counts[k] += 1.0