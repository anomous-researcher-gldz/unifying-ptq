"""Test that SmoothQuant's _collect_act_scales does not crash on a 2D
(N, T) calibration tensor, which is the format produced by
scripts/run_training_free_full_table.py._calib_batch_llm.
"""
import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")


class _MiniLlama(nn.Module):
    """Minimal stand-in: an embedding + one Linear, with a 3D-shape sanity
    check inside that mirrors Llama's `bsz, q_len, _ = hidden_states.size()`.
    """
    def __init__(self, vocab=256, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, ids):
        h = self.embed(ids)
        bsz, q_len, _ = h.size()  # crashes if h is 2D
        return self.proj(h)


def test_collect_act_scales_handles_2d_calib_tensor():
    """Mirror of _calib_batch_llm output: 2D (n, seq_len) tensor."""
    from flatquant.baselines.smoothquant import _collect_act_scales

    torch.manual_seed(0)
    model = _MiniLlama(vocab=256, hidden=32)
    # _calib_batch_llm-style: shape (N, T), values are token ids
    calib = torch.randint(0, 256, (4, 16))  # 4 sequences, 16 tokens each

    scales = _collect_act_scales(model, calib, alpha=0.5)
    assert "proj" in scales, "expected per-Linear scale dict entry"
    assert scales["proj"].shape == (32,), \
        f"per-channel scale should be [d_in=32], got {scales['proj'].shape}"
    assert torch.isfinite(scales["proj"]).all(), "scale should be finite"
