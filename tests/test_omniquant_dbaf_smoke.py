"""Smoke test: DBAF building blocks (gate + fold + unfold) reduce MSE on a
sparse-outlier tensor more than naive RTN. Used as a regression check for the
OmniQuant integration which composes the same building blocks at calibration."""
import sys
import torch
sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")

from ahcptq.quantization.fake_quant import (
    is_like_normal_plus_3sigma_outliers, fold_outliers, unfold_outliers,
)
from flatquant.baselines.rtn import _quantize_tensor_uniform


def test_dbaf_better_than_rtn_on_sparse():
    rng = torch.Generator().manual_seed(0)
    # 1024 weights, mostly N(0,1), with 5 outliers at ±10
    w = torch.randn(1024, generator=rng)
    idx = torch.randperm(1024, generator=rng)[:5]
    w[idx] = torch.tensor([10.0, -10.0, 10.0, -10.0, 10.0])
    w = w.unsqueeze(0)  # [1, 1024]

    # RTN baseline
    w_rtn = _quantize_tensor_uniform(w, 4, per_channel=True)
    rtn_mse = ((w - w_rtn) ** 2).mean().item()

    # DBAF
    gate = is_like_normal_plus_3sigma_outliers(w)
    assert gate["is_like_c"], "Constructed tensor should pass the gate"
    T = float(3.0 * gate["stats"]["std"])
    alpha = 0.95
    w_fold, tag = fold_outliers(w, T, alpha)
    w_q = _quantize_tensor_uniform(w_fold, 4, per_channel=True)
    w_dbaf = unfold_outliers(w_q, tag, T, alpha)
    dbaf_mse = ((w - w_dbaf) ** 2).mean().item()

    assert dbaf_mse < rtn_mse, f"dbaf_mse={dbaf_mse:.4g} >= rtn_mse={rtn_mse:.4g}"
