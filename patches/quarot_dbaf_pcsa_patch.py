"""Monkey-patch QuaRot's ActQuantizer + WeightQuantizer to add DBAF + PCSA-tf.

QuaRot (fake_quant/quant_utils.py) defines:
  - ActQuantizer       — per-token act quant; forward(x) returns quantized x.
  - WeightQuantizer    — per-channel weight quant; configure() + quantize().

QuaRot rotates the model first (random Hadamard, training-free, applied via
rotation_utils.py + monkeypatch.py), then per-layer ActQuantizer/WeightQuantizer
handles low-bit quantization. The rotation "spreads" outliers but leaves
two gaps we close:

  1) The post-rotation activation still has prompt-conditioned variation
     that a static rotation can't capture → PCSA-tf routes per-prompt scales.
  2) Per-token quant scale comes from current-token max abs only, which
     amplifies any residual outlier still present after rotation → DBAF
     folds those before the per-token scale is computed.

Both wraps are surgical: they only fire on tensors matching the dense-core +
sparse-outlier signature; everything else passes through unchanged so the
rotation's normal effect is preserved.

Usage:

    import sys
    sys.path.insert(0, '/home/ubuntu/unifying-ptq')
    sys.path.insert(0, '/home/ubuntu/unifying-ptq/QuaRot/fake_quant')
    import quarot_dbaf_pcsa_patch as p

    p.install_dbaf_patches(dbaf_alpha=0.95)
    p.fit_pcsa_tf_on_calib_data(descriptors, activations, K=8)
    p.install_pcsa_tf()
    p.set_descriptor(desc)   # per-prompt descriptor

Then run QuaRot's main.py normally (the wraps are in place).
"""
from __future__ import annotations
import sys

sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")

import torch
from ahcptq.quantization.fake_quant import (
    is_like_normal_plus_3sigma_outliers, fold_outliers, unfold_outliers,
)
from flatquant.baselines.pcsa_tf import (
    fit_pcsa_tf, route_pcsa_tf,
)


_PCSA_STATE: dict | None = None
_DBAF_ALPHA: float = 0.95
_DBAF_INSTALLED: bool = False
_PCSA_INSTALLED: bool = False
_CURRENT_DESCRIPTOR: torch.Tensor | None = None


def set_descriptor(desc: torch.Tensor):
    """Per-prompt descriptor for PCSA-tf routing. Call before each forward."""
    global _CURRENT_DESCRIPTOR
    _CURRENT_DESCRIPTOR = desc


def fit_pcsa_tf_on_calib_data(descs: torch.Tensor, acts: torch.Tensor, K: int = 8):
    """Fit anchors + scales from collected calibration data."""
    global _PCSA_STATE
    _PCSA_STATE = fit_pcsa_tf(descs, acts, K=K)
    print(f"[pcsa_tf] fitted K={K} anchors; scales={_PCSA_STATE['scales'].tolist()}",
          flush=True)


def install_dbaf_patches(dbaf_alpha: float = 0.95):
    """Wrap QuaRot's ActQuantizer.forward + WeightQuantizer.quantize_dequantize.

    Both gated by the dense-with-outliers signature so we only fire on tensors
    where folding actually helps (rotation handles most cases on its own).
    """
    global _DBAF_ALPHA, _DBAF_INSTALLED
    if _DBAF_INSTALLED:
        print("[dbaf_patch] already installed; skipping", flush=True)
        return
    _DBAF_ALPHA = dbaf_alpha

    import quant_utils as qu

    orig_act_forward = qu.ActQuantizer.forward
    orig_w_forward = (qu.WeightQuantizer.forward
                      if hasattr(qu.WeightQuantizer, 'forward') else None)

    def wrapped_act_forward(self, x):
        if self.bits == 16:
            return orig_act_forward(self, x)
        try:
            fires = bool(is_like_normal_plus_3sigma_outliers(x.detach()))
        except Exception:
            fires = False
        if not fires:
            return orig_act_forward(self, x)
        folded, meta = fold_outliers(x, alpha=_DBAF_ALPHA)
        q_folded = orig_act_forward(self, folded)
        return unfold_outliers(q_folded, meta)

    qu.ActQuantizer.forward = wrapped_act_forward

    if orig_w_forward is not None:
        def wrapped_w_forward(self, x):
            try:
                fires = bool(is_like_normal_plus_3sigma_outliers(x.detach()))
            except Exception:
                fires = False
            if not fires:
                return orig_w_forward(self, x)
            folded, meta = fold_outliers(x, alpha=_DBAF_ALPHA)
            q_folded = orig_w_forward(self, folded)
            return unfold_outliers(q_folded, meta)
        qu.WeightQuantizer.forward = wrapped_w_forward

    _DBAF_INSTALLED = True
    print(f"[dbaf_patch] installed on QuaRot ActQuantizer (alpha={dbaf_alpha})",
          flush=True)


def install_pcsa_tf():
    """Stack PCSA-tf scale routing on top of (post-DBAF) ActQuantizer.forward.

    The wrapped order at inference: x -> [DBAF if gate fires] -> PCSA-tf rescale
                                       -> ActQuantizer per-token quant ->
                                       -> [DBAF unfold if folded].
    """
    global _PCSA_INSTALLED
    if _PCSA_INSTALLED:
        print("[pcsa_tf] already installed; skipping", flush=True)
        return
    if _PCSA_STATE is None:
        raise RuntimeError("PCSA-tf state not fit. Call fit_pcsa_tf_on_calib_data first.")

    import quant_utils as qu
    cur_forward = qu.ActQuantizer.forward

    def wrapped_pcsa_forward(self, x):
        if self.bits == 16:
            return cur_forward(self, x)
        if _CURRENT_DESCRIPTOR is None:
            return cur_forward(self, x)
        try:
            rescaled = route_pcsa_tf(x, _CURRENT_DESCRIPTOR, _PCSA_STATE)
        except Exception:
            return cur_forward(self, x)
        return cur_forward(self, rescaled)

    qu.ActQuantizer.forward = wrapped_pcsa_forward
    _PCSA_INSTALLED = True
    print("[pcsa_tf] installed on QuaRot ActQuantizer.forward", flush=True)


def install_both(dbaf_alpha: float = 0.95):
    install_dbaf_patches(dbaf_alpha=dbaf_alpha)
    if _PCSA_STATE is not None:
        install_pcsa_tf()
    else:
        print("[install_both] DBAF installed; PCSA-tf skipped (state not fit)",
              flush=True)
