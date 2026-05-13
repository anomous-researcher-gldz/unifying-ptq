"""Monkey-patch OmniQuant's UniformAffineQuantizer to add DBAF + PCSA-tf hooks.

OmniQuant uses one quantizer class for both weights and activations
(quantize.quantizer.UniformAffineQuantizer). We subclass it and override
forward() to wrap the existing fake-quant call with DBAF gate + fold/unfold
when the gate fires.

For PCSA-tf, a process-global state is filled by fit_pcsa_tf_on_calib_data()
during the calibration setup, then activation quantizers route per-prompt
to anchor-specific scales by replacing self.scale before calling fake_quant.

Usage:
  import sys; sys.path.insert(0, '/home/ubuntu/unifying-ptq/OmniQuant')
  from omniquant_dbaf_pcsa_patch import install_dbaf_patches, install_pcsa_tf
  install_dbaf_patches(dbaf_alpha=0.95)            # wraps UniformAffineQuantizer
  fit_pcsa_tf_on_calib_data(descs, acts, K=8)      # fills global PCSA state
  install_pcsa_tf()                                # activates per-prompt routing

Copy this file into OmniQuant/ at use-time:
  cp /home/ubuntu/unifying-ptq/patches/omniquant_dbaf_pcsa_patch.py \\
     /home/ubuntu/unifying-ptq/OmniQuant/omniquant_dbaf_pcsa_patch.py
"""
from __future__ import annotations
import sys

# Make our DBAF + PCSA-tf utilities importable
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
_CURRENT_DESCRIPTOR: torch.Tensor | None = None  # set externally per forward pass


def set_descriptor(desc: torch.Tensor):
    """Called by the calibration loop before each prompt's forward pass."""
    global _CURRENT_DESCRIPTOR
    _CURRENT_DESCRIPTOR = desc


def fit_pcsa_tf_on_calib_data(descs: torch.Tensor, acts: torch.Tensor, K: int = 8):
    """Fit the PCSA-tf state once from collected calibration descriptors + acts."""
    global _PCSA_STATE
    _PCSA_STATE = fit_pcsa_tf(descs, acts, K=K)
    print(f"[pcsa_tf] fitted K={K} anchors; scales={_PCSA_STATE['scales'].tolist()}",
          flush=True)


def install_dbaf_patches(dbaf_alpha: float = 0.95):
    """Wrap UniformAffineQuantizer.forward with the DBAF gate + fold/unfold.

    Safe to call multiple times; second call is a no-op.
    """
    global _DBAF_ALPHA, _DBAF_INSTALLED
    if _DBAF_INSTALLED:
        print("[dbaf_patch] already installed; skipping", flush=True)
        return
    _DBAF_ALPHA = dbaf_alpha

    from quantize.quantizer import UniformAffineQuantizer
    orig_forward = UniformAffineQuantizer.forward

    def wrapped_forward(self, x, *args, **kwargs):
        # Only fire DBAF if this quantizer is enabled (matches OmniQuant's check)
        if not getattr(self, "enable", True):
            return orig_forward(self, x, *args, **kwargs)
        gate = is_like_normal_plus_3sigma_outliers(x)
        if gate["is_like_c"]:
            T = float(3.0 * gate["stats"]["std"])
            x_fold, tag = fold_outliers(x, T, _DBAF_ALPHA)
            q = orig_forward(self, x_fold, *args, **kwargs)
            return unfold_outliers(q, tag, T, _DBAF_ALPHA)
        return orig_forward(self, x, *args, **kwargs)

    UniformAffineQuantizer.forward = wrapped_forward
    _DBAF_INSTALLED = True
    print(f"[dbaf_patch] wrapped UniformAffineQuantizer.forward (alpha={dbaf_alpha})",
          flush=True)


def install_pcsa_tf():
    """Activate per-prompt scale routing on activation quantizers.

    Must be called AFTER fit_pcsa_tf_on_calib_data(...) has populated the state.
    Replaces UniformAffineQuantizer.forward (the version possibly already wrapped
    by DBAF) with a version that, for activation quantizers (sym=False per
    OmniQuant convention for activations), divides the scale by the routed
    anchor's scale before calling the (DBAF-wrapped) forward.

    For LLM activations OmniQuant typically uses asymmetric activation quant
    (sym=False) and symmetric weights (sym=True); the routing fires only when
    sym==False AND _CURRENT_DESCRIPTOR is set AND _PCSA_STATE is fit.
    """
    global _PCSA_INSTALLED
    if _PCSA_INSTALLED:
        print("[pcsa_patch] already installed; skipping", flush=True)
        return
    if _PCSA_STATE is None:
        raise RuntimeError("fit_pcsa_tf_on_calib_data must be called before install_pcsa_tf")

    from quantize.quantizer import UniformAffineQuantizer
    pre_pcsa_forward = UniformAffineQuantizer.forward  # may be DBAF-wrapped

    def routed_forward(self, x, *args, **kwargs):
        # Only re-scale for activation quantizers AND when a current descriptor exists
        if (not getattr(self, "sym", True)) and (_CURRENT_DESCRIPTOR is not None) \
                and (_PCSA_STATE is not None):
            anchor_id = route_pcsa_tf(_CURRENT_DESCRIPTOR, _PCSA_STATE)
            anchor_scale = float(_PCSA_STATE["scales"][anchor_id[0]].item())
            # Use the anchor's max-abs as the effective dynamic range:
            # OmniQuant's per_token_dynamic_calibration usually sets self.scale
            # from x.amax during forward; we override that by pre-clipping x to
            # the anchor band so the resulting per-token scale tightens to the
            # anchor's pre-fitted bound. Cheap and avoids monkey-patching the
            # calibration method directly.
            x = x.clamp(min=-anchor_scale, max=anchor_scale)
        return pre_pcsa_forward(self, x, *args, **kwargs)

    UniformAffineQuantizer.forward = routed_forward
    _PCSA_INSTALLED = True
    print("[pcsa_patch] wrapped UniformAffineQuantizer.forward for PCSA-tf routing",
          flush=True)
