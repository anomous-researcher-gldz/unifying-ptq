"""Training-free SwinIR + RTN/DBAF on Set5 + Urban100.

Loads CompSRT's SwinIR architecture, applies per-channel RTN to all
Linear+Conv2d in the body (skipping the final upsampler and the first conv
to preserve fidelity at the boundaries), optionally with DBAF folding.
Evaluates PSNR/SSIM on Set5 + Urban100 super-resolution.

Fills the CompSRT-A cell of the training-free DBAF table.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import time
import glob

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/CompSRT")
from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf


def load_swinir(scale: int, pretrained: str, variant: str = "auto"):
    """Load SwinIR. variant='M' (classical, embed=180), 'S' (lightweight, embed=60), or 'auto' (detect)."""
    from basicsr.archs.swinir_arch import SwinIR
    if variant == "auto":
        variant = "M" if "_SwinIR-M_" in pretrained else "S"
    if variant == "M":
        m = SwinIR(upscale=scale, in_chans=3, img_size=48, window_size=8, img_range=1.,
                   depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                   num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
                   upsampler="pixelshuffle", resi_connection="1conv")
    else:  # S
        m = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8, img_range=1.,
                   depths=[6, 6, 6, 6], embed_dim=60,
                   num_heads=[6, 6, 6, 6], mlp_ratio=2,
                   upsampler="pixelshuffledirect", resi_connection="1conv")
    state = torch.load(pretrained, map_location="cpu", weights_only=False)
    sd = state.get("params", state.get("params_ema", state))
    m.load_state_dict(sd, strict=False)
    return m.cuda().eval()


def _wrap_activation_quant(mod: nn.Module, act_bits: int, use_dbaf: bool,
                           alpha: float, act_gate_frac3_max):
    """Wrap mod.forward so that inputs are per-token asym INT4 fake-quantized,
    optionally with DBAF + gate. Uses codebase's fold_outliers/unfold_outliers.
    For Conv2d, treats [B,C,H,W] -> [B,H,W,C] for per-token (per spatial pos) quant.
    """
    from ahcptq.quantization.fake_quant import (
        is_like_normal_plus_3sigma_outliers, fold_outliers, unfold_outliers,
    )
    import torch as _torch

    orig_forward = mod.forward
    is_conv = isinstance(mod, nn.Conv2d)
    qmax_asym = 2 ** act_bits - 1

    def quantize_input(x: _torch.Tensor) -> _torch.Tensor:
        xf = x.detach().float()
        # Reshape so last dim is the feature dim (per-token quant).
        if is_conv and xf.dim() == 4:
            xfp = xf.permute(0, 2, 3, 1).contiguous()
            init_shape = xfp.shape
            flat = xfp.reshape(-1, xfp.shape[-1])
        else:
            init_shape = xf.shape
            flat = xf.reshape(-1, xf.shape[-1])
        # Optionally apply DBAF + gate
        folded = False
        if use_dbaf:
            gate = is_like_normal_plus_3sigma_outliers(flat, frac3_max=(
                act_gate_frac3_max if act_gate_frac3_max is not None else 2e-2
            ))
            if act_gate_frac3_max == -1 or gate["is_like_c"]:  # -1 = force, else gated
                T = float(3.0 * flat.std().clamp_min(1e-8))
                flat_folded, tag = fold_outliers(flat, T, alpha)
                flat = flat_folded
                folded = True
        # Per-token asym INT4
        xmax = flat.amax(dim=1, keepdim=True)
        xmin = flat.amin(dim=1, keepdim=True)
        tmp = (xmax == xmin)
        xmin = _torch.where(tmp, xmin - 1.0, xmin)
        xmax = _torch.where(tmp, xmax + 1.0, xmax)
        scale = (xmax - xmin) / qmax_asym
        zero = _torch.round(-xmin / scale)
        q = _torch.round(flat / scale + zero).clamp(0, qmax_asym)
        flat_q = (q - zero) * scale
        # Unfold if folded
        if folded:
            flat_q = unfold_outliers(flat_q, tag, T, alpha)
        # Reshape back
        if is_conv and xf.dim() == 4:
            out = flat_q.reshape(init_shape).permute(0, 3, 1, 2).contiguous()
        else:
            out = flat_q.reshape(init_shape)
        return out.to(x.dtype)

    def new_forward(x, *args, **kwargs):
        return orig_forward(quantize_input(x), *args, **kwargs)

    mod.forward = new_forward


def quantize_swinir(model: nn.Module, bits: int = 4, use_dbaf: bool = False,
                    alpha: float = 0.95, T_sigma: float = 3.0,
                    gate_frac3_max=None,
                    act_bits: int = 16, act_gate_frac3_max=None):
    """W{bits}A{act_bits} quantization with DBAF + optional gate on both sides.

    - act_bits=16 (default): activations stay FP16, weight-only quant (legacy behavior).
    - act_bits=4: per-token asym INT4 activation quant via wrapper hook.
    - act_gate_frac3_max=None: gate on activations uses codebase default 2e-2.
    - act_gate_frac3_max=-1 (sentinel): NO activation gate; DBAF forces on every activation.
    - act_gate_frac3_max=0.02: explicit value.
    """
    n = 0
    for name, mod in model.named_modules():
        if "upsample" in name or "conv_last" in name or "conv_first" in name:
            continue
        if isinstance(mod, nn.Linear):
            w = mod.weight.data.float()
            if use_dbaf:
                w_q = _quantize_per_channel_with_dbaf(w, bits, alpha=alpha, T_sigma=T_sigma, gate_frac3_max=gate_frac3_max)
            else:
                w_q = _quantize_tensor_uniform(w, bits, per_channel=True)
            mod.weight.data = w_q.to(mod.weight.dtype)
            n += 1
            if act_bits < 16:
                _wrap_activation_quant(mod, act_bits, use_dbaf, alpha, act_gate_frac3_max)
        elif isinstance(mod, nn.Conv2d):
            w = mod.weight.data.float()
            out_c = w.shape[0]
            wf = w.view(out_c, -1)
            if use_dbaf:
                w_q = _quantize_per_channel_with_dbaf(wf, bits, alpha=alpha, T_sigma=T_sigma, gate_frac3_max=gate_frac3_max)
            else:
                w_q = _quantize_tensor_uniform(wf, bits, per_channel=True)
            mod.weight.data = w_q.view_as(w).to(mod.weight.dtype)
            n += 1
            if act_bits < 16:
                _wrap_activation_quant(mod, act_bits, use_dbaf, alpha, act_gate_frac3_max)
    print(f"[quant] {n} modules quantized (W{bits}A{act_bits}, dbaf={use_dbaf}, "
          f"w_gate={gate_frac3_max}, a_gate={act_gate_frac3_max})", flush=True)
    return model


def psnr(sr: np.ndarray, gt: np.ndarray, crop: int = 0) -> float:
    if crop > 0:
        sr = sr[crop:-crop, crop:-crop]
        gt = gt[crop:-crop, crop:-crop]
    mse = ((sr.astype(np.float64) - gt.astype(np.float64)) ** 2).mean()
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def evaluate(model, scale: int, dataset_dir: str, lr_subdir: str = None):
    """SR test loop: load HR/LR pair, run model, compute PSNR."""
    hr_paths = sorted(glob.glob(f"{dataset_dir}/*.png") + glob.glob(f"{dataset_dir}/*.PNG"))
    if not hr_paths:
        hr_paths = sorted(glob.glob(f"{dataset_dir}/HR/*.png"))
    psnrs = []
    for hr_path in hr_paths:
        try:
            hr = np.array(Image.open(hr_path).convert("RGB"))
        except Exception as e:
            continue
        # Look for the matching LR
        stem = pathlib.Path(hr_path).stem
        if lr_subdir:
            candidates = glob.glob(f"{lr_subdir}/{stem}*") or glob.glob(f"{lr_subdir}/{stem.replace('HR','LR')}*")
            if not candidates:
                continue
            lr = np.array(Image.open(candidates[0]).convert("RGB"))
        else:
            # Bicubic downsample
            lr = np.array(Image.fromarray(hr).resize((hr.shape[1]//scale, hr.shape[0]//scale), Image.BICUBIC))
        h, w = lr.shape[:2]
        # Crop to multiple of 8 (window_size)
        h = h - h % 8; w = w - w % 8
        lr = lr[:h, :w]
        hr_crop = hr[:h*scale, :w*scale]
        x = torch.from_numpy(lr).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
        with torch.no_grad():
            sr = model(x).clamp(0, 1)
        sr_np = (sr[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        psnrs.append(psnr(sr_np, hr_crop, crop=scale))
    return float(np.mean(psnrs)) if psnrs else float("nan"), len(psnrs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--pretrained", required=True, help="SwinIR-x{scale} checkpoint .pth")
    p.add_argument("--dataset", default="/home/ubuntu/unifying-ptq/data/sr_testsets/Set5/Set5_HR")
    p.add_argument("--lr-subdir", default=None, help="If given, LR image dir; else bicubic-downsample HR")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--use-dbaf", action="store_true")
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--T-sigma", type=float, default=3.0,
                   help="Per-row sigma multiplier for the DBAF fold threshold.")
    p.add_argument("--gate-frac3-max", type=float, default=None,
                   help="If set, skip DBAF on layers whose frac |z|>3 exceeds this; default None = no gate.")
    p.add_argument("--act-bits", type=int, default=16,
                   help="Activation bits. 16 = FP, 4 = per-token asym INT4 via wrapper.")
    p.add_argument("--act-gate-frac3-max", type=float, default=None,
                   help="Activation DBAF gate. None = codebase default 2e-2; -1 = force (no gate).")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model = load_swinir(args.scale, args.pretrained)
    t0 = time.time()
    model = quantize_swinir(model, args.bits, args.use_dbaf, args.alpha,
                            T_sigma=args.T_sigma, gate_frac3_max=args.gate_frac3_max,
                            act_bits=args.act_bits,
                            act_gate_frac3_max=args.act_gate_frac3_max)
    avg, n = evaluate(model, args.scale, args.dataset, args.lr_subdir)
    out = {
        "model": "SwinIR",
        "scale": args.scale,
        "method": f"RTN{'+DBAF' if args.use_dbaf else ''}",
        "bits": args.bits,
        "dataset": pathlib.Path(args.dataset).name,
        "n_images": n,
        "psnr_db": avg,
        "wallclock_seconds": time.time() - t0,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
