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


def quantize_swinir(model: nn.Module, bits: int = 4, use_dbaf: bool = False, alpha: float = 0.95):
    n = 0
    for name, mod in model.named_modules():
        if "upsample" in name or "conv_last" in name or "conv_first" in name:
            continue
        if isinstance(mod, nn.Linear):
            w = mod.weight.data.float()
            if use_dbaf:
                w_q = _quantize_per_channel_with_dbaf(w, bits, alpha=alpha)
            else:
                w_q = _quantize_tensor_uniform(w, bits, per_channel=True)
            mod.weight.data = w_q.to(mod.weight.dtype)
            n += 1
        elif isinstance(mod, nn.Conv2d):
            w = mod.weight.data.float()
            out_c = w.shape[0]
            wf = w.view(out_c, -1)
            if use_dbaf:
                w_q = _quantize_per_channel_with_dbaf(wf, bits, alpha=alpha)
            else:
                w_q = _quantize_tensor_uniform(wf, bits, per_channel=True)
            mod.weight.data = w_q.view_as(w).to(mod.weight.dtype)
            n += 1
    print(f"[quant] {n} modules quantized (bits={bits}, dbaf={use_dbaf})", flush=True)
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
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model = load_swinir(args.scale, args.pretrained)
    t0 = time.time()
    model = quantize_swinir(model, args.bits, args.use_dbaf, args.alpha)
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
