"""Weight and activation outlier statistics across SwinIR-light x2/x3/x4.

For each scale and each Conv2d/Linear layer, compute:
  - Weight: kurtosis, frac |z|>3, frac |z|>4, DBAF gate (is_like_c).
  - Activation: same stats on inputs captured via forward hooks on 2-3 Set5 LRs.

Goal: test the hypothesis that x3 weights/activations are less heavy-tailed
than x2/x4, explaining why DBAF helps x2/x4 but not x3.
"""
from __future__ import annotations
import sys, json, pathlib, glob, os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/ubuntu/unifying-ptq")
from ahcptq.quantization.fake_quant import (
    profile_with_3sigma_outliers, is_like_normal_plus_3sigma_outliers,
)
sys.path.insert(0, "/home/ubuntu/unifying-ptq/scripts")
from run_training_free_swinir import load_swinir

CKPTS = {
    2: "/home/ubuntu/unifying-ptq/ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth",
    3: "/home/ubuntu/unifying-ptq/ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth",
    4: "/home/ubuntu/unifying-ptq/ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
}
SET5 = "/home/ubuntu/unifying-ptq/data/sr_testsets/Set5_HR"


def stats_tuple(x: torch.Tensor):
    s = profile_with_3sigma_outliers(x)
    flat = x.detach().float().reshape(-1)
    mu, sd = flat.mean(), flat.std().clamp_min(1e-8)
    z = (flat - mu) / sd
    frac4 = float((z.abs() > 4.0).float().mean().item())
    gate = is_like_normal_plus_3sigma_outliers(x)["is_like_c"]
    return s["kurtosis"], s["frac_out_3"], frac4, bool(gate)


def bicubic_downsample(arr_uint8, scale, window=8):
    h, w = arr_uint8.shape[:2]
    h -= h % scale; w -= w % scale
    arr_uint8 = arr_uint8[:h, :w]
    img = Image.fromarray(arr_uint8)
    lr = np.asarray(img.resize((w // scale, h // scale), Image.BICUBIC))
    # Crop LR to multiple of window_size (SwinIR uses 8).
    lh, lw = lr.shape[:2]
    lh -= lh % window; lw -= lw % window
    return lr[:lh, :lw], arr_uint8


def measure_scale(scale, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_swinir(scale, ckpt_path).to(device).eval()

    # Per-layer weight stats.
    weight_rows = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and mod.weight.numel() >= 64:
            k, f3, f4, gate = stats_tuple(mod.weight.data)
            weight_rows.append({"name": name, "type": type(mod).__name__,
                                "shape": list(mod.weight.shape), "kurt": k,
                                "frac3": f3, "frac4": f4, "dbaf_gate": gate})

    # Hooks: capture inputs to each Conv2d/Linear during a small forward pass.
    act_acc = {r["name"]: [] for r in weight_rows}
    handles = []
    name_to_mod = {n: m for n, m in model.named_modules()}
    for r in weight_rows:
        m = name_to_mod[r["name"]]
        def make_hook(nm):
            def hook(mod, inp, out):
                x = inp[0] if isinstance(inp, tuple) else inp
                # Subsample to avoid huge memory.
                flat = x.detach().float().reshape(-1)
                if flat.numel() > 200_000:
                    idx = torch.randperm(flat.numel(), device=flat.device)[:200_000]
                    flat = flat[idx]
                act_acc[nm].append(flat.cpu())
            return hook
        handles.append(m.register_forward_hook(make_hook(r["name"])))

    # Forward 2 Set5 LR images.
    imgs = sorted(glob.glob(os.path.join(SET5, "*.png")) +
                  glob.glob(os.path.join(SET5, "*.bmp")))[:2]
    with torch.no_grad():
        for ip in imgs:
            hr = np.asarray(Image.open(ip).convert("RGB"))
            lr, _ = bicubic_downsample(hr, scale)
            x = torch.from_numpy(lr).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            _ = model(x).clamp(0, 1)

    for h in handles:
        h.remove()

    # Compute activation stats from accumulated samples.
    act_stats = {}
    for nm, chunks in act_acc.items():
        if not chunks:
            continue
        t = torch.cat(chunks)
        if t.numel() < 64:
            continue
        k, f3, f4, gate = stats_tuple(t)
        act_stats[nm] = {"kurt": k, "frac3": f3, "frac4": f4, "dbaf_gate": gate}

    # Aggregate.
    def agg(rows, key):
        return [r[key] for r in rows if key in r]

    w_kurt = [r["kurt"] for r in weight_rows]
    w_f3 = [r["frac3"] for r in weight_rows]
    w_f4 = [r["frac4"] for r in weight_rows]
    w_gate = [r["dbaf_gate"] for r in weight_rows]

    a_kurt = [v["kurt"] for v in act_stats.values()]
    a_f3 = [v["frac3"] for v in act_stats.values()]
    a_f4 = [v["frac4"] for v in act_stats.values()]
    a_gate = [v["dbaf_gate"] for v in act_stats.values()]

    summary = {
        "scale": scale,
        "n_layers": len(weight_rows),
        # Weight aggregates.
        "weight_pct_dbaf_gated": sum(w_gate) / len(w_gate),
        "weight_mean_kurt": float(np.mean(w_kurt)),
        "weight_median_kurt": float(np.median(w_kurt)),
        "weight_max_kurt": float(np.max(w_kurt)),
        "weight_mean_frac3": float(np.mean(w_f3)),
        "weight_mean_frac4": float(np.mean(w_f4)),
        # Activation aggregates.
        "act_n_layers_observed": len(a_kurt),
        "act_pct_dbaf_gated": (sum(a_gate) / len(a_gate)) if a_gate else 0.0,
        "act_mean_kurt": float(np.mean(a_kurt)) if a_kurt else 0.0,
        "act_median_kurt": float(np.median(a_kurt)) if a_kurt else 0.0,
        "act_max_kurt": float(np.max(a_kurt)) if a_kurt else 0.0,
        "act_mean_frac3": float(np.mean(a_f3)) if a_f3 else 0.0,
        "act_mean_frac4": float(np.mean(a_f4)) if a_f4 else 0.0,
    }
    return summary, weight_rows, act_stats


def main():
    rows = []
    for scale, p in CKPTS.items():
        s, w, a = measure_scale(scale, p)
        outdir = pathlib.Path("/home/ubuntu/unifying-ptq/results/S8-compsrt/swinir-light-x3-alpha-sweep")
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"weight_act_stats_x{scale}.json").write_text(
            json.dumps({"summary": s, "weights": w, "activations": a}, indent=2))
        rows.append(s)
        print(json.dumps(s, indent=2))

    print("\n=== Summary ===")
    cols = ["scale", "n_layers",
            "weight_pct_dbaf_gated", "weight_mean_kurt", "weight_max_kurt", "weight_mean_frac3", "weight_mean_frac4",
            "act_pct_dbaf_gated", "act_mean_kurt", "act_max_kurt", "act_mean_frac3", "act_mean_frac4"]
    print(" | ".join(f"{c:>22}" for c in cols))
    for r in rows:
        print(" | ".join(f"{r[c]:>22}" if not isinstance(r[c], float)
                         else f"{r[c]:>22.5f}" for c in cols))


if __name__ == "__main__":
    main()
