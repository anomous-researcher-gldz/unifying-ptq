"""Per-layer DBAF ablation for training-free SAM-B image_encoder.

For each Linear/Conv2d in the image_encoder:
  baseline  = no-DBAF (RTN per-channel on every layer)
  per_layer = DBAF applied to EXACTLY ONE layer; RTN on all others.

Measures segmentation mAP (pycocotools) on a small COCO val2017 subset.
For speed, use --coco-imgs 30 --max-layers 12 on first pass.

Output JSON mirrors the SwinIR version:
  {
    "summary": {...},
    "rows": [{"layer", "shape", "frac3", "gate", "metric", "delta"}, ...]
  }
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ── vendor path for segment_anything (mmdet-free) ──────────────────────────
sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/projects/instance_segment_anything/models")

from ahcptq.quantization.fake_quant import is_like_normal_plus_3sigma_outliers
from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf

COCO_ROOT = "/home/ubuntu/unifying-ptq/data/coco"


# ── SAM loading ───────────────────────────────────────────────────────────
def load_sam(model_type: str = "vit_b", checkpoint: Optional[str] = None):
    from segment_anything import sam_model_registry, SamPredictor

    if checkpoint is None:
        _defaults = {
            "vit_b": "/home/ubuntu/unifying-ptq/ckpt/sam_vit_b_01ec64.pth",
            "vit_l": "/home/ubuntu/unifying-ptq/ckpt/sam_vit_l_0b3195.pth",
            "vit_h": "/home/ubuntu/unifying-ptq/ckpt/sam_vit_h_4b8939.pth",
        }
        checkpoint = _defaults[model_type]
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.eval().cuda()
    return SamPredictor(sam), sam


# ── quantisation helpers ──────────────────────────────────────────────────
def _rtn_layer(w: torch.Tensor, bits: int) -> torch.Tensor:
    out_c = w.shape[0]
    w_flat = w.view(out_c, -1) if w.dim() > 2 else w
    return _quantize_tensor_uniform(w_flat, bits, per_channel=True).view_as(w).to(w.dtype)


def _dbaf_layer(w: torch.Tensor, bits: int, alpha: float) -> torch.Tensor:
    out_c = w.shape[0]
    w_flat = w.view(out_c, -1) if w.dim() > 2 else w
    return _quantize_per_channel_with_dbaf(w_flat.float(), bits, alpha=alpha).view_as(w).to(w.dtype)


def list_target_layers(encoder: nn.Module) -> list[str]:
    names = []
    for name, mod in encoder.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and mod.weight.numel() >= 64:
            names.append(name)
    return names


def snapshot_fp_weights(encoder: nn.Module) -> dict[str, torch.Tensor]:
    out = {}
    for name, mod in encoder.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and mod.weight.numel() >= 64:
            out[name] = mod.weight.data.clone()
    return out


def apply_quant_with_one_dbaf(
    encoder: nn.Module,
    fp_weights: dict[str, torch.Tensor],
    dbaf_layer: Optional[str],
    bits: int,
    alpha: float,
) -> None:
    """Reset all encoder weights to FP, then quantize: DBAF for dbaf_layer, RTN for all others."""
    for name, mod in encoder.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and mod.weight.numel() >= 64:
            w_fp = fp_weights[name]
            w_q = (
                _dbaf_layer(w_fp, bits, alpha)
                if name == dbaf_layer
                else _rtn_layer(w_fp, bits)
            )
            mod.weight.data = w_q.to(mod.weight.dtype).clone()


# ── detector helpers ──────────────────────────────────────────────────────
def load_detector():
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    )

    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights).eval().cuda()
    return model, weights.transforms()


def detector_boxes(detector, transforms, image_pil, score_thresh: float = 0.5):
    img_t = transforms(image_pil).cuda().unsqueeze(0)
    with torch.no_grad():
        preds = detector(img_t)[0]
    keep = preds["scores"] >= score_thresh
    return (
        preds["boxes"][keep].cpu().numpy(),
        preds["scores"][keep].cpu().numpy(),
        preds["labels"][keep].cpu().numpy(),
    )


COCO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
    89, 90,
]


# ── evaluation ────────────────────────────────────────────────────────────
def evaluate_map(
    predictor,
    sam,
    detector,
    det_transforms,
    img_ids: list,
    coco_gt,
    tmp_pred_path: str,
) -> float:
    import cv2
    from PIL import Image
    from pycocotools import mask as mask_util
    from pycocotools.cocoeval import COCOeval

    results = []
    for img_id in img_ids:
        info = coco_gt.loadImgs(img_id)[0]
        img_path = f"{COCO_ROOT}/val2017/{info['file_name']}"
        pil = Image.open(img_path).convert("RGB")
        img_np = np.array(pil)
        boxes, scores, labels = detector_boxes(detector, det_transforms, pil, score_thresh=0.5)
        if len(boxes) == 0:
            continue
        predictor.set_image(img_np)
        for box, score, label in zip(boxes, scores, labels):
            masks, _, _ = predictor.predict(box=np.array(box), multimask_output=False)
            mask = masks[0].astype(np.uint8)
            rle = mask_util.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("ascii")
            cat = int(COCO80_TO_COCO91[label - 1]) if 1 <= label <= 80 else int(label)
            results.append(
                {
                    "image_id": int(img_id),
                    "category_id": cat,
                    "segmentation": rle,
                    "score": float(score),
                }
            )

    if not results:
        return float("nan")

    with open(tmp_pred_path, "w") as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(tmp_pred_path)
    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0])


# ── main ──────────────────────────────────────────────────────────────────
def main():
    from pycocotools.coco import COCO

    p = argparse.ArgumentParser(
        description="Per-layer DBAF ablation for SAM-B image_encoder (COCO mAP)"
    )
    p.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--sam-ckpt", default=None)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--coco-imgs", type=int, default=30, help="Number of COCO val2017 images (0=all)")
    p.add_argument("--max-layers", type=int, default=None, help="Cap number of ablation layers")
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()

    print(f"[sam-ablation] loading SAM-{args.model_type}", flush=True)
    predictor, sam = load_sam(args.model_type, args.sam_ckpt)
    encoder = sam.image_encoder

    fp_weights = snapshot_fp_weights(encoder)
    layers = list_target_layers(encoder)
    if args.max_layers:
        layers = layers[: args.max_layers]

    print(f"[sam-ablation] target layers: {len(layers)}", flush=True)

    detector, det_transforms = load_detector()

    coco_gt = COCO(f"{COCO_ROOT}/annotations/instances_val2017.json")
    img_ids = coco_gt.getImgIds()
    if args.coco_imgs > 0:
        img_ids = img_ids[: args.coco_imgs]
    print(f"[sam-ablation] evaluating on {len(img_ids)} COCO val2017 images", flush=True)

    tmp_pred_path = str(pathlib.Path(args.out).with_suffix(".tmp_preds.json"))

    # ── baseline: RTN everywhere, no DBAF ──────────────────────────────────
    apply_quant_with_one_dbaf(encoder, fp_weights, dbaf_layer=None, bits=args.bits, alpha=args.alpha)
    t0 = time.time()
    map_baseline = evaluate_map(
        predictor, sam, detector, det_transforms, img_ids, coco_gt, tmp_pred_path
    )
    print(f"[baseline] no-DBAF mAP = {map_baseline:.4f} ({time.time()-t0:.1f}s)", flush=True)

    rows = []
    for i, name in enumerate(layers):
        fp = fp_weights[name]
        flat = fp.detach().float().reshape(-1)
        mu, sd = flat.mean(), flat.std().clamp_min(1e-8)
        z = (flat - mu) / sd
        frac3 = float((z.abs() > 3.0).float().mean().item())
        gate = bool(is_like_normal_plus_3sigma_outliers(fp)["is_like_c"])

        apply_quant_with_one_dbaf(
            encoder, fp_weights, dbaf_layer=name, bits=args.bits, alpha=args.alpha
        )
        map_l = evaluate_map(
            predictor, sam, detector, det_transforms, img_ids, coco_gt, tmp_pred_path
        )
        delta = map_l - map_baseline

        rows.append(
            {
                "layer": name,
                "shape": list(fp.shape),
                "frac3": frac3,
                "gate": gate,
                "metric": map_l,
                "delta": delta,
            }
        )
        if (i + 1) % 5 == 0 or i == len(layers) - 1:
            print(
                f"  [{i+1:3d}/{len(layers)}] {name}: gate={gate} frac3={frac3:.4f} "
                f"mAP={map_l:.4f} Δ={delta:+.4f}",
                flush=True,
            )

    gate_pass = [r["delta"] for r in rows if r["gate"]]
    gate_fail = [r["delta"] for r in rows if not r["gate"]]
    summary = {
        "model": f"SAM-{args.model_type}",
        "bits": args.bits,
        "alpha": args.alpha,
        "n_coco_imgs": len(img_ids),
        "baseline_map": map_baseline,
        "n_layers": len(rows),
        "n_gate_pass": len(gate_pass),
        "n_gate_fail": len(gate_fail),
        "mean_delta_gate_pass": float(np.mean(gate_pass)) if gate_pass else None,
        "mean_delta_gate_fail": float(np.mean(gate_fail)) if gate_fail else None,
        "max_delta": max(r["delta"] for r in rows) if rows else None,
        "min_delta": min(r["delta"] for r in rows) if rows else None,
        "n_positive_delta": sum(1 for r in rows if r["delta"] > 0),
        "n_negative_delta": sum(1 for r in rows if r["delta"] < 0),
    }

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))

    # Clean up temp predictions file
    tmp = pathlib.Path(tmp_pred_path)
    if tmp.exists():
        tmp.unlink()


if __name__ == "__main__":
    main()
