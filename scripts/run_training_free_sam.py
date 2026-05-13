"""Training-free SAM-B + DBAF on COCO val2017.

Loads SAM-B from the vendored segment_anything package (mmdet-free), uses
torchvision's pretrained Faster-RCNN-R50-FPN as the detector (bounding-box
prompts), applies per-channel RTN to SAM image encoder weights (optionally
+ DBAF fold), runs on COCO val2017, and computes segmentation mAP via
pycocotools.

This fills the "training-free across architectures" cell for SAM in the
EMNLP paper (the LLM cell is RTN+DBAF on LLaMA-3-8B; the SR cell is
RTN+DBAF on SwinIR).
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import time

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util
from PIL import Image

sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
# Bypass the mmdet-2.x wrapper __init__.py by importing segment_anything directly
sys.path.insert(0, "/home/ubuntu/unifying-ptq/projects/instance_segment_anything/models")

from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf


COCO_ROOT = "/home/ubuntu/unifying-ptq/data/coco"


def load_sam(model_type: str = "vit_b", checkpoint: str = None):
    """Load SAM from the vendored segment_anything package (mmdet-free).
    model_type: 'vit_b' | 'vit_l' | 'vit_h'.
    """
    from segment_anything import sam_model_registry, SamPredictor
    if checkpoint is None:
        default = {
            "vit_b": "/home/ubuntu/unifying-ptq/ckpt/sam_vit_b_01ec64.pth",
            "vit_l": "/home/ubuntu/unifying-ptq/ckpt/sam_vit_l_0b3195.pth",
            "vit_h": "/home/ubuntu/unifying-ptq/ckpt/sam_vit_h_4b8939.pth",
        }
        checkpoint = default[model_type]
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.eval().cuda()
    return SamPredictor(sam), sam


def quantize_sam_encoder(sam, bits: int = 4, use_dbaf: bool = False, alpha: float = 0.75):
    """Apply per-channel RTN (optionally + DBAF) to all Linear + Conv2d weights in image_encoder."""
    encoder = sam.image_encoder
    n_quant = 0
    for name, mod in encoder.named_modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight.data.float()
            if use_dbaf:
                w_q = _quantize_per_channel_with_dbaf(w, bits, alpha=alpha)
            else:
                w_q = _quantize_tensor_uniform(w, bits, per_channel=True)
            mod.weight.data = w_q.to(mod.weight.dtype)
            n_quant += 1
        elif isinstance(mod, nn.Conv2d):
            # Reshape [out, in, k, k] -> [out, in*k*k] for per-output-channel scaling
            w = mod.weight.data.float()
            out_c = w.shape[0]
            w_flat = w.view(out_c, -1)
            if use_dbaf:
                w_q = _quantize_per_channel_with_dbaf(w_flat, bits, alpha=alpha)
            else:
                w_q = _quantize_tensor_uniform(w_flat, bits, per_channel=True)
            mod.weight.data = w_q.view_as(w).to(mod.weight.dtype)
            n_quant += 1
    print(f"[SAM-quant] quantized {n_quant} modules in image_encoder (bits={bits}, dbaf={use_dbaf})", flush=True)
    return sam


def load_detector():
    """Pretrained torchvision Faster-RCNN R50-FPN (COCO pretrained)."""
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights).eval().cuda()
    return model, weights.transforms()


def detector_boxes(detector, transforms, image_pil, score_thresh: float = 0.5):
    """Run detector; return (boxes, scores, labels) in image coords."""
    img_t = transforms(image_pil).cuda().unsqueeze(0)
    with torch.no_grad():
        preds = detector(img_t)[0]
    keep = preds["scores"] >= score_thresh
    return (preds["boxes"][keep].cpu().numpy(),
            preds["scores"][keep].cpu().numpy(),
            preds["labels"][keep].cpu().numpy())


COCO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
    89, 90,
]


def evaluate(args):
    print(f"[eval] loading SAM-{args.model_type[-1].upper()} + Faster-RCNN", flush=True)
    predictor, sam = load_sam(args.model_type, args.sam_ckpt)
    sam = quantize_sam_encoder(sam, bits=args.bits, use_dbaf=args.use_dbaf, alpha=args.alpha)
    detector, det_transforms = load_detector()

    print(f"[eval] opening COCO annotations", flush=True)
    coco_gt = COCO(f"{COCO_ROOT}/annotations/instances_val2017.json")
    img_ids = coco_gt.getImgIds()
    if args.max_images > 0:
        img_ids = img_ids[:args.max_images]
    print(f"[eval] processing {len(img_ids)} images", flush=True)

    results = []
    t0 = time.time()
    for i, img_id in enumerate(img_ids):
        info = coco_gt.loadImgs(img_id)[0]
        img_path = f"{COCO_ROOT}/val2017/{info['file_name']}"
        pil = Image.open(img_path).convert("RGB")
        img_np = np.array(pil)
        boxes, scores, labels = detector_boxes(detector, det_transforms, pil, score_thresh=0.5)
        if len(boxes) == 0:
            continue
        predictor.set_image(img_np)
        for box, score, label in zip(boxes, scores, labels):
            input_box = np.array(box)
            masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
            mask = masks[0].astype(np.uint8)
            rle = mask_util.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("ascii")
            cat = int(COCO80_TO_COCO91[label - 1]) if 1 <= label <= 80 else int(label)
            results.append({
                "image_id": int(img_id),
                "category_id": cat,
                "segmentation": rle,
                "score": float(score),
            })
        if (i + 1) % 50 == 0:
            print(f"[eval] {i+1}/{len(img_ids)} ({(time.time()-t0):.1f}s)", flush=True)

    print(f"[eval] {len(results)} predictions, computing mAP", flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    preds_path = str(pathlib.Path(args.out).with_suffix(".preds.json"))
    with open(preds_path, "w") as f:
        json.dump(results, f)
    if not results:
        print("[eval] no predictions, mAP undefined")
        ap = -1.0
    else:
        coco_dt = coco_gt.loadRes(preds_path)
        ev = COCOeval(coco_gt, coco_dt, "segm")
        ev.params.imgIds = img_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
        ap = float(ev.stats[0])

    out = {
        "model": "SAM-" + args.model_type[-1].upper(),
        "method": f"RTN{'+DBAF' if args.use_dbaf else ''}",
        "bits": args.bits,
        "detector": "torchvision-fasterrcnn-r50-fpn",
        "n_images": len(img_ids),
        "n_predictions": len(results),
        "segm_mAP": ap,
        "wallclock_seconds": time.time() - t0,
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--sam-ckpt", default=None)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--use-dbaf", action="store_true")
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--max-images", type=int, default=0, help="0 = all val2017")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
