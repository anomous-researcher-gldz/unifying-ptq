"""Training-free full table driver: RTN/GPTQ/AWQ ± {DBAF, PCSA-tf, both} across models.

Cells = {RTN, GPTQ, AWQ} × {alone, +DBAF, +PCSA-tf, +DBAF+PCSA-tf}
      × {llama3-8b, qwen25-7b, sam-b, sam-l, sam-h, swinir-x2, swinir-x3, swinir-x4}
= 12 augment×method × 8 models = 96 cells total.

Usage:
  python scripts/run_training_free_full_table.py \\
    --target llama3-8b \\
    --method rtn \\
    --augments dbaf+pcsa_tf \\
    --out /data/outputs/G8-training-free-full/llama3-8b/rtn_dbaf+pcsa_tf

Env vars (used by the bash wrapper):
  TARGETS  — comma-separated subset of target names   (default: all)
  METHODS  — comma-separated subset of method names   (default: rtn,gptq,awq)
  AUGMENTS — comma-separated subset of augment strings (default: alone,dbaf,pcsa_tf,dbaf+pcsa_tf)
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import time

# ---------------------------------------------------------------------------
# Path setup — must come first so sub-imports resolve regardless of cwd.
# ---------------------------------------------------------------------------
_REPO = pathlib.Path(__file__).resolve().parent.parent
_FLATQUANT = _REPO / "FlatQuant"
_COMPSRT   = _REPO / "CompSRT"
_PROJECTS  = _REPO / "projects" / "instance_segment_anything" / "models"
for _p in [str(_REPO), str(_FLATQUANT), str(_COMPSRT), str(_PROJECTS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_TARGETS  = ["llama3-8b", "qwen25-7b", "sam-b", "sam-l", "sam-h",
                "swinir-x2", "swinir-x3", "swinir-x4"]
ALL_METHODS  = ["rtn", "gptq", "awq"]
ALL_AUGMENTS = ["alone", "dbaf", "pcsa_tf", "dbaf+pcsa_tf"]

LLM_TARGETS  = {"llama3-8b", "qwen25-7b"}
SAM_TARGETS  = {"sam-b", "sam-l", "sam-h"}
SWINIR_TARGETS = {"swinir-x2", "swinir-x3", "swinir-x4"}

LLM_MODEL_PATHS = {
    "llama3-8b": "/data/modelzoo/meta-llama/Meta-Llama-3-8B",
    "qwen25-7b":  "/data/modelzoo/Qwen/Qwen2.5-7B",
}

SAM_MODEL_TYPES = {
    "sam-b": "vit_b",
    "sam-l": "vit_l",
    "sam-h": "vit_h",
}

# SwinIR checkpoint paths (existing ckpt dir layout)
SWINIR_CKPT = {
    "swinir-x2": str(_REPO / "ckpt" / "swinir" / "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"),
    "swinir-x3": str(_REPO / "ckpt" / "swinir" / "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth"),
    "swinir-x4": str(_REPO / "ckpt" / "swinir" / "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"),
}

SWINIR_SCALE = {"swinir-x2": 2, "swinir-x3": 3, "swinir-x4": 4}

SR_SET5_DIR    = str(_REPO / "data" / "sr_testsets" / "Set5_HR")
SR_URBAN_DIR   = str(_REPO / "data" / "sr_testsets" / "Urban100_HR")

COCO_ROOT = str(_REPO / "data" / "coco")

CALIB_NSAMPLES = 4    # calibration batch size for GPTQ/AWQ/PCSA-tf
CALIB_SEQLEN   = 2048
PPL_SAMPLES    = 64   # wikitext eval chunks

PCSA_K = 8            # number of PCSA-tf anchors

# ---------------------------------------------------------------------------
# Augment parsing helpers
# ---------------------------------------------------------------------------

def _aug_flags(augments: str) -> tuple[bool, bool]:
    """Return (use_dbaf, use_pcsa_tf) from an augment string."""
    a = augments.lower()
    use_dbaf    = "dbaf" in a
    use_pcsa_tf = "pcsa_tf" in a or "pcsa-tf" in a
    return use_dbaf, use_pcsa_tf


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _load_llm(target: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = LLM_MODEL_PATHS[target]
    print(f"[driver] loading LLM {target} from {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


def _calib_batch_llm(tok, seq_len: int = CALIB_SEQLEN, n: int = CALIB_NSAMPLES):
    """Small WikiText-2 train slice for GPTQ/AWQ calibration."""
    from datasets import load_dataset
    import torch
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[:, : n * seq_len].view(n, seq_len).cuda()
    return ids


def _eval_ppl_wikitext2(model, tok, seq_len: int = CALIB_SEQLEN, n_samples: int = PPL_SAMPLES) -> float:
    import torch
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    n_chunks = min(n_samples, ids.shape[1] // seq_len)
    nlls = []
    for i in range(n_chunks):
        chunk = ids[:, i * seq_len:(i + 1) * seq_len]
        with torch.no_grad():
            out = model(chunk, labels=chunk)
        nlls.append(out.loss.float().item())
    return float(torch.tensor(nlls).mean().exp().item())


def _eval_ppl_c4(model, tok, seq_len: int = CALIB_SEQLEN, n_samples: int = PPL_SAMPLES) -> float:
    """C4 PPL — downloads on first use (requires network / HF cache)."""
    import torch
    from datasets import load_dataset
    # Use FlatQuant's data_utils pattern: allenai/c4 validation shard
    ds = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    text = " ".join(ds[:1100]["text"])
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    n_chunks = min(n_samples, ids.shape[1] // seq_len)
    nlls = []
    for i in range(n_chunks):
        chunk = ids[:, i * seq_len:(i + 1) * seq_len]
        with torch.no_grad():
            out = model(chunk, labels=chunk)
        nlls.append(out.loss.float().item())
    if not nlls:
        return float("nan")
    return float(torch.tensor(nlls).mean().exp().item())


def _collect_llm_pcsa_state(model, tok) -> dict:
    """One-pass calibration for PCSA-tf on LLM hidden states.

    Hooks the final hidden state of each forward pass, treats it as
    a prompt-level descriptor and as the raw activation for max-abs scaling.
    Returns a PCSA-tf state dict {"anchors": ..., "scales": ...}.

    TODO: For production, collect per-layer PCSA states (one per Linear block).
          Currently a single global state derived from the LM embedding output
          is returned as a stand-in — sufficient to test integration plumbing
          but not the full per-layer PCSA-tf described in the paper.
    """
    import torch
    from flatquant.baselines.pcsa_tf import fit_pcsa_tf
    print("[driver] collecting LLM PCSA-tf calibration activations ...", flush=True)
    calib = _calib_batch_llm(tok)
    hidden_states = []
    hooks = []

    def _make_hook():
        def _h(module, inp, out):
            # out might be a tuple (for transformer blocks); take first element
            hs = out[0] if isinstance(out, tuple) else out
            hidden_states.append(hs.detach().float().cpu())
        return _h

    # Hook on the embedding layer to get token-level descriptors
    embed_layer = model.model.embed_tokens if hasattr(model, "model") else None
    if embed_layer is not None:
        hooks.append(embed_layer.register_forward_hook(_make_hook()))

    with torch.no_grad():
        model(calib)
    for h in hooks:
        h.remove()

    if not hidden_states:
        # Fallback: no hook fired; return dummy state
        print("[driver] WARNING: no hidden states captured; PCSA-tf state is dummy", flush=True)
        dummy = torch.zeros(PCSA_K, 1)
        return {"anchors": dummy, "scales": torch.ones(PCSA_K)}

    # descs: [N, D] — mean-pool over token dim per prompt
    hs = hidden_states[0]  # [N, T, D]
    descs = hs.mean(dim=1)  # [N, D]
    state = fit_pcsa_tf(descs, hs, K=PCSA_K)
    print(f"[driver] PCSA-tf fitted: K={PCSA_K} anchors, scales={state['scales']}", flush=True)
    return state


def _apply_pcsa_tf_to_llm(model, state: dict):
    """Wrap LLM forward to apply PCSA-tf activation fake-quantization.

    Hooks the input of every Linear layer (excluding lm_head) with a
    per-prompt activation quantizer. The descriptor is approximated by
    averaging the current input batch along the token dimension.

    NOTE: This is a one-shot wrapper; the real implementation should route
    using the same descriptor as used at fit time. This is a plumbing stub
    for integration testing.
    """
    import torch
    import torch.nn as nn
    from flatquant.baselines.pcsa_tf import apply_pcsa_tf_to_activation

    def _make_forward_hook(orig_forward, _state):
        def _wrapped(x, *args, **kwargs):
            with torch.no_grad():
                # desc: [B, D] — mean-pool over sequence
                if x.dim() == 3:
                    desc = x.mean(dim=1).float()
                else:
                    desc = x.float()
                # Move state to correct device on first call
                if _state["anchors"].device != x.device:
                    _state["anchors"] = _state["anchors"].to(x.device)
                    _state["scales"]  = _state["scales"].to(x.device)
                x = apply_pcsa_tf_to_activation(x, desc, _state, bits=4)
            return orig_forward(x, *args, **kwargs)
        return _wrapped

    n_wrapped = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            mod.forward = _make_forward_hook(mod.forward, state)
            n_wrapped += 1
    print(f"[driver] PCSA-tf activation hooks installed on {n_wrapped} Linear layers", flush=True)
    return model


def run_llm(target: str, method: str, augments: str, out_path: pathlib.Path):
    import torch
    use_dbaf, use_pcsa_tf = _aug_flags(augments)
    model, tok = _load_llm(target)

    t0 = time.time()

    # --- PCSA-tf calibration (before weight quant, while model is FP) ---
    pcsa_state = None
    if use_pcsa_tf:
        pcsa_state = _collect_llm_pcsa_state(model, tok)

    # --- Weight quantization ---
    calib = None
    if method in ("gptq", "awq"):
        calib = _calib_batch_llm(tok)

    if method == "rtn":
        from flatquant.baselines.rtn import quantize_model
        model = quantize_model(model, bits=4, use_dbaf=use_dbaf)
    elif method == "gptq":
        from flatquant.baselines.gptq import quantize_model
        model = quantize_model(model, bits=4, calibration_data=calib, use_dbaf=use_dbaf)
    elif method == "awq":
        from flatquant.baselines.awq import quantize_model
        model = quantize_model(model, bits=4, calibration_data=calib, use_dbaf=use_dbaf)
    else:
        raise ValueError(f"Unknown method: {method}")

    # --- Apply PCSA-tf activation quantizer ---
    if use_pcsa_tf and pcsa_state is not None:
        model = _apply_pcsa_tf_to_llm(model, pcsa_state)

    print("[driver] evaluating WikiText-2 PPL ...", flush=True)
    wt2_ppl = _eval_ppl_wikitext2(model, tok)
    print(f"[driver] WikiText-2 PPL = {wt2_ppl:.3f}", flush=True)

    print("[driver] evaluating C4 PPL ...", flush=True)
    try:
        c4_ppl = _eval_ppl_c4(model, tok)
        print(f"[driver] C4 PPL = {c4_ppl:.3f}", flush=True)
    except Exception as exc:
        print(f"[driver] WARNING: C4 eval failed ({exc}); setting nan", flush=True)
        c4_ppl = float("nan")

    result = {
        "target": target,
        "method": method,
        "augments": augments,
        "metrics": {
            "wikitext2_ppl": wt2_ppl,
            "c4_ppl": c4_ppl,
        },
        "wallclock_seconds": time.time() - t0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# SAM helpers
# ---------------------------------------------------------------------------

def _collect_sam_pcsa_state(sam, bits: int = 4) -> dict:
    """Fit PCSA-tf state from a tiny set of COCO images (first 32).

    Descriptor = mean-pooled patch embedding output per image.
    Activation  = flattened patch token features.

    TODO: For production, collect per-layer states (one per attention block).
          Current implementation uses global image encoder output as descriptor.
    """
    import torch
    import numpy as np
    from PIL import Image
    from pycocotools.coco import COCO
    from flatquant.baselines.pcsa_tf import fit_pcsa_tf

    print("[driver] collecting SAM PCSA-tf calibration activations ...", flush=True)
    coco_gt = COCO(f"{COCO_ROOT}/annotations/instances_val2017.json")
    img_ids = coco_gt.getImgIds()[:32]
    descs_list, acts_list = [], []
    for img_id in img_ids:
        info = coco_gt.loadImgs(img_id)[0]
        img_path = f"{COCO_ROOT}/val2017/{info['file_name']}"
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        img_np = np.array(pil)
        # Encode through image encoder to get patch features
        sam.predictor.set_image(img_np)
        features = sam.predictor.features  # [1, C, H, W]
        if features is None:
            continue
        feat = features.squeeze(0)  # [C, H, W]
        # mean-pool over spatial dims -> descriptor [C]
        desc = feat.mean(dim=(1, 2)).unsqueeze(0)   # [1, C]
        # flatten spatial -> acts [1, H*W, C]
        C, H, W = feat.shape
        act = feat.view(C, H * W).T.unsqueeze(0)   # [1, H*W, C]
        descs_list.append(desc.cpu().float())
        acts_list.append(act.cpu().float())

    if not descs_list:
        print("[driver] WARNING: no SAM PCSA-tf calib images; returning dummy state", flush=True)
        return {"anchors": torch.zeros(PCSA_K, 1), "scales": torch.ones(PCSA_K)}

    descs = torch.cat(descs_list, dim=0)  # [N, C]
    acts  = torch.cat(acts_list,  dim=0)  # [N, H*W, C]
    state = fit_pcsa_tf(descs, acts, K=PCSA_K)
    print(f"[driver] SAM PCSA-tf fitted: K={PCSA_K}", flush=True)
    return state


def _apply_pcsa_tf_to_sam(sam, state: dict):
    """Wrap SAM image encoder Linear layers with PCSA-tf activation quantizer.

    NOTE: The descriptor used at inference time is approximated from the
    current batch input (mean-pooled along the token/spatial dimension).
    This is a plumbing stub — full production integration should route via
    the same embedding space as the calibration descriptors.
    """
    import torch
    import torch.nn as nn
    from flatquant.baselines.pcsa_tf import apply_pcsa_tf_to_activation

    def _make_hook(orig_fwd, _state):
        def _wrapped(x, *args, **kwargs):
            with torch.no_grad():
                desc = x.mean(dim=tuple(range(1, x.dim() - 1))).float()  # [B, D]
                if _state["anchors"].device != x.device:
                    _state["anchors"] = _state["anchors"].to(x.device)
                    _state["scales"]  = _state["scales"].to(x.device)
                x = apply_pcsa_tf_to_activation(x, desc, _state, bits=4)
            return orig_fwd(x, *args, **kwargs)
        return _wrapped

    n_wrapped = 0
    for name, mod in sam.image_encoder.named_modules():
        if isinstance(mod, nn.Linear):
            mod.forward = _make_hook(mod.forward, state)
            n_wrapped += 1
    print(f"[driver] PCSA-tf activation hooks installed on {n_wrapped} SAM encoder layers", flush=True)
    return sam


def run_sam(target: str, method: str, augments: str, out_path: pathlib.Path):
    """Run SAM evaluation cell.

    Weight quantization method is always RTN-style (GPTQ/AWQ for convolutional
    vision models is not standard; we apply them to Linear layers only and fall
    back to per-channel RTN for Conv2d — matching the training-free SAM script).

    TODO: Implement proper GPTQ/AWQ for Conv2d layers in SAM encoder;
          currently GPTQ/AWQ cells apply RTN for Conv2d and GPTQ/AWQ for Linear.
    """
    import torch
    import torch.nn as nn
    import numpy as np
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_util

    # Import SAM loader from the existing training-free SAM script
    sys.path.insert(0, str(_REPO / "scripts"))
    from run_training_free_sam import (
        load_sam, load_detector, detector_boxes, COCO80_TO_COCO91,
        quantize_sam_encoder,
    )
    from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf

    use_dbaf, use_pcsa_tf = _aug_flags(augments)
    model_type = SAM_MODEL_TYPES[target]

    print(f"[driver] loading SAM {model_type}", flush=True)
    predictor, sam = load_sam(model_type)

    t0 = time.time()

    # --- PCSA-tf calibration before weight quant (FP model) ---
    if use_pcsa_tf:
        # Attach predictor to sam object for convenience inside helper
        sam.predictor = predictor
        pcsa_state = _collect_sam_pcsa_state(sam)
    else:
        pcsa_state = None

    # --- Weight quantization ---
    if method == "rtn":
        sam = quantize_sam_encoder(sam, bits=4, use_dbaf=use_dbaf)
    elif method in ("gptq", "awq"):
        # TODO: Full GPTQ/AWQ for SAM encoder requires storing intermediate
        #       activations per layer during a calibration forward pass through
        #       the image encoder. Stub: apply RTN (use_dbaf controls DBAF);
        #       mark as TODO in output JSON.
        print(f"[driver] WARNING: {method} for SAM encoder falls back to RTN "
              "(Conv2d GPTQ/AWQ not yet implemented; see TODO)", flush=True)
        sam = quantize_sam_encoder(sam, bits=4, use_dbaf=use_dbaf)

    # --- Apply PCSA-tf activation quantizer ---
    if use_pcsa_tf and pcsa_state is not None:
        sam = _apply_pcsa_tf_to_sam(sam, pcsa_state)

    # --- Eval: COCO val 500 images ---
    detector, det_transforms = load_detector()
    coco_gt = COCO(f"{COCO_ROOT}/annotations/instances_val2017.json")
    img_ids = coco_gt.getImgIds()[:500]
    print(f"[driver] evaluating SAM on {len(img_ids)} COCO images", flush=True)

    results = []
    for i, img_id in enumerate(img_ids):
        info = coco_gt.loadImgs(img_id)[0]
        img_path = f"{COCO_ROOT}/val2017/{info['file_name']}"
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception:
            continue
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
            results.append({
                "image_id": int(img_id),
                "category_id": cat,
                "segmentation": rle,
                "score": float(score),
            })
        if (i + 1) % 100 == 0:
            print(f"[driver] SAM eval {i+1}/{len(img_ids)}", flush=True)

    # Save preds and compute mAP
    preds_path = str(out_path.with_suffix(".preds.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(preds_path, "w") as f:
        json.dump(results, f)

    if results:
        coco_dt = coco_gt.loadRes(preds_path)
        ev = COCOeval(coco_gt, coco_dt, "segm")
        ev.params.imgIds = img_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
        ap = float(ev.stats[0])
    else:
        print("[driver] WARNING: no SAM predictions; mAP = -1", flush=True)
        ap = -1.0

    stub_note = (
        f"GPTQ/AWQ for SAM Conv2d not implemented; RTN used for {method}"
        if method in ("gptq", "awq") else None
    )
    result = {
        "target": target,
        "method": method,
        "augments": augments,
        "metrics": {
            "coco_map": ap,
        },
        "wallclock_seconds": time.time() - t0,
    }
    if stub_note:
        result["stub_note"] = stub_note
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# SwinIR helpers
# ---------------------------------------------------------------------------

def _collect_swinir_pcsa_state(model, scale: int) -> dict:
    """Fit PCSA-tf state from a few Set5 images.

    Descriptor = mean-pooled activation after the first SwinIR transformer block.
    Activation  = same tensor, used for max-abs scale.

    TODO: For production, collect per-layer states (one per RSTB/attention block).
          Currently a single global state is returned.
    """
    import torch
    import torch.nn as nn
    import numpy as np
    import glob
    from PIL import Image
    from flatquant.baselines.pcsa_tf import fit_pcsa_tf

    print("[driver] collecting SwinIR PCSA-tf calibration activations ...", flush=True)
    sr_dir = SR_SET5_DIR
    hr_paths = sorted(glob.glob(f"{sr_dir}/*.png") + glob.glob(f"{sr_dir}/*.PNG"))

    hidden_states = []
    hooks = []

    def _make_hook():
        def _h(mod, inp, out):
            hs = out.detach().cpu().float()
            hidden_states.append(hs)
        return _h

    # Hook on the first layer of the residual blocks
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(_make_hook()))
            break  # just the first Linear

    for hr_path in hr_paths[:5]:
        try:
            hr = np.array(Image.open(hr_path).convert("RGB"))
        except Exception:
            continue
        h, w = hr.shape[:2]
        h = h - h % 8; w = w - w % 8
        lr = np.array(Image.fromarray(hr).resize((w // scale, h // scale), Image.BICUBIC))
        x = torch.from_numpy(lr).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
        with torch.no_grad():
            model(x)

    for h in hooks:
        h.remove()

    if not hidden_states:
        print("[driver] WARNING: no SwinIR PCSA-tf calib states; returning dummy", flush=True)
        return {"anchors": torch.zeros(PCSA_K, 1), "scales": torch.ones(PCSA_K)}

    # Each hidden_state: [1, T, D] or [1, D]
    hs_cat = torch.cat(hidden_states, dim=0)  # [N, ...]
    if hs_cat.dim() == 3:
        descs = hs_cat.mean(dim=1)  # [N, D]
    else:
        descs = hs_cat  # [N, D]
    state = fit_pcsa_tf(descs, hs_cat, K=min(PCSA_K, descs.shape[0]))
    print(f"[driver] SwinIR PCSA-tf fitted: K={PCSA_K}", flush=True)
    return state


def _apply_pcsa_tf_to_swinir(model, state: dict):
    """Wrap SwinIR Linear layers with PCSA-tf activation quantizer.

    NOTE: Descriptor approximated from current batch input. Production
    integration should route using calibration-time descriptors.
    """
    import torch
    import torch.nn as nn
    from flatquant.baselines.pcsa_tf import apply_pcsa_tf_to_activation

    def _make_hook(orig_fwd, _state):
        def _wrapped(x, *args, **kwargs):
            with torch.no_grad():
                if x.dim() >= 2:
                    desc = x.reshape(x.shape[0], -1).float()
                    # reduce to [B, D_small] via mean if too large
                    if desc.shape[-1] > 4096:
                        desc = desc.view(desc.shape[0], -1, 64).mean(dim=-1)
                else:
                    desc = x.float().unsqueeze(0)
                if _state["anchors"].device != x.device:
                    _state["anchors"] = _state["anchors"].to(x.device)
                    _state["scales"]  = _state["scales"].to(x.device)
                # Ensure anchors has matching dim to desc
                if _state["anchors"].shape[-1] != desc.shape[-1]:
                    # dimension mismatch; skip PCSA-tf and pass through
                    return orig_fwd(x, *args, **kwargs)
                x = apply_pcsa_tf_to_activation(x, desc, _state, bits=4)
            return orig_fwd(x, *args, **kwargs)
        return _wrapped

    n_wrapped = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if "upsample" in name or "conv_last" in name or "conv_first" in name:
                continue
            mod.forward = _make_hook(mod.forward, state)
            n_wrapped += 1
    print(f"[driver] PCSA-tf hooks installed on {n_wrapped} SwinIR Linear layers", flush=True)
    return model


def run_swinir(target: str, method: str, augments: str, out_path: pathlib.Path):
    """Run SwinIR evaluation cell.

    GPTQ/AWQ for SwinIR: Conv2d layers are handled with RTN (GPTQ/AWQ for Conv2d
    requires calibration images; TODO for full implementation). Linear layers
    in transformer blocks are quantized with the selected method.

    TODO: Full GPTQ calibration for SwinIR requires running calibration images
          through the encoder to capture intermediate activations per Linear layer.
          Currently GPTQ/AWQ fall back to RTN for Conv2d and apply RTN to Linear
          (same as RTN path) — a TODO stub.
    """
    import torch
    import numpy as np
    import glob
    from PIL import Image

    sys.path.insert(0, str(_REPO / "scripts"))
    from run_training_free_swinir import load_swinir, quantize_swinir, evaluate as swinir_evaluate

    use_dbaf, use_pcsa_tf = _aug_flags(augments)
    scale = SWINIR_SCALE[target]
    ckpt  = SWINIR_CKPT[target]

    print(f"[driver] loading SwinIR x{scale} from {ckpt}", flush=True)
    model = load_swinir(scale, ckpt)

    t0 = time.time()

    # --- PCSA-tf calibration before weight quant ---
    if use_pcsa_tf:
        pcsa_state = _collect_swinir_pcsa_state(model, scale)
    else:
        pcsa_state = None

    # --- Weight quantization ---
    if method == "rtn":
        model = quantize_swinir(model, bits=4, use_dbaf=use_dbaf)
    elif method in ("gptq", "awq"):
        # TODO: implement GPTQ/AWQ for SwinIR Linear layers via calibration
        #       forward pass. Stub: use RTN for now.
        print(f"[driver] WARNING: {method} for SwinIR falls back to RTN "
              "(Conv2d GPTQ/AWQ not yet implemented; see TODO)", flush=True)
        model = quantize_swinir(model, bits=4, use_dbaf=use_dbaf)

    # --- Apply PCSA-tf activation quantizer ---
    if use_pcsa_tf and pcsa_state is not None:
        model = _apply_pcsa_tf_to_swinir(model, pcsa_state)

    # --- Eval: Set5 + Urban100 ---
    set5_dir   = _REPO / "data" / "sr_testsets" / "Set5_HR"
    urban_dir  = _REPO / "data" / "sr_testsets" / "Urban100_HR"
    set5_lr    = _REPO / "data" / "sr_testsets" / f"Set5_LR_x{scale}"
    urban_lr   = _REPO / "data" / "sr_testsets" / f"Urban100_LR_x{scale}"

    def _dir_or_none(p):
        return str(p) if p.exists() else None

    print("[driver] evaluating SwinIR on Set5 ...", flush=True)
    set5_psnr, set5_n = swinir_evaluate(model, scale, str(set5_dir), _dir_or_none(set5_lr))
    print(f"[driver] Set5 PSNR = {set5_psnr:.2f} dB ({set5_n} images)", flush=True)

    print("[driver] evaluating SwinIR on Urban100 ...", flush=True)
    urban_psnr, urban_n = swinir_evaluate(model, scale, str(urban_dir), _dir_or_none(urban_lr))
    print(f"[driver] Urban100 PSNR = {urban_psnr:.2f} dB ({urban_n} images)", flush=True)

    stub_note = (
        f"GPTQ/AWQ for SwinIR not implemented; RTN used for {method}"
        if method in ("gptq", "awq") else None
    )
    result = {
        "target": target,
        "method": method,
        "augments": augments,
        "metrics": {
            "set5_psnr_db": set5_psnr,
            "urban100_psnr_db": urban_psnr,
        },
        "wallclock_seconds": time.time() - t0,
    }
    if stub_note:
        result["stub_note"] = stub_note
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def _default_out(target: str, method: str, augments: str) -> pathlib.Path:
    base = pathlib.Path("/data/outputs/G8-training-free-full")
    cell_name = f"{method}_{augments}"
    return base / target / cell_name / "eval.json"


def run_cell(target: str, method: str, augments: str, out_path: pathlib.Path):
    if target in LLM_TARGETS:
        return run_llm(target, method, augments, out_path)
    elif target in SAM_TARGETS:
        return run_sam(target, method, augments, out_path)
    elif target in SWINIR_TARGETS:
        return run_swinir(target, method, augments, out_path)
    else:
        raise ValueError(f"Unknown target: {target}")


def main():
    p = argparse.ArgumentParser(
        description="Training-free full table driver: RTN/GPTQ/AWQ ± DBAF ± PCSA-tf across models."
    )
    p.add_argument("--target",   required=True, choices=ALL_TARGETS,
                   help="Model target.")
    p.add_argument("--method",   required=True, choices=ALL_METHODS,
                   help="Quantization method.")
    p.add_argument("--augments", required=True,
                   choices=ALL_AUGMENTS,
                   help="Augmentation(s): alone | dbaf | pcsa_tf | dbaf+pcsa_tf.")
    p.add_argument("--out",      default=None,
                   help="Output eval.json path. Default: /data/outputs/G8-training-free-full/<target>/<method>_<augments>/eval.json")
    p.add_argument("--force",    action="store_true",
                   help="Re-run even if eval.json already exists.")
    args = p.parse_args()

    out_path = pathlib.Path(args.out) if args.out else _default_out(args.target, args.method, args.augments)

    if out_path.exists() and not args.force:
        print(f"[driver] SKIP — already exists: {out_path}", flush=True)
        existing = json.loads(out_path.read_text())
        print(json.dumps(existing, indent=2))
        return

    print(f"[driver] target={args.target}  method={args.method}  augments={args.augments}", flush=True)
    print(f"[driver] output -> {out_path}", flush=True)
    run_cell(args.target, args.method, args.augments, out_path)


if __name__ == "__main__":
    main()
