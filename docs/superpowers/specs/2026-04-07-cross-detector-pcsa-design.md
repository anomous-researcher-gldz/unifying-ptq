# Cross-Detector PCSA Generalization

## Goal

Enable calibrating PCSA anchors with one detector (YOLOX) and evaluating with another (H-DETR) without recalibrating anchors. This demonstrates that PCSA anchors generalize across prompt distributions from different detectors.

## Background

PCSA (Prompt-Conditioned Scale Anchoring) maintains K anchor vectors in descriptor space. During calibration, prompt descriptors are matched to the nearest anchor via cosine distance, and per-anchor quantization scales are learned via EMA. At inference, the same matching selects the frozen scale for each input.

Each `QuantDecoderOurAttentionBlock` has its own `PromptAnchorBank` (anchors + counts) and `AnchorAwareFakeQuantize` (per-anchor scale/zero_point). SAM-B's decoder has multiple such blocks, so a checkpoint must capture all of them.

## Architecture

### New CLI Argument

Add `--load-pcsa <path>` to `parse_args()` in `ahcptq/solver/test_quant.py`. Optional string, default `''`.

Add `--save-pcsa <path>` to explicitly control save location. Default: `<work_dir>/pcsa_checkpoint.pt`.

### Save Function

`save_pcsa_checkpoint(model, path)` in `test_quant.py`:

- Walk `model.named_modules()`
- For each `PromptAnchorBank`: save `module.state_dict()` (contains `anchors` [K, D] and `counts` [K])
- For each `AnchorAwareFakeQuantize`: save `scale` (Parameter [K, ...]), `zero_point` (buffer [K, ...]), and `num_anchors`
- Key everything by the dotted module path name
- Call `torch.save(state, path)`

Trigger: runs automatically after `calibrate()` in `main()` when `--load-pcsa` is NOT provided.

### Load Function

`load_pcsa_checkpoint(model, path)` in `test_quant.py`:

- `torch.load(path)` the state dict
- Walk `model.named_modules()`
- For each `PromptAnchorBank` with a matching key: call `module.load_state_dict()`
- For each `AnchorAwareFakeQuantize` with a matching key: set `module.scale = nn.Parameter(saved_scale)` and `module.zero_point = saved_zp`

Trigger: called after `quantize_model()` but before `calibrate()` when `--load-pcsa` is provided.

### Modified Calibration Flow

When `--load-pcsa` is provided:

1. `quantize_model()` runs normally (module replacement creates `PromptAnchorBank` and `AnchorAwareFakeQuantize` instances)
2. `load_pcsa_checkpoint()` populates anchors and per-anchor scales from checkpoint
3. `calibrate()` receives a flag (`pcsa_loaded=True`) that:
   - Skips the activation calibration loop (`for i in range(len(cali_data))` and the BIG bimodal adjust)
   - Still runs weight calibration (`enable_calibration_woquantization(model, 'weight_fake_quant')` + one forward pass)
4. Reconstruction and evaluation proceed as normal

When `--load-pcsa` is NOT provided:

1. Calibration runs as before
2. After calibration, `save_pcsa_checkpoint()` saves the PCSA state

### Inference Path (Unchanged)

`QuantDecoderOurAttentionBlock.forward()` already handles both cases:
- `observer_enabled=1` (calibration): calls `assign_and_update(desc)`
- `observer_enabled=0` (inference): calls `assign(desc, update=False)`

Loaded anchors sit in the same buffers. Cosine matching works identically regardless of whether anchors were freshly calibrated or loaded.

## Files Modified

| File | Change |
|------|--------|
| `ahcptq/solver/test_quant.py` | Add `--load-pcsa`, `--save-pcsa` args; add `save_pcsa_checkpoint()`, `load_pcsa_checkpoint()`; modify `main()` flow and `calibrate()` signature |

No changes to `prompt_anchor.py`, `fake_quant.py`, `quant_model.py`, or any config files.

## Experiment Plan

### Run 1: Calibrate with YOLOX

```bash
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/yolox/yolo_l-sam-vit-b.py \
  --q_config ./exp/config44.yaml \
  --quant-encoder \
  --save-pcsa result/tmp/pcsa_yolox_samb.pt
```

Output: `pcsa_yolox_samb.pt` containing all anchor banks and anchor-aware scales.

### Run 2: Evaluate with H-DETR using YOLOX anchors

```bash
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/hdetr/hdetr-sam-vit-b.py \
  --q_config ./exp/config44.yaml \
  --quant-encoder \
  --load-pcsa result/tmp/pcsa_yolox_samb.pt
```

Loads YOLOX-calibrated anchors, skips PCSA calibration, calibrates weights with H-DETR data, evaluates.

### Baseline: H-DETR with its own PCSA calibration

```bash
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/hdetr/hdetr-sam-vit-b.py \
  --q_config ./exp/config44.yaml \
  --quant-encoder
```

### Expected Result

Cross-detector PCSA (Run 2) should perform comparably to native PCSA (Baseline), demonstrating that PCSA anchors capture prompt-distribution structure that transfers across detectors.

## Prerequisites

- COCO dataset in `data/coco/`
- Model weights in `ckpt/`: SAM-B, YOLOX, HDETR
- Environment set up per README (conda env `ahcptq`, MMCV, CUDA ops compiled)
