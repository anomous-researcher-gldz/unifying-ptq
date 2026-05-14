# G8 Training-Free Full Table — Driver Status

Driver: `scripts/run_training_free_full_table.py`  
Wrapper: `scripts/run_training_free_full_table.sh`  
Output root: `/data/outputs/G8-training-free-full/<target>/<method>_<augments>/eval.json`

---

## Cell Implementation Status

### 96 cells = 3 methods × 4 augment variants × 8 targets

| Method | Augment      | LLaMA-3-8B | Qwen-2.5-7B | SAM-B | SAM-L | SAM-H | SwinIR-x2 | SwinIR-x3 | SwinIR-x4 |
|--------|-------------|-----------|------------|-------|-------|-------|-----------|-----------|-----------|
| RTN    | alone        | FULL      | FULL       | FULL  | FULL  | FULL  | FULL      | FULL      | FULL      |
| RTN    | dbaf         | FULL      | FULL       | FULL  | FULL  | FULL  | FULL      | FULL      | FULL      |
| RTN    | pcsa_tf      | PARTIAL   | PARTIAL    | PARTIAL | PARTIAL | PARTIAL | PARTIAL | PARTIAL | PARTIAL |
| RTN    | dbaf+pcsa_tf | PARTIAL   | PARTIAL    | PARTIAL | PARTIAL | PARTIAL | PARTIAL | PARTIAL | PARTIAL |
| GPTQ   | alone        | FULL      | FULL       | STUB  | STUB  | STUB  | STUB      | STUB      | STUB      |
| GPTQ   | dbaf         | FULL      | FULL       | STUB  | STUB  | STUB  | STUB      | STUB      | STUB      |
| GPTQ   | pcsa_tf      | PARTIAL   | PARTIAL    | STUB  | STUB  | STUB  | STUB      | STUB      | STUB      |
| GPTQ   | dbaf+pcsa_tf | PARTIAL   | PARTIAL    | STUB  | STUB  | STUB  | STUB      | STUB      | STUB      |
| AWQ    | alone        | FULL      | FULL       | STUB  | STUB  | STUB  | STUB      | STUB      | STUB      |
| AWQ    | dbaf         | FULL      | FULL       | STUB  | STUB  | STUB  | STUB      | STUB      | STUB      |
| AWQ    | pcsa_tf      | PARTIAL   | PARTIAL    | STUB  | STUB  | STUB  | STUB      | STUB      | STUB      |
| AWQ    | dbaf+pcsa_tf | PARTIAL   | PARTIAL    | STUB  | STUB  | STUB  | STUB      | STUB      | STUB      |

**Legend:**
- **FULL**: Method is fully wired; weight quantization + eval loop are complete.
- **PARTIAL**: Weight quantization is complete; PCSA-tf activation quantizer is installed but uses an approximate prompt descriptor (mean-pool from current batch) rather than the calibration-time routing. Results will be valid for ablation but not paper-quality.
- **STUB**: Method falls back to RTN weight quantization (the `_quantize_per_channel_with_dbaf` or `_quantize_tensor_uniform` path). A `stub_note` key is written to the eval JSON. See TODOs below.

---

## Fully Wired Cells (run as-is)

1. **LLM × {RTN, GPTQ, AWQ} × {alone, +DBAF}** — 12 cells  
   Uses existing `flatquant/baselines/{rtn,gptq,awq}.py` from task #48.  
   Evals WikiText-2 + C4 PPL (reuses FlatQuant's `eval_utils.ppl_eval` pattern).

---

## PARTIAL Cells (PCSA-tf plumbing present, approximate descriptor)

**All 32 cells with `pcsa_tf` or `dbaf+pcsa_tf` augments** are wired and will run without errors. They produce numerically valid output. The PCSA-tf descriptor is approximated as the mean-pool of the current-batch input activations rather than being routed via the calibration-set cluster centroids. This is acceptable for initial sweep; for paper-quality numbers refine the descriptor routing.

---

## STUB Cells (method falls back to RTN)

### GPTQ/AWQ for SAM encoder (24 cells)
**Reason:** SAM image encoder has Conv2d layers (patch embed, attention projections) that require storing intermediate activations during a calibration forward pass of COCO images. The `quantize_sam_encoder()` helper in `run_training_free_sam.py` only supports per-channel RTN. GPTQ/AWQ cells silently fall back to RTN and write `stub_note` in the JSON.

**Fix needed:** Implement `quantize_sam_encoder_gptq(sam, coco_calibration_images, bits, use_dbaf)` that:
1. Runs 32 COCO images through the encoder with forward hooks.
2. Applies `_GPTQ` from `flatquant/baselines/gptq.py` per Linear, and per-channel RTN for Conv2d.

### GPTQ/AWQ for SwinIR (24 cells)
**Reason:** SwinIR uses Conv2d and Linear layers; the existing `quantize_swinir()` only supports per-channel RTN. GPTQ/AWQ cells fall back to RTN.

**Fix needed:** Implement `quantize_swinir_gptq(model, calib_images, bits, use_dbaf)` that hooks Linear layers during a Set5-based calibration pass.

---

## Infrastructure Notes

### C4 dataset download
C4 evaluation requires `allenai/c4` from HuggingFace Hub. First run will download ~1 GB (single validation shard). If offline, set `HF_DATASETS_OFFLINE=1` and pre-cache with:
```bash
python -c "from datasets import load_dataset; load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')"
```

### SwinIR checkpoints
The driver uses lightweight SwinIR-S checkpoints:
- x2: `ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth`
- x3: `ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth`
- x4: `ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth`

All three exist at `ckpt/swinir/` — confirmed.

### SR eval directories
The driver looks for:
- `data/sr_testsets/Set5_HR/`       — exists
- `data/sr_testsets/Urban100_HR/`   — exists
- `data/sr_testsets/Set5_LR_x{2,3,4}/` — used if present; falls back to bicubic
- `data/sr_testsets/Urban100_LR_x{2,3,4}/` — same

### COCO val2017
Required for SAM eval. Expected at `data/coco/val2017/` and `data/coco/annotations/instances_val2017.json`.

---

## Estimated Runtime Per Cell

| Target      | Method | Time Estimate | Notes |
|-------------|--------|---------------|-------|
| LLaMA-3-8B  | RTN    | 5-8 min       | Weight quant fast; 64-chunk PPL eval dominates |
| LLaMA-3-8B  | GPTQ   | 8-12 min      | Hessian build + PPL |
| LLaMA-3-8B  | AWQ    | 6-10 min      | Act-scale pass + PPL |
| Qwen-2.5-7B | RTN    | 5-8 min       | Same as LLaMA-3-8B |
| SAM-B       | RTN    | 4-6 min       | 500 COCO images |
| SAM-L       | RTN    | 8-12 min      | Larger encoder |
| SAM-H       | RTN    | 25-45 min     | ViT-H encoder; slow per image |
| SwinIR-x2   | RTN    | 30-60 s       | Set5 (5 imgs) + Urban100 (100 imgs) |
| SwinIR-x3   | RTN    | 30-60 s       | Same |
| SwinIR-x4   | RTN    | 30-60 s       | Same |

**Total for full 96-cell sweep (rough):**
- LLM cells (24): ~4-5 hours
- SAM cells (24): ~6-12 hours (SAM-H dominates)
- SwinIR cells (24): ~1-2 hours
- **Grand total: ~12-20 GPU hours**

Recommended: run LLM + SwinIR first (fastest iteration); schedule SAM-H cells overnight.

---

## Invocation Examples

```bash
# Full sweep (serial, background)
bash scripts/run_training_free_full_table.sh > /data/outputs/G8-sweep.log 2>&1 &

# LLM RTN/GPTQ only
TARGETS=llama3-8b,qwen25-7b METHODS=rtn,gptq AUGMENTS=alone,dbaf bash scripts/run_training_free_full_table.sh

# SwinIR all augments
TARGETS=swinir-x2,swinir-x3,swinir-x4 bash scripts/run_training_free_full_table.sh

# Single cell
python scripts/run_training_free_full_table.py \
  --target llama3-8b --method rtn --augments dbaf \
  --out /data/outputs/G8-training-free-full/llama3-8b/rtn_dbaf/eval.json

# Re-run a cell (ignore existing output)
python scripts/run_training_free_full_table.py \
  --target swinir-x2 --method awq --augments alone --force
```
