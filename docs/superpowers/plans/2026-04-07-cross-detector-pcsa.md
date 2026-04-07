# Cross-Detector PCSA Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable saving PCSA anchors+scales from one detector's calibration run and loading them in another detector's evaluation run, demonstrating cross-detector generalization of prompt-conditioned quantization.

**Architecture:** Add `save_pcsa_checkpoint()` and `load_pcsa_checkpoint()` functions to `test_quant.py` that walk `model.named_modules()` to collect/restore state from all `PromptAnchorBank` and `AnchorAwareFakeQuantize` instances. Modify `calibrate()` to accept a `pcsa_loaded` flag that skips the activation calibration loop while still calibrating weights. All changes are in a single file.

**Tech Stack:** PyTorch, mmdetection, mmcv, CUDA

---

### Task 1: Add CLI Arguments

**Files:**
- Modify: `ahcptq/solver/test_quant.py:153-298` (in `parse_args()`)

- [ ] **Step 1: Add `--load-pcsa` and `--save-pcsa` arguments to `parse_args()`**

In `ahcptq/solver/test_quant.py`, add these two arguments after the existing `--load_sam_path` block (after line 233):

```python
    parser.add_argument(
        '--load-pcsa',
        type=str,
        default='',
        help='Path to a PCSA checkpoint to load (skips PCSA calibration)')

    parser.add_argument(
        '--save-pcsa',
        type=str,
        default='',
        help='Path to save PCSA checkpoint after calibration')
```

- [ ] **Step 2: Verify the arguments parse correctly**

Run:
```bash
cd /home/unifying-ptq && python -c "
import sys; sys.argv = ['test', '--config', 'x', '--load-pcsa', 'foo.pt', '--save-pcsa', 'bar.pt']
# Patch to avoid missing file error
import ahcptq.solver.test_quant as tq
args = tq.parse_args()
print('load_pcsa:', args.load_pcsa)
print('save_pcsa:', args.save_pcsa)
"
```

Expected: prints `load_pcsa: foo.pt` and `save_pcsa: bar.pt`.

- [ ] **Step 3: Commit**

```bash
git add ahcptq/solver/test_quant.py
git commit -m "feat: add --load-pcsa and --save-pcsa CLI arguments"
```

---

### Task 2: Add Import for PromptAnchorBank and AnchorAwareFakeQuantize

**Files:**
- Modify: `ahcptq/solver/test_quant.py:1-52` (imports section)

- [ ] **Step 1: Add imports**

In `ahcptq/solver/test_quant.py`, add after line 41 (`from ahcptq.model.quant_model import QuantDecoderOurAttentionBlock`):

```python
from ahcptq.model.prompt_anchor import PromptAnchorBank
from ahcptq.quantization.fake_quant import AnchorAwareFakeQuantize
```

- [ ] **Step 2: Commit**

```bash
git add ahcptq/solver/test_quant.py
git commit -m "feat: import PromptAnchorBank and AnchorAwareFakeQuantize"
```

---

### Task 3: Implement `save_pcsa_checkpoint()`

**Files:**
- Modify: `ahcptq/solver/test_quant.py` (add new function before `calibrate()`, around line 700)

- [ ] **Step 1: Add `save_pcsa_checkpoint()` function**

Insert before the `calibrate()` function (before line 723):

```python
def save_pcsa_checkpoint(model, path):
    """Save all PromptAnchorBank and AnchorAwareFakeQuantize state keyed by module path."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, PromptAnchorBank):
            state[name] = {'type': 'PromptAnchorBank', 'state_dict': module.state_dict()}
        elif isinstance(module, AnchorAwareFakeQuantize):
            state[name] = {
                'type': 'AnchorAwareFakeQuantize',
                'scale': module.scale.data.clone().cpu(),
                'zero_point': module.zero_point.clone().cpu(),
                'num_anchors': module.num_anchors,
            }
    torch.save(state, path)
    logger.info('Saved PCSA checkpoint to {} ({} modules)'.format(path, len(state)))
```

- [ ] **Step 2: Commit**

```bash
git add ahcptq/solver/test_quant.py
git commit -m "feat: add save_pcsa_checkpoint() function"
```

---

### Task 4: Implement `load_pcsa_checkpoint()`

**Files:**
- Modify: `ahcptq/solver/test_quant.py` (add right after `save_pcsa_checkpoint()`)

- [ ] **Step 1: Add `load_pcsa_checkpoint()` function**

Insert right after `save_pcsa_checkpoint()`:

```python
def load_pcsa_checkpoint(model, path):
    """Load PromptAnchorBank and AnchorAwareFakeQuantize state from a PCSA checkpoint."""
    state = torch.load(path, map_location='cpu')
    loaded = 0
    for name, module in model.named_modules():
        if name not in state:
            continue
        entry = state[name]
        if isinstance(module, PromptAnchorBank) and entry['type'] == 'PromptAnchorBank':
            module.load_state_dict(entry['state_dict'])
            loaded += 1
        elif isinstance(module, AnchorAwareFakeQuantize) and entry['type'] == 'AnchorAwareFakeQuantize':
            saved_scale = entry['scale'].to(module.scale.device)
            saved_zp = entry['zero_point'].to(module.zero_point.device)
            module.scale = torch.nn.Parameter(saved_scale)
            module.zero_point = saved_zp
            loaded += 1
    logger.info('Loaded PCSA checkpoint from {} ({}/{} modules matched)'.format(path, loaded, len(state)))
```

- [ ] **Step 2: Commit**

```bash
git add ahcptq/solver/test_quant.py
git commit -m "feat: add load_pcsa_checkpoint() function"
```

---

### Task 5: Modify `calibrate()` to Accept `pcsa_loaded` Flag

**Files:**
- Modify: `ahcptq/solver/test_quant.py:723-760` (the `calibrate()` function)

- [ ] **Step 1: Add `pcsa_loaded` parameter and skip activation calibration when set**

Replace the current `calibrate()` function (lines 723-760) with:

```python
@torch.no_grad()
def calibrate(model, cali_data, BIG, pcsa_loaded=False):
    st = time.time()
    if not pcsa_loaded:
        if BIG:
            model.extract_feat(cali_data[0])
            bimodal_adjust(model, logger=logger)
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')

        for i in range(len(cali_data)):
            model.extract_feat(cali_data[i]) #HERE IS WHERE TO DO TIME EVALS
        # model.extract_feat(cali_data[0])
        rank, world_size = get_dist_info()
        observer = False
        if world_size!=1:
            for name, module in model.named_modules():
                if isinstance(module, ObserverBase):
                    observer=True
                    module.min_val.data /= world_size
                    module.max_val.data /= world_size
                    dist.all_reduce(module.min_val.data)
                    dist.all_reduce(module.max_val.data)
            if not observer:
                for name, module in model.predictor.model.named_modules():
                    if isinstance(module, ObserverBase):
                        observer=True
                        module.min_val.data /= world_size
                        module.max_val.data /= world_size
                        dist.all_reduce(module.min_val.data)
                        dist.all_reduce(module.max_val.data)
    else:
        logger.info('PCSA loaded from checkpoint — skipping activation calibration')

    enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
    model.extract_feat(cali_data[0])

    ed = time.time()
    rank, _ = get_dist_info()
    if rank == 0:
        logger.info('the calibration time is {}'.format(ed - st))
```

Key change: the activation calibration loop (BIG adjust, act_fake_quant observer pass, distributed observer sync) is wrapped in `if not pcsa_loaded`. Weight calibration always runs.

- [ ] **Step 2: Commit**

```bash
git add ahcptq/solver/test_quant.py
git commit -m "feat: add pcsa_loaded flag to calibrate() to skip activation calibration"
```

---

### Task 6: Modify `main()` Flow to Wire Everything Together

**Files:**
- Modify: `ahcptq/solver/test_quant.py:300-589` (the `main()` function)

- [ ] **Step 1: Add PCSA load/save logic to `main()`**

In `main()`, find the block starting at line 488 (`model = quantize_model(model, q_config, args)`). The section from line 488 to line 518 currently reads:

```python
            model = quantize_model(model, q_config, args)
            model.cuda()
            # # print(model.predictor.device)
            model.eval()
            fp_model = copy.deepcopy(model)
            fp_model = fp_model.to("cuda") 
            ...
            calibrate(model, cali_data, q_config.ptq4sam.BIG)
```

Replace the `calibrate(...)` call at line 510 with:

```python
            pcsa_loaded = False
            if args.load_pcsa:
                load_pcsa_checkpoint(model, args.load_pcsa)
                pcsa_loaded = True

            calibrate(model, cali_data, q_config.ptq4sam.BIG, pcsa_loaded=pcsa_loaded)

            if not pcsa_loaded:
                save_path = args.save_pcsa if args.save_pcsa else os.path.join(args.work_dir, 'pcsa_checkpoint.pt')
                save_pcsa_checkpoint(model, save_path)
```

This replaces the single line `calibrate(model, cali_data, q_config.ptq4sam.BIG)` at line 510. Everything else in `main()` stays the same.

- [ ] **Step 2: Commit**

```bash
git add ahcptq/solver/test_quant.py
git commit -m "feat: wire PCSA save/load into main() flow"
```

---

### Task 7: Update Data Root Paths in Config Files

**Files:**
- Modify: `projects/configs/yolox/yolo_l-sam-vit-b.py:43` (data_root)
- Modify: `projects/configs/hdetr/r50-hdetr_sam-vit-b.py:72` (data_root)

The YOLOX config has `data_root = '/data1/user/zhang/coco/'` and the HDETR config has `data_root = '/home/AHCPTQ/data/coco/'`. Both need to point to wherever COCO is actually placed on this machine.

- [ ] **Step 1: Update data_root in both config files**

In `projects/configs/yolox/yolo_l-sam-vit-b.py`, change line 43:
```python
data_root = './data/coco/'
```

In `projects/configs/hdetr/r50-hdetr_sam-vit-b.py`, change line 72:
```python
data_root = './data/coco/'
```

- [ ] **Step 2: Also fix the hardcoded sys.path in test_quant.py**

In `ahcptq/solver/test_quant.py`, line 33 has:
```python
sys.path.append("/home/zhang/Project/ahcptq")
```

Change to:
```python
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
```

- [ ] **Step 3: Commit**

```bash
git add projects/configs/yolox/yolo_l-sam-vit-b.py projects/configs/hdetr/r50-hdetr_sam-vit-b.py ahcptq/solver/test_quant.py
git commit -m "fix: update data_root and sys.path to be portable"
```

---

### Task 8: Set Up Environment

**Files:** None (shell commands only)

- [ ] **Step 1: Create conda environment and install dependencies**

```bash
conda create -n ahcptq python=3.7 -y
conda activate ahcptq
pip install torch torchvision
pip install -U openmim
mim install "mmcv-full<2.0.0"
pip install -r requirements.txt
```

- [ ] **Step 2: Compile CUDA operators**

```bash
cd /home/unifying-ptq/projects/instance_segment_anything/ops
python setup.py build install
cd /home/unifying-ptq
```

- [ ] **Step 3: Install mmdetection**

```bash
cd /home/unifying-ptq/mmdetection
python3 setup.py build develop
cd /home/unifying-ptq
```

- [ ] **Step 4: Verify environment**

```bash
python -c "import torch; import mmcv; import mmdet; print('torch:', torch.__version__); print('mmcv:', mmcv.__version__); print('mmdet:', mmdet.__version__)"
```

Expected: versions print without errors.

---

### Task 9: Download Models and Data

**Files:** None (shell commands only)

- [ ] **Step 1: Download COCO dataset**

Download from the Google Drive link in the README and extract:
```bash
mkdir -p /home/unifying-ptq/data/coco
# Download https://drive.google.com/file/d/1j92XnlzQZwPff2sP_nwU3LE9Npemkn7Q/view?usp=sharing
# Extract so structure is:
# data/coco/annotations/instances_train2017.json
# data/coco/annotations/instances_val2017.json  
# data/coco/train2017/
# data/coco/val2017/
```

Use `gdown` to download:
```bash
pip install gdown
gdown 1j92XnlzQZwPff2sP_nwU3LE9Npemkn7Q -O /home/unifying-ptq/data/coco.tar.gz
cd /home/unifying-ptq/data && tar xzf coco.tar.gz && cd /home/unifying-ptq
```

- [ ] **Step 2: Download model weights to `ckpt/`**

```bash
mkdir -p /home/unifying-ptq/ckpt
# SAM-B
gdown 1UlwYWVRsS4SbSPDXlR5_dVmcuqT8CzeI -O /home/unifying-ptq/ckpt/sam_vit_b_01ec64.pth
# YOLOX
gdown 1FQeKOaDJzwqXq4zz8-VHJbn6iKFT4HLt -O /home/unifying-ptq/ckpt/yolox_l.pth
# HDETR (R50)
gdown 1i7iMAicmoif8tUbuHEntVtmEsJrpXTZ4 -O /home/unifying-ptq/ckpt/r50_hdetr.pth
```

- [ ] **Step 3: Verify all files exist**

```bash
ls -la /home/unifying-ptq/ckpt/sam_vit_b_01ec64.pth /home/unifying-ptq/ckpt/yolox_l.pth /home/unifying-ptq/ckpt/r50_hdetr.pth
ls /home/unifying-ptq/data/coco/annotations/instances_val2017.json
ls /home/unifying-ptq/data/coco/val2017/ | head -3
```

Expected: all files exist, val2017 contains images.

---

### Task 10: Run Experiment 1 — YOLOX Calibration + Save PCSA

**Files:** None (experiment run)

- [ ] **Step 1: Run YOLOX SAM-B calibration with PCSA save**

```bash
cd /home/unifying-ptq
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/yolox/yolo_l-sam-vit-b.py \
  --q_config ./exp/config44.yaml \
  --quant-encoder \
  --save-pcsa result/tmp/pcsa_yolox_samb.pt \
  --work-dir result/yolox_samb_w4a4
```

Expected: calibration runs, PCSA checkpoint saved, evaluation prints segm mAP.

- [ ] **Step 2: Record YOLOX baseline result**

Save the output `segm` mAP value. This is the YOLOX-detector SAM-B W4A4 baseline.

- [ ] **Step 3: Verify PCSA checkpoint was saved**

```bash
ls -la result/tmp/pcsa_yolox_samb.pt
python -c "import torch; s = torch.load('result/tmp/pcsa_yolox_samb.pt'); print(len(s), 'modules saved'); print(list(s.keys())[:5])"
```

Expected: checkpoint file exists, contains multiple module entries.

---

### Task 11: Run Experiment 2 — H-DETR Baseline (Native PCSA)

**Files:** None (experiment run)

- [ ] **Step 1: Run H-DETR SAM-B with its own PCSA calibration**

```bash
cd /home/unifying-ptq
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/hdetr/r50-hdetr_sam-vit-b.py \
  --q_config ./exp/config44_hdetr.yaml \
  --quant-encoder \
  --work-dir result/hdetr_samb_w4a4_native
```

Expected: full calibration (including PCSA), evaluation prints segm mAP. This is the H-DETR native baseline.

- [ ] **Step 2: Record H-DETR native result**

Save the output `segm` mAP value. This is the comparison baseline.

---

### Task 12: Run Experiment 3 — H-DETR with Cross-Detector PCSA (YOLOX Anchors)

**Files:** None (experiment run)

- [ ] **Step 1: Run H-DETR SAM-B loading YOLOX PCSA checkpoint**

```bash
cd /home/unifying-ptq
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/hdetr/r50-hdetr_sam-vit-b.py \
  --q_config ./exp/config44_hdetr.yaml \
  --quant-encoder \
  --load-pcsa result/tmp/pcsa_yolox_samb.pt \
  --work-dir result/hdetr_samb_w4a4_cross_pcsa
```

Expected: PCSA loaded from YOLOX checkpoint, activation calibration skipped, weight calibration still runs, evaluation prints segm mAP.

- [ ] **Step 2: Record cross-detector PCSA result**

Save the output `segm` mAP value. Compare with Task 11 baseline.

---

### Task 13: Compile Results Table

**Files:** None (documentation)

- [ ] **Step 1: Create results summary**

Compile a table:

| Experiment | Detector | PCSA Source | segm mAP |
|-----------|----------|-------------|----------|
| YOLOX Baseline | YOLOX | YOLOX (native) | _from Task 10_ |
| H-DETR Baseline | H-DETR | H-DETR (native) | _from Task 11_ |
| Cross-Detector | H-DETR | YOLOX (loaded) | _from Task 12_ |

The key comparison is H-DETR Baseline vs Cross-Detector. If mAP is comparable, PCSA anchors generalize across detectors.

- [ ] **Step 2: Save results to a file**

```bash
# Save to result/cross_detector_pcsa_results.txt with the table above filled in
```
