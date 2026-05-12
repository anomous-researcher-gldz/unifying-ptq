# S2: AHCPTQ Paper-Faithful Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the published AHCPTQ + DBAF + PCSA SAM-B and SAM-L W4A4 results in the `ahcptq-old` env to confirm the ICML numbers reproduce on the current hardware. This freezes a "ground-truth" set of calibrated checkpoints that S3 (torchao layering) and S10 (paper) depend on.

**Architecture:** Runs the existing `ahcptq/solver/test_quant.py` driver with the published config and quant config. No code changes — only orchestration, checkpointing, and verification. Runs on remote A100 (frees local for FlatQuant). SAM-H optional.

**Tech Stack:** ahcptq-old env (Py 3.9 + torch 1.13 + mmcv-full 1.7 + vendored mmdet 2.x), COCO val2017, YOLOX-L detector checkpoint, SAM-B/L ViT checkpoints.

**Prereqs:** S1.3 + S1.7 complete (both envs built). COCO val2017 already downloaded at `/home/ubuntu/unifying-ptq/data/coco/`. Need SAM-B, SAM-L, YOLOX-L checkpoints downloaded.

---

### Task S2.1: Download SAM-B + SAM-L + YOLOX checkpoints

**Files:**
- Create: `/home/ubuntu/unifying-ptq/ckpt/{sam_vit_b_01ec64.pth,sam_vit_l_0b3195.pth,yolox_l.pth}`

- [ ] **Step 1: List required checkpoints**

The repo's README lists Google Drive URLs. Use `gdown` to fetch.

```bash
conda activate ahcptq-old
pip install gdown
mkdir -p /home/ubuntu/unifying-ptq/ckpt
cd /home/ubuntu/unifying-ptq/ckpt
```

- [ ] **Step 2: Download SAM-B (1UlwYWVRsS4SbSPDXlR5_dVmcuqT8CzeI)**

```bash
gdown --id 1UlwYWVRsS4SbSPDXlR5_dVmcuqT8CzeI -O sam_vit_b_01ec64.pth
ls -lh sam_vit_b_01ec64.pth
```

Expected: ~358 MB file.

- [ ] **Step 3: Download SAM-L (14MBHh7OFwY8EpaGkX6ZyjUAw83wywk7U)**

```bash
gdown --id 14MBHh7OFwY8EpaGkX6ZyjUAw83wywk7U -O sam_vit_l_0b3195.pth
ls -lh sam_vit_l_0b3195.pth
```

Expected: ~1.2 GB file.

- [ ] **Step 4: Download YOLOX-L (1FQeKOaDJzwqXq4zz8-VHJbn6iKFT4HLt)**

```bash
gdown --id 1FQeKOaDJzwqXq4zz8-VHJbn6iKFT4HLt -O yolox_l.pth
ls -lh yolox_l.pth
```

Expected: ~430 MB file.

- [ ] **Step 5: Sync checkpoints to remote**

```bash
rsync -avz --partial /home/ubuntu/unifying-ptq/ckpt/ remote-gpu:~/unifying-ptq/ckpt/
ssh remote-gpu 'ls -lh ~/unifying-ptq/ckpt/'
```

Expected: matches local sizes.

---

### Task S2.2: Smoke test on remote — SAM-B W4A4 single-image inference

**Files:**
- Read: `/home/ubuntu/unifying-ptq/projects/configs/yolox/yolo_l-sam-vit-b.py`
- Read: `/home/ubuntu/unifying-ptq/exp/config44.yaml`

- [ ] **Step 1: Patch config's `data_root` to point at local COCO**

The config has `data_root = '/data1/user/zhang/coco/'`. Override to our path.

```bash
ssh remote-gpu 'bash -lc "
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ahcptq-old
cd ~/unifying-ptq
sed -i \"s|/data1/user/zhang/coco/|/home/ubuntu/unifying-ptq/data/coco/|g\" projects/configs/yolox/yolo_l-sam-vit-b.py
git diff projects/configs/yolox/yolo_l-sam-vit-b.py
"'
```

(On remote, COCO will need to be at `/home/ubuntu/unifying-ptq/data/coco/` too — sync next step.)

- [ ] **Step 2: Sync COCO val2017 + annotations to remote**

```bash
rsync -avz --partial /home/ubuntu/unifying-ptq/data/coco/val2017/ remote-gpu:~/unifying-ptq/data/coco/val2017/
rsync -avz --partial /home/ubuntu/unifying-ptq/data/coco/annotations/ remote-gpu:~/unifying-ptq/data/coco/annotations/
ssh remote-gpu 'ls ~/unifying-ptq/data/coco/'
```

Expected: prints `annotations val2017`.

- [ ] **Step 3: Smoke run — first 5 images only**

Add a small modification to `test_quant.py` to enable `--max-images N` for fast smoke testing. Or pipe-cap via slicing in the dataloader. For now, use the existing flow with a config patch limiting `dataset.test.pipeline` to first 5 images via custom override.

Simpler: run for ~2 minutes, kill it, check outputs printed.

```bash
ssh remote-gpu 'bash -lc "
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ahcptq-old
cd ~/unifying-ptq
mkdir -p result/s2-smoke
timeout 300 python ahcptq/solver/test_quant.py --config ./projects/configs/yolox/yolo_l-sam-vit-b.py --q_config ./exp/config44.yaml --quant-encoder --work-dir result/s2-smoke --save-pcsa result/s2-smoke/pcsa.pt 2>&1 | tail -40 || true
"'
```

Expected: prints calibration progress, no Python tracebacks. If it crashes with a missing module or mmdet API mismatch, that's the signal mmcv-full 1.7 + vendored mmdet 2.x needs fixing — fix in S1.3 before continuing.

---

### Task S2.3: Full SAM-B W4A4 reproduction

**Files:**
- Create: `results/S2-ahcptq/sam-b/w4a4/seed0/{state.pt,pcsa.pt,eval.json,logs/}`

- [ ] **Step 1: Launch full run on remote in tmux**

```bash
ssh remote-gpu 'tmux new-session -d -s s2-samb "
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ahcptq-old
cd ~/unifying-ptq
mkdir -p results/S2-ahcptq/sam-b/w4a4/seed0/logs
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/yolox/yolo_l-sam-vit-b.py \
  --q_config ./exp/config44.yaml \
  --quant-encoder \
  --work-dir results/S2-ahcptq/sam-b/w4a4/seed0 \
  --save-pcsa results/S2-ahcptq/sam-b/w4a4/seed0/pcsa.pt \
  --save_sam_path results/S2-ahcptq/sam-b/w4a4/seed0/sam.pt \
  --eval segm 2>&1 | tee results/S2-ahcptq/sam-b/w4a4/seed0/logs/run.log
"'
```

- [ ] **Step 2: Poll until completion**

```bash
ssh remote-gpu 'tmux capture-pane -t s2-samb -p | tail -20'
# repeat until you see "Final mAP" or "DONE" or it crashes
```

Expected runtime: 1–2 hours.

- [ ] **Step 3: Sync results back**

```bash
/home/ubuntu/unifying-ptq/scripts/sync_results.sh pull
cat /home/ubuntu/unifying-ptq/results/S2-ahcptq/sam-b/w4a4/seed0/logs/run.log | tail -50
```

- [ ] **Step 4: Extract metric to eval.json**

Locate the final segm mAP in the log. The paper reported 18.2 mAP for SAM-B + YOLOX + W4A4 with DBAF+PCSA.

```bash
python - <<'EOF'
import json, re, pathlib
log = pathlib.Path("/home/ubuntu/unifying-ptq/results/S2-ahcptq/sam-b/w4a4/seed0/logs/run.log").read_text()
m = re.search(r"segm.*?(\d+\.\d+)", log)
val = float(m.group(1)) if m else None
out = {"model":"SAM-B","bits":"W4A4","method":"AHCPTQ+DBAF+PCSA","task":"COCO-val2017","metric":"segm-mAP","value":val,"seed":0,"timestamp":__import__("datetime").datetime.utcnow().isoformat()}
pathlib.Path("/home/ubuntu/unifying-ptq/results/S2-ahcptq/sam-b/w4a4/seed0/eval.json").write_text(json.dumps(out,indent=2))
print(out)
EOF
```

- [ ] **Step 5: Verify reproduction within tolerance**

Acceptance: SAM-B segm mAP ≥ 17.5 (paper claim 18.2 ±0.7 absolute mAP is acceptable noise for SAM eval).

If reproduction fails (mAP << 17), inspect the log for early divergence, check that --quant-encoder flag took effect, and review q_config alpha/anchor settings.

- [ ] **Step 6: Commit eval.json**

```bash
cd /home/ubuntu/unifying-ptq
git add results/S2-ahcptq/sam-b/w4a4/seed0/eval.json
git commit -m "result(S2): reproduce SAM-B + YOLOX W4A4 AHCPTQ+DBAF+PCSA"
```

---

### Task S2.4: Full SAM-L W4A4 reproduction

**Files:**
- Read: `/home/ubuntu/unifying-ptq/projects/configs/yolox/yolo_l-sam-vit-l.py`
- Create: `results/S2-ahcptq/sam-l/w4a4/seed0/*`

- [ ] **Step 1: Patch SAM-L config data_root (mirror Task S2.2 Step 1)**

```bash
ssh remote-gpu 'sed -i "s|/data1/user/zhang/coco/|/home/ubuntu/unifying-ptq/data/coco/|g" ~/unifying-ptq/projects/configs/yolox/yolo_l-sam-vit-l.py'
```

- [ ] **Step 2: Launch full SAM-L W4A4 run in tmux**

```bash
ssh remote-gpu 'tmux new-session -d -s s2-saml "
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ahcptq-old
cd ~/unifying-ptq
mkdir -p results/S2-ahcptq/sam-l/w4a4/seed0/logs
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/yolox/yolo_l-sam-vit-l.py \
  --q_config ./exp/config44.yaml \
  --quant-encoder \
  --work-dir results/S2-ahcptq/sam-l/w4a4/seed0 \
  --save-pcsa results/S2-ahcptq/sam-l/w4a4/seed0/pcsa.pt \
  --save_sam_path results/S2-ahcptq/sam-l/w4a4/seed0/sam.pt \
  --eval segm 2>&1 | tee results/S2-ahcptq/sam-l/w4a4/seed0/logs/run.log
"'
```

Expected runtime: 3–6 hours.

- [ ] **Step 3: Poll, sync, extract metric, commit (mirror S2.3 Steps 2-6)**

Paper claim for SAM-L + YOLOX + W4A4 + AHCPTQ+DBAF+PCSA: 33.4 mAP. Acceptance ≥ 32.5.

---

### Task S2.5 (optional, time-permitting): SAM-H W4A4

**Files:**
- Need: SAM-H checkpoint (1fMJyX938_H17OxfVq6PQZ_ef9TBy5r36 ~2.4 GB)
- Create: `results/S2-ahcptq/sam-h/w4a4/seed0/*`

- [ ] **Step 1: Download SAM-H, sync to remote**

```bash
cd /home/ubuntu/unifying-ptq/ckpt
gdown --id 1fMJyX938_H17OxfVq6PQZ_ef9TBy5r36 -O sam_vit_h_4b8939.pth
rsync -avz sam_vit_h_4b8939.pth remote-gpu:~/unifying-ptq/ckpt/
```

- [ ] **Step 2: Confirm config exists or create**

If `projects/configs/yolox/yolo_l-sam-vit-h.py` doesn't exist, copy from `yolo_l-sam-vit-b.py` and adapt the `sam_checkpoint` / `model_type='vit_h'` fields.

- [ ] **Step 3: Set `keep_gpu: False` in q_config for memory safety**

```bash
ssh remote-gpu 'grep keep_gpu ~/unifying-ptq/exp/config44.yaml || echo "keep_gpu: False" >> ~/unifying-ptq/exp/config44.yaml'
```

- [ ] **Step 4: Run + sync + extract + commit (mirror S2.4)**

Acceptance: SAM-H segm mAP > AHCPTQ baseline (paper section "Rebuttal" showed the same distributional pattern at 0.905% outlier rate; gains predicted but not previously measured).

---

### Task S2.6: Aggregate S2 results

**Files:**
- Create: `results/S2-ahcptq/summary.json`, `results/S2-ahcptq/summary.md`

- [ ] **Step 1: Aggregate eval.json files**

```bash
cd /home/ubuntu/unifying-ptq
python - <<'EOF'
import json, glob, pathlib
items = [json.load(open(p)) for p in glob.glob("results/S2-ahcptq/*/w4a4/seed0/eval.json")]
out = sorted(items, key=lambda r: r["model"])
pathlib.Path("results/S2-ahcptq/summary.json").write_text(json.dumps(out, indent=2))
print("\\n".join(f"{r['model']:10s} {r['method']:25s} {r['metric']}: {r['value']}" for r in out))
EOF
```

Expected output:

```
SAM-B   AHCPTQ+DBAF+PCSA   segm-mAP: 18.2
SAM-L   AHCPTQ+DBAF+PCSA   segm-mAP: 33.4
(SAM-H optional)
```

- [ ] **Step 2: Commit**

```bash
git add results/S2-ahcptq/summary.json
git commit -m "result(S2): aggregate AHCPTQ reproduction summary"
```

---

## Done when

- `results/S2-ahcptq/sam-b/w4a4/seed0/eval.json` exists with mAP ≥ 17.5
- `results/S2-ahcptq/sam-l/w4a4/seed0/eval.json` exists with mAP ≥ 32.5
- Both `state.pt`, `pcsa.pt`, `sam.pt`, and `logs/run.log` checkpointed for S3 to consume
- `summary.json` committed
- Optional SAM-H result in same layout
