# Towards a Unified Distribution-Centric Post-Training Quantization

## Abstract

Post-training quantization is widely used for efficient deployment of models, yet existing methods remain largely architecture or task-specific. In contrast to this model/task-specific view of quantization, we find that across tasks and model families, quantization errors repeatedly arise from similar distributional properties such as dense cores and sparse outliers, and input-conditioned distribution shifts. Based on this, we propose viewing quantization through a unified distribution-centered rather than model or task-centered perspective and introduce a novel Dual-Band Affine Folding (DBAF) for outlier suppression and Prompt-Conditioned Scale Anchoring (PCSA) for adaptive scaling to prompt distribution shifts. Applied to the Segment Anything Model, Large Language Models and image super resolution under W4A4 quantization, our method improves performance over prior baselines, especially for SAM as distributions with outliers are largely unaddressed in SAM quantization, validating distribution-centric techniques as effective strategies for low-bit quantization.

## Repository Structure

```
unifying-ptq/
├── ahcptq/          # AHCPTQ: SAM quantization with DBAF+PCSA
├── FlatQuant/       # FlatQuant: LLM quantization with DBAF+PCSA
├── CompSRT/         # CompSRT: Image super-resolution quantization with DBAF
├── ckpt/            # Model checkpoints (SAM)
├── exp/             # Quantization configs (SAM)
├── mmdetection/     # MMDetection (SAM detector dependency)
└── projects/        # SAM project configs
```

---

## 1. AHCPTQ — Segment Anything Model (SAM)

### 1.1 Environment Setup

```bash
# Create environment
conda create -n ahcptq python=3.7 -y
conda activate ahcptq
pip install torch torchvision

# Install MMCV
pip install -U openmim
mim install "mmcv-full<2.0.0"

# Install other requirements
pip install -r requirements.txt

# Compile CUDA operators
cd projects/instance_segment_anything/ops
python setup.py build install
cd ../../..

# Install mmdet
cd mmdetection/
python3 setup.py build develop
cd ..
```

### 1.2 Prepare Dataset

Download the [COCO](https://drive.google.com/file/d/1j92XnlzQZwPff2sP_nwU3LE9Npemkn7Q/view?usp=sharing) dataset:

```
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

### 1.3 Download Model Weights

Save to `ckpt/`:

| Model       | Download |
|-------------|----------|
| SAM-B       | [Link](https://drive.google.com/file/d/1UlwYWVRsS4SbSPDXlR5_dVmcuqT8CzeI/view?usp=sharing) |
| SAM-L       | [Link](https://drive.google.com/file/d/14MBHh7OFwY8EpaGkX6ZyjUAw83wywk7U/view?usp=sharing) |
| SAM-H       | [Link](https://drive.google.com/file/d/1fMJyX938_H17OxfVq6PQZ_ef9TBy5r36/view?usp=sharing) |
| Faster-RCNN | [Link](https://drive.google.com/file/d/1RKTLk07E4apoRzwoeQbnaY8ZxEX1SlbG/view?usp=sharing) |
| YOLOX       | [Link](https://drive.google.com/file/d/1FQeKOaDJzwqXq4zz8-VHJbn6iKFT4HLt/view?usp=sharing) |
| HDETR       | [Link](https://drive.google.com/file/d/1i7iMAicmoif8tUbuHEntVtmEsJrpXTZ4/view?usp=sharing) |
| DINO        | [Link](https://drive.google.com/file/d/1DDHkZcVI9TwmN9vqEYXFBjRZVsBK4yLO/view?usp=sharing) |

### 1.4 Run Experiments

```bash
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/<DETECTOR>/<MODEL.py> \
  --q_config ./exp/<QCONFIG>.yaml \
  --quant-encoder
```

Example (W4A4, SAM-B, YOLO detector):

```bash
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/yolox/yolo_l-sam-vit-b.py \
  --q_config ./exp/config44.yaml \
  --quant-encoder
```

**Note:** For HDETR/DINO, set `keep_gpu: False` in the YAML config if memory is insufficient. See the original AHCPTQ README for details.

---

## 2. FlatQuant — Large Language Models (LLM)

### 2.1 Environment Setup

```bash
cd FlatQuant

# Create environment
python -m venv venv_flatquant
source venv_flatquant/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Install FlatQuant
pip install -e .

# Compile CUDA extensions (for real INT4 deployment)
cd third-party/fast-hadamard-transform
pip install -e .
cd ../..
python setup.py install
```

### 2.2 Download Models

Download models from HuggingFace to `modelzoo/`:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B \
  --local-dir ./modelzoo/meta-llama/Meta-Llama-3-8B
```

Supported models: LLaMA-2 (7B/13B/70B), LLaMA-3 (8B/70B), LLaMA-3.1 (8B/70B), Qwen-2.5 (7B/32B).

### 2.3 Download Datasets

```bash
# Calibration + PPL eval
huggingface-cli download wikitext --repo-type dataset --local-dir ./datasets/wikitext
huggingface-cli download allenai/c4 en/c4-train.00000-of-01024.json.gz en/c4-validation.00000-of-00008.json.gz --repo-type dataset --local-dir ./datasets/allenai/c4
```

### 2.4 Run Experiments

W4A4KV4 calibration + evaluation:

```bash
python ./main.py \
  --model ./modelzoo/meta-llama/Meta-Llama-3-8B \
  --w_bits 4 --a_bits 4 \
  --k_bits 4 --k_asym --k_groupsize 128 \
  --v_bits 4 --v_asym --v_groupsize 128 \
  --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
  --lwc --lac --cali_trans --add_diag \
  --output_dir ./outputs --save_matrix \
  --lm_eval --lm_eval_batch_size 16 \
  --tasks arc_challenge arc_easy hellaswag lambada_openai piqa winogrande
```

Reload saved matrices for evaluation only:

```bash
python ./main.py \
  --model ./modelzoo/meta-llama/Meta-Llama-3-8B \
  --w_bits 4 --a_bits 4 \
  --k_bits 4 --k_asym --k_groupsize 128 \
  --v_bits 4 --v_asym --v_groupsize 128 \
  --lwc --lac --add_diag \
  --reload_matrix --matrix_path ./outputs/Meta-Llama-3-8B/w4a4/exp \
  --lm_eval --lm_eval_batch_size 16 \
  --tasks arc_challenge arc_easy hellaswag lambada_openai piqa winogrande
```

Use `--disable_dbaf` to run ablation experiments without DBAF.

---

## 3. CompSRT — Image Super-Resolution

### 3.1 Environment Setup

```bash
cd CompSRT

# Create environment
conda create -n srtquant python=3.9 -y
conda activate srtquant

pip install six
pip install --no-cache-dir \
  torch==2.0.1+cu117 \
  torchvision==0.15.2+cu117 \
  torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt
pip install -e . --no-build-isolation -v

pip install -v --no-build-isolation causal_conv1d==1.0.0
pip install -v --no-build-isolation mamba_ssm==1.0.1
```

### 3.2 Datasets

Download and place in `CompSRT/datasets/`:

* [Training set (DF2K)](https://drive.google.com/file/d/1TubDkirxl4qAWelfOnpwaSKoj3KLAIG4/view?usp=share_link)
* [Testing set](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing)
* [Calibration data](https://drive.google.com/file/d/1UxgyQWrToZHxsMrPursuMBtyCcNjFwUA/view?usp=drive_link)
* [Pretrained models](https://drive.google.com/file/d/12g_64n-hhJJbvd6cpU7VakxruGRpzhP-/view?usp=drive_link)

### 3.3 Run Experiments

Training (with optional pruning):

```bash
# Example: 4-bit x4 SR
python basicsr/train.py -opt options/train/train_srtquant_x4.yml --pruning 0.4 \
  --force_yml bit=4 name=train_srtquant_x4_bit4
```

Testing:

```bash
python basicsr/test.py -opt options/test/test_srtquant_x2.yml --pruning 0.4 \
  --force_yml bit=4 name=test_srtquant_x2_bit4 \
  path:pretrain_network_Q=experiments/train_srtquant_x2_bit4/models/<best_model.pth>
```

See `CompSRT/README.md` for Docker/Singularity setup and statistical analysis scripts.

---

## Acknowledgments

This work builds upon [PTQ4SAM](https://github.com/chengtao-lv/PTQ4SAM), [FlatQuant](https://github.com/ruikangliu/FlatQuant), [CompSRT](https://github.com/anonymous-researcher-99/CompSRT), and [BasicSR](https://github.com/XPixelGroup/BasicSR).

