# TesseraQ EMNLP-pivot configs

TesseraQ tree is gitignored; tracking the new Llama-3-8B configs here.

## Workflow

TesseraQ requires AWQ first (saves scales + clips), then TesseraQ block-recon
loads them. Two configs:

1. `awq_llama3_8b_w4a4.yml` — AWQ W4A4 on Llama-3-8B, saves scales/clips to
   `../cache/activations/L3_8b/awq_w4a4`.
2. `tesseraq_w4a4_L3_8b.yml` — TesseraQ W4A4 block-recon on Llama-3-8B,
   loads the AWQ scales/clips, runs PAR (Progressive Adaptive Rounding) +
   scale tuning, then evaluates wikitext2 + c4 PPL.

## Quick run

```bash
# Copy configs into TesseraQ tree:
cp patches/tesseraq_emnlp_configs/*.yml TesseraQ/configs/quantization/wa_quant/

# AWQ first (saves scales/clips):
cd TesseraQ
torchrun --nproc_per_node 1 llmc/__main__.py \
  --config configs/quantization/wa_quant/awq_llama3_8b_w4a4.yml \
  --task_id 12345

# Then TesseraQ:
torchrun --nproc_per_node 1 llmc/__main__.py \
  --config configs/quantization/wa_quant/tesseraq_w4a4_L3_8b.yml \
  --task_id 12346
```

## Expected baseline (vanilla TesseraQ, no DBAF/PCSA)

Per arxiv 2410.19103 Table 3: Llama-3.1-8B W4A4 wikitext2 PPL = 25.73 (AWQ-init).
Llama-3-8B is the same architecture (Llama-3.1 only differs in training data and
context length), so we expect a similar number.

## DBAF/PCSA integration plan (TODO)

Following the `omniquant_dbaf_pcsa_patch.py` / `twodquant_dbaf_pcsa_patch.py`
pattern. Injection sites in TesseraQ's `llmc/compression/quantization/`:
- `BaseBlockwiseQuantization.fake_quant_weight_dynamic` (weights, line 37)
- `BaseBlockwiseQuantization.fake_quant_act_dynamic` (activations, line 43)

These are called from inside the TesseraQ block-recon loop in `tesseraq.py`.
Monkey-patching both before importing `llmc.__main__` (mirrors our pattern in
the existing 2DQuant + OmniQuant patches) gives us DBAF folding on every
fake-quant op without touching the upstream code.

PCSA-tf in the trained-host setting needs anchor scales fit from a calibration
descriptor stream (similar to how omniquant_dbaf_pcsa_patch fits anchors during
calibration). The descriptor for Llama-3 can be the hidden-state mean per layer.

## Env caveat

Installing TesseraQ's requirements.txt into base conda accidentally upgraded
transformers, accelerate, peft, etc. Strongly recommend creating a dedicated
`tesseraq` conda env before running:

```bash
conda create -n tesseraq python=3.10 -y
conda activate tesseraq
pip install -r TesseraQ/requirements.txt
```
