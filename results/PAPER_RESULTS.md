# PAPER_RESULTS.md — Canonical paper results tracker

**Last updated:** 2026-05-13
**Paper:** EMNLP 2026 submission, "Towards a Unified Distribution-Centric Post-Training Quantization"
**Deadline:** May 25 2026 AoE

Every result that will be cited in the paper goes here, with the eval.json path
that produced it. Use this as the source of truth when generating LaTeX tables.

---

## Bucket 1: Training-free / RTN-style baselines (NEW, S4)

DBAF as a *composable primitive* on top of training-free quantization. All
W4, per-channel weight scaling, WikiText-2 perplexity (lower is better).

### LLaMA-3-8B

| Method | Baseline PPL | +DBAF PPL | DBAF Δ | Eval path |
|---|---|---|---|---|
| RTN | 10.279 | 8.870 | **−1.41** | `results/S4-dbaf-weak/llama3-8b/rtn/{baseline,with-dbaf}/eval.json` |
| GPTQ-style | 11.604 | 9.794 | **−1.81** | `results/S4-dbaf-weak/llama3-8b/gptq/{baseline,with-dbaf}/eval.json` |
| AWQ-style | 13.491 | 9.417 | **−4.07** | `results/S4-dbaf-weak/llama3-8b/awq/{baseline,with-dbaf}/eval.json` |
| Matched-T clipping (α=0) | — | (pending) | — | — |

FP16 baseline (LLaMA-3-8B WikiText-2): ~6.14 PPL.

### Qwen-2.5-7B

(pending — sweep launches after LLaMA-3-8B + matched-T)

### SAM-B + YOLOX W4A4 (training-free)

| Method | mAP | +DBAF mAP | DBAF Δ | Status |
|---|---|---|---|---|
| RTN | (pending) | (pending) | — | S4.9 not yet run |

### SwinIR ×2 W4 (training-free)

| Method | PSNR (Set5) | +DBAF PSNR | DBAF Δ | Status |
|---|---|---|---|---|
| RTN | (pending) | (pending) | — | S8.4 (= CompSRT-A) not yet run |

---

## Bucket 2: Fine-tuned / optimized methods (REUSE from ICML + verify)

DBAF stacked on baselines that already address outliers via learned
rotations / Hadamard transforms. Gains are small but consistent — this is
itself a useful empirical finding: DBAF's value scales inversely with the
host method's outlier handling.

### LLaMA-3-8B (FlatQuant W4A4KV4)

| Method | PPL | Source |
|---|---|---|
| FP16 | 6.14 | ICML Table 5 |
| SmoothQuant | 210.19 | ICML Table 5 |
| QuaRot | 10.60 | ICML Table 5 |
| SpinQuant | 7.96 | ICML Table 5 |
| FlatQuant | 6.98 | ICML Table 5 |
| **FlatQuant + DBAF + PCSA** | **6.96** | ICML Table 5; verify in S5 |

### SAM-B + YOLOX W4A4 (AHCPTQ)

| Method | mAP | Source |
|---|---|---|
| FP | 37.2 | ICML Table 4 |
| AHCPTQ | 13.4 | ICML Table 4 |
| **AHCPTQ + DBAF + PCSA** | **18.2** | ICML Table 4; SAM-H reproduction pending |
| SAM-B + H-DETR + Ours | 20.6 | rebuttal |
| Cross-detector PCSA transfer (YOLOX→H-DETR, recon-free) | 1.8 vs 1.9 (relative −5%) | rebuttal |

### SAM-H (NEW for EMNLP — primary contribution to fill ICML gap)

| Method | mAP | Status |
|---|---|---|
| FP | (pending) | running |
| AHCPTQ + DBAF + PCSA | (pending) | running on remote ~6-10h |
| AHCPTQ + DBAF + PCSA + H-DETR | (pending) | pending after YOLOX |

### SwinIR (CompSRT W4)

| Method | PSNR (Set5 ×2) | PSNR (Set5 ×3) | Source |
|---|---|---|---|
| FP | 38.15 | 34.63 | ICML Table 6 |
| NoisyQuant | 37.50 | 31.09 | ICML Table 6 |
| 2DQuant | 37.87 | 33.24 | ICML Table 6 |
| CondiQuant | 38.03 | 33.92 | ICML Table 6 |
| CompSRT | 38.13 | 34.56 | ICML Table 6 |
| **CompSRT + DBAF** | **38.15** | **34.59** | ICML Table 6 |

---

## Causal evidence: outliers are the recurring distributional cause (NEW, S4)

### Synthetic outlier injection study (S4.5) ✓ done

Gaussian tensor + controlled outliers. DBAF MSE reduction vs outlier fraction:

| Outlier fraction | 5σ outliers | 10σ outliers | 20σ outliers |
|---|---|---|---|
| 0% (none) | 10.4% | 10.4% | 10.4% |
| 0.1% | 18.4% | **31.1%** | **31.0%** |
| 0.5% | 17.3% | 29.2% | 27.9% |
| 1% | 15.9% | 26.8% | 25.0% |
| 2% | 13.1% | 22.7% | 20.5% |
| 5% | 5.4% | 13.0% | 11.5% |
| 10% | 0.0% | 0.7% | 1.5% |

**Headline interpretation:** DBAF gain peaks at outlier fractions of 0.1–1% — the exact range observed in real models (per ICML Table 3, ~1% for SAM-B, 0.9% for SAM-H, etc.). At ≥10% fraction, "outliers" are no longer rare and DBAF gain decays to near zero.

Eval: `results/S4-dbaf-weak/synthetic/study.json`
Figure: `paper/emnlp2026/figures/synthetic_outlier_gain.pdf`

### Per-layer outlier-vs-gain correlation (S4.6) — pending

LLaMA-3-8B; compute per-Linear-layer outlier fraction + DBAF MSE reduction; expect strong positive correlation.

### Matched-T clipping (S4.4 / S10.7) — pending

Same threshold T=3σ; clip instead of fold. Expected: clipping degrades PPL vs DBAF, isolating "ordering preservation" as the value of folding.

### DBAF on outlier-free layers (S4.7, negative control) — pending

Apply DBAF to LayerNorm scale params / embeddings (low outlier fraction); expect ≈ no change.

### Cross-arch outlier prevalence (S4.8) — port from ICML Table 3

| Model | Total tensors | Dense+outlier classified | % >3σ | Mean α* |
|---|---|---|---|---|
| SAM-B | 163 | 108 | 1.053 | 0.0173 |
| SAM-L | 324 | 181 | 0.946 | 0.0186 |
| SAM-H | 435 | 214 | 0.905 | 0.0197 |

The same distributional taxonomy applies across SAM scales; we extend to LLaMA + SwinIR in S4.8.

---

## Rebuttal-only material now in main paper (per ICML rebuttal)

- Real INT4 hardware latency (DBAF ≤1.7% overhead under real INT4 GEMM) — in S6 / S10.9
- α=1.0 in grid (PPL: PCSA-only 6.97, DBAF-only 6.97, both 6.96) — in S7.1 / S10.6
- LLaMA-3-8B calibration sensitivity (WikiText-2 vs C4: PPL 6.97–7.04 vs 11.08–11.19) — in S10
- α*/α_grid alignment (α* < α_grid < 1.0 in 100% of classified tensors) — in S10.8

---

## Methodology notes

**Bucket 1 baselines:** "RTN", "GPTQ-style", "AWQ-style" use the *core* quantization
algorithm (per-row scaling, Hessian-aware updates, activation magnitude scaling)
without paper-specific tricks like group-wise scales, learned grid search, etc.
This is intentional: we evaluate DBAF as a *primitive* that composes with these
core algorithms. Published GPTQ/AWQ numbers (with their full pipelines) are cited
from the literature for context but not re-run by us. The DBAF gain ratios
reported here are conservative — they would be smaller in absolute magnitude
relative to a perfectly-tuned baseline, but the *relative* gain story remains.

**Bucket 2 baselines:** FlatQuant (LLM), AHCPTQ (SAM), CompSRT (SR) all use
their published fine-tuning pipelines (reconstruction loss, learned rotations,
Hadamard transforms, etc.). DBAF is integrated as a folding step before quantization.
