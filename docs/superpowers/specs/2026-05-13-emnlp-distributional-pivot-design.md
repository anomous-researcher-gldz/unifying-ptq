# EMNLP 2026 Submission — Distributional-Pivot Reframe Design

**Status:** approved (2026-05-13)
**Author:** anomous-researcher-gldz
**Deadline:** May 25 2026 AoE (12 days)

---

## 1. Problem statement

The ICML 2026 submission claimed DBAF + PCSA stack on top of FlatQuant /
AHCPTQ / CompSRT to give better numbers. Empirically reproducible at submission
time (LLaMA-3-8B W4A4 KV16: 6.96 vs 6.98 pure FlatQuant; SAM-B AHCPTQ + DBAF:
+5 mAP; CompSRT + DBAF ≈ FP at ×2). The EMNLP rewrite has surfaced two
problems with this framing:

1. **Gain disappears at W4A4 KV4.** New experiments today: pure FlatQuant W4A4
   KV4 = 6.964 PPL, FlatQuant + DBAF + PCSA W4A4 KV4 = 6.966 PPL. The 0.02 PPL
   improvement seen at KV16 does not replicate at KV4. The DBAF + PCSA
   contribution on rotation-based hosts is in the noise once KV is also
   quantized.

2. **DBAF is not universally helpful at the per-layer task level.** SwinIR-light
   ×3 regresses by −0.32 dB with DBAF (training-free) at α=0.95 and at every
   gate strictness; the only "winning" gate is one that disables DBAF
   completely. Per-layer ablation shows damage compounds nonlinearly through
   depth (each layer ±0.04 dB; total −0.32 dB on ×3).

The current paper claim "DBAF + PCSA improve quantization" is too strong for
the data and would be easy to refute by a reviewer running a single ablation.

## 2. Reframed contribution

**Headline:** Quantization is distributionally motivated, and the same
distribution types recur across architectures. We propose composable
distribution-guided primitives (DBAF + training-free PCSA) that improve
quantization on hosts with explicit headroom, demonstrating the same primitives
generalize across LLM, SAM, and SR.

### 2.1 Three-level claim hierarchy

| Level | Claim | Evidence required |
|---|---|---|
| **Top (conceptual)** | PTQ error is dominated by a small number of recurring distribution types (sparse-outlier, post-ReLU asymmetric, bimodal post-softmax, heavy-tailed unimodal, ...). Prior work addresses these per-architecture. | Distribution taxonomy + literature fragmentation table |
| **Middle (methodological)** | DBAF + PCSA are distribution-guided primitives that *compose* on any host method without architecture-specific machinery. | Same primitive code applied across LLM / SAM / SR (training-free + trained host composition) |
| **Bottom (empirical)** | On hosts with headroom (OmniQuant for LLM, AHCPTQ for SAM, 2DQuant for SR), our primitives add consistent moderate gains across all three architectures. On rotation-based hosts (FlatQuant / CompSRT) primitives add ~0 — evidence the same distributions are being absorbed implicitly there. | Headline composability table + saturation evidence |

### 2.2 What we explicitly drop

- **"Beats SOTA" claim.** We do not beat SpinQuant on LLM, do not beat
  FlatQuant on LLaMA-3-8B W4A4 KV4. We're explicit about this.
- **CompSRT as primary host** — moved to saturation evidence.
- **FlatQuant as primary host** — moved to saturation evidence.

### 2.3 What we keep + sharpen

- Fragmentation table (already in `intro.tex:74-88`) — extend with a
  cross-arch citation column showing per-distribution prior work.
- Six-panel distributional figure in `03-method.tex` — keep, possibly add
  one panel.
- Mechanism trio (synthetic, per-layer, matched-T) — keep, supplement
  with the new dense-vs-sparse synthetic + distribution taxonomy.

## 3. Architecture & components

### 3.1 Primitives (already in codebase)

- **DBAF (Dual-Band Affine Fold)**: per-tensor outlier compression governed by
  the gate `is_like_normal_plus_3sigma_outliers` (`ahcptq/quantization/fake_quant.py`).
  Closed-form α*; fold/unfold preserves order on outlier band.
- **PCSA (gradient-trained)**: per-prompt anchor routing on activations with
  learnable per-anchor scales. Currently plumbed in FlatQuant / AHCPTQ /
  CompSRT training paths.

### 3.2 NEW primitive variants

- **PCSA-tf (training-free)**: k-means on prompt-level descriptors from
  calibration set; per-anchor max-abs activation scales. No gradients.
  Calibration cost: seconds. Composes on RTN / GPTQ / AWQ.
- **KV-PCSA-tf (training-free)**: k-means on prompt-level descriptors;
  per-anchor max-abs K and V cache scales. No gradients. For LLMs only.

### 3.3 Host methods (composition targets)

**With training (gradient-based calibration):**
- **OmniQuant** (LLM) — to be added under `/home/ubuntu/unifying-ptq/OmniQuant/`
- **AHCPTQ** (SAM) — already in codebase
- **2DQuant** (SR) — to be added under `/home/ubuntu/unifying-ptq/2DQuant/`

**Without training (calibration-stat-only):**
- **RTN** (per-channel) — already in `flatquant/baselines/rtn.py`
- **GPTQ-simplified** — already in `flatquant/baselines/`
- **AWQ-simplified** — already in `flatquant/baselines/`

**Saturation-evidence hosts (carryover, marginal-gain reporting):**
- **FlatQuant** (LLM) — C1 no-gate run currently in-flight
- **CompSRT** (SR) — ICML numbers carryover

## 4. Experiment matrix

### 4.1 Headline composability table (§4.2)

Trained hosts × {alone, +DBAF, +PCSA, +both}:

| Host | LLM (LLaMA-3-8B WikiText / C4 PPL) | SAM (mAP) | SR (PSNR) |
|---|---|---|---|
| OmniQuant | NEW base + 3 variants | — | — |
| AHCPTQ | — | NEW base + 3 variants × {YOLOX, H-DETR} × {B, L, H} | — |
| 2DQuant | — | — | NEW base + 3 variants × {×2, ×3, ×4} |

LLM hosts evaluated on Qwen-2.5-7B as a second model.

### 4.2 Training-free table (§4.3)

RTN / GPTQ / AWQ × {alone, +DBAF, +PCSA-tf, +both} on every model:

| Model | RTN | RTN+D | RTN+P | RTN+D+P | GPTQ | GPTQ+D | GPTQ+P | GPTQ+D+P | AWQ | AWQ+D | AWQ+P | AWQ+D+P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaMA-3-8B | ✓ | ✓ | NEW | NEW | ✓ | ✓ | NEW | NEW | ✓ | ✓ | NEW | NEW |
| Qwen-2.5-7B | ✓ | ✓ | NEW | NEW | ✓ | ✓ | NEW | NEW | ✓ | ✓ | NEW | NEW |
| SAM-B/L/H | partial | partial | NEW | NEW | NEW | NEW | NEW | NEW | NEW | NEW | NEW | NEW |
| SwinIR ×2/×3/×4 | ✓ | ✓ | NEW | NEW | NEW | NEW | NEW | NEW | NEW | NEW | NEW | NEW |

D = DBAF, P = PCSA-tf.

### 4.3 Long-context KV (§4.5)

RULER NIAH at 4k + 8k context, two arms:
- Baseline: FlatQuant + DBAF + PCSA (no KV-PCSA), trained calibration
- KV-PCSA-tf: same + training-free k-means KV anchors

If KV-PCSA-tf wins at long context: keep in §4.5. Otherwise: move to Limitations.

### 4.4 Saturation evidence (§4.6)

- FlatQuant W4A4 KV4 alone: 6.964 PPL (have)
- FlatQuant + DBAF + PCSA W4A4 KV4: 6.966 PPL (have)
- FlatQuant + DBAF (no-gate) + PCSA W4A4 KV4: C1 in-flight, expected ~6.97
- CompSRT + DBAF SwinIR ×2: 38.15 vs 38.13 (have)

Single paragraph + small table; framed as positive evidence: rotation-based
hosts implicitly handle the same distributions our primitives target.

### 4.5 Mechanism (§4.7)

- Synthetic outlier injection (have)
- Synthetic dense-vs-sparse outlier study (NEW, have)
- Per-layer outlier-vs-gain correlation: LLaMA-3-8B (have), SAM-B (NEW), SwinIR ×3 (have, NEW)
- Matched-T clipping ablation (have)
- Distribution taxonomy classifier across all models (NEW, have)
- Taxonomy-predicts-x3 narrative: pre-experiment, our taxonomy showed 22 dense-outlier activation layers at SwinIR ×3 vs 3 at ×2 / 8 at ×4; the post-experiment ×3 regression confirms the prediction. The framework is *predictive*, not just descriptive.

### 4.6 Cost-vs-quality figure (§4.8)

x-axis: calibration FLOPs (or wall-clock seconds), log scale
y-axis: quality metric per task (PPL / mAP / PSNR)
Curves: base methods and their +DBAF+PCSA-tf variants. Our claim: DBAF +
PCSA-tf shift the entire curve down at zero added calibration cost (for
training-free) or marginal added cost (for trained hosts).

### 4.7 Real INT4 deployment (§4.9)

Carryover — torchao plumbing + latency numbers in PAPER_RESULTS.md.

## 5. Paper section structure

```
§1 Introduction
   - Hook, three-level claim
   - Fragmentation Table (extend existing, add citation column)

§2 Related Work
   - Reorganized per-distribution-type:
     - Sparse-Gaussian outliers
     - Rotational outlier absorption
     - Post-ReLU asymmetric
     - Bimodal post-Softmax / post-Softmax peaked
     - Per-prompt distribution shift
   - Each subsection cites prior work and identifies which arch it targets

§3 Method
   - 3.1 Distributional taxonomy + gate predicate
   - 3.2 DBAF (existing)
   - 3.3 PCSA (gradient) + PCSA-tf (k-means)
   - 3.4 KV-PCSA-tf
   - 3.5 Composability API

§4 Experiments
   - 4.1 Distribution recurrence + cross-arch figure
   - 4.2 Composability on trained hosts (HEADLINE TABLE)
   - 4.3 Training-free table
   - 4.4 Cross-detector AHCPTQ matrix (SAM-B/L/H × YOLOX/H-DETR)
   - 4.5 Long-context KV (RULER 4k/8k)
   - 4.6 Saturation evidence
   - 4.7 Mechanism (synthetic + per-layer + taxonomy-predicts-x3)
   - 4.8 Cost vs Quality figure
   - 4.9 Real INT4 deployment

§5 Conclusion

§6 Limitations
   - Bimodal post-Softmax not addressed by current primitives
   - KV-PCSA-tf on standard PPL marginal — long-context only
   - SR ×3 regression as in-the-wild distributional mismatch
```

## 6. Experiment program (12-day plan)

### 6.1 Engineering (Days 1-5)

| ID | Item | GPU | Where | Day |
|---|---|---|---|---|
| G1 | Clone OmniQuant + 2DQuant under `/home/ubuntu/unifying-ptq/` | — | local | 1 |
| G2 | Training-free PCSA-tf primitive (k-means anchors) | — | local CPU | 1-2 |
| G3 | Training-free KV-PCSA-tf primitive | — | local CPU | 2 |
| G4 | Plumb DBAF + (gradient) PCSA + PCSA-tf into OmniQuant | small | local | 2-3 |
| G5 | Plumb DBAF + (gradient) PCSA + PCSA-tf into 2DQuant | small | local | 3-4 |

### 6.2 Calibration & evaluation (Days 4-9)

| ID | Item | GPU-hours | Where |
|---|---|---|---|
| G6 | OmniQuant × {alone, +DBAF, +PCSA, +D+P} × LLaMA-3-8B + Qwen-2.5-7B | 16h | local |
| G7 | 2DQuant × {alone, +DBAF, +PCSA, +D+P} × SwinIR ×2/×3/×4 | 6h | local |
| G8 | RTN/GPTQ/AWQ × {alone, +DBAF, +PCSA-tf, +D+P} × all 8 model/scale combos | 10h | local + remote |
| G9 | AHCPTQ + DBAF + PCSA cross-detector matrix (SAM-B/L + H-DETR; SAM-H + YOLOX running; SAM-H + H-DETR queued) | 21h | remote (sequential) |
| G10 | RULER 4k/8k eval baseline vs KV-PCSA-tf | 2h | local |
| G11 | Cross-model layer analysis on LLaMA + Qwen + SAM-H | 3h | local |
| G12 | Per-layer ablation on SAM-B + LLaMA-3-8B | 5h | local |
| G13 | C1 (FlatQuant no-gate, currently in-flight) | 2h | local | 

### 6.3 Paper writing (Days 9-12)

| ID | Item | Time |
|---|---|---|
| G14 | §1 intro + abstract rewrite | 1 day |
| G15 | §2 related rewrite (per-distribution organization) | 0.5 day |
| G16 | §3 method rewrite (PCSA-tf, KV-PCSA-tf, composability) | 1 day |
| G17 | §4 experiments rewrite + table generation | 1.5 days |
| G18 | §5 + §6 + figures + Overleaf push + 8-page check | 1 day |

### 6.4 Supplementary (unlimited pages)

- α derivation (existing appendix)
- DBAF clipping proof (existing appendix)
- Full per-model layer JSON tables
- All per-layer ablation curves
- Cost accounting details
- Saturation-evidence numbers (FlatQuant / CompSRT raw)

## 7. Risks + mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| OmniQuant integration delay | Med | High | Fallback: use published numbers + apply DBAF/PCSA as post-hoc forward-only |
| 2DQuant integration delay | Med | Med | Same fallback |
| AHCPTQ SAM-H reproduction differs from ICML | Med | Med | Report observed + cite ICML |
| KV-PCSA-tf flat on RULER | Low-Med | Low | Move to Limitations |
| 8-page overflow | High | Medium | Supplementary unlimited; move appendix proofs there |
| Reviewer asks about FlatQuant | High | Low | Saturation evidence answers this |
| GPU shortage | Low | Med | Both A100s utilized; sequential queues planned |

## 8. Drop / demote list

**Dropped (genuinely redundant)**:
- #55 S4.7 "DBAF on outlier-free layers": superseded by synthetic dense-vs-sparse (df=10 row)
- #56 S4.8 "Cross-arch outlier prevalence (port from ICML)": superseded by F3 distribution taxonomy

**Demoted to saturation evidence (run only if GPU idle)**:
- C2 no-gate Qwen FlatQuant
- C3a no-gate AHCPTQ SAM-B (full retrain)
- C3b no-gate CompSRT SwinIR ×2

**Kept (still in scope)**:
- All paper rewrite tasks
- B4 per-layer ablation on SAM-B + LLaMA (now central, was almost-dropped)
- A3 cross-model analysis extension to LLM + Qwen + SAM-H
- C1 (currently in-flight, becomes saturation evidence)
- D2 RULER eval

## 9. Definition of done

- All cells in §4.1 (headline table) and §4.2 (training-free table) have numbers
- Saturation evidence section has numbers from C1 + ICML carryover
- RULER 4k/8k eval done (baseline + KV-PCSA-tf)
- Mechanism §4.7 has the taxonomy-predicts-x3 narrative explicit
- Cost-vs-quality figure rendered
- Paper compiles in 8 pages; supplementary unlimited
- Final compile + Overleaf push by May 25 23:59 AoE

## 10. Open questions for later

- Should we report seed variance on the headline cells? (need re-runs at multiple seeds)
- Do we have time for Mistral-7B / Phi-3 as third LLM? (deferred unless paper has page)
- Do we include W3 results anywhere? (not in current plan; possible supplementary)
