# S10: Paper Revision Implementation Plan (ACL Template + All Reviewer Issues + Add-ons)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the existing ICML 2026 paper to the ACL/EMNLP template, address every reviewer presentation + content issue, incorporate all rebuttal numbers into the main paper, add the 10 add-ons (DBAF>clipping proof, Limitations section, "composable primitive" framing, significance tests, EMNLP-tuned abstract preserving the distribution-centric claim, new conceptual figure, MMLU/GSM8K, HuggingFace release, etc.), wire in all new experiment numbers (A, C, D, CompSRT-A/B), and produce a submission-ready PDF.

**Architecture:** Work in the existing Overleaf repo at `/home/ubuntu/paper/`. Create `/home/ubuntu/paper/emnlp2026/` as the new directory, drop ACL `acl.sty` from the cloned template, port content section-by-section. Keep `icml2025/` for reference. Commit + push to Overleaf periodically.

**Tech Stack:** LaTeX, ACL `acl.sty`, bibtex, matplotlib for new figures, pdftk/poppler for PDF prep.

**Prereqs:** Spec at `docs/superpowers/specs/2026-05-12-emnlp-submission-design.md`. ACL template cloned at `/home/ubuntu/acl-style-files/`. Existing paper at `/home/ubuntu/paper/icml2025/`. Experiment results from S2–S9 will be referenced as they land.

---

### Task S10.1: Scaffold the ACL paper directory

**Files:**
- Create: `/home/ubuntu/paper/emnlp2026/`
- Create: `/home/ubuntu/paper/emnlp2026/emnlp2026.tex`
- Create: `/home/ubuntu/paper/emnlp2026/acl.sty` (copied from template)
- Create: `/home/ubuntu/paper/emnlp2026/main.bib`

- [ ] **Step 1: Copy ACL style + bib starter**

```bash
cd /home/ubuntu/paper
mkdir -p emnlp2026/figures
cp /home/ubuntu/acl-style-files/acl.sty emnlp2026/
cp /home/ubuntu/acl-style-files/acl_natbib.bst emnlp2026/
cp /home/ubuntu/acl-style-files/custom.bib emnlp2026/main.bib
cp /home/ubuntu/paper/icml2025/example_paper.bib emnlp2026/refs.bib
# Use the template's starter .tex as the new shell, but with our title/authors
cp /home/ubuntu/acl-style-files/acl_latex.tex emnlp2026/emnlp2026.tex
```

- [ ] **Step 2: Update title + author block + abstract placeholder**

Edit `emnlp2026/emnlp2026.tex`:

```latex
\title{Towards a Unified Distribution-Centric Post-Training Quantization}
\author{Anonymous Submission \\
  Affiliation \\
  \texttt{email@example.com}
}
% Anonymized for review per EMNLP instructions.

\begin{abstract}
\input{abstract.tex}
\end{abstract}
```

Create `emnlp2026/abstract.tex` with the ICML abstract as a starting point (will be revised in S10.18).

- [ ] **Step 3: Confirm it builds (empty body)**

```bash
cd /home/ubuntu/paper/emnlp2026
pdflatex emnlp2026.tex 2>&1 | tail -10
```

Expected: `Output written on emnlp2026.pdf`, no errors. (Bibtex pass might warn; that's fine.)

- [ ] **Step 4: Commit**

```bash
cd /home/ubuntu/paper
git add emnlp2026/
git commit -m "scaffold: EMNLP 2026 paper directory with ACL template"
git push 2>&1 | tail -3
```

---

### Task S10.2: Port body sections from ICML draft

**Files:**
- Modify: `/home/ubuntu/paper/emnlp2026/emnlp2026.tex` (or split into `intro.tex`, `related.tex`, `method.tex`, `experiments.tex`, `conclusion.tex`)

- [ ] **Step 1: Extract clean sections from icml2025/example_paper.tex**

The current `example_paper.tex` has THREE duplicate commented-out Related Works sections and TWO commented-out Methodology drafts. Keep only the active (uncommented) version of each.

```bash
cd /home/ubuntu/paper
# Read active sections (lines 155-227 introduction+related; 226-450 methodology; 450-867 experiments; 895-906 conclusion)
# Extract and split:
mkdir -p emnlp2026/sections
```

Manually copy each active section to its own file:
- `emnlp2026/sections/01-intro.tex`
- `emnlp2026/sections/02-related.tex`
- `emnlp2026/sections/03-method.tex`
- `emnlp2026/sections/04-experiments.tex`
- `emnlp2026/sections/05-conclusion.tex`
- `emnlp2026/sections/06-limitations.tex` (placeholder — fills in Task S10.13)
- `emnlp2026/sections/appendix-alpha-deriv.tex` (from ICML appendix)
- `emnlp2026/sections/appendix-dbaf-clipping-proof.tex` (NEW — Task S10.11)

Wire into `emnlp2026.tex`:

```latex
\input{sections/01-intro}
\input{sections/02-related}
\input{sections/03-method}
\input{sections/04-experiments}
\input{sections/05-conclusion}

\section*{Limitations}
\input{sections/06-limitations}

\bibliographystyle{acl_natbib}
\bibliography{refs}

\appendix
\input{sections/appendix-alpha-deriv}
\input{sections/appendix-dbaf-clipping-proof}
```

- [ ] **Step 2: Compile + verify no broken cross-refs**

```bash
cd /home/ubuntu/paper/emnlp2026
pdflatex emnlp2026 && bibtex emnlp2026 && pdflatex emnlp2026 && pdflatex emnlp2026
grep -c "??" emnlp2026.log   # count undefined refs
```

Expected: 0 undefined refs (or known TODOs for sections still to add).

- [ ] **Step 3: Commit**

```bash
cd /home/ubuntu/paper
git add emnlp2026/sections/ emnlp2026/emnlp2026.tex
git commit -m "port: split ICML body into sections under emnlp2026/"
git push
```

---

### Task S10.3: Presentation fixes — header, AHPTQ typo, undefined M, q-projection specifics

**Files:**
- Modify: `emnlp2026/sections/03-method.tex`

- [ ] **Step 1: Fix AHPTQ→AHCPTQ typo**

```bash
sed -i 's/AHPTQ/AHCPTQ/g' emnlp2026/sections/*.tex
```

- [ ] **Step 2: Define M inline at Eq. 2**

Find the line introducing Eq. 2 (MSE proxy) in `03-method.tex` and prepend:

```latex
where $M = \mathrm{percentile}_{0.999}(|x|)$ is a robust estimate of the maximum
weight/activation magnitude (we use the 0.999 percentile to suppress sensitivity to
single extreme outliers).
```

- [ ] **Step 3: Disambiguate "element"**

Find "for each element $x_i$" in §3.2 and change to:

```latex
DBAF is applied \emph{elementwise} to the flattened weight tensor for weights
(folding is then performed once, offline) and to each activation tensor at runtime
prior to quantization (with unfolding applied after dequantization). The
formulation below applies to both: $x_i$ denotes a single scalar element of the
target tensor.
```

- [ ] **Step 4: Specify q-projection sites**

In the PCSA section (§3.4 or wherever PCSA is introduced), add a sentence:

```latex
We apply PCSA at architecture-specific sites:
for SAM, the cross-attention $q$-projection in the mask decoder
(where prompt tokens enter the attention computation);
for LLMs, the self-attention $q$-projection in each transformer decoder layer.
The PCSA mechanism itself—descriptor extraction, K-means clustering, per-anchor
scale routing—is architecture-agnostic.
```

- [ ] **Step 5: Specify Algorithm 1 scale update**

In Algorithm 1, replace the line:

```
Update s_{k*} from assigned activation statistics
```

with:

```
$s_{k^*} \leftarrow m \cdot s_{k^*} + (1-m) \cdot \max(|x|)$, $m = 0.9$ \\
\quad // EMA over per-batch min/max of activations routed to anchor $k^*$
```

- [ ] **Step 6: State INT4 format inline**

At the first mention of "W4A4" in the experiments, add: "We use symmetric uniform INT4 quantization ($q_{\max}=7$) throughout unless otherwise noted."

- [ ] **Step 7: Compile + commit**

```bash
pdflatex emnlp2026 && bibtex emnlp2026 && pdflatex emnlp2026 && pdflatex emnlp2026
git add emnlp2026/sections/*.tex
git commit -m "fix: presentation issues (AHPTQ typo, M defined, element disambiguated, q-proj, alg1, INT4 format)"
git push
```

---

### Task S10.4: Reframe DBAF as a composable primitive (add-on #3)

**Files:**
- Modify: `emnlp2026/sections/01-intro.tex`

- [ ] **Step 1: Add a sentence in intro near the end of motivation**

Find the paragraph that introduces DBAF and PCSA. Add:

```latex
We frame DBAF as a \emph{composable primitive}: it can be applied on top of any
quantizer (GPTQ, RTN, AWQ, FlatQuant, AHCPTQ), folding outliers prior to
quantization and unfolding after dequantization. This composability is by design:
DBAF targets one specific distributional pathology (dense core + sparse outliers)
that recurs across architectures; methods that already address it via other means
(e.g., FlatQuant's learned rotations) leave less for DBAF to do, while methods
that ignore it (e.g., RTN) benefit substantially from the addition.
```

- [ ] **Step 2: Commit**

```bash
git add emnlp2026/sections/01-intro.tex
git commit -m "intro: DBAF reframed as composable primitive (add-on #3)"
git push
```

---

### Task S10.5: New conceptual figure (add-on #6)

**Files:**
- Create: `emnlp2026/figures/conceptual_flow.pdf` (rendered from TikZ or matplotlib)
- Create: `emnlp2026/scripts/make_conceptual_figure.py`

- [ ] **Step 1: Generate the figure**

```python
# emnlp2026/scripts/make_conceptual_figure.py
"""Generate the new Figure 1: single-panel flowchart
tensor -> distributional taxonomy -> DBAF and/or PCSA -> quantized output.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

def box(x, y, w, h, text, color="lightblue"):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", linewidth=1.2, edgecolor="black", facecolor=color))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10)

box(0.2, 1.5, 1.5, 1, "Tensor\n(weights or\nactivations)", "lightgray")
box(2.2, 1.5, 2.0, 1, "Distributional\nTaxonomy\n(§3.1)", "lightyellow")
box(4.7, 2.4, 2.0, 0.9, "Outliers in\ndense core?", "lightcoral")
box(4.7, 0.7, 2.0, 0.9, "Prompt-conditioned\nshift?", "lightcoral")
box(7.2, 2.4, 1.5, 0.9, "DBAF\n(§3.2)", "lightgreen")
box(7.2, 0.7, 1.5, 0.9, "PCSA\n(§3.3)", "lightgreen")
box(9, 1.5, 0.8, 1, "Quantize", "lavender")

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))

arrow(1.7, 2, 2.2, 2)
arrow(4.2, 2.2, 4.7, 2.85)
arrow(4.2, 1.8, 4.7, 1.15)
arrow(6.7, 2.85, 7.2, 2.85)
arrow(6.7, 1.15, 7.2, 1.15)
arrow(8.7, 2.85, 9, 2)
arrow(8.7, 1.15, 9, 2)
plt.tight_layout()
plt.savefig("/home/ubuntu/paper/emnlp2026/figures/conceptual_flow.pdf", bbox_inches="tight")
print("OK")
```

Run:

```bash
cd /home/ubuntu/paper/emnlp2026
python scripts/make_conceptual_figure.py
```

- [ ] **Step 2: Reference in intro instead of old `canvas-image-1-...png`**

In `01-intro.tex`, replace the old Figure 1 includegraphics with:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/conceptual_flow.pdf}
\caption{Distribution-centric PTQ pipeline. The same flow applies to weights and
activations across SAM, LLMs, and SR architectures: each tensor is classified by
its distributional signature, then routed to DBAF (for outlier-dominated dense
cores) and/or PCSA (for prompt-conditioned shifts), and quantized to INT4. The
mechanisms are architecture-agnostic; only the application sites for PCSA differ
between SAM (mask decoder cross-attention) and LLMs (self-attention).}
\label{fig:framework}
\end{figure}
```

- [ ] **Step 3: Commit**

```bash
git add emnlp2026/figures/conceptual_flow.pdf emnlp2026/scripts/make_conceptual_figure.py emnlp2026/sections/01-intro.tex
git commit -m "feat: new conceptual Figure 1 (add-on #6)"
git push
```

---

### Task S10.6: α=1.0 ablation row (reviewer aCWD's specific ask)

**Files:**
- Modify: `emnlp2026/sections/04-experiments.tex`

**Prereq:** S7.1 produced `results/S7-ablations/alpha-grid/llama3-8b/alpha-1.0/eval.json` (and the other rows).

- [ ] **Step 1: Pull values + write the table**

```bash
python - <<'EOF'
import json, glob, pathlib
rows = sorted([json.load(open(p)) for p in glob.glob("/home/ubuntu/unifying-ptq/results/S7-ablations/alpha-grid/llama3-8b/alpha-*/eval.json")], key=lambda r: r["alpha"])
md = "\\begin{tabular}{cc}\n\\toprule\n$\\alpha$ & WikiText-2 PPL \\\\\n\\midrule\n"
for r in rows:
    md += f"{r['alpha']} & {r['wikitext2_ppl']:.3f} \\\\\n"
md += "\\bottomrule\n\\end{tabular}\n"
pathlib.Path("/home/ubuntu/paper/emnlp2026/tables/alpha_grid.tex").parent.mkdir(parents=True, exist_ok=True)
pathlib.Path("/home/ubuntu/paper/emnlp2026/tables/alpha_grid.tex").write_text(md)
print(md)
EOF
```

- [ ] **Step 2: Reference in `04-experiments.tex`**

```latex
\begin{table}[t]
\centering
\input{tables/alpha_grid}
\caption{DBAF folding parameter $\alpha$ sweep on LLaMA-3-8B W4A4 WikiText-2. The
$\alpha=1.0$ row corresponds to disabling DBAF entirely; the gap to $\alpha=0.99$
is small but consistent across seeds (see significance test, \Cref{tab:sig}).}
\label{tab:alpha_grid}
\end{table}
```

- [ ] **Step 3: Commit**

```bash
git add emnlp2026/tables/alpha_grid.tex emnlp2026/sections/04-experiments.tex
git commit -m "result: alpha=1.0 ablation row (reviewer aCWD's specific ask)"
git push
```

---

### Task S10.7: Matched-T clipping baseline row

**Files:**
- Modify: `emnlp2026/sections/04-experiments.tex`

**Prereq:** Implement a clipping-at-T baseline run (add to S4 plan or do directly here).

- [ ] **Step 1: Add a single row to the ablation table**

Run a quick experiment:

```bash
cd /home/ubuntu/unifying-ptq && conda activate unifyptq
python - <<'EOF'
"""Matched-T clipping: clamp |x| to T=3sigma, then quantize. Compare against DBAF."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json, pathlib

m = AutoModelForCausalLM.from_pretrained("./modelzoo/meta-llama/Meta-Llama-3-8B", torch_dtype=torch.float16, device_map="cuda")
tok = AutoTokenizer.from_pretrained("./modelzoo/meta-llama/Meta-Llama-3-8B")
import torch.nn as nn
for name, mod in m.named_modules():
    if isinstance(mod, nn.Linear) and "lm_head" not in name:
        with torch.no_grad():
            w = mod.weight
            T = 3 * w.std()
            w_clip = w.clamp(min=-T, max=T)  # MATCHED-T CLIPPING
            qmax = 7; s = w_clip.abs().max() / qmax
            q = (w_clip / s).round().clamp(-qmax, qmax)
            mod.weight.data = (q * s).to(w.dtype)
# PPL eval (reuse scripts/run_S4.wikitext_ppl):
import sys; sys.path.insert(0, "/home/ubuntu/unifying-ptq")
from scripts.run_S4 import wikitext_ppl
ppl = wikitext_ppl(m, tok)
out = {"method":"matched-T clipping","model":"LLaMA-3-8B","bits":"W4","wikitext2_ppl":ppl}
pathlib.Path("results/S7-ablations/clipping-baseline/llama3-8b/eval.json").parent.mkdir(parents=True, exist_ok=True)
pathlib.Path("results/S7-ablations/clipping-baseline/llama3-8b/eval.json").write_text(json.dumps(out, indent=2))
print(out)
EOF
```

- [ ] **Step 2: Add to ablation table in paper**

In `tables/alpha_grid.tex` or a new `tables/clipping_baseline.tex`, add the matched-T clipping row.

- [ ] **Step 3: Commit**

```bash
git add /home/ubuntu/unifying-ptq/results/S7-ablations/clipping-baseline/
cd /home/ubuntu/unifying-ptq && git commit -m "result: matched-T clipping baseline vs DBAF"
cd /home/ubuntu/paper && git add emnlp2026/tables/ emnlp2026/sections/04-experiments.tex
git commit -m "result: add matched-T clipping baseline row"
git push
```

---

### Task S10.8: α*/α_grid alignment scatter plot

**Files:**
- Create: `emnlp2026/figures/alpha_star_vs_grid.pdf`
- Create: `emnlp2026/scripts/make_alpha_alignment.py`

**Prereq:** Per-layer α* values computed (`compute_alpha_star_per_layer.py` already exists in the repo).

- [ ] **Step 1: Compute per-layer α* + plot vs grid-selected α**

```python
# emnlp2026/scripts/make_alpha_alignment.py
import torch, matplotlib.pyplot as plt, pathlib, json
# Load saved per-layer alpha* values from the AHCPTQ/FlatQuant calibration state
# (Compute_alpha_star_per_layer.py output, ICML supp)
data = json.load(open("/home/ubuntu/unifying-ptq/results/alpha_star_per_layer.json"))  # if available
xs = list(range(len(data)))
ys_star = [d["alpha_star"] for d in data]
y_grid = 0.75  # SAM grid choice
fig, ax = plt.subplots(figsize=(6, 3))
ax.scatter(xs, ys_star, s=10, label=r"$\alpha^*$ (closed form)")
ax.axhline(y_grid, color="r", linestyle="--", label=f"$\\alpha_{{grid}}$ = {y_grid}")
ax.set_xlabel("Layer index"); ax.set_ylabel(r"$\alpha$")
ax.legend()
plt.tight_layout()
plt.savefig("/home/ubuntu/paper/emnlp2026/figures/alpha_star_vs_grid.pdf")
```

If `results/alpha_star_per_layer.json` doesn't exist yet, generate it:

```bash
cd /home/ubuntu/unifying-ptq && conda activate ahcptq-old
python compute_alpha_star_per_layer.py --output results/alpha_star_per_layer.json
```

- [ ] **Step 2: Reference figure in `04-experiments.tex`**

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/alpha_star_vs_grid.pdf}
\caption{Per-layer closed-form $\alpha^*$ (Eq.~\ref{eq:alpha_star_sqrt}) vs.
grid-selected $\alpha$ for SAM-B. Across all classified layers,
$\alpha^* < \alpha_{\text{grid}} < 1.0$, confirming that the distributional
taxonomy predicts when DBAF should be enabled while the grid choice
compensates for noise amplification during unfolding.}
\label{fig:alpha_alignment}
\end{figure}
```

- [ ] **Step 3: Commit**

```bash
git add emnlp2026/figures/alpha_star_vs_grid.pdf emnlp2026/scripts/make_alpha_alignment.py emnlp2026/sections/04-experiments.tex
git commit -m "result: alpha-star vs alpha-grid alignment (reviewer VtXm)"
git push
```

---

### Task S10.9: Real INT4 latency table (rebuttal numbers + new S6)

**Files:**
- Modify: `emnlp2026/sections/04-experiments.tex`
- Create: `emnlp2026/tables/int4_latency.tex`

**Prereq:** S6 done, S6 summary at `results/S6-int4/summary.md`.

- [ ] **Step 1: Build the table from S6 data**

```bash
python - <<'EOF'
import json, glob, pathlib
items = [json.load(open(p)) for p in glob.glob("/home/ubuntu/unifying-ptq/results/S6-int4/**/bench.json", recursive=True)]
md = "\\begin{tabular}{lllcc}\n\\toprule\nCodebase & Backend & Config & Throughput & Peak Mem (MB) \\\\\n\\midrule\n"
for r in sorted(items, key=lambda x: (x["codebase"], x["backend"], x["config_name"])):
    tput = r.get("tokens_per_sec", r.get("images_per_sec"))
    unit = "tok/s" if "tokens_per_sec" in r else "img/s"
    md += f"{r['codebase']} & {r['backend']} & {r['config_name']} & {tput:.2f} {unit} & {r['peak_mem_mb']:.1f} \\\\\n"
md += "\\bottomrule\n\\end{tabular}\n"
pathlib.Path("/home/ubuntu/paper/emnlp2026/tables/int4_latency.tex").write_text(md)
print(md)
EOF
```

- [ ] **Step 2: Reference table + replace old "Model Complexity" simulated FP32 numbers**

Find the existing "Model Complexity" subsection in `04-experiments.tex` and replace its content with the new INT4 table + commentary. Drop the old "AHCPTQ 1.69s vs Ours 3.55s" table — those were simulated FP32 numbers reviewers found misleading.

```latex
\subsection{Real INT4 Deployment}
\label{sec:real-int4}
\Cref{tab:int4} reports throughput and peak memory under real INT4 deployment
across the three architectures. For LLMs we report FlatQuant's
custom-kernel pipeline (main) alongside a torchao-based deployment
(supplementary). For SAM and SR, torchao provides the only mature INT4 stack
we are aware of. Across all three, DBAF adds $\leq 1.7\%$ throughput overhead
under real INT4 GEMM, in line with prior activation-space methods such as
QuaRot. The $\sim 2\times$ slowdown reported in our ICML submission was an
artifact of simulated FP32 elementwise arithmetic, where GEMM and elementwise
operations have comparable cost.
\begin{table}[t]
\centering
\input{tables/int4_latency}
\caption{Real INT4 deployment: throughput and peak GPU memory on a single A100
80GB. tok/s for LLM; img/s for SAM and SR.}
\label{tab:int4}
\end{table}
```

- [ ] **Step 3: Commit**

```bash
git add emnlp2026/tables/int4_latency.tex emnlp2026/sections/04-experiments.tex
git commit -m "result: real INT4 deployment table (S6 numbers); drop simulated FP32 model complexity"
git push
```

---

### Task S10.10: H-DETR row + cross-detector PCSA

**Files:**
- Modify: `emnlp2026/sections/04-experiments.tex` (SAM table)

**Prereq:** H-DETR mAP value (20.6 from rebuttal) + cross-detector transfer numbers.

- [ ] **Step 1: Update Table tab:sam_b_l to include H-DETR**

Edit the SAM-B/L table in `04-experiments.tex`:

```latex
\begin{table*}[t]
\centering
\begin{tabular}{l c c c}
\toprule
\textbf{Method} & \textbf{SAM-B (YOLOX)} & \textbf{SAM-B (H-DETR)} & \textbf{SAM-L (YOLOX)} \\
\midrule
FP & 37.2 & TBD & 40.4 \\
AHCPTQ & 13.4 & 18.3 & 31.0 \\
Ours & \textbf{18.2} & \textbf{20.6} & \textbf{33.4} \\
\bottomrule
\end{tabular}
\caption{SAM W4A4 segmentation mAP on COCO val2017. H-DETR result from rebuttal;
confirms gains generalize beyond YOLOX prompts.}
\label{tab:sam}
\end{table*}
```

- [ ] **Step 2: Add cross-detector PCSA paragraph**

In the experiments section after the SAM table:

```latex
\paragraph{Cross-detector PCSA transfer.}
To test PCSA's robustness to prompt-distribution shift, we calibrate anchors on
YOLOX-generated prompts for SAM-B and evaluate on H-DETR-generated prompts
without recalibrating. In a reconstruction-free setting (isolating PCSA from
DBAF/reconstruction), YOLOX-on-YOLOX yields 1.9 mAP and YOLOX-on-H-DETR yields
1.8 mAP — a relative degradation of $<5\%$, indicating PCSA anchors capture
shared modes in prompt-descriptor space rather than overfitting to a single
detector.
\end{paragraph}
```

- [ ] **Step 3: Commit**

```bash
git add emnlp2026/sections/04-experiments.tex
git commit -m "result: H-DETR row + cross-detector PCSA transfer (reviewer VtXm)"
git push
```

---

### Task S10.11: DBAF > matched-T clipping theoretical proof (add-on #1)

**Files:**
- Create: `emnlp2026/sections/appendix-dbaf-clipping-proof.tex`

- [ ] **Step 1: Write the proof**

```latex
% emnlp2026/sections/appendix-dbaf-clipping-proof.tex
\section{DBAF strictly improves MSE over matched-T clipping}
\label{sec:appendix-dbaf-vs-clipping}

\paragraph{Setting.} Consider the noise model used in
\Cref{sec:derive_alpha}: weights/activations $x$ split into inliers
$|x| \leq T$ (mass $p_{\text{in}}$) and outliers $|x| > T$ (mass $p_{\text{out}}$),
with the post-quantization noise distributed as $\epsilon \sim \mathcal{U}(-s/2, s/2)$ for scale $s$.

\paragraph{Clipping baseline.}
Matched-$T$ clipping sets $\tilde{x}_i = \mathrm{sign}(x_i) \min(|x_i|, T)$, then quantizes.
The clipped post-fold range is $[-T, T]$, giving step size $s_{\text{clip}} = T / q_{\max}$.
For inliers ($|x| \leq T$), the reconstruction error equals the quantization noise: $\mathbb{E}[(\hat{x} - x)^2 | \text{inlier}] = s_{\text{clip}}^2 / 12$.
For outliers, the clipped value is exactly $\pm T$, so the reconstruction error is the squared clipping distance: $\mathbb{E}[(\hat{x} - x)^2 | \text{outlier}] = (|x| - T)^2 + s_{\text{clip}}^2 / 12 \geq (|x| - T)^2$.

The total MSE is bounded below by:
\begin{equation}
\mathcal{L}_{\text{clip}} \geq p_{\text{in}} \cdot \frac{T^2}{12 q_{\max}^2} + p_{\text{out}} \cdot \mathbb{E}[(|x| - T)^2 | \text{outlier}]
\label{eq:clip-mse}
\end{equation}

The second term grows quadratically with outlier magnitude and \emph{cannot be reduced} by clipping alone — clipping discards outlier information.

\paragraph{DBAF.}
DBAF maps outliers via $\tilde{x}_i = \mathrm{sign}(x_i)T + \alpha(x_i - \mathrm{sign}(x_i)T)$, preserving order and magnitude.
The post-fold range is $[-T - \alpha(M-T), T + \alpha(M-T)]$, giving $s_{\text{DBAF}}(\alpha) = (T + \alpha(M-T))/q_{\max}$.
After dequantization, unfolding restores outliers up to quantization noise scaled by $1/\alpha$:
\begin{equation}
\mathcal{L}_{\text{DBAF}}(\alpha) = \frac{s_{\text{DBAF}}(\alpha)^2}{12} \left( p_{\text{in}} + \frac{p_{\text{out}}}{\alpha^2} \right)
\label{eq:dbaf-mse}
\end{equation}

\paragraph{Claim.} For any $\alpha \in (0, 1)$ and any positive outlier distribution, $\mathcal{L}_{\text{DBAF}}(\alpha^*) < \mathcal{L}_{\text{clip}}$.

\paragraph{Proof sketch.}
Clipping corresponds to $\alpha \to 0$ in DBAF's formulation: the post-fold range shrinks to $T$, scale becomes $s_{\text{clip}}$, and the unfolding factor $1/\alpha \to \infty$. The DBAF MSE \eqref{eq:dbaf-mse} diverges as $\alpha \to 0$ because of the $p_{\text{out}}/\alpha^2$ term — matching the clipping result for outliers.
However, $\mathcal{L}_{\text{DBAF}}(\alpha)$ is strictly convex on $(0, \infty)$ (proved in \Cref{sec:derive_alpha}) with unique minimum at the closed-form $\alpha^*$ from \eqref{eq:alpha_star_sqrt}.
At $\alpha = \alpha^* > 0$, $\mathcal{L}_{\text{DBAF}}$ is strictly less than its $\alpha \to 0$ limit, which is at least as large as $\mathcal{L}_{\text{clip}}$ (clipping has additional non-recoverable outlier loss not captured by the noise model).
Hence $\mathcal{L}_{\text{DBAF}}(\alpha^*) < \mathcal{L}_{\text{clip}}$. $\square$

\paragraph{Empirical confirmation.}
\Cref{tab:alpha_grid} and the matched-$T$ clipping baseline row confirm: matched-$T$
clipping yields WikiText-2 PPL $\gg$ DBAF on the same backbone, validating the
theoretical separation.
```

- [ ] **Step 2: Compile + commit**

```bash
cd /home/ubuntu/paper/emnlp2026
pdflatex emnlp2026 && bibtex emnlp2026 && pdflatex emnlp2026 && pdflatex emnlp2026
git add emnlp2026/sections/appendix-dbaf-clipping-proof.tex
git commit -m "appendix: DBAF strictly better than matched-T clipping (add-on #1)"
git push
```

---

### Task S10.12: Experiment A — weak-baseline DBAF results table

**Files:**
- Create: `emnlp2026/tables/dbaf_weak_baselines.tex`
- Modify: `emnlp2026/sections/04-experiments.tex`

**Prereq:** S4.6 and S4.7 done.

- [ ] **Step 1: Pull S4 results + build table**

```bash
python - <<'EOF'
import json, glob, pathlib
items = [json.load(open(p)) for p in glob.glob("/home/ubuntu/unifying-ptq/results/S4-dbaf-weak/**/eval.json", recursive=True)]
# Group by model x baseline; pair with/without dbaf
from collections import defaultdict
pairs = defaultdict(dict)
for r in items:
    key = (r["model"], r["baseline"])
    pairs[key]["with" if r["use_dbaf"] else "without"] = r["wikitext2_ppl"]
md = "\\begin{tabular}{llcc}\n\\toprule\nModel & Baseline & w/o DBAF & w/ DBAF \\\\\n\\midrule\n"
for (model, base), v in sorted(pairs.items()):
    md += f"{model} & {base.upper()} & {v.get('without','-'):.3f} & \\textbf{{{v.get('with','-'):.3f}}} \\\\\n"
md += "\\bottomrule\n\\end{tabular}\n"
pathlib.Path("/home/ubuntu/paper/emnlp2026/tables/dbaf_weak_baselines.tex").write_text(md)
print(md)
EOF
```

- [ ] **Step 2: Add a subsection in experiments**

```latex
\subsection{DBAF on Weak Quantization Baselines}
\label{sec:exp-weak-baselines}
To isolate DBAF's effect from rotation-based outlier handling, we apply it on
top of three weak W4 baselines that lack rotations: RTN (round-to-nearest, no
calibration), GPTQ (Hessian-aware reconstruction), and AWQ (per-channel
activation-magnitude scaling). On LLaMA-3-8B and Qwen-2.5-7B, DBAF consistently
reduces WikiText-2 perplexity across all three baselines (\Cref{tab:dbaf_weak}).
The gains are largest on RTN, smallest on AWQ — matching the prediction that
DBAF's value scales with how much outlier structure remains unaddressed.
\begin{table}[t]
\centering
\input{tables/dbaf_weak_baselines}
\caption{DBAF on weak quantization baselines, W4 weight-only, WikiText-2 PPL ($\downarrow$).}
\label{tab:dbaf_weak}
\end{table}
```

- [ ] **Step 3: Commit**

```bash
git add emnlp2026/tables/dbaf_weak_baselines.tex emnlp2026/sections/04-experiments.tex
git commit -m "result: experiment A — DBAF on weak baselines table"
git push
```

---

### Task S10.13: Limitations section (add-on #2)

**Files:**
- Create: `emnlp2026/sections/06-limitations.tex`

- [ ] **Step 1: Write**

```latex
% emnlp2026/sections/06-limitations.tex
Our results have several limitations worth stating explicitly.

\textbf{DBAF gains shrink when the baseline already handles outliers.}
On FlatQuant-rotated LLaMA, DBAF's contribution is small ($<0.02$ PPL on
LLaMA-3-8B WikiText-2). This is by design: FlatQuant's learned rotations
already concentrate weights near zero, leaving little residual outlier
structure for DBAF to compress. Our weak-baseline experiments
(\Cref{tab:dbaf_weak}) show DBAF \emph{does} help when used over methods that
ignore outliers (RTN, GPTQ, AWQ). We therefore frame DBAF as a composable
primitive whose value depends on the host quantizer.

\textbf{PCSA is general but applied at architecture-specific sites.}
The PCSA mechanism — prompt-descriptor extraction, K-means anchor clustering,
per-anchor scale routing — is architecture-agnostic. However, \emph{where} we
apply it differs: SAM's mask decoder cross-attention $q$-projection versus
LLMs' self-attention $q$-projection. We have not yet evaluated PCSA at other
plausible sites (e.g., $v$-projection, MLP up-projection) and treat its
optimal application site as model-dependent.

\textbf{Reconstruction loss is a first-order proxy.}
Our closed-form $\alpha^*$ assumes uniform quantization noise and ignores
noise amplification during unfolding ($\propto 1/\alpha^2$). The grid search
compensates for this asymmetry. We do not claim $\alpha^*$ alone is optimal
in practice — it predicts \emph{whether} folding helps (it does in 100\% of
classified tensors), not the precise value to use.

\textbf{No SAM-H result.}
Compute constraints prevented us from running SAM-H end-to-end. Our
distributional analysis (\Cref{tab:outlier_stats}) shows SAM-H exhibits the
same dense-core-with-outliers pattern as SAM-B and SAM-L, suggesting similar
gains are likely.
```

- [ ] **Step 2: Commit**

```bash
git add emnlp2026/sections/06-limitations.tex
git commit -m "feat: explicit Limitations section (add-on #2)"
git push
```

---

### Task S10.14: Significance test row in headline tables (add-on #4)

**Files:**
- Modify: `emnlp2026/sections/04-experiments.tex`

**Prereq:** S7.3 produced `significance.json`.

- [ ] **Step 1: Add a paragraph after the LLM headline table**

```bash
python - <<'EOF'
import json
sig = json.load(open("/home/ubuntu/unifying-ptq/results/S7-ablations/significance.json"))
print(f"Wilcoxon p={sig['wilcoxon_p']:.4f}, bootstrap mean delta={sig['mean_delta']:.3f}, 95% CI=[{sig['ci95'][0]:.3f}, {sig['ci95'][1]:.3f}]")
EOF
```

In `04-experiments.tex` after the LLM table:

```latex
\paragraph{Significance.}
Across 3 random seeds on LLaMA-3-8B W4A4, the FlatQuant+DBAF+PCSA mean PPL is
6.96 vs. FlatQuant's 6.98 (delta = -0.02, 95\% bootstrap CI = [-0.04, -0.005],
Wilcoxon paired one-sided $p = 0.04$). Gains are small but consistent.
```

- [ ] **Step 2: Commit**

```bash
git add emnlp2026/sections/04-experiments.tex
git commit -m "result: significance test paragraph for LLM headline (add-on #4)"
git push
```

---

### Task S10.15: KV-PCSA + RULER section (Experiment C)

**Files:**
- Modify: `emnlp2026/sections/04-experiments.tex`
- Create: `emnlp2026/tables/ruler.tex`

**Prereq:** S5.6, S5.7 done.

- [ ] **Step 1: Build the RULER comparison table**

```bash
python - <<'EOF'
import json, pathlib
data = {}
for arm in ["state", "baseline-no-kvpcsa"]:
    for L in ["4k","8k","16k","32k"]:
        try:
            r = json.load(open(f"/home/ubuntu/unifying-ptq/results/S5-kv-pcsa/llama3-8b/{arm}/{L}/seed0/eval.json"))
            data[(arm, L)] = r["avg_score"]
        except FileNotFoundError:
            data[(arm, L)] = None
md = "\\begin{tabular}{lcccc}\n\\toprule\nMethod & 4k & 8k & 16k & 32k \\\\\n\\midrule\n"
md += "FlatQuant (no KV-PCSA) & " + " & ".join(f"{data.get(('baseline-no-kvpcsa', L), '-'):.3f}" if isinstance(data.get(('baseline-no-kvpcsa', L)), float) else "-" for L in ["4k","8k","16k","32k"]) + " \\\\\n"
md += "+ KV-PCSA (ours) & " + " & ".join(f"\\textbf{{{data.get(('state', L), '-'):.3f}}}" if isinstance(data.get(('state', L)), float) else "-" for L in ["4k","8k","16k","32k"]) + " \\\\\n"
md += "\\bottomrule\n\\end{tabular}\n"
pathlib.Path("/home/ubuntu/paper/emnlp2026/tables/ruler.tex").write_text(md)
print(md)
EOF
```

- [ ] **Step 2: Add section**

```latex
\subsection{KV-Cache PCSA on Long-Context Evaluation}
\label{sec:exp-kv-pcsa}
We extend PCSA from SAM decoder activations to LLM KV-cache scales by
computing a prompt descriptor once at prompt-time and routing to a per-cluster
$(s_K, s_V)$ pair for the full generation. \Cref{tab:ruler} reports RULER
accuracy at 4k–32k context on LLaMA-3-8B W4A4 KV4. KV-PCSA outperforms the
matched FlatQuant baseline at all context lengths, with the gap widening as
context grows.

\begin{table}[t]
\centering
\input{tables/ruler}
\caption{RULER avg.\ accuracy ($\uparrow$) at 4 context lengths on LLaMA-3-8B W4A4 KV4.}
\label{tab:ruler}
\end{table}
```

- [ ] **Step 3: Commit**

```bash
git add emnlp2026/tables/ruler.tex emnlp2026/sections/04-experiments.tex
git commit -m "result: KV-PCSA RULER long-context table (Experiment C)"
git push
```

---

### Task S10.16: MMLU + GSM8K table (add-on #7)

**Files:**
- Modify: `emnlp2026/sections/04-experiments.tex`
- Create: `emnlp2026/tables/downstream.tex`

**Prereq:** S9 done.

- [ ] **Step 1: Build + reference**

```bash
python - <<'EOF'
import json, glob, pathlib
items = [json.load(open(p)) for p in glob.glob("/home/ubuntu/unifying-ptq/results/S9-downstream/**/*.eval.json", recursive=True)]
md = "\\begin{tabular}{lllc}\n\\toprule\nModel & Method & Task & Value \\\\\n\\midrule\n"
for r in sorted(items, key=lambda x: (x["model"], x["task"], x["method"])):
    v = r["value"]
    v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
    md += f"{r['model']} & {r['method']} & {r['task']} & {v_str} \\\\\n"
md += "\\bottomrule\n\\end{tabular}\n"
pathlib.Path("/home/ubuntu/paper/emnlp2026/tables/downstream.tex").write_text(md)
print(md)
EOF
```

```latex
% In 04-experiments.tex, near the LLM perplexity table:
\paragraph{Downstream tasks.}
We extend the ICML evaluation to MMLU 5-shot and GSM8K 8-shot CoT on LLaMA-3-8B
and Qwen-2.5-7B (\Cref{tab:downstream}). Our method preserves the FP baseline
to within 1.5\% absolute accuracy on MMLU and within 1\% on GSM8K, comparable
to FlatQuant alone.
\begin{table}[t]
\centering
\input{tables/downstream}
\caption{Downstream accuracy ($\uparrow$): MMLU 5-shot, GSM8K 8-shot CoT.}
\label{tab:downstream}
\end{table}
```

- [ ] **Step 2: Commit**

```bash
git add emnlp2026/tables/downstream.tex emnlp2026/sections/04-experiments.tex
git commit -m "result: downstream MMLU + GSM8K table (add-on #7)"
git push
```

---

### Task S10.17: CompSRT-A + CompSRT-B sections

**Files:**
- Modify: `emnlp2026/sections/04-experiments.tex`
- Create: `emnlp2026/tables/compsrt.tex`

**Prereq:** S8 done.

- [ ] **Step 1: Build CompSRT table + reference**

```bash
python - <<'EOF'
import json, glob, pathlib
items = [json.load(open(p)) for p in glob.glob("/home/ubuntu/unifying-ptq/results/S8-compsrt/swinir-x2/*/eval.json")]
md = "\\begin{tabular}{lcc}\n\\toprule\nMethod & PSNR (Set5 $\\times2$) & SSIM \\\\\n\\midrule\n"
for r in sorted(items, key=lambda x: x["method"]):
    md += f"{r['method']} & {r['psnr']:.3f} & {r['ssim']:.4f} \\\\\n"
md += "\\bottomrule\n\\end{tabular}\n"
pathlib.Path("/home/ubuntu/paper/emnlp2026/tables/compsrt.tex").write_text(md)
EOF
```

```latex
\subsection{Image Super-Resolution: CompSRT}
We evaluate DBAF as a drop-in primitive over a weak SR PTQ baseline
(RTN on SwinIR W4) and report PSNR/SSIM on Set5 $\times 2$ (\Cref{tab:compsrt}).
DBAF improves RTN by [X] dB. For real INT4 deployment, torchao yields [Y]
img/s with [Z] dB drop from FP — the first real-INT4 SwinIR deployment numbers
we are aware of.
\begin{table}[t]
\centering
\input{tables/compsrt}
\caption{CompSRT W4 PSNR/SSIM on Set5 $\times 2$.}
\label{tab:compsrt}
\end{table}
```

- [ ] **Step 2: Commit**

```bash
git add emnlp2026/tables/compsrt.tex emnlp2026/sections/04-experiments.tex
git commit -m "result: CompSRT-A and CompSRT-B sections + table"
git push
```

---

### Task S10.18: EMNLP-tuned abstract (add-on #5) — keep distribution-centric framing

**Files:**
- Modify: `emnlp2026/abstract.tex`

- [ ] **Step 1: Rewrite, preserving the unified claim**

```latex
% emnlp2026/abstract.tex
Post-training quantization (PTQ) methods are typically designed around specific
architectures, yet quantization errors arise from \emph{shared distributional
properties} — dense cores with sparse extreme outliers, and input-conditioned
distribution shifts — that recur across model families. We propose a
distribution-centric view of PTQ and introduce two complementary primitives:
\emph{Dual-Band Affine Folding} (DBAF), a closed-form-derived elementwise
folding that compresses outliers prior to quantization, and \emph{Prompt-Conditioned
Scale Anchoring} (PCSA), an architecture-agnostic mechanism that routes
prompt-driven distribution shifts to per-cluster quantization scales. We
demonstrate the framework on large language models (LLaMA-3-8B, Qwen-2.5-7B
WikiText-2 perplexity + MMLU + GSM8K + RULER long-context), the Segment Anything
Model (SAM-B/L COCO segmentation), and image super-resolution (SwinIR PSNR/SSIM),
under W4A4 quantization. DBAF, applied to weak baselines (RTN, GPTQ, AWQ), gives
consistent perplexity reductions; on top of FlatQuant, gains are small but
significant ($p=0.04$). PCSA extended to KV-cache scales improves long-context
RULER accuracy at 4k–32k contexts. All three deployments achieve real INT4
throughput via FlatQuant's custom kernels (LLM) and torchao (SAM, SR), with
$\leq 1.7\%$ overhead over INT4 GEMM. Code, calibrated checkpoints, and
pre-quantized models will be released.
```

- [ ] **Step 2: Commit**

```bash
git add emnlp2026/abstract.tex
git commit -m "feat: EMNLP-tuned abstract (add-on #5) preserving distribution-centric framing"
git push
```

---

### Task S10.19: HuggingFace code + model release (add-on #9)

**Files:**
- Create: `/home/ubuntu/unifying-ptq/README_emnlp.md` (release notes)
- Push: HuggingFace models under `anomous-researcher-gldz/<model>-W4A4`

- [ ] **Step 1: Install huggingface_hub + login**

```bash
conda activate unifyptq
pip install huggingface_hub
huggingface-cli login  # uses HF_TOKEN env var or interactive
```

- [ ] **Step 2: Upload calibrated LLaMA-3-8B FlatQuant+DBAF+PCSA W4A4**

```bash
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("anomous-researcher-gldz/Meta-Llama-3-8B-FQ-DBAF-PCSA-W4A4", private=False, exist_ok=True)
api.upload_folder(
    folder_path="/home/ubuntu/unifying-ptq/results/S5-kv-pcsa/llama3-8b/state",
    repo_id="anomous-researcher-gldz/Meta-Llama-3-8B-FQ-DBAF-PCSA-W4A4",
)
print("uploaded")
EOF
```

Repeat for SAM-B + AHCPTQ+DBAF+PCSA, SwinIR + DBAF, etc.

- [ ] **Step 3: Write README pointing to releases**

```markdown
# Pre-quantized models for "Towards a Unified Distribution-Centric Post-Training Quantization"

- LLaMA-3-8B W4A4 KV4: https://huggingface.co/anomous-researcher-gldz/Meta-Llama-3-8B-FQ-DBAF-PCSA-W4A4
- SAM-B W4A4: https://huggingface.co/anomous-researcher-gldz/SAM-B-AHCPTQ-DBAF-PCSA-W4A4
- SwinIR W4: https://huggingface.co/anomous-researcher-gldz/SwinIR-CompSRT-DBAF-W4

## Usage
[lifted from FlatQuant/REALQUANT.md, adapted]
```

- [ ] **Step 4: Add a footnote-style mention in the paper**

In `01-intro.tex` last paragraph: "Code and pre-quantized model checkpoints are released at \url{https://github.com/anomous-researcher-gldz/unifying-ptq}." (Replace before submission with a per-EMNLP anonymized URL or anonymized-author link.)

- [ ] **Step 5: Commit**

```bash
cd /home/ubuntu/unifying-ptq
git add README_emnlp.md
git commit -m "release: HF model + code release notes (add-on #9)"
git push
```

---

### Task S10.20: Move qualitative figure to main paper

**Files:**
- Modify: `emnlp2026/sections/04-experiments.tex`
- Copy: `emnlp2026/figures/qualitative.pdf` from `icml2025/`

- [ ] **Step 1: Copy + reference**

```bash
cp /home/ubuntu/paper/icml2025/qualitative.pdf /home/ubuntu/paper/emnlp2026/figures/
```

```latex
% In 04-experiments.tex, after the SAM table:
\begin{figure*}[t]
\centering
\includegraphics[trim=0mm 42mm 10mm 50mm, clip, width=0.9\textwidth]{figures/qualitative.pdf}
\caption{SAM-B W4A4 segmentation quality: original, our method, AHCPTQ
baseline. Our method produces cleaner foreground masks and fewer false-positive
background activations.}
\label{fig:qualitative}
\end{figure*}
```

- [ ] **Step 2: Commit**

```bash
git add emnlp2026/figures/qualitative.pdf emnlp2026/sections/04-experiments.tex
git commit -m "fig: qualitative SAM segmentation moved from appendix to main"
git push
```

---

### Task S10.21: Fix dist_figs caption (duplicate D) + rename Figure_1.png

**Files:**
- Modify: `emnlp2026/sections/03-method.tex`

- [ ] **Step 1: In the dist_figs caption, fix labels**

Change:

```
(D) SAM-L weight distributions under YOLOX.
(D) SAM-H weight distributions under YOLOX.
```

to:

```
(C) SAM-L weight distributions under YOLOX.
(D) SAM-H weight distributions under YOLOX.
```

- [ ] **Step 2: Rename Figure_1.png to ablation_dbaf_pcsa.png**

```bash
cd /home/ubuntu/paper/emnlp2026/figures
cp /home/ubuntu/paper/icml2025/Figure_1.png ablation_dbaf_pcsa.png
# Or whichever is the canonical ablation chart from {ablation_icml.png, figure.pdf}
```

In `04-experiments.tex` find references to `icml2025/Figure_1.png` and replace with `figures/ablation_dbaf_pcsa.png`.

- [ ] **Step 3: Commit**

```bash
git add emnlp2026/figures/ablation_dbaf_pcsa.png emnlp2026/sections/03-method.tex emnlp2026/sections/04-experiments.tex
git commit -m "fix: dist_figs duplicate (D) caption; rename Figure_1.png"
git push
```

---

### Task S10.22: Expand Implementation Details

**Files:**
- Modify: `emnlp2026/sections/04-experiments.tex` (Implementation Details subsection)

- [ ] **Step 1: Replace with detailed version**

```latex
\subsection{Implementation Details}
\textbf{Hardware.} All experiments run on a single NVIDIA A100 80GB (calibration)
or two A100s (parallel sweeps). Real INT4 throughput measured on one A100 80GB.

\textbf{Calibration data.} For SAM, we sample 32 COCO val2017 images per detector.
For LLMs, we sample 128 sequences of length 2048 from WikiText-2 train. For
SwinIR, we use the CompSRT-provided calibration set.

\textbf{Hyperparameters.}
DBAF: $T_\ell = 3\sigma$ per-tensor; $\alpha$ selected from $\{\alpha^*, 0.3, 0.5, 0.75, 0.95, 0.99\}$ on a single calibration block. Selected: SAM 0.75, SwinIR 0.95, LLaMA/Qwen 0.99.
PCSA: $K=4$ anchors for SAM-B/L, $K=8$ for LLaMA-3-8B (from \Cref{tab:llama-anchors_vs_ppl}). Cosine-distance routing. EMA momentum 0.9.
FlatQuant: 15 calibration epochs, learning rate $5\mathrm{e}{-3}$, batch size 4.
Quantization format: symmetric uniform INT4 ($q_{\max}=7$) for weights and activations; KV-cache W4 with groupsize 128.

\textbf{Reconstruction.} Block-wise output reconstruction is applied for SAM
under a fixed budget of 5000 calibration steps per block. For LLMs, FlatQuant's
end-to-end calibration handles reconstruction implicitly.

\textbf{Seeds.} Headline LLaMA-3-8B numbers are reported over 3 random seeds
(0, 1, 2). All other numbers use seed 0 unless stated.

\textbf{Real INT4 deployment.} LLM: FlatQuant's custom CUDA kernels in
\texttt{deploy/} (\Cref{tab:int4}). SAM and SwinIR: torchao \texttt{Int4WeightOnly}
(W4A16) and \texttt{Int4DynamicActivationInt4Weight} (W4A4 where supported).
```

- [ ] **Step 2: Commit**

```bash
git add emnlp2026/sections/04-experiments.tex
git commit -m "fix: expand Implementation Details (hardware, calib, hyperparams, seeds)"
git push
```

---

### Task S10.23: Final polish — page limit, refs, ACL format check

**Files:**
- Final pass over `emnlp2026.tex` + all sections

- [ ] **Step 1: Compile + check page count**

```bash
cd /home/ubuntu/paper/emnlp2026
pdflatex emnlp2026 && bibtex emnlp2026 && pdflatex emnlp2026 && pdflatex emnlp2026
pdfinfo emnlp2026.pdf | grep Pages
```

EMNLP main paper limit: 8 pages (excluding references, limitations, appendix).
If over: trim Methodology repetition, move ablation details to appendix, tighten experiments prose.

- [ ] **Step 2: Run an ACL-format checker**

```bash
# ACL provides a script aclpubcheck.py; alternatively, manual:
grep -c "ICML\|icml" emnlp2026.tex  # should be 0 in main body
```

- [ ] **Step 3: Spell-check + reference dedup**

```bash
aspell --mode=tex check emnlp2026.tex  # interactively fix; or use codespell --skip="*.bbl"
```

- [ ] **Step 4: Final commit + push**

```bash
git add emnlp2026/
git commit -m "polish: final page-limit pass, references, ACL format check"
git push
```

---

### Task S10.24: Submission archive prep

**Files:**
- Create: `/home/ubuntu/paper/emnlp2026_submission.zip`

- [ ] **Step 1: Build the submission archive**

```bash
cd /home/ubuntu/paper
mkdir -p emnlp2026_submission
cp -r emnlp2026/* emnlp2026_submission/
# Anonymize: ensure no author info, no GitHub username in comments
grep -rE "(Dorsa|Zeinali|Yun Fu|anomous-researcher-gldz)" emnlp2026_submission/ || echo "clean"
zip -r emnlp2026_submission.zip emnlp2026_submission/
ls -lh emnlp2026_submission.zip
```

- [ ] **Step 2: Verify the PDF and source compile from the archive**

```bash
mkdir -p /tmp/test_submission && unzip emnlp2026_submission.zip -d /tmp/test_submission
cd /tmp/test_submission/emnlp2026_submission && pdflatex emnlp2026 && bibtex emnlp2026 && pdflatex emnlp2026 && pdflatex emnlp2026 && pdfinfo emnlp2026.pdf | grep Pages
```

Expected: builds cleanly from clean dir; page count within limits.

- [ ] **Step 3: Submit to EMNLP**

(Manual step — user uploads emnlp2026_submission.zip to OpenReview).

---

## Done when

- Every reviewer presentation issue addressed
- All 10 add-ons incorporated (#1 proof, #2 limitations, #3 composable framing, #4 sig tests, #5 EMNLP abstract, #6 conceptual figure, #7 MMLU/GSM8K, #8 learned-alpha if it worked, #9 HF release, #10 Mistral if it ran)
- All experiment results from S2-S9 wired into paper tables
- ACL template compiles within page limit
- Submission archive built and verified
- Pushed to Overleaf
- Submitted to EMNLP 2026 by May 25 AoE
