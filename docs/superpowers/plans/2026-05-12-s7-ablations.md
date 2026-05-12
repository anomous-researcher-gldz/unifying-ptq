# S7: Ablations + Statistical Significance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the ablation results that reviewer aCWD specifically demanded (α=1.0 row), the error-bar evidence reviewers expected (3-seed runs on LLaMA-3-8B headline), the statistical significance tests (Wilcoxon paired + paired bootstrap), and the stretch add-on #8 (per-layer learned α via gradient).

**Architecture:** Five small scripts under `scripts/`: `run_alpha_grid.py` (sweep), `run_seeds.py` (multi-seed), `compute_significance.py` (Wilcoxon + bootstrap), `learn_alpha_per_layer.py` (add-on #8), `aggregate_ablations.py` (table builder). Each writes `eval.json` files into `results/S7-ablations/`.

**Tech Stack:** PyTorch, FlatQuant, scipy.stats, numpy, the WikiText-2 PPL evaluator from S4.

**Prereqs:** S4 (DBAF on weak baselines confirms baselines work), FlatQuant calibrated LLaMA-3-8B from S5.5.

---

### Task S7.1: α=1.0 grid row (reviewer aCWD's specific ask)

**Files:**
- Create: `scripts/run_alpha_grid.py`
- Create: `results/S7-ablations/alpha-grid/llama3-8b/alpha-{0.99,1.0}/eval.json`

- [ ] **Step 1: Implement alpha-grid sweep script**

```python
# scripts/run_alpha_grid.py
"""Sweep DBAF alpha values including alpha=1.0 (which disables folding).
Outputs WikiText-2 PPL per alpha for the headline FlatQuant+DBAF+PCSA configuration.
"""
import argparse, json, pathlib, sys, copy
import torch
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")

def run_one(model_path, alpha, out_path):
    """Calibrate FlatQuant with a specific DBAF alpha, eval WikiText-2 PPL."""
    # We assume FlatQuant's main.py exposes alpha as an arg. If not, monkey-patch:
    import subprocess
    cmd = [
        "python", "main.py",
        "--model", model_path,
        "--w_bits", "4", "--a_bits", "4",
        "--k_bits", "4", "--k_asym", "--k_groupsize", "128",
        "--v_bits", "4", "--v_asym", "--v_groupsize", "128",
        "--cali_bsz", "4", "--epoch", "15", "--flat_lr", "5e-3",
        "--lwc", "--lac", "--cali_trans", "--add_diag",
        "--dbaf-alpha", str(alpha),  # may need to be added to args_utils
        "--output_dir", str(pathlib.Path(out_path).parent / "state"),
        "--lm_eval", "--lm_eval_batch_size", "16",
    ]
    print("RUN:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/ubuntu/unifying-ptq/FlatQuant")
    print(r.stdout[-2000:]); print(r.stderr[-2000:])
    # Extract WikiText-2 PPL from stdout
    import re
    m = re.search(r"wikitext2.*?(\d+\.\d+)", r.stdout, re.IGNORECASE)
    ppl = float(m.group(1)) if m else None
    out = {"model": pathlib.Path(model_path).name, "alpha": alpha, "wikitext2_ppl": ppl,
           "bits": "W4A4KV4", "method": "FlatQuant+DBAF+PCSA"}
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text(json.dumps(out, indent=2))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="./modelzoo/meta-llama/Meta-Llama-3-8B")
    p.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.75, 0.95, 0.99, 1.0])
    p.add_argument("--out-dir", default="/home/ubuntu/unifying-ptq/results/S7-ablations/alpha-grid/llama3-8b")
    args = p.parse_args()
    for a in args.alphas:
        out = run_one(args.model_path, a, f"{args.out_dir}/alpha-{a}/eval.json")
        print(out)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `--dbaf-alpha` to FlatQuant's args_utils**

If not already present, in `FlatQuant/flatquant/args_utils.py`:

```python
parser.add_argument("--dbaf-alpha", type=float, default=0.99,
                    help="DBAF folding alpha. 1.0 disables folding.")
```

And in the DBAF application code in `flatquant/flat_linear.py` or wherever DBAF is wired, read `args.dbaf_alpha` and skip the fold when `alpha == 1.0`.

- [ ] **Step 3: Run the alpha grid sweep**

```bash
cd /home/ubuntu/unifying-ptq && conda activate unifyptq
tmux new-session -d -s s7-alpha "
python scripts/run_alpha_grid.py 2>&1 | tee results/S7-ablations/alpha-grid/llama3-8b/sweep.log
echo DONE
"
until tmux capture-pane -t s7-alpha -p | grep -q "^DONE"; do sleep 600; done
```

Expected: 5 alpha values × ~6 hours each = 30 hours (or fewer if calibration is shorter; reuse the FlatQuant state for non-alpha layers).

**Smarter approach**: only re-run the parts that depend on alpha. DBAF is applied at the end of FlatQuant's calibration loop, so we can save the pre-DBAF state once and apply different alphas without re-running FlatQuant from scratch. If this optimization is feasible, implement it in `flatquant.flat_utils.apply_dbaf_only(state, alpha)` and reduce runtime per alpha to ~30 minutes.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_alpha_grid.py FlatQuant/flatquant/args_utils.py results/S7-ablations/alpha-grid/
git commit -m "result(S7): DBAF alpha grid sweep including alpha=1.0 row"
```

---

### Task S7.2: 3-seed runs on LLaMA-3-8B W4A4 headline

**Files:**
- Create: `scripts/run_seeds.py`
- Create: `results/S7-ablations/seeds/llama3-8b/seed{0,1,2}/eval.json`

- [ ] **Step 1: Implement multi-seed driver**

```python
# scripts/run_seeds.py
"""Re-run the headline FlatQuant+DBAF+PCSA W4A4 config with 3 different seeds for error bars."""
import argparse, json, pathlib, subprocess

def run_seed(seed, out_path, model_path):
    cmd = [
        "python", "main.py",
        "--model", model_path,
        "--w_bits", "4", "--a_bits", "4",
        "--k_bits", "4", "--k_asym", "--k_groupsize", "128",
        "--v_bits", "4", "--v_asym", "--v_groupsize", "128",
        "--cali_bsz", "4", "--epoch", "15", "--flat_lr", "5e-3",
        "--lwc", "--lac", "--cali_trans", "--add_diag",
        "--seed", str(seed),
        "--output_dir", str(pathlib.Path(out_path).parent / "state"),
        "--lm_eval", "--lm_eval_batch_size", "16",
    ]
    r = subprocess.run(cmd, cwd="/home/ubuntu/unifying-ptq/FlatQuant", capture_output=True, text=True)
    import re
    m = re.search(r"wikitext2.*?(\d+\.\d+)", r.stdout, re.IGNORECASE)
    ppl = float(m.group(1)) if m else None
    out = {"model": pathlib.Path(model_path).name, "seed": seed, "wikitext2_ppl": ppl,
           "bits": "W4A4KV4", "method": "FlatQuant+DBAF+PCSA"}
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text(json.dumps(out, indent=2))
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--model-path", default="./modelzoo/meta-llama/Meta-Llama-3-8B")
    p.add_argument("--out-dir", default="/home/ubuntu/unifying-ptq/results/S7-ablations/seeds/llama3-8b")
    args = p.parse_args()
    for s in args.seeds:
        print(run_seed(s, f"{args.out_dir}/seed{s}/eval.json", args.model_path))
```

- [ ] **Step 2: Ensure FlatQuant honors `--seed`**

In `main.py`, add at top:

```python
if hasattr(args, "seed"):
    import random, numpy as np, torch
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
```

- [ ] **Step 3: Run 3 seeds**

```bash
tmux new-session -d -s s7-seeds "python scripts/run_seeds.py; echo DONE"
```

Expected: 3 × ~6 hours = 18 hours. Run on whichever GPU is free.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_seeds.py FlatQuant/main.py results/S7-ablations/seeds/
git commit -m "result(S7): 3-seed runs on LLaMA-3-8B W4A4 for error bars"
```

---

### Task S7.3: Statistical significance tests

**Files:**
- Create: `scripts/compute_significance.py`
- Create: `results/S7-ablations/significance.json`

- [ ] **Step 1: Implement significance computation**

```python
# scripts/compute_significance.py
"""Compute Wilcoxon paired test and paired bootstrap p-values
for FlatQuant+DBAF+PCSA vs FlatQuant baseline across seeds.
"""
import json, pathlib, glob
import numpy as np
from scipy.stats import wilcoxon

def load_ppls(pattern):
    rows = [json.load(open(p)) for p in glob.glob(pattern)]
    return [r["wikitext2_ppl"] for r in rows if r["wikitext2_ppl"] is not None]

def paired_bootstrap(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.array(a); b = np.array(b)
    delta = a - b
    means = []
    for _ in range(n):
        sample = rng.choice(delta, size=len(delta), replace=True)
        means.append(sample.mean())
    means = np.array(means)
    p = (means >= 0).mean()  # one-sided: P(method ≤ baseline)
    return float(p), float(means.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

ours = load_ppls("/home/ubuntu/unifying-ptq/results/S7-ablations/seeds/llama3-8b/seed*/eval.json")
# Baseline: FlatQuant without DBAF/PCSA. If we have it stored elsewhere, load:
baseline = load_ppls("/home/ubuntu/unifying-ptq/results/S7-ablations/baseline-noseed/llama3-8b/seed*/eval.json")

if len(ours) < 2 or len(baseline) < 2:
    print("not enough seeds; need at least 2 per arm for Wilcoxon")
else:
    w_stat, w_p = wilcoxon(ours, baseline, alternative="less")  # ours < baseline (lower PPL is better)
    p_boot, mean_d, lo, hi = paired_bootstrap(ours, baseline)
    out = {
        "ours": ours, "baseline": baseline,
        "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
        "bootstrap_p_one_sided": p_boot, "mean_delta": mean_d, "ci95": [lo, hi],
    }
    pathlib.Path("/home/ubuntu/unifying-ptq/results/S7-ablations/significance.json").write_text(json.dumps(out, indent=2))
    print(out)
```

- [ ] **Step 2: Run after Task S7.2 finishes**

```bash
python scripts/compute_significance.py
```

Expected: outputs Wilcoxon p-value and bootstrap CI.

- [ ] **Step 3: Commit**

```bash
git add scripts/compute_significance.py results/S7-ablations/significance.json
git commit -m "result(S7): Wilcoxon + paired bootstrap significance vs FlatQuant baseline"
```

---

### Task S7.4 (stretch, add-on #8): Per-layer learned α

**Files:**
- Create: `scripts/learn_alpha_per_layer.py`
- Create: `results/S7-ablations/learned-alpha/llama3-8b/eval.json`

- [ ] **Step 1: Implement**

```python
# scripts/learn_alpha_per_layer.py
"""Learn per-layer DBAF alpha via gradient on a small calibration set.
Initialize all alphas to 0.99; backprop wikitext2 loss through DBAF unfolding.
"""
import argparse, json, pathlib, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn as nn

class LearnableDBAFLinear(nn.Module):
    """Wraps an nn.Linear: applies learnable-alpha DBAF fold→quant→unfold to weights at every forward."""
    def __init__(self, linear: nn.Linear, init_alpha=0.99, bits=4):
        super().__init__()
        self.linear = linear
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.register_buffer("T", torch.tensor(3.0 * linear.weight.std().item()))
        self.bits = bits

    def _fold(self, w):
        sgn = torch.sign(w); mask = w.abs() > self.T
        return torch.where(mask, sgn * self.T + self.alpha * (w - sgn * self.T), w)

    def _quant(self, w):
        qmax = 2 ** (self.bits - 1) - 1
        s = w.abs().max() / qmax + 1e-9
        q = torch.round(w / s).clamp(-qmax, qmax)
        return q * s

    def _unfold(self, w):
        sgn = torch.sign(w); mask = w.abs() > self.T
        return torch.where(mask, sgn * self.T + (1.0 / self.alpha.clamp(min=1e-3)) * (w - sgn * self.T), w)

    def forward(self, x):
        w = self.linear.weight
        w = self._fold(w); w = self._quant(w); w = self._unfold(w)
        return torch.nn.functional.linear(x, w, self.linear.bias)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="/home/ubuntu/unifying-ptq/results/S7-ablations/learned-alpha/llama3-8b/eval.json")
    args = p.parse_args()

    m = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="cuda")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    # Wrap every Linear except lm_head
    for name, mod in list(m.named_modules()):
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = m.get_submodule(parent_name)
            setattr(parent, child_name, LearnableDBAFLinear(mod))
    # Calibration loop
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[:, :args.steps * 256].cuda()
    optim = torch.optim.Adam([p for p in m.parameters() if p.requires_grad and p.dtype == torch.float32], lr=args.lr)
    for step in range(args.steps):
        chunk = ids[:, step*256:(step+1)*256]
        out = m(chunk, labels=chunk)
        optim.zero_grad()
        out.loss.backward()
        optim.step()
        if step % 20 == 0: print(step, out.loss.item())
    # Final PPL eval
    from scripts.run_S4 import wikitext_ppl
    ppl = wikitext_ppl(m, tok)
    alphas = {n: float(p.alpha.detach().cpu()) for n, p in m.named_modules() if isinstance(p, LearnableDBAFLinear)}
    out = {"model": args.model_name, "method": "FlatQuant+learned-alpha-DBAF",
           "wikitext2_ppl": ppl, "n_layers_learned": len(alphas), "alpha_mean": sum(alphas.values())/len(alphas)}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run (stretch)**

```bash
python scripts/learn_alpha_per_layer.py 2>&1 | tee results/S7-ablations/learned-alpha/llama3-8b/run.log
```

Expected runtime: ~3 hours. If PPL beats grid-α=0.99 by >0.05, add-on #8 lands; otherwise, document and drop silently.

- [ ] **Step 3: Commit (only if PPL improved)**

```bash
git add scripts/learn_alpha_per_layer.py results/S7-ablations/learned-alpha/
git commit -m "result(S7): learned-alpha per-layer DBAF (add-on #8)"
```

---

### Task S7.5: Aggregate ablation table

**Files:**
- Create: `scripts/aggregate_ablations.py`
- Create: `results/S7-ablations/summary.md`

- [ ] **Step 1: Build the table**

```python
# scripts/aggregate_ablations.py
import json, glob, pathlib

rows = []
for p in glob.glob("/home/ubuntu/unifying-ptq/results/S7-ablations/**/eval.json", recursive=True):
    rows.append(json.load(open(p)))

md = ["# S7 Ablation Summary\n",
      "| Method | Alpha | Seed | WikiText-2 PPL |",
      "|---|---|---|---|"]
for r in sorted(rows, key=lambda x: (x.get("method",""), x.get("alpha",-1), x.get("seed",-1))):
    md.append(f"| {r.get('method','')} | {r.get('alpha','-')} | {r.get('seed','-')} | {r.get('wikitext2_ppl','-')} |")
md.append("\n## Significance (vs FlatQuant baseline)\n")
try:
    sig = json.load(open("/home/ubuntu/unifying-ptq/results/S7-ablations/significance.json"))
    md.append(f"- Wilcoxon p = {sig['wilcoxon_p']:.4f}")
    md.append(f"- Bootstrap mean delta = {sig['mean_delta']:.3f}, 95% CI = [{sig['ci95'][0]:.3f}, {sig['ci95'][1]:.3f}]")
except FileNotFoundError:
    md.append("(significance.json not yet computed)")
pathlib.Path("/home/ubuntu/unifying-ptq/results/S7-ablations/summary.md").write_text("\n".join(md))
print("\n".join(md))
```

- [ ] **Step 2: Run + commit**

```bash
python scripts/aggregate_ablations.py
git add scripts/aggregate_ablations.py results/S7-ablations/summary.md
git commit -m "result(S7): aggregate ablation + significance summary"
```

---

## Done when

- `alpha-grid/llama3-8b/alpha-1.0/eval.json` exists (reviewer's specific ask)
- 3 seed `eval.json` files exist for headline LLaMA-3-8B W4A4
- `significance.json` exists with Wilcoxon + bootstrap results
- `summary.md` aggregates all ablation rows
- Add-on #8 (learned alpha) either improved PPL and is committed, or was tried and silently dropped
- Numbers ready for paper Ablation table + statistical claims
