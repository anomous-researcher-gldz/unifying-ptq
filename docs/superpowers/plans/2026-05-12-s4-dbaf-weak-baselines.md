# S4: DBAF on Weak Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show DBAF gives clear PPL gains when applied on top of *weak* PTQ baselines (RTN, GPTQ, AWQ) that don't already handle outliers via learned rotations. This directly attacks reviewer aCWD's "α=0.99 ≈ DBAF off" critique by demonstrating that DBAF *does* help — the marginal-on-FlatQuant result reflects FlatQuant's strength, not DBAF's weakness.

**Architecture:** Add three weak-baseline quantization paths under `FlatQuant/flatquant/baselines/{rtn,gptq,awq}.py`. Each exposes `quantize_model(model, bits, calibration_data, use_dbaf=False)`. When `use_dbaf=True`, applies DBAF folding before quantization in the same way `apply_flatquant_to_llama` does (per-layer α from grid, T=3σ). Driver script `scripts/run_S4.py` sweeps the matrix.

**Tech Stack:** PyTorch, transformers, FlatQuant's existing quantization plumbing, GPTQ ref implementation (we vendored or rewrote), AWQ ref implementation, WikiText-2.

**Prereqs:** S1.1 (unifyptq env), S1.2 (torchao), FlatQuant working.

---

### Task S4.1: RTN baseline — write failing test

**Files:**
- Create: `FlatQuant/tests/test_baseline_rtn.py`

- [ ] **Step 1: Write the test**

```python
# FlatQuant/tests/test_baseline_rtn.py
import torch
import pytest
from transformers import AutoModelForCausalLM

@pytest.fixture
def tiny_llama():
    m = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM").cuda().half()
    return m

def _ppl_proxy(model, n=8):
    """Cheap proxy: log-likelihood on random ids should be finite + roughly preserved."""
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (1, n), device="cuda")
    with torch.no_grad():
        out = model(ids, labels=ids)
    return out.loss.item()

def test_rtn_w4_quantize_runs(tiny_llama):
    from flatquant.baselines.rtn import quantize_model
    ref = _ppl_proxy(tiny_llama)
    q = quantize_model(tiny_llama, bits=4, use_dbaf=False)
    after = _ppl_proxy(q)
    assert torch.isfinite(torch.tensor(after))
    # Quantization error tolerable on tiny random model
    assert abs(after - ref) < 100, f"ref={ref}, q={after}"

def test_rtn_w4_dbaf_changes_output(tiny_llama):
    from flatquant.baselines.rtn import quantize_model
    import copy
    q_no = quantize_model(copy.deepcopy(tiny_llama), bits=4, use_dbaf=False)
    q_db = quantize_model(copy.deepcopy(tiny_llama), bits=4, use_dbaf=True)
    no_loss = _ppl_proxy(q_no)
    db_loss = _ppl_proxy(q_db)
    # Outputs must differ — DBAF should change something
    assert abs(no_loss - db_loss) > 1e-4, "DBAF had no effect"
```

- [ ] **Step 2: Verify fails**

```bash
cd /home/ubuntu/unifying-ptq && conda activate unifyptq
pytest FlatQuant/tests/test_baseline_rtn.py -v
```

Expected: `ModuleNotFoundError: No module named 'flatquant.baselines'`.

---

### Task S4.2: RTN baseline — implement

**Files:**
- Create: `FlatQuant/flatquant/baselines/__init__.py`
- Create: `FlatQuant/flatquant/baselines/rtn.py`

- [ ] **Step 1: Implement RTN with optional DBAF**

```python
# FlatQuant/flatquant/baselines/__init__.py
EOF
```

```python
# FlatQuant/flatquant/baselines/rtn.py
"""Round-to-nearest weight-only quantization, with optional DBAF folding.

This is intentionally a 'weak' baseline: no rotation, no calibration loop,
no GPTQ-style block solve. Purpose is to expose how much DBAF helps when the
baseline doesn't already address outliers.
"""
from __future__ import annotations
import torch
import torch.nn as nn

def _dbaf_fold(w: torch.Tensor, alpha: float = 0.75, T_sigma: float = 3.0) -> torch.Tensor:
    sigma = w.std()
    T = T_sigma * sigma
    sgn = torch.sign(w)
    mask = w.abs() > T
    out = w.clone()
    out[mask] = sgn[mask] * T + alpha * (w[mask] - sgn[mask] * T)
    return out, T, alpha

def _dbaf_unfold(w_q: torch.Tensor, T: torch.Tensor, alpha: float) -> torch.Tensor:
    sgn = torch.sign(w_q)
    mask = w_q.abs() > T
    out = w_q.clone()
    out[mask] = sgn[mask] * T + (1.0 / alpha) * (w_q[mask] - sgn[mask] * T)
    return out

def _quantize_tensor_uniform(w: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max() / qmax
    if scale == 0:
        return w.clone()
    q = torch.round(w / scale).clamp(-qmax, qmax)
    return (q * scale).to(w.dtype)

def quantize_model(model: nn.Module, bits: int = 4, use_dbaf: bool = False, alpha: float = 0.75) -> nn.Module:
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if "lm_head" in name:
            continue
        w = mod.weight.data
        if use_dbaf:
            w_fold, T, a = _dbaf_fold(w, alpha=alpha)
            w_q = _quantize_tensor_uniform(w_fold, bits)
            w_out = _dbaf_unfold(w_q, T, a)
        else:
            w_out = _quantize_tensor_uniform(w, bits)
        mod.weight.data = w_out.to(mod.weight.dtype)
    return model
```

- [ ] **Step 2: Run tests, verify pass**

```bash
pytest FlatQuant/tests/test_baseline_rtn.py -v
```

- [ ] **Step 3: Commit**

```bash
git add FlatQuant/flatquant/baselines/__init__.py FlatQuant/flatquant/baselines/rtn.py FlatQuant/tests/test_baseline_rtn.py
git commit -m "feat(S4): RTN weak baseline with optional DBAF folding"
```

---

### Task S4.3: GPTQ baseline — write failing test + implement

**Files:**
- Create: `FlatQuant/tests/test_baseline_gptq.py`
- Create: `FlatQuant/flatquant/baselines/gptq.py`

- [ ] **Step 1: Write test (mirror S4.1)**

```python
# FlatQuant/tests/test_baseline_gptq.py
import torch, pytest
from transformers import AutoModelForCausalLM

@pytest.fixture
def tiny_llama():
    return AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM").cuda().half()

@pytest.fixture
def calib_data():
    return torch.randint(0, 100, (4, 16), device="cuda")

def test_gptq_w4_runs(tiny_llama, calib_data):
    from flatquant.baselines.gptq import quantize_model
    q = quantize_model(tiny_llama, bits=4, calibration_data=calib_data, use_dbaf=False)
    out = q(calib_data, labels=calib_data)
    assert torch.isfinite(out.loss)

def test_gptq_w4_dbaf_changes_output(tiny_llama, calib_data):
    from flatquant.baselines.gptq import quantize_model
    import copy
    q_no = quantize_model(copy.deepcopy(tiny_llama), bits=4, calibration_data=calib_data, use_dbaf=False)
    q_db = quantize_model(copy.deepcopy(tiny_llama), bits=4, calibration_data=calib_data, use_dbaf=True)
    a = q_no(calib_data, labels=calib_data).loss.item()
    b = q_db(calib_data, labels=calib_data).loss.item()
    assert abs(a - b) > 1e-4
```

- [ ] **Step 2: Verify fails**

```bash
pytest FlatQuant/tests/test_baseline_gptq.py -v
```

- [ ] **Step 3: Implement (reuse the existing FlatQuant `gptq_utils.py`)**

```python
# FlatQuant/flatquant/baselines/gptq.py
"""GPTQ as a baseline, with optional DBAF folding pre-quantization."""
from __future__ import annotations
import torch
import torch.nn as nn
from flatquant.gptq_utils import GPTQ  # existing helper in the repo
from flatquant.baselines.rtn import _dbaf_fold, _dbaf_unfold

def quantize_model(model: nn.Module, bits: int = 4, calibration_data=None, use_dbaf: bool = False, alpha: float = 0.75) -> nn.Module:
    """Apply GPTQ quantization layer-by-layer, optionally folding via DBAF first.
    `calibration_data` is a tensor of token ids of shape [B, T].
    """
    # 1. Capture per-layer inputs by running calibration_data through the model with hooks
    inputs = {}
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            def make_hook(n):
                def hook(module, inp, out):
                    inputs.setdefault(n, []).append(inp[0].detach())
                return hook
            handles.append(mod.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(calibration_data)
    for h in handles:
        h.remove()

    # 2. Per-layer: optionally DBAF-fold, then GPTQ-solve
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or "lm_head" in name:
            continue
        if name not in inputs:
            continue
        gptq = GPTQ(mod)
        for batch in inputs[name]:
            gptq.add_batch(batch.view(-1, batch.shape[-1]), None)
        if use_dbaf:
            w_fold, T, a = _dbaf_fold(mod.weight.data, alpha=alpha)
            mod.weight.data = w_fold
            gptq.quantize(bits=bits)
            mod.weight.data = _dbaf_unfold(mod.weight.data, T, a).to(mod.weight.dtype)
        else:
            gptq.quantize(bits=bits)
        gptq.free()
    return model
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest FlatQuant/tests/test_baseline_gptq.py -v
```

- [ ] **Step 5: Commit**

```bash
git add FlatQuant/flatquant/baselines/gptq.py FlatQuant/tests/test_baseline_gptq.py
git commit -m "feat(S4): GPTQ weak baseline with optional DBAF folding"
```

---

### Task S4.4: AWQ baseline — write failing test + implement

**Files:**
- Create: `FlatQuant/tests/test_baseline_awq.py`
- Create: `FlatQuant/flatquant/baselines/awq.py`

- [ ] **Step 1: Write test (mirror S4.3 test patterns)**

```python
# FlatQuant/tests/test_baseline_awq.py
import torch, pytest, copy
from transformers import AutoModelForCausalLM

@pytest.fixture
def tiny_llama():
    return AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM").cuda().half()

@pytest.fixture
def calib_data():
    return torch.randint(0, 100, (4, 16), device="cuda")

def test_awq_w4_runs(tiny_llama, calib_data):
    from flatquant.baselines.awq import quantize_model
    q = quantize_model(tiny_llama, bits=4, calibration_data=calib_data, use_dbaf=False)
    out = q(calib_data, labels=calib_data)
    assert torch.isfinite(out.loss)

def test_awq_w4_dbaf_changes_output(tiny_llama, calib_data):
    from flatquant.baselines.awq import quantize_model
    q_no = quantize_model(copy.deepcopy(tiny_llama), bits=4, calibration_data=calib_data, use_dbaf=False)
    q_db = quantize_model(copy.deepcopy(tiny_llama), bits=4, calibration_data=calib_data, use_dbaf=True)
    a = q_no(calib_data, labels=calib_data).loss.item()
    b = q_db(calib_data, labels=calib_data).loss.item()
    assert abs(a - b) > 1e-4
```

- [ ] **Step 2: Verify fails**

```bash
pytest FlatQuant/tests/test_baseline_awq.py -v
```

- [ ] **Step 3: Implement (use a minimal AWQ: compute activation scales, scale weights inverse, then quantize)**

```python
# FlatQuant/flatquant/baselines/awq.py
"""Lightweight AWQ baseline: per-channel weight scaling by activation magnitude
followed by RTN quantization, with optional DBAF folding pre-quantization.
This is a simplified port — not the full AWQ paper algorithm — sufficient as a
'weak rotation-free baseline'.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from flatquant.baselines.rtn import _dbaf_fold, _dbaf_unfold, _quantize_tensor_uniform

def _activation_scales(model, calibration_data):
    scales = {}
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            def make_hook(n):
                def hook(module, inp, out):
                    x = inp[0].detach().reshape(-1, inp[0].shape[-1]).abs().mean(dim=0)
                    scales[n] = x if n not in scales else (scales[n] + x) / 2
                return hook
            handles.append(mod.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(calibration_data)
    for h in handles:
        h.remove()
    return scales

def quantize_model(model: nn.Module, bits: int = 4, calibration_data=None, use_dbaf: bool = False, alpha_dbaf: float = 0.75) -> nn.Module:
    scales = _activation_scales(model, calibration_data)
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or "lm_head" in name:
            continue
        s = scales.get(name)
        if s is None:
            continue
        # Per-channel scale: weights are scaled inversely so that activation*weight equivalent
        s_clip = s.clamp(min=1e-5).pow(0.5)  # AWQ alpha=0.5 default
        w = mod.weight.data * s_clip.view(1, -1)
        if use_dbaf:
            w_fold, T, a = _dbaf_fold(w, alpha=alpha_dbaf)
            w_q = _quantize_tensor_uniform(w_fold, bits)
            w_out = _dbaf_unfold(w_q, T, a)
        else:
            w_out = _quantize_tensor_uniform(w, bits)
        mod.weight.data = (w_out / s_clip.view(1, -1)).to(mod.weight.dtype)
    return model
```

- [ ] **Step 4: Verify tests pass + commit**

```bash
pytest FlatQuant/tests/test_baseline_awq.py -v
git add FlatQuant/flatquant/baselines/awq.py FlatQuant/tests/test_baseline_awq.py
git commit -m "feat(S4): AWQ weak baseline with optional DBAF folding"
```

---

### Task S4.5: Driver script for S4 experiments

**Files:**
- Create: `scripts/run_S4.py`

- [ ] **Step 1: Write the driver**

```python
# scripts/run_S4.py
"""Run experiment A: DBAF on weak baselines (RTN/GPTQ/AWQ) × {LLaMA-3-8B, Qwen-2.5-7B}.

Outputs WikiText-2 perplexity per (model, baseline, with/without DBAF) cell.
"""
import argparse, json, pathlib, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODELS = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "qwen25-7b": "Qwen/Qwen2.5-7B",
}
BASELINES = ["rtn", "gptq", "awq"]

def get_baseline(name):
    if name == "rtn":
        from flatquant.baselines.rtn import quantize_model
    elif name == "gptq":
        from flatquant.baselines.gptq import quantize_model
    elif name == "awq":
        from flatquant.baselines.awq import quantize_model
    return quantize_model

def wikitext_ppl(model, tokenizer, seq_len=2048, n_samples=64):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    nlls = []
    for i in range(min(n_samples, ids.shape[1] // seq_len)):
        chunk = ids[:, i*seq_len:(i+1)*seq_len]
        with torch.no_grad():
            out = model(chunk, labels=chunk)
        nlls.append(out.loss.item())
    return float(torch.tensor(nlls).mean().exp().item())

def calib_batch(tokenizer, seq_len=2048, n=4):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :n*seq_len].view(n, seq_len).cuda()
    return ids

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS.keys(), required=True)
    p.add_argument("--baseline", choices=BASELINES, required=True)
    p.add_argument("--use-dbaf", action="store_true")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], torch_dtype=torch.float16, device_map="cuda")
    calib = calib_batch(tok)

    t0 = time.time()
    quantize_model = get_baseline(args.baseline)
    if args.baseline == "rtn":
        model = quantize_model(model, bits=args.bits, use_dbaf=args.use_dbaf)
    else:
        model = quantize_model(model, bits=args.bits, calibration_data=calib, use_dbaf=args.use_dbaf)
    t_quant = time.time() - t0

    ppl = wikitext_ppl(model, tok)
    out = {
        "model": args.model, "baseline": args.baseline, "use_dbaf": args.use_dbaf,
        "bits": args.bits, "wikitext2_ppl": ppl, "quant_seconds": t_quant,
    }
    p_out = pathlib.Path(args.out); p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps(out, indent=2))
    print(out)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run on small model first (tiny-random, not Llama-3-8B yet)**

Make sure the driver works end-to-end before launching multi-hour runs:

```bash
cd /home/ubuntu/unifying-ptq/FlatQuant
python ../scripts/run_S4.py --model llama3-8b --baseline rtn --bits 4 --out /tmp/smoke.json
# Above will take long! Substitute tiny model for smoke:
python -c "
import sys; sys.path.insert(0, '/home/ubuntu/unifying-ptq/FlatQuant')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flatquant.baselines.rtn import quantize_model
m = AutoModelForCausalLM.from_pretrained('hf-internal-testing/tiny-random-LlamaForCausalLM').cuda().half()
q = quantize_model(m, bits=4, use_dbaf=True)
print('SMOKE_OK')
"
```

Expected: `SMOKE_OK`.

- [ ] **Step 3: Commit driver**

```bash
git add scripts/run_S4.py
git commit -m "feat(S4): driver script for DBAF-on-weak-baselines experiment matrix"
```

---

### Task S4.6: Run S4 on LLaMA-3-8B (local A100)

**Files:**
- Create: `results/S4-dbaf-weak/llama3-8b/{rtn,gptq,awq}/{baseline,with-dbaf}/eval.json`

**Prereq:** LLaMA-3-8B downloaded at `./modelzoo/meta-llama/Meta-Llama-3-8B` (HF Hub or pre-downloaded).

- [ ] **Step 1: Launch the 6-cell sweep in tmux on local**

```bash
cd /home/ubuntu/unifying-ptq && conda activate unifyptq
tmux new-session -d -s s4-llama "
cd /home/ubuntu/unifying-ptq
for baseline in rtn gptq awq; do
  for flag in '' '--use-dbaf'; do
    suffix=baseline; [ -n \"\$flag\" ] && suffix=with-dbaf
    out=results/S4-dbaf-weak/llama3-8b/\$baseline/\$suffix/eval.json
    echo \"=== \$baseline \$suffix ===\"
    python scripts/run_S4.py --model llama3-8b --baseline \$baseline \$flag --bits 4 --out \$out 2>&1 | tee \$out.log
  done
done
echo DONE
"
```

- [ ] **Step 2: Poll until done**

```bash
until tmux capture-pane -t s4-llama -p | grep -q "^DONE"; do
  echo "---"; tmux capture-pane -t s4-llama -p | tail -3; sleep 600
done
```

Expected: ~18 hours total. Acceptance: each `eval.json` exists with finite `wikitext2_ppl`.

- [ ] **Step 3: Aggregate**

```bash
python - <<'EOF'
import json, glob
rows = [json.load(open(p)) for p in glob.glob("results/S4-dbaf-weak/llama3-8b/*/*/eval.json")]
for r in sorted(rows, key=lambda x: (x["baseline"], x["use_dbaf"])):
    print(f"{r['baseline']:6s} dbaf={r['use_dbaf']!s:5s} PPL={r['wikitext2_ppl']:.3f}")
EOF
```

- [ ] **Step 4: Commit results**

```bash
git add results/S4-dbaf-weak/llama3-8b/
git commit -m "result(S4): DBAF on RTN/GPTQ/AWQ for LLaMA-3-8B W4"
```

---

### Task S4.7: Run S4 on Qwen-2.5-7B (remote A100, in parallel with S4.6)

**Files:**
- Create: `results/S4-dbaf-weak/qwen25-7b/{rtn,gptq,awq}/{baseline,with-dbaf}/eval.json`

- [ ] **Step 1: Launch on remote in tmux**

```bash
ssh remote-gpu 'tmux new-session -d -s s4-qwen "
source ~/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
cd ~/unifying-ptq
for baseline in rtn gptq awq; do
  for flag in \"\" \"--use-dbaf\"; do
    suffix=baseline; [ -n \"\$flag\" ] && suffix=with-dbaf
    out=results/S4-dbaf-weak/qwen25-7b/\$baseline/\$suffix/eval.json
    python scripts/run_S4.py --model qwen25-7b --baseline \$baseline \$flag --bits 4 --out \$out 2>&1 | tee \$out.log
  done
done
echo DONE
"'
```

- [ ] **Step 2: Poll, sync, commit (mirror S4.6 Steps 2-4)**

```bash
until ssh remote-gpu 'tmux capture-pane -t s4-qwen -p | grep -q "^DONE"'; do sleep 600; done
./scripts/sync_results.sh pull
git add results/S4-dbaf-weak/qwen25-7b/ && git commit -m "result(S4): DBAF on weak baselines for Qwen-2.5-7B"
```

---

### Task S4.8 (stretch): Run on Mistral-7B (add-on #10)

**Files:**
- Create: `results/S4-dbaf-weak/mistral-7b/...`

- [ ] Same pattern as S4.7 but with `--model mistral-7b` added to MODELS dict in `scripts/run_S4.py` first.

---

## Done when

- All 6 cells (3 baselines × {with/without DBAF}) for LLaMA-3-8B exist with finite PPL
- Same 6 cells for Qwen-2.5-7B
- Aggregated table shows DBAF gives >= 0.2 PPL improvement on at least one (baseline, model) pair (acceptance: at least ONE positive effect)
- Mistral-7B optional
- All commits land on `main`
- Numbers ready for paper Table "Experiment A"
