# S5: KV-Cache PCSA + RULER Long-Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend PCSA from SAM-decoder activations to LLM KV-cache quantization. Use design 1 (per-prompt routing): compute a prompt descriptor once at prompt-time, look up a per-cluster scale for K and V caches, and reuse it for the whole generation. Evaluate on RULER at 4k/8k/16k/32k context lengths to show PCSA helps as context grows.

**Architecture:** Add a new `flatquant/kv_pcsa.py` module exposing `KVPCSAQuantizer(num_anchors, descriptor_dim, k_bits=4, v_bits=4)` with API:
- `set_prompt(input_ids, hidden_states)`: compute descriptor + route to anchor + freeze K/V scales for this prompt
- forward hooks on each `forward` of K/V projection layers to swap in the per-cluster scale

Wire into FlatQuant's main.py via a new `--kv-pcsa` flag. Evaluate using NVIDIA's RULER benchmark (cloned + installed). Each context-length run produces an `eval.json`.

**Tech Stack:** PyTorch, transformers, FlatQuant's KV-cache plumbing, RULER eval harness, datasets.

**Prereqs:** S1.1 (unifyptq env), S4.6 (LLaMA-3-8B FlatQuant baseline confirmed working).

---

### Task S5.1: Write failing tests for KVPCSAQuantizer

**Files:**
- Create: `FlatQuant/tests/test_kv_pcsa.py`

- [ ] **Step 1: Write the tests**

```python
# FlatQuant/tests/test_kv_pcsa.py
import torch
import pytest

@pytest.fixture
def kv_pcsa():
    from flatquant.kv_pcsa import KVPCSAQuantizer
    return KVPCSAQuantizer(num_anchors=4, descriptor_dim=128, k_bits=4, v_bits=4)

def test_set_prompt_picks_anchor(kv_pcsa):
    desc = torch.randn(1, 128).cuda()
    kv_pcsa.cuda()
    idx = kv_pcsa.set_prompt(desc)
    assert 0 <= idx < 4

def test_quantize_kv_returns_int4(kv_pcsa):
    """K/V tensors quantized via PCSA should round-trip with bounded error."""
    kv_pcsa.cuda()
    kv_pcsa.set_prompt(torch.randn(1, 128).cuda())
    k = torch.randn(1, 8, 64, 32, dtype=torch.float16, device="cuda")  # B, n_heads, T, D
    kq = kv_pcsa.quantize_k(k)
    assert kq.shape == k.shape
    assert (k - kq).abs().mean() < 0.5

def test_anchor_routing_deterministic(kv_pcsa):
    kv_pcsa.cuda()
    desc = torch.randn(1, 128).cuda()
    i1 = kv_pcsa.set_prompt(desc)
    i2 = kv_pcsa.set_prompt(desc)
    assert i1 == i2

def test_calibrate_updates_anchors_ema(kv_pcsa):
    """During calibration, anchors should drift toward seen descriptors."""
    kv_pcsa.cuda(); kv_pcsa.train()
    a0 = kv_pcsa.anchors.detach().clone()
    descs = torch.randn(100, 128).cuda()
    for d in descs:
        kv_pcsa.calibrate_step(d.unsqueeze(0))
    a1 = kv_pcsa.anchors.detach().clone()
    assert (a0 - a1).abs().mean() > 1e-4, "anchors didn't move during calibration"
```

- [ ] **Step 2: Verify fails**

```bash
cd /home/ubuntu/unifying-ptq && conda activate unifyptq
pytest FlatQuant/tests/test_kv_pcsa.py -v
```

Expected: `ModuleNotFoundError: No module named 'flatquant.kv_pcsa'`.

---

### Task S5.2: Implement KVPCSAQuantizer

**Files:**
- Create: `FlatQuant/flatquant/kv_pcsa.py`

- [ ] **Step 1: Implement**

```python
# FlatQuant/flatquant/kv_pcsa.py
"""Per-prompt KV-cache scale routing via PCSA.

Compute a prompt descriptor once when the prompt arrives, route to one of
K anchors, and use the per-anchor K/V quantization scale for the full
generation. Calibration uses EMA on descriptors (mirrors original PCSA).
"""
from __future__ import annotations
import torch
import torch.nn as nn

class KVPCSAQuantizer(nn.Module):
    def __init__(self, num_anchors: int = 4, descriptor_dim: int = 4096, k_bits: int = 4, v_bits: int = 4, momentum: float = 0.9):
        super().__init__()
        self.K = num_anchors
        self.D = descriptor_dim
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.momentum = momentum
        # Anchors: K x D, L2-normalized
        anchors = torch.randn(num_anchors, descriptor_dim)
        anchors = anchors / anchors.norm(dim=1, keepdim=True).clamp(min=1e-6)
        self.register_buffer("anchors", anchors)
        # Per-anchor scales for K and V (scalar per anchor for simplicity)
        self.register_buffer("k_scales", torch.ones(num_anchors))
        self.register_buffer("v_scales", torch.ones(num_anchors))
        self.register_buffer("counts", torch.zeros(num_anchors))
        self.current_anchor = 0

    def _descriptor(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> normalized [B, D]."""
        d = hidden_states.mean(dim=1)
        return d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)

    def set_prompt(self, descriptor_or_hidden: torch.Tensor) -> int:
        """Called once per prompt. Returns the selected anchor index."""
        if descriptor_or_hidden.dim() == 3:
            d = self._descriptor(descriptor_or_hidden)
        else:
            d = descriptor_or_hidden
            d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)
        sims = d @ self.anchors.T  # [B, K]
        idx = int(sims.argmax(dim=1)[0].item())
        self.current_anchor = idx
        return idx

    def calibrate_step(self, descriptor: torch.Tensor):
        d = descriptor / descriptor.norm(dim=1, keepdim=True).clamp(min=1e-6)
        d = d.squeeze(0)
        sims = d @ self.anchors.T
        idx = int(sims.argmax().item())
        self.anchors[idx] = self.momentum * self.anchors[idx] + (1 - self.momentum) * d
        self.anchors[idx] = self.anchors[idx] / self.anchors[idx].norm().clamp(min=1e-6)
        self.counts[idx] += 1

    def update_scales(self, k: torch.Tensor, v: torch.Tensor):
        """Call during calibration after set_prompt to track per-anchor scales."""
        with torch.no_grad():
            new_k = k.abs().max()
            new_v = v.abs().max()
            i = self.current_anchor
            self.k_scales[i] = self.momentum * self.k_scales[i] + (1 - self.momentum) * new_k
            self.v_scales[i] = self.momentum * self.v_scales[i] + (1 - self.momentum) * new_v

    def _quant(self, x: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
        qmax = 2 ** (bits - 1) - 1
        s = scale / qmax
        if s.item() == 0:
            return x
        q = torch.round(x / s).clamp(-qmax, qmax)
        return (q * s).to(x.dtype)

    def quantize_k(self, k: torch.Tensor) -> torch.Tensor:
        return self._quant(k, self.k_scales[self.current_anchor], self.k_bits)

    def quantize_v(self, v: torch.Tensor) -> torch.Tensor:
        return self._quant(v, self.v_scales[self.current_anchor], self.v_bits)
```

- [ ] **Step 2: Verify tests pass**

```bash
pytest FlatQuant/tests/test_kv_pcsa.py -v
```

- [ ] **Step 3: Commit**

```bash
git add FlatQuant/flatquant/kv_pcsa.py FlatQuant/tests/test_kv_pcsa.py
git commit -m "feat(S5): KVPCSAQuantizer with per-prompt anchor routing"
```

---

### Task S5.3: Integrate KVPCSAQuantizer into FlatQuant's LLaMA path

**Files:**
- Modify: `FlatQuant/flatquant/model_tools/llama_utils.py`
- Modify: `FlatQuant/main.py`

- [ ] **Step 1: Add hook injection into K/V projection forwards**

In `llama_utils.py`, find the function that wraps attention layers and add a hook that calls `kv_pcsa.quantize_k()` and `quantize_v()` on the projected K/V tensors before they're written to cache. The user can locate this by searching for `k_proj` and `v_proj` in the existing FlatQuant LlamaAttention wrapper.

Add inside the wrapped attention's forward:

```python
# In whatever LlamaAttention wrapper exists:
def forward(self, hidden_states, ...):
    # ... existing rotation, quant ...
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    if getattr(self, "kv_pcsa", None) is not None:
        # Route on the first call (set_prompt should have been called by main.py before generation)
        k = self.kv_pcsa.quantize_k(k)
        v = self.kv_pcsa.quantize_v(v)
    # ... rest of attention ...
```

- [ ] **Step 2: Add `--kv-pcsa` CLI flag to `main.py`**

In `FlatQuant/main.py`, in `args_utils.py` arg parser, add:

```python
parser.add_argument("--kv-pcsa", action="store_true", help="Use KV-cache PCSA with per-prompt routing")
parser.add_argument("--kv-pcsa-anchors", type=int, default=4)
```

After the model is built, if `args.kv_pcsa`:

```python
from flatquant.kv_pcsa import KVPCSAQuantizer
for layer in model.model.layers:
    layer.self_attn.kv_pcsa = KVPCSAQuantizer(
        num_anchors=args.kv_pcsa_anchors,
        descriptor_dim=model.config.hidden_size,
        k_bits=args.k_bits, v_bits=args.v_bits,
    ).cuda()
```

- [ ] **Step 3: Smoke test wiring**

```bash
cd /home/ubuntu/unifying-ptq/FlatQuant
python main.py --model hf-internal-testing/tiny-random-LlamaForCausalLM --w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4 --kv-pcsa --output_dir /tmp/s5-smoke --cali_bsz 1 --epoch 1 2>&1 | tail -20
```

Expected: prints "KVPCSAQuantizer attached" (or similar) and no AttributeError.

- [ ] **Step 4: Commit**

```bash
git add FlatQuant/flatquant/model_tools/llama_utils.py FlatQuant/main.py FlatQuant/flatquant/args_utils.py
git commit -m "feat(S5): wire KVPCSAQuantizer into FlatQuant LlamaAttention path + --kv-pcsa flag"
```

---

### Task S5.4: Install RULER eval harness

**Files:**
- Create: `/home/ubuntu/unifying-ptq/ruler/`

- [ ] **Step 1: Clone RULER**

```bash
cd /home/ubuntu/unifying-ptq
git clone https://github.com/NVIDIA/RULER.git ruler
cd ruler && pip install -r requirements.txt 2>&1 | tail -5
```

- [ ] **Step 2: Download synthetic data for 4k context (smallest)**

```bash
cd /home/ubuntu/unifying-ptq/ruler
bash scripts/synthetic/prepare.sh 4096
ls scripts/synthetic/data/4096/
```

Expected: a set of `.jsonl` files for each RULER subtask (niah, multikey, etc.) at 4k context length.

- [ ] **Step 3: Commit RULER as submodule pointer**

```bash
cd /home/ubuntu/unifying-ptq
git submodule add https://github.com/NVIDIA/RULER.git ruler 2>&1 | tail -3 || echo "(if not a submodule, just .gitignore the dir)"
echo "ruler/" >> .gitignore
git add .gitignore
git commit -m "feat(S5): add RULER long-context eval harness"
```

---

### Task S5.5: Run RULER 4k context on LLaMA-3-8B FlatQuant + KV-PCSA

**Files:**
- Create: `results/S5-kv-pcsa/llama3-8b/4k/seed0/eval.json`

- [ ] **Step 1: Calibrate KV-PCSA anchors on FlatQuant LLaMA-3-8B**

We need a calibrated model with `--kv-pcsa` flag set. Reuse FlatQuant's main.py calibration path.

```bash
cd /home/ubuntu/unifying-ptq/FlatQuant && conda activate unifyptq
tmux new-session -d -s s5-calib "
python main.py \
  --model ./modelzoo/meta-llama/Meta-Llama-3-8B \
  --w_bits 4 --a_bits 4 --k_bits 4 --k_asym --k_groupsize 128 --v_bits 4 --v_asym --v_groupsize 128 \
  --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
  --lwc --lac --cali_trans --add_diag \
  --kv-pcsa --kv-pcsa-anchors 4 \
  --output_dir /home/ubuntu/unifying-ptq/results/S5-kv-pcsa/llama3-8b/state \
  --save_matrix \
  2>&1 | tee /home/ubuntu/unifying-ptq/results/S5-kv-pcsa/llama3-8b/state/calib.log
"
```

Poll: `tmux capture-pane -t s5-calib -p | tail -10` until "calibration complete" or DONE marker. Expected: ~8 hours.

- [ ] **Step 2: Write RULER eval glue**

```python
# scripts/eval_ruler.py
"""Evaluate a FlatQuant-calibrated model on RULER at a given context length."""
import argparse, json, pathlib, sys
import torch
from transformers import AutoTokenizer
sys.path.insert(0, "/home/ubuntu/unifying-ptq/ruler")
# RULER provides a model wrapper; we adapt to our calibrated model:

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Calibrated checkpoint dir")
    p.add_argument("--context-len", type=int, required=True, choices=[4096, 8192, 16384, 32768])
    p.add_argument("--tasks", nargs="+", default=["niah_single_1", "niah_multikey_1", "vt"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # Load calibrated model (FlatQuant saves W4A4 + KV-PCSA state)
    from flatquant.model_utils import load_calibrated_model
    model = load_calibrated_model(args.model_path)
    tok = AutoTokenizer.from_pretrained(args.model_path)

    # For each task, load JSONL, run generation, score
    results = {}
    for task in args.tasks:
        ds_path = f"/home/ubuntu/unifying-ptq/ruler/scripts/synthetic/data/{args.context_len}/{task}.jsonl"
        scores = []
        with open(ds_path) as f:
            for line in f:
                ex = json.loads(line)
                prompt = ex["input"]
                gold = ex["outputs"]
                # PCSA: set prompt descriptor BEFORE generation
                input_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=args.context_len).input_ids.cuda()
                # Get hidden states for descriptor:
                with torch.no_grad():
                    h = model.model.embed_tokens(input_ids)
                for layer in model.model.layers:
                    if getattr(layer.self_attn, "kv_pcsa", None) is not None:
                        layer.self_attn.kv_pcsa.set_prompt(h)
                # Generate
                with torch.no_grad():
                    gen = model.generate(input_ids, max_new_tokens=32, do_sample=False)
                pred = tok.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
                # Score: exact match (RULER convention)
                ok = any(g.lower() in pred.lower() for g in gold)
                scores.append(ok)
        results[task] = sum(scores) / len(scores)
    out = {
        "model": "llama3-8b", "method": "FlatQuant+DBAF+KV-PCSA",
        "context_len": args.context_len, "task_scores": results,
        "avg_score": sum(results.values()) / len(results),
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(out)

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run RULER 4k**

```bash
python scripts/eval_ruler.py \
  --model-path results/S5-kv-pcsa/llama3-8b/state \
  --context-len 4096 \
  --out results/S5-kv-pcsa/llama3-8b/4k/seed0/eval.json
```

- [ ] **Step 4: Commit**

```bash
git add results/S5-kv-pcsa/llama3-8b/4k/ scripts/eval_ruler.py
git commit -m "result(S5): RULER 4k on LLaMA-3-8B FlatQuant+KV-PCSA"
```

---

### Task S5.6: Run RULER 8k, 16k, 32k

**Files:**
- Create: `results/S5-kv-pcsa/llama3-8b/{8k,16k,32k}/seed0/eval.json`

- [ ] **Step 1: Prepare RULER data at higher context lengths**

```bash
cd /home/ubuntu/unifying-ptq/ruler
bash scripts/synthetic/prepare.sh 8192
bash scripts/synthetic/prepare.sh 16384
bash scripts/synthetic/prepare.sh 32768
```

- [ ] **Step 2: Sequentially run each length**

```bash
cd /home/ubuntu/unifying-ptq
for L in 8192 16384 32768; do
  k=$((L / 1024))k
  python scripts/eval_ruler.py \
    --model-path results/S5-kv-pcsa/llama3-8b/state \
    --context-len $L \
    --out results/S5-kv-pcsa/llama3-8b/${k}/seed0/eval.json 2>&1 | tee results/S5-kv-pcsa/llama3-8b/${k}/seed0/run.log
done
```

Expected runtime: 4k=2h, 8k=4h, 16k=8h, 32k=16h. If 32k OOMs on the A100 80GB, document the OOM, drop to 16k.

- [ ] **Step 3: Commit per-length results**

```bash
git add results/S5-kv-pcsa/llama3-8b/
git commit -m "result(S5): RULER 8k/16k/32k on LLaMA-3-8B FlatQuant+KV-PCSA"
```

---

### Task S5.7: Comparison baseline — FlatQuant without KV-PCSA at same lengths

**Files:**
- Create: `results/S5-kv-pcsa/llama3-8b/baseline-no-kvpcsa/{4k,8k,16k,32k}/eval.json`

To claim KV-PCSA helps, need the matched baseline:

- [ ] **Step 1: Calibrate FlatQuant without --kv-pcsa**

(Re-use existing FlatQuant calibrated model if available; otherwise calibrate fresh.)

- [ ] **Step 2: Run RULER at each length on the non-PCSA model**

Mirror S5.6 with `--model-path` pointing at the non-KV-PCSA calibrated dir.

- [ ] **Step 3: Build comparison table**

```python
import json, pathlib
rows = []
for arm in ["state", "baseline-no-kvpcsa"]:
    for L in ["4k", "8k", "16k", "32k"]:
        p = f"results/S5-kv-pcsa/llama3-8b/{arm}/{L}/seed0/eval.json"
        try:
            rows.append(json.load(open(p)))
        except FileNotFoundError:
            pass
for r in rows:
    print(f"{r['method']:30s} {r['context_len']:6d} avg_score={r['avg_score']:.3f}")
```

Acceptance: KV-PCSA avg_score ≥ baseline avg_score at ≥2 context lengths.

- [ ] **Step 4: Commit summary**

```bash
git add results/S5-kv-pcsa/llama3-8b/
git commit -m "result(S5): KV-PCSA vs baseline at 4k/8k/16k/32k RULER"
```

---

## Done when

- `KVPCSAQuantizer` exists in `flatquant/kv_pcsa.py` with passing tests
- FlatQuant main.py accepts `--kv-pcsa` and `--kv-pcsa-anchors`
- RULER eval glue runs end-to-end
- Results at ≥3 context lengths (4k, 8k, 16k) for both KV-PCSA-on and KV-PCSA-off
- Aggregated comparison committed
- Numbers ready for paper Table "Experiment C"
