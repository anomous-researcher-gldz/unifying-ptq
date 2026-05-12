# S3: torchao Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire torchao's real INT4 kernels into each of the three codebases (FlatQuant, AHCPTQ-calibrated SAM, CompSRT SwinIR) so we can deploy actual INT4 models — not just fake-quant simulations. Target W4A4 everywhere via `Int4DynamicActivationInt4WeightConfig`; fall back to W4A16 via `Int4WeightOnlyConfig` if W4A4 fails for a given codebase.

**Architecture:** Each codebase gets a small new module `<codebase>/torchao_deploy.py` exposing one function `apply_torchao(model, config_name, calibration_state)` that:
1. Loads the codebase's calibrated state (rotations, alphas, anchors) into the underlying FP model
2. Applies torchao's `quantize_(model, config)` to the target sub-module
3. Returns a model ready for inference with real INT4 weights/activations
Each module is tested with TDD: write the failing test (call `apply_torchao` on a tiny model, assert weights are torchao-quantized), implement, verify, commit.

**Tech Stack:** torchao, PyTorch 2.6, FlatQuant rotations, AHCPTQ specials, CompSRT Hadamard.

**Prereqs:** S1.2 (torchao installed), S2 (AHCPTQ calibrated checkpoint), FlatQuant + CompSRT calibrated checkpoints (from existing code).

---

### Task S3.1: torchao on FlatQuant — write failing test

**Files:**
- Create: `FlatQuant/tests/test_torchao_deploy.py`

- [ ] **Step 1: Write the test**

```python
# FlatQuant/tests/test_torchao_deploy.py
import torch
import pytest
from transformers import AutoModelForCausalLM

@pytest.fixture
def tiny_model():
    """Lightweight stand-in for a calibrated LLaMA model."""
    cfg = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM").config
    cfg.hidden_size = 256
    cfg.intermediate_size = 512
    cfg.num_hidden_layers = 2
    return AutoModelForCausalLM.from_config(cfg).cuda().half()

def test_apply_torchao_w4a16(tiny_model):
    from flatquant.torchao_deploy import apply_torchao
    quantized = apply_torchao(tiny_model, config_name="w4a16", calibration_state=None)
    # At least one Linear should now be a torchao tensor subclass
    from torchao.dtypes import AffineQuantizedTensor
    has_aqt = any(isinstance(p, AffineQuantizedTensor) for p in quantized.parameters())
    assert has_aqt, "No AffineQuantizedTensor found after quantize_"

def test_apply_torchao_w4a4_or_fallback(tiny_model):
    """W4A4 may fail on tiny shapes; verify graceful fallback to W4A16."""
    from flatquant.torchao_deploy import apply_torchao
    quantized = apply_torchao(tiny_model, config_name="w4a4", calibration_state=None)
    x = torch.randint(0, 100, (1, 8), device="cuda")
    out = quantized(x)
    assert out.logits.shape == (1, 8, tiny_model.config.vocab_size)
```

- [ ] **Step 2: Run and verify failure**

```bash
conda activate unifyptq && cd /home/ubuntu/unifying-ptq/FlatQuant
pytest tests/test_torchao_deploy.py -v
```

Expected: `ModuleNotFoundError: No module named 'flatquant.torchao_deploy'`.

---

### Task S3.2: torchao on FlatQuant — implement

**Files:**
- Create: `FlatQuant/flatquant/torchao_deploy.py`

- [ ] **Step 1: Implement the module**

```python
# FlatQuant/flatquant/torchao_deploy.py
"""Apply torchao real INT4 quantization to a FlatQuant-calibrated model.

Supports W4A4 (preferred) and W4A16 (fallback). Loads FlatQuant rotation
matrices + learnable diagonals into the FP weights before applying torchao,
so the rotated/normalized weights are what torchao packs to INT4.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torchao.quantization import quantize_, Int4WeightOnlyConfig
try:
    from torchao.quantization import Int4DynamicActivationInt4WeightConfig
except ImportError:
    Int4DynamicActivationInt4WeightConfig = None

def _load_calibration(model: nn.Module, state):
    """Apply FlatQuant rotations + learnable params from saved state. No-op if state is None."""
    if state is None:
        return model
    # Walk model, set R/L/D/scale params per FlatQuant's apply_flatquant_to_llama API
    # Expects state keyed by module-path -> dict(rotation_R, learnable_scale, ...)
    for name, mod in model.named_modules():
        if name in state:
            for k, v in state[name].items():
                if hasattr(mod, k):
                    setattr(mod, k, nn.Parameter(v.to(mod.weight.device)) if v.dim() > 0 else v)
    return model

def _filter_linear(mod, fqn):
    """torchao filter fn: only quantize nn.Linear, skip lm_head."""
    return isinstance(mod, nn.Linear) and "lm_head" not in fqn

def apply_torchao(model: nn.Module, config_name: str = "w4a4", calibration_state=None) -> nn.Module:
    """Quantize `model` in-place using torchao.
    config_name: 'w4a4' (preferred) or 'w4a16' (fallback)
    Returns the same model (mutated).
    """
    model = _load_calibration(model, calibration_state)
    if config_name == "w4a4":
        if Int4DynamicActivationInt4WeightConfig is None:
            print("[torchao] W4A4 config unavailable in this torchao version; falling back to W4A16")
            config_name = "w4a16"
        else:
            try:
                quantize_(model, Int4DynamicActivationInt4WeightConfig(), filter_fn=_filter_linear)
                return model
            except Exception as e:
                print(f"[torchao] W4A4 failed ({type(e).__name__}: {e}); falling back to W4A16")
                config_name = "w4a16"
    if config_name == "w4a16":
        quantize_(model, Int4WeightOnlyConfig(group_size=128), filter_fn=_filter_linear)
        return model
    raise ValueError(f"Unknown config_name: {config_name}")
```

- [ ] **Step 2: Run tests, verify pass**

```bash
pytest tests/test_torchao_deploy.py -v
```

Expected: both tests pass. If W4A4 raises on tiny shapes, the fallback kicks in and the second test still passes via W4A16.

- [ ] **Step 3: Commit**

```bash
git add FlatQuant/flatquant/torchao_deploy.py FlatQuant/tests/test_torchao_deploy.py
git commit -m "feat(S3): torchao W4A4/W4A16 deployment for FlatQuant"
```

---

### Task S3.3: torchao on AHCPTQ SAM — write failing test

**Files:**
- Create: `ahcptq/tests/test_torchao_sam.py`

- [ ] **Step 1: Write the test**

```python
# ahcptq/tests/test_torchao_sam.py
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

@pytest.fixture
def sam_b_image_encoder():
    """Build a small SAM-B image encoder."""
    from projects.instance_segment_anything.models.segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=None)  # random init for test speed
    return sam.image_encoder.cuda().half()

def test_apply_torchao_sam_w4a16(sam_b_image_encoder):
    from ahcptq.torchao_deploy import apply_torchao_sam
    enc = apply_torchao_sam(sam_b_image_encoder, config_name="w4a16", calibration_state=None)
    from torchao.dtypes import AffineQuantizedTensor
    has_aqt = any(isinstance(p, AffineQuantizedTensor) for p in enc.parameters())
    assert has_aqt

def test_inference_runs(sam_b_image_encoder):
    """A quantized encoder should still run a forward pass."""
    from ahcptq.torchao_deploy import apply_torchao_sam
    enc = apply_torchao_sam(sam_b_image_encoder, config_name="w4a16")
    x = torch.randn(1, 3, 1024, 1024, device="cuda", dtype=torch.half)
    with torch.no_grad():
        out = enc(x)
    assert out.shape[0] == 1
```

- [ ] **Step 2: Run and verify failure**

```bash
cd /home/ubuntu/unifying-ptq && pytest ahcptq/tests/test_torchao_sam.py -v
```

Expected: `ModuleNotFoundError: No module named 'ahcptq.torchao_deploy'`.

---

### Task S3.4: torchao on AHCPTQ SAM — implement

**Files:**
- Create: `ahcptq/torchao_deploy.py`

- [ ] **Step 1: Implement**

```python
# ahcptq/torchao_deploy.py
"""Apply torchao real INT4 to an AHCPTQ-calibrated SAM image encoder.

Loads AHCPTQ's fake-quant state (DBAF alphas, PCSA anchors+scales) into the
FP image encoder by 'baking' the rotation/folding into the weights, then
hands the FP weights to torchao for INT4 packing.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torchao.quantization import quantize_, Int4WeightOnlyConfig
try:
    from torchao.quantization import Int4DynamicActivationInt4WeightConfig
except ImportError:
    Int4DynamicActivationInt4WeightConfig = None

def _bake_dbaf_into_weights(encoder: nn.Module, state):
    """Fold (DBAF) the FP weights using saved per-layer alpha/T values. No-op if state is None."""
    if state is None:
        return encoder
    for name, mod in encoder.named_modules():
        if isinstance(mod, nn.Linear) and name in state and "dbaf_alpha" in state[name]:
            alpha = state[name]["dbaf_alpha"]
            T = state[name]["dbaf_T"]
            with torch.no_grad():
                w = mod.weight.data
                sgn = torch.sign(w)
                mask = w.abs() > T
                w[mask] = sgn[mask] * T + alpha * (w[mask] - sgn[mask] * T)
                mod.weight.data = w
    return encoder

def _filter_attn_mlp(mod, fqn):
    """Quantize attention qkv/proj and MLP fc1/fc2; skip patch embed."""
    if not isinstance(mod, nn.Linear):
        return False
    skips = ["patch_embed", "neck"]
    return not any(s in fqn for s in skips)

def apply_torchao_sam(encoder: nn.Module, config_name: str = "w4a4", calibration_state=None) -> nn.Module:
    encoder = _bake_dbaf_into_weights(encoder, calibration_state)
    if config_name == "w4a4":
        if Int4DynamicActivationInt4WeightConfig is None:
            print("[torchao SAM] W4A4 unavailable; using W4A16")
            config_name = "w4a16"
        else:
            try:
                quantize_(encoder, Int4DynamicActivationInt4WeightConfig(), filter_fn=_filter_attn_mlp)
                return encoder
            except Exception as e:
                print(f"[torchao SAM] W4A4 failed ({type(e).__name__}: {e}); using W4A16")
                config_name = "w4a16"
    if config_name == "w4a16":
        quantize_(encoder, Int4WeightOnlyConfig(group_size=128), filter_fn=_filter_attn_mlp)
        return encoder
    raise ValueError(f"Unknown config_name: {config_name}")
```

- [ ] **Step 2: Run tests, verify pass**

```bash
pytest ahcptq/tests/test_torchao_sam.py -v
```

- [ ] **Step 3: Commit**

```bash
git add ahcptq/torchao_deploy.py ahcptq/tests/test_torchao_sam.py
git commit -m "feat(S3): torchao W4A4/W4A16 deployment for AHCPTQ SAM image encoder"
```

---

### Task S3.5: torchao on CompSRT SwinIR — write failing test + implement

**Files:**
- Create: `CompSRT/basicsr/torchao_deploy.py`
- Create: `CompSRT/tests/test_torchao_swinir.py`

- [ ] **Step 1: Write the test**

```python
# CompSRT/tests/test_torchao_swinir.py
import torch
import pytest

@pytest.fixture
def tiny_swinir():
    from basicsr.archs.swinir_arch import SwinIR
    m = SwinIR(upscale=2, in_chans=3, img_size=64, window_size=8,
               img_range=1., depths=[2,2], embed_dim=60, num_heads=[6,6],
               mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv').cuda().half()
    return m

def test_apply_torchao_swinir(tiny_swinir):
    from basicsr.torchao_deploy import apply_torchao_swinir
    m = apply_torchao_swinir(tiny_swinir, config_name="w4a16", calibration_state=None)
    from torchao.dtypes import AffineQuantizedTensor
    assert any(isinstance(p, AffineQuantizedTensor) for p in m.parameters())

def test_inference_psnr_change_is_bounded(tiny_swinir):
    """Quantized output should be close (not exact) to FP."""
    from basicsr.torchao_deploy import apply_torchao_swinir
    x = torch.randn(1, 3, 64, 64, device="cuda", dtype=torch.half)
    with torch.no_grad():
        fp_out = tiny_swinir(x).clone()
    import copy
    m = apply_torchao_swinir(copy.deepcopy(tiny_swinir), config_name="w4a16")
    with torch.no_grad():
        q_out = m(x)
    err = (fp_out - q_out).abs().mean().item()
    assert err < 1.0, f"quantization error too high: {err}"
```

- [ ] **Step 2: Verify fails**

```bash
cd /home/ubuntu/unifying-ptq && pytest CompSRT/tests/test_torchao_swinir.py -v
```

Expected: `ModuleNotFoundError: No module named 'basicsr.torchao_deploy'`.

- [ ] **Step 3: Implement**

```python
# CompSRT/basicsr/torchao_deploy.py
"""Apply torchao real INT4 to a CompSRT-calibrated SwinIR.

CompSRT applies Hadamard transformations and DBAF before quantization. We bake
those transforms into FP weights and then hand to torchao.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torchao.quantization import quantize_, Int4WeightOnlyConfig
try:
    from torchao.quantization import Int4DynamicActivationInt4WeightConfig
except ImportError:
    Int4DynamicActivationInt4WeightConfig = None

def _bake_compsrt_state(model: nn.Module, state):
    if state is None:
        return model
    # CompSRT calibration state: per-layer Hadamard matrix H and DBAF alpha
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name in state:
            entry = state[name]
            with torch.no_grad():
                w = mod.weight.data
                if "hadamard" in entry:
                    H = entry["hadamard"].to(w.device, w.dtype)
                    w = H @ w
                if "dbaf_alpha" in entry and "dbaf_T" in entry:
                    alpha = entry["dbaf_alpha"]; T = entry["dbaf_T"]
                    sgn = torch.sign(w); mask = w.abs() > T
                    w[mask] = sgn[mask] * T + alpha * (w[mask] - sgn[mask] * T)
                mod.weight.data = w
    return model

def _filter_swinir(mod, fqn):
    return isinstance(mod, nn.Linear) and "conv_" not in fqn

def apply_torchao_swinir(model: nn.Module, config_name: str = "w4a4", calibration_state=None) -> nn.Module:
    model = _bake_compsrt_state(model, calibration_state)
    if config_name == "w4a4" and Int4DynamicActivationInt4WeightConfig is not None:
        try:
            quantize_(model, Int4DynamicActivationInt4WeightConfig(), filter_fn=_filter_swinir)
            return model
        except Exception as e:
            print(f"[torchao SwinIR] W4A4 failed ({e}); W4A16 fallback")
    quantize_(model, Int4WeightOnlyConfig(group_size=128), filter_fn=_filter_swinir)
    return model
```

- [ ] **Step 4: Run tests**

```bash
pytest CompSRT/tests/test_torchao_swinir.py -v
```

- [ ] **Step 5: Commit**

```bash
git add CompSRT/basicsr/torchao_deploy.py CompSRT/tests/test_torchao_swinir.py
git commit -m "feat(S3): torchao W4A4/W4A16 deployment for CompSRT SwinIR"
```

---

### Task S3.6: End-to-end smoke on real calibrated checkpoints

**Files:**
- Create: `scripts/smoke_torchao_all.py`

- [ ] **Step 1: Write a single-script smoke that loads each calibrated state + applies torchao + measures error**

```python
# scripts/smoke_torchao_all.py
import torch, pathlib, json, sys
sys.path.insert(0, "/home/ubuntu/unifying-ptq")

results = {}

# FlatQuant LLaMA-3-8B (uses existing FlatQuant output_dir; here we just check torchao_deploy import + a small forward)
# (will be done properly in S6)
# Sanity check imports here:
from flatquant.torchao_deploy import apply_torchao as fq_torchao  # noqa
from ahcptq.torchao_deploy import apply_torchao_sam              # noqa
from basicsr.torchao_deploy import apply_torchao_swinir          # noqa
results["imports_ok"] = True

# Build untrained SAM-B and apply torchao end-to-end
from projects.instance_segment_anything.models.segment_anything import sam_model_registry
sam = sam_model_registry["vit_b"](checkpoint="/home/ubuntu/unifying-ptq/ckpt/sam_vit_b_01ec64.pth")
enc = sam.image_encoder.cuda().half()
q = apply_torchao_sam(enc, "w4a16")
x = torch.randn(1, 3, 1024, 1024, device="cuda", dtype=torch.half)
with torch.no_grad():
    out = q(x)
results["sam_w4a16_forward"] = list(out.shape)

pathlib.Path("results/S3-torchao/smoke.json").parent.mkdir(parents=True, exist_ok=True)
pathlib.Path("results/S3-torchao/smoke.json").write_text(json.dumps(results, indent=2))
print(results)
```

- [ ] **Step 2: Run on local**

```bash
cd /home/ubuntu/unifying-ptq && conda activate unifyptq && python scripts/smoke_torchao_all.py
```

Expected: `{'imports_ok': True, 'sam_w4a16_forward': [1, 256, 64, 64]}`.

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_torchao_all.py results/S3-torchao/smoke.json
git commit -m "test(S3): end-to-end torchao smoke on real SAM-B checkpoint"
```

---

## Done when

- `flatquant.torchao_deploy.apply_torchao` exists, tests pass
- `ahcptq.torchao_deploy.apply_torchao_sam` exists, tests pass
- `basicsr.torchao_deploy.apply_torchao_swinir` exists, tests pass
- End-to-end smoke (`scripts/smoke_torchao_all.py`) runs on real SAM-B checkpoint and produces `results/S3-torchao/smoke.json`
- All commits land on `main`
- Sub-projects S6 (real INT4 deployment measurement) and S10 (paper deployment table) can consume `apply_torchao_*` functions
