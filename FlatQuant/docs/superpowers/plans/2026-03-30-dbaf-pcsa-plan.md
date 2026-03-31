# DBAF + PCSA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DBAF (outlier-fold-before-quantization) and PCSA (per-anchor activation scaling for q_proj) to FlatQuant's Llama quantization pipeline.

**Architecture:** DBAF helper functions are imported from `ahcptq` and called inside `ActivationQuantizer.fake_quant` (activations) and `FlatQuantizedLinear._train_forward` (weights) using a runtime profile check. PCSA is a new `AnchorAwareActivationQuantizer` subclass that stores 8 per-anchor scale/zero parameters and receives `anchor_id` from a `PromptBank` in the attention wrapper. The two features are mutually exclusive on the activation side per layer: `q_proj` activations use PCSA only; all other activations use DBAF only.

**Tech Stack:** PyTorch, FlatQuant (`flatquant/`), AHCPTQ (`ahcptq/`), transformers LlamaAttention/LlamaMLP

---

> **Dependency note:** `ahcptq/quantization/fake_quant.py` has top-level imports of `pandas`, `scipy`, and `scikit-learn`. Task 1 installs these so the import works at runtime.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `flatquant/prompt_anchor.py` | **Create** | `PromptBank` class — cosine-distance anchor assignment with EMA update |
| `flatquant/quant_utils.py` | **Modify** | Import DBAF functions from ahcptq; add `dbaf_alpha` to `ActivationQuantizer`; DBAF in `fake_quant`; new `AnchorAwareActivationQuantizer` |
| `flatquant/flat_linear.py` | **Modify** | Add `dbaf_alpha` to `FlatQuantizedLinear`; DBAF fold/unfold in weight path of `_train_forward`; thread `anchor_id` to `act_quantizer` |
| `flatquant/model_tools/llama_utils.py` | **Modify** | Init `PromptBank`; PCSA descriptor in `_trans_forward_after_ln`; swap `q_proj.act_quantizer` |
| `flatquant/model_tools/llama31_utils.py` | **Modify** | Same as `llama_utils.py` |
| `tests/test_dbaf_pcsa.py` | **Create** | Smoke tests for all components |

---

## Task 1: Install ahcptq runtime dependencies

**Files:**
- No code changes

- [ ] **Step 1: Install missing packages**

```bash
pip install pandas scipy scikit-learn
```

Expected: all three install without error.

- [ ] **Step 2: Verify the import works**

```bash
python3 -c "
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), '..'))
# adjust if needed:
sys.path.insert(0, '/home/unifying-ptq-sam1')
from ahcptq.quantization.fake_quant import (
    fold_outliers, unfold_outliers,
    profile_with_3sigma_outliers, is_like_normal_plus_3sigma_outliers,
)
print('ahcptq DBAF imports OK')
"
```

Expected output: `ahcptq DBAF imports OK`

---

## Task 2: Create `flatquant/prompt_anchor.py`

**Files:**
- Create: `flatquant/prompt_anchor.py`

Port `PromptAnchorBank` from `ahcptq/model/prompt_anchor.py`, renamed to `PromptBank`. Replace hardcoded `.cuda()` calls with `.to(desc.device)` for device agnosticism.

- [ ] **Step 1: Write the file**

Create `/home/unifying-ptq-sam1/FlatQuant/flatquant/prompt_anchor.py`:

```python
# Ported from ahcptq/model/prompt_anchor.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptBank(nn.Module):
    """
    Maintains K prompt anchors in descriptor space.
    Each anchor corresponds to one set of quantization scales.

    During training:
        desc = hidden_states.mean(dim=1)
        desc = desc / (desc.norm(dim=-1, keepdim=True) + 1e-6)
        anchor_ids = bank.assign(desc, update=True)

    During inference:
        anchor_ids = bank.assign(desc, update=False)
    """

    def __init__(
        self,
        num_anchors: int,
        descriptor_dim: int,
        ema_momentum: float = 0.9,
        normalize: bool = True,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.descriptor_dim = descriptor_dim
        self.ema_momentum = ema_momentum
        self.normalize = normalize

        anchors = torch.randn(num_anchors, descriptor_dim)
        anchors = F.normalize(anchors, dim=-1)
        self.register_buffer("anchors", anchors)

        counts = torch.zeros(num_anchors)
        self.register_buffer("counts", counts)

    @torch.no_grad()
    def _cosine_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x: [B, D], y: [K, D] -> cosine distance [B, K]"""
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1).to(x.device)
        return 1.0 - (x @ y.t())

    @torch.no_grad()
    def assign(self, desc: torch.Tensor, update: bool = False) -> torch.Tensor:
        """
        desc: [B, D]  (should already be L2-normalized by caller)
        returns anchor_ids: [B] in [0, K-1]
        """
        dist = self._cosine_distance(desc, self.anchors)  # [B, K]
        anchor_ids = dist.argmin(dim=-1)                   # [B]
        if update:
            self._update_anchors(desc, anchor_ids)
        return anchor_ids

    @torch.no_grad()
    def _update_anchors(self, desc: torch.Tensor, anchor_ids: torch.Tensor):
        """EMA update: a_k <- m * a_k + (1 - m) * desc[assigned_to_k]"""
        for b in range(desc.size(0)):
            k = anchor_ids[b].item()
            d = desc[b].to(self.anchors.device)
            c = self.counts[k].item()
            m = 0.0 if c < 1 else self.ema_momentum
            self.anchors[k] = F.normalize(
                m * self.anchors[k] + (1.0 - m) * d,
                dim=-1,
            )
            self.counts[k] += 1.0
```

- [ ] **Step 2: Write the smoke test**

Create `/home/unifying-ptq-sam1/FlatQuant/tests/test_dbaf_pcsa.py`:

```python
import sys
sys.path.insert(0, '/home/unifying-ptq-sam1')
sys.path.insert(0, '/home/unifying-ptq-sam1/FlatQuant')

import torch
from flatquant.prompt_anchor import PromptBank


def test_prompt_bank_assign_returns_valid_ids():
    bank = PromptBank(num_anchors=8, descriptor_dim=64)
    desc = torch.randn(4, 64)
    desc = desc / (desc.norm(dim=-1, keepdim=True) + 1e-6)
    ids = bank.assign(desc, update=False)
    assert ids.shape == (4,)
    assert ids.min() >= 0
    assert ids.max() < 8


def test_prompt_bank_ema_update_changes_anchors():
    bank = PromptBank(num_anchors=4, descriptor_dim=16)
    anchors_before = bank.anchors.clone()
    desc = torch.randn(2, 16)
    desc = desc / (desc.norm(dim=-1, keepdim=True) + 1e-6)
    bank.assign(desc, update=True)
    # at least one anchor should have changed
    assert not torch.allclose(bank.anchors, anchors_before)
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
cd /home/unifying-ptq-sam1/FlatQuant && python3 -m pytest tests/test_dbaf_pcsa.py::test_prompt_bank_assign_returns_valid_ids tests/test_dbaf_pcsa.py::test_prompt_bank_ema_update_changes_anchors -v
```

Expected: both PASS.

- [ ] **Step 4: Commit**

```bash
cd /home/unifying-ptq-sam1/FlatQuant && git add flatquant/prompt_anchor.py tests/test_dbaf_pcsa.py && git commit -m "feat: add PromptBank and initial smoke tests"
```

---

## Task 3: Add DBAF to `quant_utils.py`

**Files:**
- Modify: `flatquant/quant_utils.py`

Add the ahcptq DBAF imports at the top of the file, add `dbaf_alpha` to `ActivationQuantizer.__init__`, and wrap `fake_quant` with the runtime profile check + fold/unfold. Also update `forward` to accept and pass through `anchor_id`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_dbaf_pcsa.py`:

```python
def test_activation_quantizer_dbaf_fold_unfold():
    """DBAF fold/unfold should be a no-op on a tensor without outliers."""
    import sys
    sys.path.insert(0, '/home/unifying-ptq-sam1')
    from flatquant.quant_utils import ActivationQuantizer
    q = ActivationQuantizer(bits=8, sym=True)
    x = torch.randn(2, 16, 64)   # normal distribution — no outliers expected
    out = q(x)
    assert out.shape == x.shape


def test_activation_quantizer_dbaf_applied_on_outlier_tensor():
    """On a tensor with clear 3-sigma outliers the quantizer should still produce correct shape."""
    import sys
    sys.path.insert(0, '/home/unifying-ptq-sam1')
    from flatquant.quant_utils import ActivationQuantizer
    q = ActivationQuantizer(bits=8, sym=True)
    x = torch.randn(2, 16, 64)
    x[0, 0, 0] = 1000.0   # inject an outlier
    out = q(x)
    assert out.shape == x.shape
```

- [ ] **Step 2: Run tests to verify they fail (ActivationQuantizer missing dbaf_alpha)**

```bash
cd /home/unifying-ptq-sam1/FlatQuant && python3 -m pytest tests/test_dbaf_pcsa.py::test_activation_quantizer_dbaf_fold_unfold tests/test_dbaf_pcsa.py::test_activation_quantizer_dbaf_applied_on_outlier_tensor -v
```

Expected: both FAIL with ImportError or AttributeError.

- [ ] **Step 3: Modify `flatquant/quant_utils.py`**

At the very top of the file, after `import torch`, add:

```python
import sys as _sys
_sys.path.insert(0, '/home/unifying-ptq-sam1')
from ahcptq.quantization.fake_quant import (
    fold_outliers,
    unfold_outliers,
    profile_with_3sigma_outliers,
    is_like_normal_plus_3sigma_outliers,
)
```

Change `ActivationQuantizer.__init__` signature from:
```python
def __init__(self, bits, sym=False, lac=False, groupsize=-1, clip_ratio=None, ):
```
to:
```python
def __init__(self, bits, sym=False, lac=False, groupsize=-1, clip_ratio=None, dbaf_alpha=0.99):
```

Add `self.dbaf_alpha = dbaf_alpha` inside `__init__`, after `self.enable = True`.

Change `ActivationQuantizer.forward` from:
```python
def forward(self, x):
    if self.bits == 16 or (not self.enable):
        return x
    fq_x = self.fake_quant(x)
    return fq_x
```
to:
```python
def forward(self, x, anchor_id=None):
    if self.bits == 16 or (not self.enable):
        return x
    fq_x = self.fake_quant(x, anchor_id=anchor_id)
    return fq_x
```

Change `ActivationQuantizer.fake_quant` from:
```python
def fake_quant(self, x):
    x_dtype = x.dtype
    scale, zero = self.get_scale_zero(x)
    if self.sym:
        return sym_quant_dequant(x, scale, self.q_max.to(x)).to(x_dtype)
    else:
        return asym_quant_dequant(x, scale, zero, self.q_max.to(x)).to(x_dtype)  # TODO
```
to:
```python
def fake_quant(self, x, anchor_id=None):
    x_dtype = x.dtype
    _apply_dbaf = is_like_normal_plus_3sigma_outliers(x)['is_like_c']
    if _apply_dbaf:
        T = float(3.0 * x.detach().float().std().clamp_min(1e-8))
        x, _tag = fold_outliers(x, T, self.dbaf_alpha)
    scale, zero = self.get_scale_zero(x)
    if self.sym:
        result = sym_quant_dequant(x, scale, self.q_max.to(x)).to(x_dtype)
    else:
        result = asym_quant_dequant(x, scale, zero, self.q_max.to(x)).to(x_dtype)
    if _apply_dbaf:
        result = unfold_outliers(result, _tag, T, self.dbaf_alpha)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/unifying-ptq-sam1/FlatQuant && python3 -m pytest tests/test_dbaf_pcsa.py::test_activation_quantizer_dbaf_fold_unfold tests/test_dbaf_pcsa.py::test_activation_quantizer_dbaf_applied_on_outlier_tensor -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/unifying-ptq-sam1/FlatQuant && git add flatquant/quant_utils.py tests/test_dbaf_pcsa.py && git commit -m "feat: add DBAF imports and runtime profile-gated fold/unfold to ActivationQuantizer"
```

---

## Task 4: Add `AnchorAwareActivationQuantizer` to `quant_utils.py`

**Files:**
- Modify: `flatquant/quant_utils.py`

New subclass of `ActivationQuantizer`. Stores `[num_anchors, 1]` learnable scale and zero parameters. Overrides `fake_quant` to select per-anchor scale/zero and apply quantization — no DBAF on this path.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_dbaf_pcsa.py`:

```python
def test_anchor_aware_quantizer_basic():
    import sys
    sys.path.insert(0, '/home/unifying-ptq-sam1')
    from flatquant.quant_utils import AnchorAwareActivationQuantizer
    q = AnchorAwareActivationQuantizer(bits=8, sym=True, num_anchors=8)
    x = torch.randn(3, 16, 64)
    anchor_id = torch.tensor([0, 3, 7])
    out = q(x, anchor_id=anchor_id)
    assert out.shape == x.shape


def test_anchor_aware_quantizer_no_anchor_id_falls_back():
    import sys
    sys.path.insert(0, '/home/unifying-ptq-sam1')
    from flatquant.quant_utils import AnchorAwareActivationQuantizer
    q = AnchorAwareActivationQuantizer(bits=8, sym=True, num_anchors=8)
    x = torch.randn(2, 16, 64)
    out = q(x)   # no anchor_id — should fall back to anchor 0
    assert out.shape == x.shape
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/unifying-ptq-sam1/FlatQuant && python3 -m pytest tests/test_dbaf_pcsa.py::test_anchor_aware_quantizer_basic tests/test_dbaf_pcsa.py::test_anchor_aware_quantizer_no_anchor_id_falls_back -v
```

Expected: FAIL with `ImportError: cannot import name 'AnchorAwareActivationQuantizer'`.

- [ ] **Step 3: Add `AnchorAwareActivationQuantizer` to `flatquant/quant_utils.py`**

Append the following class at the end of the `ActivationQuantizer` section (before `WeightQuantizer`):

```python
class AnchorAwareActivationQuantizer(ActivationQuantizer):
    """
    Per-anchor activation quantizer for PCSA.
    One scale/zero per anchor; anchor_id selects which to use.
    DBAF is NOT applied on this path (q_proj activations only).
    """

    def __init__(self, bits, sym=False, lac=False, groupsize=-1, clip_ratio=None,
                 dbaf_alpha=0.99, num_anchors=8):
        super().__init__(bits, sym=sym, lac=lac, groupsize=groupsize,
                         clip_ratio=clip_ratio, dbaf_alpha=dbaf_alpha)
        self.num_anchors = num_anchors
        # Per-anchor scale and zero; initialized lazily from first-batch stats.
        self.anchor_scale = torch.nn.Parameter(
            torch.ones(num_anchors, 1), requires_grad=True
        )
        self.register_buffer('anchor_zero', torch.zeros(num_anchors, 1))
        self.register_buffer(
            '_anchor_initialized', torch.zeros(num_anchors, dtype=torch.bool)
        )

    def _init_anchor(self, x_anchor: torch.Tensor, k: int):
        """Initialize scale/zero for anchor k from observed statistics."""
        q_max = self.q_max.to(x_anchor)
        flat = x_anchor.detach().reshape(-1)
        xmax = flat.max().clamp(min=0)
        xmin = flat.min().clamp(max=0)
        if self.sym:
            xmax = torch.maximum(xmax.abs(), xmin.abs())
            scale = (xmax / q_max).clamp(min=1e-8)
            zero = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) and (xmax == 0)
            if tmp:
                xmin, xmax = torch.tensor(-1.0), torch.tensor(1.0)
            scale = ((xmax - xmin) / q_max).clamp(min=1e-8)
            zero = torch.round(-xmin / scale)
        self.anchor_scale.data[k] = scale.reshape(1)
        self.anchor_zero[k] = zero.reshape(1)

    def fake_quant(self, x: torch.Tensor, anchor_id=None) -> torch.Tensor:
        # No DBAF on this path.
        x_dtype = x.dtype
        B = x.shape[0]
        if anchor_id is None:
            anchor_id = torch.zeros(B, dtype=torch.long, device=x.device)

        # Lazy per-anchor initialization from first batch
        for k in range(self.num_anchors):
            if not self._anchor_initialized[k].item():
                mask = (anchor_id == k)
                if mask.any():
                    self._init_anchor(x[mask], k)
                    self._anchor_initialized[k] = True

        # Gather per-sample scale/zero: [B, *1s] for broadcasting over x
        extra_dims = x.dim() - 1
        scale = self.anchor_scale[anchor_id].view(B, *([1] * extra_dims))
        zero  = self.anchor_zero[anchor_id].view(B, *([1] * extra_dims))

        q_max = self.q_max.to(x)
        if self.sym:
            return sym_quant_dequant(x, scale, q_max).to(x_dtype)
        else:
            return asym_quant_dequant(x, scale, zero, q_max).to(x_dtype)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/unifying-ptq-sam1/FlatQuant && python3 -m pytest tests/test_dbaf_pcsa.py::test_anchor_aware_quantizer_basic tests/test_dbaf_pcsa.py::test_anchor_aware_quantizer_no_anchor_id_falls_back -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/unifying-ptq-sam1/FlatQuant && git add flatquant/quant_utils.py tests/test_dbaf_pcsa.py && git commit -m "feat: add AnchorAwareActivationQuantizer for PCSA"
```

---

## Task 5: Update `flat_linear.py`

**Files:**
- Modify: `flatquant/flat_linear.py`

Add `dbaf_alpha` to `FlatQuantizedLinear.__init__`. Add DBAF fold/unfold around the weight quantization step in `_train_forward`. Thread `anchor_id` through `forward` and `_train_forward` to `act_quantizer`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_dbaf_pcsa.py`:

```python
def test_flat_quantized_linear_forward_with_anchor_id():
    import sys
    sys.path.insert(0, '/home/unifying-ptq-sam1')
    import torch.nn as nn
    from types import SimpleNamespace
    from flatquant.flat_linear import FlatQuantizedLinear
    from flatquant.quant_utils import AnchorAwareActivationQuantizer

    args = SimpleNamespace(
        w_bits=8, w_asym=False, a_bits=8, a_asym=False,
        lac=False, a_groupsize=-1, lwc=False,
    )
    linear = nn.Linear(64, 32, bias=False)
    layer = FlatQuantizedLinear(args, linear)
    # Swap act_quantizer for anchor-aware version (as llama_utils will do)
    layer.act_quantizer = AnchorAwareActivationQuantizer(bits=8, sym=True, num_anchors=8)

    x = torch.randn(2, 4, 64)
    anchor_id = torch.tensor([0, 3])
    out = layer(x, anchor_id=anchor_id)
    assert out.shape == (2, 4, 32)


def test_flat_quantized_linear_forward_no_anchor_id():
    import sys
    sys.path.insert(0, '/home/unifying-ptq-sam1')
    import torch.nn as nn
    from types import SimpleNamespace
    from flatquant.flat_linear import FlatQuantizedLinear

    args = SimpleNamespace(
        w_bits=8, w_asym=False, a_bits=8, a_asym=False,
        lac=False, a_groupsize=-1, lwc=False,
    )
    linear = nn.Linear(64, 32, bias=False)
    layer = FlatQuantizedLinear(args, linear)
    x = torch.randn(2, 4, 64)
    out = layer(x)   # no anchor_id — base ActivationQuantizer ignores it
    assert out.shape == (2, 4, 32)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/unifying-ptq-sam1/FlatQuant && python3 -m pytest tests/test_dbaf_pcsa.py::test_flat_quantized_linear_forward_with_anchor_id tests/test_dbaf_pcsa.py::test_flat_quantized_linear_forward_no_anchor_id -v
```

Expected: FAIL — `forward()` doesn't accept `anchor_id`.

- [ ] **Step 3: Modify `flatquant/flat_linear.py`**

Change the import line at the top from:
```python
from flatquant.quant_utils import WeightQuantizer, ActivationQuantizer
```
to:
```python
from flatquant.quant_utils import (
    WeightQuantizer, ActivationQuantizer,
    fold_outliers, unfold_outliers, is_like_normal_plus_3sigma_outliers,
)
```

Change `FlatQuantizedLinear.__init__` signature from:
```python
def __init__(self, args, linear: nn.Linear):
```
to:
```python
def __init__(self, args, linear: nn.Linear, dbaf_alpha: float = 0.99):
```

Add `self.dbaf_alpha = dbaf_alpha` inside `__init__`, after `self._eval_mode = False`.

Change `_train_forward` from:
```python
def _train_forward(self, hidden_states, qa_trans=None, out_trans=None):
    weight = self.linear.weight.data
    # quantization-adaptive transform
    if qa_trans is not None:
        weight = self.apply_trans(weight, qa_trans)
    # learnable weight clipping
    if self.lwc:
        weight = self.apply_wclip(weight)
    if out_trans is not None:
        weight = out_trans(weight.T).T

    # quantize weight
    self.weight_quantizer.find_params(weight)
    weight = self.weight_quantizer(weight)
    # quantize activation
    hidden_states = self.act_quantizer(hidden_states)
```
to:
```python
def _train_forward(self, hidden_states, qa_trans=None, out_trans=None, anchor_id=None):
    weight = self.linear.weight.data
    # quantization-adaptive transform
    if qa_trans is not None:
        weight = self.apply_trans(weight, qa_trans)
    # learnable weight clipping
    if self.lwc:
        weight = self.apply_wclip(weight)
    if out_trans is not None:
        weight = out_trans(weight.T).T

    # DBAF: fold weight outliers if tensor matches normal-with-outliers profile
    _dbaf_w = is_like_normal_plus_3sigma_outliers(weight)['is_like_c']
    if _dbaf_w:
        _T_w = float(3.0 * weight.detach().float().std().clamp_min(1e-8))
        weight, _w_tag = fold_outliers(weight, _T_w, self.dbaf_alpha)

    # quantize weight
    self.weight_quantizer.find_params(weight)
    weight = self.weight_quantizer(weight)

    if _dbaf_w:
        weight = unfold_outliers(weight, _w_tag, _T_w, self.dbaf_alpha)

    # quantize activation
    hidden_states = self.act_quantizer(hidden_states, anchor_id=anchor_id)
```

Change `forward` from:
```python
def forward(self, hidden_states, qa_trans=None, out_trans=None):
    if not self._eval_mode:
        return self._train_forward(hidden_states, qa_trans=qa_trans, out_trans=out_trans)
    else:
        return self._eval_forward(hidden_states)
```
to:
```python
def forward(self, hidden_states, qa_trans=None, out_trans=None, anchor_id=None):
    if not self._eval_mode:
        return self._train_forward(hidden_states, qa_trans=qa_trans, out_trans=out_trans, anchor_id=anchor_id)
    else:
        return self._eval_forward(hidden_states)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/unifying-ptq-sam1/FlatQuant && python3 -m pytest tests/test_dbaf_pcsa.py::test_flat_quantized_linear_forward_with_anchor_id tests/test_dbaf_pcsa.py::test_flat_quantized_linear_forward_no_anchor_id -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/unifying-ptq-sam1/FlatQuant && git add flatquant/flat_linear.py tests/test_dbaf_pcsa.py && git commit -m "feat: add DBAF to FlatQuantizedLinear weight path and thread anchor_id to act_quantizer"
```

---

## Task 6: Update `llama_utils.py`

**Files:**
- Modify: `flatquant/model_tools/llama_utils.py`

Add `PromptBank` to `FlatQuantLlamaAttention.__init__`, swap `q_proj.act_quantizer` to `AnchorAwareActivationQuantizer`, and update `_trans_forward_after_ln` to compute the PCSA descriptor and pass `anchor_id` to `q_proj` only.

- [ ] **Step 1: Add imports**

In `flatquant/model_tools/llama_utils.py`, change:
```python
from flatquant.quant_utils import ActivationQuantizer
```
to:
```python
from flatquant.quant_utils import ActivationQuantizer, AnchorAwareActivationQuantizer
from flatquant.prompt_anchor import PromptBank
```

- [ ] **Step 2: Update `FlatQuantLlamaAttention.__init__`**

After `self.add_fq_trans()` and before the cache quantizer block, add:

```python
        # PCSA: prompt bank for q_proj anchor assignment
        self.prompt_bank = PromptBank(num_anchors=8, descriptor_dim=self.config.hidden_size)
        self.q_proj.act_quantizer = AnchorAwareActivationQuantizer(
            bits=args.a_bits,
            sym=not args.a_asym,
            lac=args.lac,
            groupsize=args.a_groupsize,
            num_anchors=8,
        )
```

- [ ] **Step 3: Update `_trans_forward_after_ln`**

Replace the existing method:
```python
    def _trans_forward_after_ln(self, hidden_states):
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)
        query_states = self.q_proj(hidden_states, qa_trans=self.ln_trans)
        key_states = self.k_proj(hidden_states, qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans)
        else:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        return query_states, key_states, value_states
```
with:
```python
    def _trans_forward_after_ln(self, hidden_states):
        # PCSA: compute descriptor from pre-transform hidden states
        desc = hidden_states.mean(dim=1)                          # [B, D]
        desc = desc / (desc.norm(dim=-1, keepdim=True) + 1e-6)
        anchor_ids = self.prompt_bank.assign(desc, update=not self._eval_mode)

        # Apply FlatQuant transform
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)

        # q_proj receives anchor_id (PCSA); k/v do not
        query_states = self.q_proj(hidden_states, qa_trans=self.ln_trans, anchor_id=anchor_ids)
        key_states   = self.k_proj(hidden_states, qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans)
        else:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        return query_states, key_states, value_states
```

- [ ] **Step 4: Run smoke test**

Add to `tests/test_dbaf_pcsa.py`:

```python
def test_llama_utils_attention_forward_smoke():
    """Smoke test: FlatQuantLlamaAttention forward pass completes without error."""
    import sys
    sys.path.insert(0, '/home/unifying-ptq-sam1')
    from types import SimpleNamespace
    import torch
    # Use transformers LlamaAttention as base
    try:
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention
        from flatquant.model_tools.llama_utils import FlatQuantLlamaAttention
    except ImportError:
        import pytest
        pytest.skip("transformers not available")

    config = LlamaConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=4,
        intermediate_size=128, max_position_embeddings=32,
    )
    args = SimpleNamespace(
        w_bits=8, w_asym=False, a_bits=8, a_asym=False,
        q_bits=16, k_bits=16, v_bits=16, q_asym=False, k_asym=False, v_asym=False,
        lac=False, a_groupsize=-1, lwc=False, direct_inv=False,
        add_diag=False, diag_init=None, separate_vtrans=False,
    )
    base_attn = LlamaAttention(config, layer_idx=0)
    attn = FlatQuantLlamaAttention(args, base_attn)

    hidden = torch.randn(2, 8, 64)
    q, k, v = attn._trans_forward_after_ln(hidden)
    assert q.shape == (2, 8, 64)
    assert k.shape == (2, 8, 64)
    assert v.shape == (2, 8, 64)
```

```bash
cd /home/unifying-ptq-sam1/FlatQuant && python3 -m pytest tests/test_dbaf_pcsa.py::test_llama_utils_attention_forward_smoke -v
```

Expected: PASS (or SKIP if transformers not installed).

- [ ] **Step 5: Commit**

```bash
cd /home/unifying-ptq-sam1/FlatQuant && git add flatquant/model_tools/llama_utils.py tests/test_dbaf_pcsa.py && git commit -m "feat: add PCSA PromptBank and anchor-aware q_proj to FlatQuantLlamaAttention (llama)"
```

---

## Task 7: Update `llama31_utils.py`

**Files:**
- Modify: `flatquant/model_tools/llama31_utils.py`

Identical changes to Task 6 but applied to the Llama 3.1 wrapper.

- [ ] **Step 1: Add imports**

In `flatquant/model_tools/llama31_utils.py`, change:
```python
from flatquant.quant_utils import ActivationQuantizer
```
to:
```python
from flatquant.quant_utils import ActivationQuantizer, AnchorAwareActivationQuantizer
from flatquant.prompt_anchor import PromptBank
```

- [ ] **Step 2: Update `FlatQuantLlamaAttention.__init__`**

After `self.add_fq_trans()` and before the cache quantizer block, add:

```python
        # PCSA: prompt bank for q_proj anchor assignment
        self.prompt_bank = PromptBank(num_anchors=8, descriptor_dim=self.config.hidden_size)
        self.q_proj.act_quantizer = AnchorAwareActivationQuantizer(
            bits=args.a_bits,
            sym=not args.a_asym,
            lac=args.lac,
            groupsize=args.a_groupsize,
            num_anchors=8,
        )
```

- [ ] **Step 3: Update `_trans_forward_after_ln`**

Replace the existing method:
```python
    def _trans_forward_after_ln(self, hidden_states):
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)
        query_states = self.q_proj(hidden_states, qa_trans=self.ln_trans)
        key_states = self.k_proj(hidden_states, qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans)
        else:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        return query_states, key_states, value_states
```
with:
```python
    def _trans_forward_after_ln(self, hidden_states):
        # PCSA: compute descriptor from pre-transform hidden states
        desc = hidden_states.mean(dim=1)                          # [B, D]
        desc = desc / (desc.norm(dim=-1, keepdim=True) + 1e-6)
        anchor_ids = self.prompt_bank.assign(desc, update=not self._eval_mode)

        # Apply FlatQuant transform
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)

        # q_proj receives anchor_id (PCSA); k/v do not
        query_states = self.q_proj(hidden_states, qa_trans=self.ln_trans, anchor_id=anchor_ids)
        key_states   = self.k_proj(hidden_states, qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans)
        else:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        return query_states, key_states, value_states
```

- [ ] **Step 4: Run the full test suite**

```bash
cd /home/unifying-ptq-sam1/FlatQuant && python3 -m pytest tests/test_dbaf_pcsa.py -v
```

Expected: all tests PASS (or SKIP for those requiring transformers).

- [ ] **Step 5: Commit**

```bash
cd /home/unifying-ptq-sam1/FlatQuant && git add flatquant/model_tools/llama31_utils.py && git commit -m "feat: add PCSA PromptBank and anchor-aware q_proj to FlatQuantLlamaAttention (llama31)"
```

---

## Self-Review

**Spec coverage check:**
- ✅ DBAF helper imports from ahcptq (Task 1, 3)
- ✅ DBAF weight path in `_train_forward` (Task 5)
- ✅ DBAF activation path in `ActivationQuantizer.fake_quant` (Task 3)
- ✅ DBAF not in `reparameterize()` — untouched
- ✅ PCSA `AnchorAwareActivationQuantizer` in `quant_utils.py` (Task 4)
- ✅ `PromptBank` in `flatquant/prompt_anchor.py` (Task 2)
- ✅ `anchor_id` from descriptor in `_trans_forward_after_ln` (Tasks 6, 7)
- ✅ `anchor_id` threaded through `flat_linear.py` (Task 5)
- ✅ `q_proj` only gets PCSA; k/v/o/MLP untouched (Tasks 6, 7)
- ✅ `q_proj` weights still get DBAF (Task 5 — weight path runs regardless of act_quantizer type)
- ✅ `AnchorAwareActivationQuantizer.fake_quant` has no DBAF (Task 4)
- ✅ Changes applied to both `llama_utils.py` and `llama31_utils.py` (Tasks 6, 7)
- ✅ `dbaf_alpha=0.99` is the default everywhere (Tasks 3, 5)
- ✅ `k=3.0` is hardcoded (not configurable) in the T computation (Tasks 3, 5)
