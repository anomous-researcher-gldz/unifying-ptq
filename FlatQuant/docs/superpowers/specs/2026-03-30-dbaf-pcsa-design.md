---
title: DBAF + PCSA Integration into FlatQuant
date: 2026-03-30
status: approved
---

# DBAF + PCSA Integration into FlatQuant

## Overview

Add two quantization aids from AHCPTQ into FlatQuant's Llama pipeline:

- **DBAF (Dynamic Block-wise Activation Folding):** fold outliers before quantization, unfold after. Applied to weights in all layers and activations in all layers except `q_proj`.
- **PCSA (Per-Cluster/anchor Scale Adjustment):** anchor-aware activation quantizer for `q_proj` only. Each forward pass assigns the input to one of 8 learned anchors; that anchor's scale/zero-point is used for quantization.

Neither DBAF nor PCSA replaces FlatQuant's transform matrices. They operate inside the quantization step, after transforms have been applied.

---

## Constraints

- DBAF lives inside quantization, not in `reparameterize()` or the transform path.
- PCSA lives only in the activation quantizer, not in weight quantization.
- DBAF and PCSA are mutually exclusive on the activation side per layer:
  - `q_proj` activations: PCSA only, no DBAF.
  - All other layer activations: DBAF only.
- `q_proj` weights still use DBAF (the exclusion is activation-side only).
- Changes apply identically to all Llama variants (`llama_utils.py`, `llama31_utils.py`).

---

## DBAF Design

### Parameters

- `k = 3.0` (fixed, same as AHCPTQ) — threshold multiplier: `T = k * std(x)`
- `alpha` — fold aggressiveness. Set to `0.99` for all Llama layers.

### Imports in `flatquant/quant_utils.py`

Import directly from AHCPTQ — do not copy or re-implement:

```python
from ahcptq.quantization.fake_quant import (
    fold_outliers,
    unfold_outliers,
    profile_with_3sigma_outliers,
    is_like_normal_plus_3sigma_outliers,
)
```

`T` is computed as `3.0 * x.std()` at the call site. DBAF is only applied when `is_like_normal_plus_3sigma_outliers(x)['is_like_c']` is `True`.

### Weight path: `flatquant/flat_linear.py → FlatQuantizedLinear._train_forward`

After the transform block, before `weight_quantizer.find_params(weight)`:

```python
_dbaf = is_like_normal_plus_3sigma_outliers(weight)['is_like_c']
if _dbaf:
    T = 3.0 * weight.std()
    weight, w_tag = dbaf_fold(weight, T, self.dbaf_alpha)

self.weight_quantizer.find_params(weight)
weight = self.weight_quantizer.quantize(weight)

if _dbaf:
    weight = dbaf_unfold(weight, w_tag, T, self.dbaf_alpha)
```

`dbaf_alpha` (float, default `0.99`) is added as a constructor arg to `FlatQuantizedLinear`. No `use_dbaf` flag — the check is always run at runtime.

### Activation path: `flatquant/quant_utils.py → ActivationQuantizer.fake_quant`

```python
def fake_quant(self, x):
    _dbaf = is_like_normal_plus_3sigma_outliers(x)['is_like_c']
    if _dbaf:
        T = 3.0 * x.std()
        x, tag = dbaf_fold(x, T, self.dbaf_alpha)

    # existing per-token scale/quantize logic
    ...

    if _dbaf:
        x = dbaf_unfold(x, tag, T, self.dbaf_alpha)
    return x
```

`dbaf_alpha` (float, default `0.99`) added to `ActivationQuantizer.__init__`. No `use_dbaf` flag. The profile check runs on every forward call; DBAF is applied only when the tensor matches the normal-with-outliers profile. `AnchorAwareActivationQuantizer` does not call this path (PCSA layers skip DBAF entirely).

---

## PCSA Design

### `flatquant/prompt_anchor.py` (new file)

Port `PromptBank` from `ahcptq/model/prompt_anchor.py`. Responsibilities:
- Stores `num_anchors` learnable anchor vectors of shape `[K, D]`.
- `assign(desc, update=False)`: cosine-distance assignment → returns `[B]` anchor_ids.
- EMA update of anchor vectors during training (`update=True`).

No structural changes to the AHCPTQ implementation; only rename/import cleanup.

### `AnchorAwareActivationQuantizer` in `flatquant/quant_utils.py`

New subclass of `ActivationQuantizer`:

```python
class AnchorAwareActivationQuantizer(ActivationQuantizer):
    def __init__(self, args, num_anchors: int = 8):
        super().__init__(args)
        self.num_anchors = num_anchors
        self.anchor_scale = None   # nn.Parameter [num_anchors, 1], initialized lazily
        self.anchor_zero  = None   # buffer [num_anchors, 1]

    def fake_quant(self, x, anchor_id=None):
        # No DBAF on this path.
        # Select scale/zero for the given anchor_id.
        # Per-token quantize using selected scale.
        # Return dequantized x.
```

Per-anchor `scale` and `zero` are `[num_anchors, 1]`, initialized from first observed statistics per anchor. `anchor_id=None` falls back to anchor 0.

### Llama wrapper changes (`llama_utils.py` and `llama31_utils.py`)

**`FlatQuantLlamaAttention.__init__`:**

```python
from flatquant.prompt_anchor import PromptBank
from flatquant.quant_utils import AnchorAwareActivationQuantizer

# After existing proj init:
self.prompt_bank = PromptBank(num_anchors=8, descriptor_dim=self.hidden_size)
self.q_proj.act_quantizer = AnchorAwareActivationQuantizer(args, num_anchors=8)

# Set dbaf_alpha on all projections (runtime profile check decides whether to apply):
for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
    proj.dbaf_alpha = 0.99
    proj.act_quantizer.dbaf_alpha = 0.99

# q_proj activation uses AnchorAwareActivationQuantizer — DBAF check is skipped there entirely.
```

**`_trans_forward_after_ln`:**

```python
def _trans_forward_after_ln(self, hidden_states):
    # PCSA descriptor
    desc = hidden_states.mean(dim=1)
    desc = desc / (desc.norm(dim=-1, keepdim=True) + 1e-6)
    anchor_ids = self.prompt_bank.assign(desc, update=not self._eval_mode)

    query_states = self.q_proj(hidden_states, anchor_id=anchor_ids)  # PCSA
    key_states   = self.k_proj(hidden_states)                         # DBAF only
    value_states = self.v_proj(hidden_states)                         # DBAF only
    return query_states, key_states, value_states
```

### `flat_linear.py` threading

`FlatQuantizedLinear._train_forward` gains `anchor_id=None`, forwarded to `self.act_quantizer.fake_quant(x, anchor_id=anchor_id)`. Base `ActivationQuantizer.fake_quant` accepts and ignores `anchor_id`.

---

## File Summary

| File | Type | Change |
|------|------|--------|
| `flatquant/quant_utils.py` | Modified | Import `fold_outliers`, `unfold_outliers`, `profile_with_3sigma_outliers`, `is_like_normal_plus_3sigma_outliers` from `ahcptq`; `dbaf_alpha` on `ActivationQuantizer`; new `AnchorAwareActivationQuantizer` |
| `flatquant/flat_linear.py` | Modified | `use_dbaf`/`dbaf_alpha` on `FlatQuantizedLinear`; DBAF fold/unfold in weight path of `_train_forward`; forward `anchor_id` to `act_quantizer` |
| `flatquant/model_tools/llama_utils.py` | Modified | `PromptBank` init; `_trans_forward_after_ln` with PCSA descriptor; swap `q_proj.act_quantizer`; set `dbaf_alpha=0.99` |
| `flatquant/model_tools/llama31_utils.py` | Modified | Same changes as `llama_utils.py` |
| `flatquant/prompt_anchor.py` | New | Port `PromptBank` from AHCPTQ |

---

## Data Flow Summary

**q_proj forward:**
```
hidden_states → mean+norm → PromptBank.assign → anchor_ids
weight path:  [transforms] → dbaf_fold → find_params → quantize → dbaf_unfold
activation:   AnchorAwareActivationQuantizer.fake_quant(x, anchor_id)  [no DBAF]
```

**k/v/o/MLP forward:**
```
weight path:  [transforms] → dbaf_fold → find_params → quantize → dbaf_unfold
activation:   ActivationQuantizer.fake_quant(x)
                └── dbaf_fold → per-token quant → dbaf_unfold
```
