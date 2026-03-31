# ahcptq/model/quant_model_sam2.py

import math
from typing import Type, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from sam2.modeling.backbones.image_encoder import ImageEncoder as SAM2ImageEncoder
from sam2.modeling.sam.transformer import (
    TwoWayTransformer,
    TwoWayAttentionBlock,
    Attention,
    RoPEAttention,
)
from sam2.modeling.sam2_utils import MLP

from ahcptq.quantization.quantized_module import (
    QuantizedBlock,
    PreQuantizedLayer,
    Quantizer,
)

# SAM2 / ISA encoder implementation
from projects.instance_segment_anything.models.segment_anything.modeling.image_encoder import (
    ImageEncoderViT as SAImageEncoderViT,
)

# Reuse the SAM1 encoder quantization implementation
from ahcptq.model.quant_model import (
    QuantImageEncoderOurViT,
)
import torch.nn.functional as F
# interpolate shim: accept `antialias` but ignore it for older Torch
_orig_interpolate = F.interpolate

def _interpolate_compat(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
    antialias=None,  # SAM2 passes this, old Torch doesn't know it
):
    return _orig_interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )

F.interpolate = _interpolate_compat

_orig_where = torch.where

def _where_compat(condition, x=None, y=None):
    # 1-arg form: torch.where(condition) -> indices
    if x is None and y is None:
        return _orig_where(condition)

    # 3-arg form: torch.where(condition, x, y)
    # Promote non-tensors to tensors when needed
    if isinstance(x, torch.Tensor) and not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=x.dtype, device=x.device)
    elif isinstance(y, torch.Tensor) and not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=y.dtype, device=y.device)
    elif not isinstance(x, torch.Tensor) and not isinstance(y, torch.Tensor):
        # both scalars: just let PyTorch handle it normally
        return _orig_where(condition, x, y)

    # Now both x and y are tensors
    # Align devices
    if x.device != y.device:
        y = y.to(x.device)

    # Align dtypes (fixes float64 vs float32, etc.)
    if x.dtype != y.dtype:
        y = y.to(x.dtype)

    return _orig_where(condition, x, y)

# Monkey-patch
torch.where = _where_compat
# ---------------------------------------------------------------------
# Torch <= 1.x / early 2.x compatibility for scaled_dot_product_attention
# ---------------------------------------------------------------------
if not hasattr(F, "scaled_dot_product_attention"):
    def _scaled_dot_product_attention_compat(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ):
        """
        Simple implementation of scaled dot-product attention
        with the same signature as torch.nn.functional.scaled_dot_product_attention.
        Shapes:
            q, k, v: (..., L_q, D), (..., L_k, D), (..., L_k, D)
        """
        d_k = q.size(-1)
        scale = 1.0 / math.sqrt(d_k)

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (..., L_q, L_k)

        if is_causal:
            L_q, L_k = attn.size(-2), attn.size(-1)
            causal_mask = torch.triu(
                torch.ones(L_q, L_k, dtype=torch.bool, device=attn.device),
                diagonal=1,
            )
            attn = attn.masked_fill(causal_mask, float("-inf"))

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = torch.softmax(attn, dim=-1)

        if dropout_p > 0.0 and torch.is_grad_enabled():
            attn = F.dropout(attn, p=dropout_p)

        out = torch.matmul(attn, v)  # (..., L_q, D)
        return out

    F.scaled_dot_product_attention = _scaled_dot_product_attention_compat

# ---------------------------------------------------------------------
# Helpers: specialized activation quantizers (AGQ / BIG) like SAM1
# ---------------------------------------------------------------------

def update_specialized_quantizer_config(base_config, quantizer_name):
    """
    Copy base_config and override quantizer / observer to match
    AGQ (softmax) or BIG (bimodal) behavior.
    """
    import copy

    specialized_config = copy.deepcopy(base_config)
    update_keys = {
        "softmax": {
            "quantizer": "AdaptiveGranularityQuantize",
            "observer": "LogAvgMSEFastObserver",
        },
        "bimodal": {
            "quantizer": "LSQSignFakeQuantize",
            "observer": "SignAvgMSEFastObserver",
        },
        "anchor_aware": {
            'quantizer': 'AnchorAwareFakeQuantize',
            'observer':  "MSEObserver",    # reuse whatever was already set
        }
    }[quantizer_name]
    specialized_config.update(update_keys)
    return specialized_config


from sam2.modeling.sam2_utils import MLP  # you already import this

import torch
import torch.nn as nn
from torch import Tensor

from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.backbones.hieradet import (
    MultiScaleBlock,
    MultiScaleAttention,
    do_pool,
)
from sam2.modeling.backbones.utils import window_partition, window_unpartition
from sam2.modeling.sam2_utils import MLP

from ahcptq.quantization.quantized_module import (
    PreQuantizedLayer,
    QuantizedBlock,
)


class QuantSAM2EncoderMLP(QuantizedBlock):
    """
    Quantized version of sam2.modeling.sam2_utils.MLP used inside MultiScaleBlock.
    Mirrors QuantEncoderMLPBlock logic (CAG/HLUQ) but for SAM2 MLP.
    """

    def __init__(self, org_module: MLP, w_qconfig, a_qconfig, ahcptq_config, qinput=True):
        super().__init__()

        # org_module.layers is a ModuleList of Linear layers, act is e.g. GELU
        assert len(org_module.layers) == 2, "Expected 2-layer MLP for SAM2 encoder."

        if ahcptq_config.cag:
            lin1_type = "group"
        else:
            lin1_type = "normal"

        if ahcptq_config.hluq:
            lin2_type = "hybrid"
        else:
            lin2_type = "normal"

        self.lin1 = PreQuantizedLayer(
            org_module.layers[0], None, w_qconfig, a_qconfig, lin1_type, qinput=qinput
        )
        self.lin2 = PreQuantizedLayer(
            org_module.layers[1], None, w_qconfig, a_qconfig, lin2_type, qinput=qinput
        )
        self.act = org_module.act

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        return x


class QuantSAM2EncoderAttention(QuantizedBlock):
    """
    Quantized MultiScaleAttention for Hiera encoder.

    - Wraps qkv and proj with PreQuantizedLayer (weight + input quant).
    - Quantizes q, k, v before scaled dot product attention.
    - Keeps the same interface as MultiScaleAttention.forward(x: BHWC).
    """

    def __init__(
        self,
        org_module: MultiScaleAttention,
        w_qconfig,
        a_qconfig,
        ahcptq_config,
        ptq4sam_config,  # kept for symmetry with SAM1; not fully used here yet
        qoutput=False,
        qinput=True,
    ):
        super().__init__()
        self.dim = org_module.dim
        self.dim_out = org_module.dim_out
        self.num_heads = org_module.num_heads
        self.q_pool = org_module.q_pool

        # For encoder attention we can optionally use "group" for qkv if CAG is enabled
        if ahcptq_config.cag:
            qkv_type = "group"
        else:
            qkv_type = "normal"

        self.qkv = PreQuantizedLayer(
            org_module.qkv,
            None,
            w_qconfig,
            a_qconfig,
            qkv_type,
            qinput=qinput,
        )
        self.proj = PreQuantizedLayer(
            org_module.proj,
            None,
            w_qconfig,
            a_qconfig,
            "normal",
            qinput=qinput,
        )

        # Simple per-tensor activation quantizers for q, k, v.
        # (You can later add AGQ/BIG style specializations if you want.)
        from ahcptq.quantization.quantized_module import Quantizer
        if ptq4sam_config.AGQ:
            softmax_a_config = update_specialized_quantizer_config(a_qconfig,'softmax')
        else:
            softmax_a_config = a_qconfig
        
        self.softmax_post_act_fake_quantize = Quantizer(None, softmax_a_config)

        self.q_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.k_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, H, W, C=dim)
        returns: (B, H' (maybe pooled), W' (maybe pooled), dim_out)
        """
        B, H, W, _ = x.shape

        # qkv with shape (B, H * W, 3, nHead, C_head)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nHead, C_head)
        q, k, v = torch.unbind(qkv, 2)

        # Quantize q, k, v before attention
        q = self.q_post_act_fake_quantize(q)
        k = self.k_post_act_fake_quantize(k)
        v = self.v_post_act_fake_quantize(v)

        # Q pooling (for downsample at stage changes), same logic as original
        if self.q_pool is not None:
            # shape: (B, H, W, nHead * C_head) for pooling
            q = q.reshape(B, H, W, -1)
            q = do_pool(q, self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)

        q_t = q.transpose(1, 2)  # (B, nH, Lq, C)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        d = q_t.shape[-1]
        attn = (q_t @ k_t.transpose(-2, -1)) / math.sqrt(d)
        attn = torch.softmax(attn, dim=-1)

        attn = self.softmax_post_act_fake_quantize(attn, value=v_t)
        # print("we did do softmaxvpost act")
        x = attn @ v_t
        x = x.transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class QuantSAM2EncoderBlock(nn.Module):
    """
    Quantized version of MultiScaleBlock.

    Mirrors MultiScaleBlock.forward but swaps:
      - self.attn -> QuantSAM2EncoderAttention
      - self.mlp -> QuantSAM2EncoderMLP
    and keeps pool/proj/drop_path/norms identical so Hiera.forward still works.
    """

    def __init__(
        self,
        org_block: MultiScaleBlock,
        w_qconfig,
        a_qconfig,
        ahcptq_config,
        ptq4sam_config,
    ):
        super().__init__()

        # Copy scalar attributes
        self.dim = org_block.dim
        self.dim_out = org_block.dim_out
        self.window_size = org_block.window_size
        self.q_stride = org_block.q_stride

        # Reuse original norms / pool / drop_path / proj
        self.norm1 = org_block.norm1
        self.pool = org_block.pool
        self.drop_path = org_block.drop_path
        self.norm2 = org_block.norm2
        self.proj = getattr(org_block, "proj", None)

        # Quantized attention + MLP
        self.attn = QuantSAM2EncoderAttention(
            org_block.attn,
            w_qconfig,
            a_qconfig,
            ahcptq_config,
            ptq4sam_config,
        )
        self.mlp = QuantSAM2EncoderMLP(
            org_block.mlp,
            w_qconfig,
            a_qconfig,
            ahcptq_config,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, H, W, C=dim)
        returns: (B, H' (maybe pooled), W' (maybe pooled), dim_out)
        """
        shortcut = x  # (B, H, W, C)
        x = self.norm1(x)

        # Skip connection path: project + pool if dim change, exactly like original
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Attention + possible Q pooling
        x = self.attn(x)

        # If Q pooling changed spatial size, recompute window size & padding as original
        if self.q_stride is not None:
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        # Residual + MLP, same as original
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class QuantSAM2ImageEncoder(nn.Module):
    """
    Quantized wrapper around sam2.modeling.backbones.image_encoder.ImageEncoder.

    - Replaces each Hiera MultiScaleBlock in trunk.blocks with QuantSAM2EncoderBlock.
    - Quantizes FpnNeck convs using PreQuantizedLayer (unless already quantized).
    - Keeps forward() API exactly the same so sam2_image_predictor continues to work.
    """

    def __init__(
        self,
        org_module: ImageEncoder,
        w_qconfig,
        a_qconfig,
        ahcptq_config,
        ptq4sam_config,
        qoutput: bool = True,
    ):
        super().__init__()

        # Keep scalar + structural attributes
        self.scalp = org_module.scalp

        # --- Trunk (Hiera) ---

       
        self.trunk = org_module.trunk  # Hiera instance
    

        # Replace each MultiScaleBlock with QuantSAM2EncoderBlock
        for i, blk in enumerate(self.trunk.blocks):
            if isinstance(blk, MultiScaleBlock):
                self.trunk.blocks[i] = QuantSAM2EncoderBlock(
                    blk, w_qconfig, a_qconfig, ahcptq_config, ptq4sam_config
                )

        # --- Neck (FpnNeck) ---
        self.neck = org_module.neck  # FpnNeck instance
        # Quantize the 1x1 convs if they are still plain Conv2d
        for seq in self.neck.convs:
            # seq is an nn.Sequential with "conv" registered
            conv = getattr(seq, "conv", None)
            if isinstance(conv, nn.Conv2d):
                seq.conv = PreQuantizedLayer(
                    conv, None, w_qconfig, a_qconfig, "normal", qinput=True
                )

        # Preserve the channel_list assertion: trunk.channel_list is unchanged,
        # and neck.backbone_channel_list is unchanged, so ImageEncoder behavior
        # is structurally identical.

    def forward(self, sample: torch.Tensor):
        # Exactly the same logic as original ImageEncoder.forward
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output
# =====================================================================
# Quantized Attention (SAM2) – like QuantDecoderOurAttentionBlock
# =====================================================================
from .prompt_anchor import PromptAnchorBank
class QuantSAM2Attention(QuantizedBlock):
    """
    Quantized version of sam2.modeling.transformer.Attention / RoPEAttention.

    - q_proj / k_proj / v_proj use PreQuantizedLayer, with
      type='group' when ahcptq_config.cag else 'normal'
    - out_proj uses 'normal'
    - optional AGQ/BIG on attention
    """

    def __init__(
        self,
        org_module: Attention,
        w_qconfig,
        a_qconfig,
        ahcptq_config,
        ptq4sam_config,
        qoutput: bool = True,
        qinput: bool = False,
    ):
        super().__init__()
        self.qoutput = qoutput
        self.embedding_dim = org_module.embedding_dim
        self.internal_dim = org_module.internal_dim
        self.num_heads = org_module.num_heads
        self.dropout_p = org_module.dropout_p

        # group for projections if CAG enabled
        if getattr(ahcptq_config, "cag", False):
            proj_type = "group"
        else:
            proj_type = "normal"
        self.q_proj = PreQuantizedLayer(
            org_module.q_proj, None, w_qconfig, a_qconfig, proj_type
        )
        self.k_proj = PreQuantizedLayer(
            org_module.k_proj, None, w_qconfig, a_qconfig, proj_type
        )
        self.v_proj = PreQuantizedLayer(
            org_module.v_proj, None, w_qconfig, a_qconfig, proj_type
        )
        self.out_proj = PreQuantizedLayer(
            org_module.out_proj, None, w_qconfig, a_qconfig
        )

        # AGQ / BIG configs
        if getattr(ptq4sam_config, "AGQ", False):
            softmax_a_config = update_specialized_quantizer_config(a_qconfig, "softmax")
        else:
            softmax_a_config = a_qconfig
        if getattr(ptq4sam_config, "BIG", False):
            sign_a_config = update_specialized_quantizer_config(a_qconfig, "bimodal")
        else:
            sign_a_config = a_qconfig

        self.softmax_post_act_fake_quantize = Quantizer(None, softmax_a_config)

        # Anchor-aware config for prompt-conditioned scales
        if getattr(ptq4sam_config, "ANCHOR_AWARE", False):
            anchor_a_config = update_specialized_quantizer_config(a_qconfig, "anchor_aware")
        else:
            anchor_a_config = a_qconfig
        self.q_post_act_fake_quantize = Quantizer(None, anchor_a_config)
        self.k_post_act_fake_quantize = Quantizer(None, sign_a_config)
        self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)

        if getattr(ptq4sam_config, "BIG", False):
            # extra BIG knobs if they exist in your Quantizer subclass
            self.k_post_act_fake_quantize.global_num = ptq4sam_config.global_num
            self.k_post_act_fake_quantize.peak_distance = ptq4sam_config.peak_distance
            self.k_post_act_fake_quantize.peak_height = ptq4sam_config.peak_height

        self.prompt_bank = PromptAnchorBank(
            num_anchors=ptq4sam_config.num_anchors,  # e.g. 8
            descriptor_dim=self.embedding_dim,
            ema_momentum=0.9,
            normalize=True,
        )


    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # 1) proj + post-activation quant
        prompt_tokens = q                 # fallback

        # descriptor: [B, C]
        desc = self.prompt_bank.compute_descriptor(prompt_tokens)  # [B, C]

        # observer_enabled => calibration => update anchors
        if self.q_post_act_fake_quantize.observer_enabled:
            anchor_ids = self.prompt_bank.assign_and_update(desc)  # [B]
        else:
            anchor_ids = self.prompt_bank.assign(desc, update=False)  # [B]
        hist = torch.bincount(anchor_ids.flatten().cpu(), minlength=num_anchors)
        print("anchor_hist:", hist.tolist())

        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        # ----- 3) Anchor-aware quantization (we’ll pass anchor_ids) -----
        q = self.q_post_act_fake_quantize(q_proj, anchor_id=anchor_ids)
        k = self.k_post_act_fake_quantize(k_proj)
        v = self.v_post_act_fake_quantize(v_proj)


        # 2) heads: [B, N, C] -> [B, H, N, D]
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        B, H, Nq, D = q.shape
        Nk = k.shape[-2]

        # 3) manual scaled dot-product attention (SAM1-style)
        # scores: [B, H, Nq, Nk]
        d = D
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

        # probs: [B, H, Nq, Nk]
        probs = torch.softmax(scores, dim=-1)

        # ---- AGQ expects [B*H, Nq, Nk] + v as [B*H, Nk, D] ----
        probs_view = probs.reshape(B * H, Nq, Nk)
        v_view = v.reshape(B * H, Nk, D)

        probs_q = self.softmax_post_act_fake_quantize(probs_view, value=v_view)
        probs_q = probs_q.reshape(B, H, Nq, Nk)

        # 4) attention output: [B, H, Nq, D]
        attn_out = torch.matmul(probs_q, v)

        # 5) recombine heads + out proj
        out = self._recombine_heads(attn_out)  # [B, Nq, H*D]
        out = self.out_proj(out)
        return out
    def bimodal_adjust(self):
        # optional BIG fold like in SAM1
        if getattr(self.k_post_act_fake_quantize, "is_bimodal", False):
            sign = self.k_post_act_fake_quantize.sign

            def adjust_linear(linear: nn.Linear, sign_vec):
                linear.weight.mul_(sign_vec.unsqueeze(1))
                if linear.bias is not None:
                    linear.bias.mul_(sign_vec)

            adjust_linear(self.k_proj.module, sign)
            adjust_linear(self.q_proj.module, sign)
            self.k_post_act_fake_quantize.is_bimodal = False


class QuantSAM2RoPEAttention(QuantSAM2Attention):
    """
    RoPEAttention has the same projections as Attention but a different
    forward. We keep SAM2's rotary logic and only quantize proj / attn.

    We assume org_module is RoPEAttention; we just reuse the q/k/v/out
    from QuantSAM2Attention and call org_module's forward for RoPE,
    but we still want group/hybrid/AGQ/BIG behavior.
    """

    def __init__(
        self,
        org_module: RoPEAttention,
        w_qconfig,
        a_qconfig,
        ahcptq_config,
        ptq4sam_config,
        qoutput: bool = True,
        qinput: bool = False,
    ):
        # init as generic attention
        super().__init__(
            org_module, w_qconfig, a_qconfig, ahcptq_config, ptq4sam_config, qoutput, qinput
        )
        # store original rope fields if needed
        self.rope = org_module

    def forward(self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0) -> Tensor:
        # 1) proj + post-activation quant
        prompt_tokens = q                 # fallback

        # descriptor: [B, C]
        desc = self.prompt_bank.compute_descriptor(prompt_tokens)  # [B, C]

        # observer_enabled => calibration => update anchors
        if self.q_post_act_fake_quantize.observer_enabled:
            anchor_ids = self.prompt_bank.assign_and_update(desc)  # [B]
        else:
            anchor_ids = self.prompt_bank.assign(desc, update=False)  # [B]

        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        # ----- 3) Anchor-aware quantization (we’ll pass anchor_ids) -----
        q = self.q_post_act_fake_quantize(q_proj, anchor_ids=anchor_ids)
        k = self.k_post_act_fake_quantize(k_proj)
        v = self.v_post_act_fake_quantize(v_proj)

        # 2) heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # 3) apply RoPE (same as original SAM2, but on quantized q/k)
        w = h = int(math.sqrt(q.shape[-2])) 
        self.rope.freqs_cis = self.rope.freqs_cis.to(q.device)
        if self.rope.freqs_cis.shape[0] != q.shape[-2]:
            self.rope.freqs_cis = self.rope.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        from sam2.modeling.position_encoding import apply_rotary_enc
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.rope.freqs_cis,
            repeat_freqs_k=self.rope.rope_k_repeat,
        )

        # 4) manual attention (same pattern as QuantSAM2Attention)
        B, H, Nq, D = q.shape
        Nk = k.shape[-2]
        d = D

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)   # [B, H, Nq, Nk]
        probs = torch.softmax(scores, dim=-1)

        probs_view = probs.reshape(B * H, Nq, Nk)
        v_view = v.reshape(B * H, Nk, D)

        probs_q = self.softmax_post_act_fake_quantize(probs_view, value=v_view)
        probs_q = probs_q.reshape(B, H, Nq, Nk)

        attn_out = torch.matmul(probs_q, v)  # [B, H, Nq, D]

        out = self._recombine_heads(attn_out)
        out = self.out_proj(out)
        return out

# =====================================================================
# Quantized MLP (decoder-side) – like QuantDecoderMLPBlock
# =====================================================================

class QuantSAM2MLP(QuantizedBlock):
    """
    Quantized version of sam2.modeling.sam2_utils.MLP.

    We don't know its internal attribute names, so we:
      - scan all child nn.Linear in order,
      - make first one 'group' (if CAG) and last one 'hybrid' (if HLUQ),
      - others 'normal'.
    """

    def __init__(self, org_module: MLP, w_qconfig, a_qconfig, ahcptq_config, qinput: bool = True):
        super().__init__()
        self.mlp = org_module
        assert hasattr(org_module, "layers"), "Expected MLP to have attribute `layers`."
        assert isinstance(org_module.layers, nn.ModuleList), "`mlp.layers` must be a ModuleList."

        old_layers = org_module.layers
        num_layers = len(old_layers)
        new_layers = nn.ModuleList()

        for idx, lin in enumerate(old_layers):
            if not isinstance(lin, nn.Linear):
                # If SAM2 MLP ever changes and includes non-linears here, just keep them
                new_layers.append(lin)
                continue

            # Decide quantization type for this layer
            quant_type = "normal"
            if idx == 0 and getattr(ahcptq_config, "cag", False):
                quant_type = "group"    # CAG on first linear
            if idx == num_layers - 1 and getattr(ahcptq_config, "hluq", False):
                quant_type = "hybrid"   # HLUQ on last linear

            q_lin = PreQuantizedLayer(
                lin,
                activation=None,
                w_qconfig=w_qconfig,
                a_qconfig=a_qconfig,
                type=quant_type,
                qinput=qinput,
            )
            new_layers.append(q_lin)

        # Replace the original ModuleList with the quantized version
        org_module.layers = new_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Just delegate to the original MLP's forward, which now uses quantized layers
        return self.mlp(x)

    #     # collect direct child linears
    #     linear_names = []
    #     for name, child in org_module.layers.named_children():
    #         if isinstance(child, nn.Linear):
    #             linear_names.append((name, child))

    #     n = len(linear_names)

    #     for idx, (name, lin) in enumerate(linear_names):
    #         quant_type = "normal"
    #         if idx == 0 and getattr(ahcptq_config, "cag", False):
    #             quant_type = "group"
    #         if idx == n - 1 and getattr(ahcptq_config, "hluq", False):
    #             quant_type = "hybrid"
    #         setattr(
    #             org_module,
    #             name,
    #             PreQuantizedLayer(lin, None, w_qconfig, a_qconfig, quant_type),
    #         )

    # def forward(self, x: Tensor) -> Tensor:
    #     return self.mlp(x)


# =====================================================================
# Quantized TwoWayAttentionBlock (decoder) – like QuantDecoderOurTwoWayAttentionBlock
# =====================================================================

class QuantSAM2TwoWayAttentionBlock(nn.Module):
    """
    Quantized version of TwoWayAttentionBlock with group/hybrid + AGQ/BIG.

    Mirrors the original forward exactly, but uses QuantSAM2Attention +
    QuantSAM2MLP for its submodules.
    """

    def __init__(
        self,
        org_module: TwoWayAttentionBlock,
        w_qconfig,
        a_qconfig,
        ahcptq_config,
        ptq4sam_config,
        qoutput: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = QuantSAM2Attention(
            org_module.self_attn, w_qconfig, a_qconfig, ahcptq_config, ptq4sam_config, qinput=True
        )
        self.norm1 = org_module.norm1

        self.cross_attn_token_to_image = QuantSAM2Attention(
            org_module.cross_attn_token_to_image,
            w_qconfig,
            a_qconfig,
            ahcptq_config,
            ptq4sam_config,
            qinput=True,
        )
        self.norm2 = org_module.norm2

        self.mlp = QuantSAM2MLP(
            org_module.mlp, w_qconfig, a_qconfig, ahcptq_config, qinput=True
        )
        self.norm3 = org_module.norm3

        self.norm4 = org_module.norm4
        self.cross_attn_image_to_token = QuantSAM2Attention(
            org_module.cross_attn_image_to_token,
            w_qconfig,
            a_qconfig,
            ahcptq_config,
            ptq4sam_config,
            qinput=True,
        )

        self.skip_first_layer_pe = org_module.skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = queries + self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


# =====================================================================
# Quantized TwoWayTransformer (top-level decoder)
# =====================================================================

class QuantSAM2TwoWayTransformer(nn.Module):
    """
    Quantized version of TwoWayTransformer:
      - layers: list of QuantSAM2TwoWayAttentionBlock
      - final_attn_token_to_image: QuantSAM2Attention
    """

    def __init__(
        self,
        org_module: TwoWayTransformer,
        w_qconfig,
        a_qconfig,
        ahcptq_config,
        ptq4sam_config,
    ):
        super().__init__()
        self.depth = org_module.depth
        self.embedding_dim = org_module.embedding_dim
        self.num_heads = org_module.num_heads
        self.mlp_dim = org_module.mlp_dim

        self.layers = nn.ModuleList()
        for layer in org_module.layers:
            self.layers.append(
                QuantSAM2TwoWayAttentionBlock(
                    layer, w_qconfig, a_qconfig, ahcptq_config, ptq4sam_config
                )
            )

        self.final_attn_token_to_image = QuantSAM2Attention(
            org_module.final_attn_token_to_image,
            w_qconfig,
            a_qconfig,
            ahcptq_config,
            ptq4sam_config,
        )
        self.norm_final_attn = org_module.norm_final_attn

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


# =====================================================================
# Specials mapping + bimodal_adjust hook (like SAM1)
# =====================================================================

specials = {
    SAM2ImageEncoder:      QuantSAM2ImageEncoder,  
    TwoWayTransformer:      QuantSAM2TwoWayTransformer,
    TwoWayAttentionBlock:   QuantSAM2TwoWayAttentionBlock,
    Attention:              QuantSAM2Attention,
    RoPEAttention:          QuantSAM2RoPEAttention,
}


def bimodal_adjust(model: nn.Module, logger):
    """
    Scan the model for SAM2 attention blocks with BIG-enabled bimodal
    K-quantizers and fold the sign into q/k projections, like in SAM1.
    """
    logger.info("start to detect bimodal distribution (SAM2)")
    for name, m in model.named_modules():
        if isinstance(m, QuantSAM2Attention) and "token_to_image" in name:
            logger.info(name)
            logger.info(getattr(m.k_post_act_fake_quantize, "is_bimodal", False))
            m.bimodal_adjust()
    logger.info("bimodal integration end (SAM2)")