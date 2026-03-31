
import torch
import torch.nn as nn

from .observer import *  # type: ignore
from .util_quant_blocks import * 
# --- NaN/Inf hardening ---
_MIN_SCALE = 1e-8
# --------------------------

__all__ = [
    "BlockQuantizeBase",
    "AdaRoundFakeQuantizeBlock",
    "LSQFakeQuantizeBlock"
]

class BlockQuantizeBase(nn.Module):
    """
    Base class mirroring QuantizeBase but operating per (bh x bw) block and using an Observer.
    Constructor matches your original signature: (observer, bit=8, symmetric=False, ch_axis=-1).
    We ignore ch_axis for block quant (we always quantize per 2D block), but keep it for API compatibility.
    """
    def __init__(self, observer=ObserverBase, bit=8, symmetric=False, ch_axis=-1, block_size=1024):
        super().__init__()
        self.observer = observer(bit=bit, symmetric=symmetric, ch_axis=ch_axis, block_size = block_size)
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.block_size = block_size 
        self.observer_enabled = 0
        self.fake_quant_enabled = 0
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max
        self.drop_prob = 1.0

    def set_bit(self, bit):
        self.bit = bit
        # re-init observer with new bit to keep quant_min/max in sync
        self.observer.bit = bit
        if self.observer.symmetric:
            self.observer.quant_min = -2 ** (bit - 1)
            self.observer.quant_max = (2 ** (bit - 1)) - 1
        else:
            self.observer.quant_min = 0
            self.observer.quant_max = (2 ** bit) - 1
        self.quant_min, self.quant_max = self.observer.quant_min, self.observer.quant_max

    def enable_observer(self):  self.observer_enabled = 1
    def disable_observer(self): self.observer_enabled = 0
    def enable_fake_quant(self):  self.fake_quant_enabled = 1
    def disable_fake_quant(self): self.fake_quant_enabled = 0


class AdaRoundFakeQuantizeBlock(BlockQuantizeBase):
    """
    Per-block AdaRound affine quantization compatible with your AdaRound interface.
    """
    def __init__(self, observer=BlockMSEObserver, bit=8, symmetric=False, ch_axis=-1, block_size=1024):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis, block_size=block_size)
        # self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        # self.scale = nn.Parameter(torch.tensor([1.0]))
        # self.zero_point = nn.Parameter(torch.tensor([0]))
        self.scale=None 
        # self.zero_point=None 
        self.adaround = False
        self.block_size = block_size 
        self.gamma, self.zeta = -0.1, 1.1
        self.round_mode = None
        self.alpha = None

    def init(self, weight_tensor: torch.Tensor, round_mode):
        self.adaround = True
        self.round_mode = round_mode
        self.init_alpha(x=weight_tensor.data.clone().detach())

    def init_alpha(self, x: torch.Tensor):
        blocks, meta = flatten_into_blocks(x, self.block_size)
        print("x shape", x.shape)
        # blocks,meta = flatten_into_channels(x)
        scale = self.scale
        # scale = _expand_param_for_blocks(self.scale,blocks)
        print("scale expanded", scale.shape)
        x_floor = torch.floor(blocks / scale)
        if self.round_mode == 'learned_hard_sigmoid':
            rest = (blocks / scale) - x_floor
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)
            
            # alpha = alpha.mean(dim=(-2), keepdim=True)
            print("alpha shape", alpha.shape)
            assert alpha.shape == blocks.shape, f"Shape mismatch: {alpha.shape} vs {blocks.shape}"
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError

    def rectified_sigmoid(self):
        return ((self.zeta - self.gamma) * torch.sigmoid(self.alpha) + self.gamma).clamp(0, 1)

    def adaround_forward(self, X, hard_value=False):
        orig_shape=X.shape
        blocks, meta = flatten_into_blocks(X, self.block_size)
        k = 3.0
        mask, stats = build_outlier_mask(X, k=k)
        idx, vals = extract_outliers(X, mask)
        # blocks,meta = flatten_into_channels(X)
        scale = self.scale
        zero_point = self.zero_point.int()
        # scale = _expand_param_for_blocks(scale.to(blocks.dtype).to(blocks.device), blocks)
        # zero_point = _expand_param_for_blocks(zero_point.to(blocks.dtype).to(blocks.device), blocks)

        # scale_expanded = scale.expand(*blocks.shape[:-2], scale.shape[-2], scale.shape[-1])
        Xq = torch.floor(blocks / scale)
        if hard_value:
            Xq = Xq + (self.alpha >= 0).float()
        else:
            Xq = Xq + self.rectified_sigmoid()
        Xq = Xq + zero_point
        Xq = torch.clamp(Xq, self.quant_min, self.quant_max)
        Xd = (Xq - zero_point) * scale
        x_out = reconstruct_from_blocks(Xd, meta)  # same 
        x_out = restore_outliers(x_out, idx, vals)
        # print("out after fold_blocks_2d shape:", out.shape)
        # print("out after final reshape shape:", out.shape, "should match original:", orig_shape)
        return x_out 
    def get_hard_value(self, X):
        return self.adaround_forward(X, hard_value=True)

    def forward(self, X):
        # if hasattr(self, "scale") and hasattr(self, "zero_point"):
            # print(self.scale.shape)

        if self.observer_enabled == 1:
            self.observer(X.clone().detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            # _scale = _scale.to(self.scale.device)
            print("weight shape", X.shape,flush=True)
            print("adaroudn scale", _scale.shape,flush=True)
            print("adaroudn zp", _zero_point.shape,flush=True)
            _zero_point = _zero_point.to(self.zero_point.device)
            if self.zero_point.shape != _zero_point.shape:
                # self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            # self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)
            self.scale=nn.Parameter(_scale).to("cuda")
            # self.zero_point = nn.Parameter(_zero_point).to("cuda")

        if self.fake_quant_enabled == 1:
           
            if not self.adaround:
                scale_q = self.scale
                zp_q = self.zero_point
                    # Sanitize params
                    #comment this out 
                # scale_q = _sanitize_scale(scale_q)
                # zp_q = torch.nan_to_num(zp_q, nan=0.0, posinf=0.0, neginf=0.0)
                grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                Y = fake_quantize_per_block_affine(
                    X, scale_q, zp_q, self.quant_min, self.quant_max,
                    self.block_size,grad_factor=grad_factor 
                )
                # print("Input X min/max:", X.min().item(), X.max().item())
                # if "Y" in locals():
                #     print("Quantized Y min/max:", Y.min().item(), Y.max().item())

                return Y
            else:
                if self.alpha is None:
                    raise NotImplementedError
                if self.round_mode == 'learned_hard_sigmoid':
                    return self.adaround_forward(X)
                else:
                    raise NotImplementedError
        return X

class LSQFakeQuantizeBlock(BlockQuantizeBase):

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, use_grad_scaling=True,block_size=4):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis,block_size=4 )
        # self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.use_grad_scaling = use_grad_scaling
        self.block_size=block_size 
        self.drop_prob = 1.0
        self.scale=None 

    def forward(self, X, value=None):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            # print()
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            if self.zero_point.shape != _zero_point.shape:
                # self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            _zero_point = _zero_point.to(self.zero_point.device)
            self.scale=nn.Parameter(_scale).to("cuda")
            self.zero_point.copy_(_zero_point)
        else:
            if self.scale!=None:
                self.scale.data.abs_()
                self.scale.data.clamp_(min=self.eps.item())
        

        if self.fake_quant_enabled == 1:
            if X.shape != self.scale.shape: 
                self.observer(X.detach())
                # print()
                _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
                if self.zero_point.shape != _zero_point.shape:
                    # self.scale.resize_(_scale.shape)
                    self.zero_point.resize_(_zero_point.shape)
                _zero_point = _zero_point.to(self.zero_point.device)
                self.scale=nn.Parameter(_scale).to("cuda")
                self.zero_point.copy_(_zero_point)
            if self.drop_prob < 1.0:
                x_orig = X
            if self.ch_axis != -1:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_per_block_affine(
                    X, self.scale, self.zero_point.data.int(), self.ch_axis,
                    self.quant_min, self.quant_max, self.block_size, grad_factor=grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_per_block_affine(
                    X, self.scale, self.zero_point, self.quant_min, self.quant_max, self.block_size, grad_factor=grad_factor)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob
        return X

