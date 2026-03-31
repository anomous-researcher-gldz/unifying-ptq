from typing import Any
from torch.nn import Module, Linear, Parameter, Conv2d, ReLU
import torch
from torch import Tensor, FloatTensor
from torch.autograd import Function
from torch.autograd.function import _ContextMethodMixin
from functools import partial
from scipy.linalg import hadamard
import torch.nn.functional as F
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
from ahcptq.quantization.fake_quant import (
    fold_outliers,
    unfold_outliers,
    is_like_normal_plus_3sigma_outliers,
)

DBAF_ALPHA = 0.95

calibrated_num = 0
total_num = 0
num_twos=0 
num_threes=0 
num_elements = 0 
def quant(input:torch.Tensor, lb:float, ub:float, bit:int,alpha:float, beta:float):
    input = input.clamp(lb, ub) #lb,ub are learned
    s = (ub - lb) / (2 ** bit -1) + alpha
    Z = lb + beta  
    if s==0: 
        input = (input - Z)/(s+1e-20)
    else: 
        input = (input - Z)/s
    input = input.round()
    input = input * s + Z
    return input

def cal_mse(input:torch.Tensor, lb:float, ub:float, bit:int,alpha:float, beta:float):
    quant_input = quant(input, lb, ub, bit,alpha,beta)
    res = float(torch.norm(input - quant_input))
    return res


def next_power_of_2(n):
    """Returns the smallest power of 2 greater than or equal to n"""
    return 1 if n == 0 else 2**(n - 1).bit_length()

def hadamard_transform_nd(orig_tensor):
    """Apply an n-dimensional Hadamard transform to an arbitrary tensor."""
    original_shape = orig_tensor.shape
    #print(original_shape)
    slices = tuple(slice(0, orig_dim) for orig_dim in original_shape)
    new_shape = torch.Tensor([next_power_of_2(s) for s in original_shape])  # Find power-of-2 shape
    #print(new_shape)

    # Pad tensor to new_shape
    pad_widths = []
    for orig_dim, new_dim in zip(original_shape, new_shape): 
        pad_widths.append(int((new_dim-orig_dim).item()))
        pad_widths.append(0)
        
    pad_widths.reverse()
    pad_widths = tuple(pad_widths)
    
    padded_tensor = F.pad(orig_tensor, pad_widths, mode='constant')
    
    #print(padded_tensor)

    H = torch.tensor(hadamard(new_shape[-1]))
    H = H.float().to("cuda")
    had_t= F.linear(padded_tensor, H) / (new_shape[-1]**0.5)

    
    # rev = F.linear(had_t, H) / (new_shape[-1]**0.5)
    # print(rev,padded_tensor)
    # assert(torch.allclose(rev,padded_tensor))
    # Crop back to original shape
    # out = had_t[slices]
 
    return had_t,slices

def inverse_hadamard_transform_nd(tensor):
    """Apply the inverse Hadamard transform (which is the same as the forward transform)."""
    return hadamard_transform_nd(tensor)  # Since H is its own inverse
def prune_bands(had_x, keep_frac):
    if keep_frac==1: 
        # print("here keep frac 1")
        return had_x 
    numel = had_x.numel()
    sample_size = 1000000 

    # Generate random flat indices, then index into tensor
    flat_idx = torch.randint(numel, (sample_size,)).to("cuda")
    sample = had_x.flatten()[flat_idx].abs()

    # total = had_x.numel()
    # zero_count = (had_x == 0).sum().item()
    # percent = zero_count / total
    # print("had_x 0 percent", percent)
    # Estimate threshold using quantile of the sample
    if keep_frac==0.0: 
        # print("here")
        out = torch.zeros(had_x.shape).to("cuda")
    else: 
        q = 1.0 - keep_frac
        thr = sample.quantile(q)
        del sample 
        del flat_idx 
        out = had_x * (had_x.abs() >= thr) 
    total = out.numel()
    zero_count = (out == 0).sum().item()
    # print(zero_count)
    percent = zero_count / total
    # print("percent", percent,1-keep_frac,flush=True)
    assert abs(percent-(1-keep_frac)) < 0.05 
    return out 

def prune_channels(had_x: torch.Tensor, keep_frac: float, channel_dim: int = 0):
    """
    Prune entire channels with the lowest average absolute magnitude until we reach
    approximately the desired pruning percentage.

    Args:
        had_x: Tensor of activations/weights, e.g. [C, H, W] or [C, ...].
        keep_frac: Fraction of channels to keep (0.0–1.0).
        channel_dim: Dimension that corresponds to channels (default 0).

    Returns:
        Tensor with some whole channels zeroed out.
    """
    if keep_frac >= 1.0:
        return had_x
    if keep_frac <= 0.0:
        return torch.zeros_like(had_x)

    device = had_x.device
    ndims = had_x.dim()
    channel_dim = channel_dim % ndims  # handle negative dims

    num_channels = had_x.size(channel_dim)
    if num_channels == 0:
        return had_x

    # Number of channels to keep (rounded)
    channels_to_keep = max(0, min(num_channels, int(round(keep_frac * num_channels))))

    if channels_to_keep == 0:
        return torch.zeros_like(had_x)
    if channels_to_keep == num_channels:
        return had_x

    # Compute mean absolute value per channel
    dims = list(range(ndims))
    reduce_dims = [d for d in dims if d != channel_dim]
    # Shape: [num_channels]
    mean_abs_per_channel = had_x.abs().mean(dim=reduce_dims)

    # Sort channels by ascending mean |value|
    _, sorted_idx = torch.sort(mean_abs_per_channel, descending=False)

    # Keep the channels with the largest mean |value|
    keep_idx = sorted_idx[-channels_to_keep:]

    # Build a channel mask and broadcast to full tensor shape
    channel_mask = torch.zeros(num_channels, dtype=torch.bool, device=device)
    channel_mask[keep_idx] = True

    # Reshape mask to broadcast over all other dims
    mask_shape = [1] * ndims
    mask_shape[channel_dim] = num_channels
    channel_mask = channel_mask.view(mask_shape)

    out = had_x * channel_mask

    # Optional sanity check on approximate sparsity
    total = out.numel()
    zero_count = (out == 0).sum().item()
    percent_zero = zero_count / total
    target_zero = 1.0 - keep_frac
    # print(percent_zero)
    # Allow some slack because we’re pruning whole channels (discrete)
    assert abs(percent_zero - target_zero) < 0.15, (
        f"Sparsity off more than 0.15: got {percent_zero:.3f}, "
        f"target {target_zero:.3f}"
    )

    return out

def DOBI(input:torch.Tensor, bit:int, one_direction = False, num:int=100):
    min_value = torch.min(input)
    max_value = torch.max(input)
    total = input.numel()
    zero_count = (input == 0).sum().item()
    percent = zero_count / total
    # print(percent)
    # print(min_value,max_value)
    diff = (max_value - min_value) / (2 * num)
    bit_change = bit 
    history_min = float('inf')
    input = input.cuda()
    #print(input.shape)
    if one_direction:
        diff = (max_value - min_value) / num
        for i in range(num):
            lb = min_value
            ub = max_value - diff * i
            alpha = 0 
            beta = 0 
            
            cur_value = cal_mse(input, lb, ub, bit,alpha,beta)
                
            if cur_value < history_min:
                best_lb = lb
                best_ub = ub
                history_min = cur_value
    else:
        diff = (max_value - min_value) / (2 * num)
        for i in range(num):
            lb = min_value + diff * i
            ub = max_value - diff * i
            alpha = 0 
            beta = 0
            cur_value = cal_mse(input, lb, ub, bit,alpha,beta)
            # print(cur_value)
            # print(history_min)
            # best_lb=
            if cur_value < history_min:
                # print("here now")
                best_lb = lb
                # print(best_lb)
                best_ub = ub
                history_min = cur_value
    # print(best_lb)
    global calibrated_num
    global total_num
    global num_threes 
    global num_twos 
    
    calibrated_num += 1
    # print(f'calibration:{calibrated_num}/{total_num}')
    global num_elements
    num_elements+=torch.numel(input)
    #print(num_elements)
    # err_inp = cal_mse(input, lb, ub, 3,0,0)
    # cur_value = cal_mse(input, lb, ub, bit,0,0)
    # # print("2bit err", cur_value) 
    # # print(cur_value - err_inp)
    # if cur_value  > 2000: #if two bit error is greater than 2000 
    #     #print("here")
    #     bit_change = 3 
    #     num_threes+=1 
    # else: 
    #     num_twos+=1 
    # print("avg_bitwidth so far", (num_twos*2+num_threes*3)/(num_twos+num_threes))
    bit_change=2 
    del input 
    # print(best_lb)
    return float(best_lb), float(best_ub),bit_change

class Differentiable_Round(Function):
    @staticmethod
    def forward(ctx: _ContextMethodMixin, x: Tensor):
        return x.round()

    @staticmethod
    def backward(ctx: _ContextMethodMixin, grad_outputs):
        return grad_outputs


class Differentiable_Clip(Function):
    @staticmethod
    def forward(
        ctx: _ContextMethodMixin,
        input: Tensor,
        min_val: Tensor,
        max_val: Tensor,
    ) -> Any:
        ctx.save_for_backward(input, min_val, max_val)
        # if isinstance(min_val, Tensor):
        #     min_val = min_val.item()
        # if isinstance(max_val, Tensor):
        #     max_val = max_val.item()
        
        # output = input.clamp(min_val, max_val)
        output = input.clamp(min_val.item(), max_val.item())
        return output

    @staticmethod
    def backward(ctx: _ContextMethodMixin, grad_outputs: Tensor) -> Any:
        input, min_val, max_val = ctx.saved_tensors

        grad_input = grad_outputs.clone()
        grad_input[(input < min_val) | (input > max_val)] = 0
        
        grad_min = grad_outputs.clone()
        grad_min[input > min_val] = 0
        grad_min = grad_min.sum().view(1)

        grad_max = grad_outputs.clone()
        grad_max[input < max_val] = 0
        grad_max = grad_max.sum().view(1)
        return grad_input, grad_min, grad_max


class FakeQuantizerBase(Module):
    def __init__(self, int_quant: bool = True, bit:int=4) -> None:
        super().__init__()
        self.lower_bound = Parameter(
            torch.randn((1,), dtype=torch.float32,requires_grad=True),
        )
        self.upper_bound = Parameter(
            torch.randn((1,), dtype=torch.float32,requires_grad=True),
        )
        self.n_bit = Parameter(
            torch.randn((1,), dtype=torch.float32,requires_grad=False),
        )
        self.alpha = Parameter(
            torch.FloatTensor([0]),requires_grad=True #False to not add trainable params 
            ) 

        self.beta_Z = Parameter(
            torch.FloatTensor([0]),requires_grad=True
        )
        # self.hadamard = None 
        # self.n_bit = bit
        self.set_n_bit_manually(bit)

        # self.quant_scaler_help = Parameter(
        #     torch.FloatTensor([0]))
        
   
        self.bit2bound = {}
        self.use_bit2bound = False
        self.size_of_input = None
        
        self.int_quant = int_quant

        self.clip = Differentiable_Clip.apply
        self.round = Differentiable_Round.apply
        
        self.calibrated = False
        self.one_direction_search = False
        
        global total_num 
        total_num += 1
        

    def set_int_quant(self, enable: bool):
        self.int_quant = enable

    def set_require_grad(self, enable_lb: bool, enable_up: bool, enable_nbit: bool):
        self.lower_bound.requires_grad = enable_lb
        self.upper_bound.requires_grad = enable_up
        
        # self.lower_bound.requires_grad = False
        # self.upper_bound.requires_grad = False
        # self.alpha.requires_grad = True
        # self.beta.requires_grad = True
        # self.n_bit.requires_grad = enable_nbit

    def set_params_manually(self, lb: Tensor, ub: Tensor, n_bit: Tensor):
        device = self.lower_bound.device
        self.lower_bound.data = FloatTensor([lb]).data.clone().to(device)
        self.upper_bound.data = FloatTensor([ub]).data.clone().to(device)
        # self.n_bit.data = FloatTensor([n_bit]).data.clone().to(device)
        
    def set_params_lb_manually(self, lb: Tensor):
        device = self.lower_bound.device
        self.lower_bound.data = FloatTensor([lb]).data.clone().to(device)
    
    def set_params_ub_manually(self, ub: Tensor):
        device = self.upper_bound.device
        self.upper_bound.data = FloatTensor([ub]).data.clone().to(device)

    def set_n_bit_manually(self, n_bit):
        device = self.n_bit.device
        self.n_bit.data = FloatTensor([n_bit]).data.clone().to(device)


class FakeQuantizerWeight(FakeQuantizerBase):
    def __init__(self,bit=4,pruning_keep_frac=0.9,name='') -> None: #,
        super(FakeQuantizerWeight, self).__init__(bit=bit)
        self.pruning_keep_frac = pruning_keep_frac
        self.name=name 
    def forward(self, x:torch.Tensor):
        if not self.calibrated:
            # DBAF: fold outliers if tensor matches normal+3sigma profile, else pass through
            _profile = is_like_normal_plus_3sigma_outliers(x)
            if _profile['is_like_c']:
                _T = float(3.0 * _profile['stats']['std'])
                folded_x, _ = fold_outliers(x, _T, DBAF_ALPHA)
                calib_x = folded_x
            else:
                calib_x = x
            pruned_calib_x = prune_bands(calib_x, keep_frac=self.pruning_keep_frac)
            lb, rb, bit_change = DOBI(pruned_calib_x, bit=self.n_bit, one_direction=self.one_direction_search)
            self.set_params_lb_manually(lb)
            self.set_params_ub_manually(rb)

            if bit_change == 3:
                self.set_n_bit_manually(bit_change)
            self.calibrated = True
            return x
        if self.size_of_input is None:
            self.size_of_input = x.numel()

        n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        if self.use_bit2bound:
            try:
                lb, ub = self.bit2bound[int(n_bits.item())]
                self.set_params_lb_manually(lb)
                self.set_params_ub_manually(ub)
            except Exception as e:
                print(f'use bit 2 bound.{int(n_bits.item())} not found.')

        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / (torch.pow(2, n_bits) - 1) + self.alpha

        # DBAF: fold outliers if tensor matches normal+3sigma profile, else quantize directly
        _profile = is_like_normal_plus_3sigma_outliers(x)
        _apply_dbaf = _profile['is_like_c']
        _dbaf_tag = None
        _T = None
        if _apply_dbaf:
            _T = float(3.0 * _profile['stats']['std'])
            x_q, _dbaf_tag = fold_outliers(x, _T, DBAF_ALPHA)
        else:
            x_q = x

        pruned_x = prune_bands(x_q, keep_frac=self.pruning_keep_frac)
        c = self.clip(pruned_x, self.lower_bound, self.upper_bound)
        del pruned_x
        # int value \in [0,2^n-1]
        Z = self.lower_bound + self.beta_Z
        r = self.round((c - Z) / s)
        out = s * r + Z

        if _apply_dbaf:
            out = unfold_outliers(out, _dbaf_tag, _T, DBAF_ALPHA)

        return out.float().to("cuda") 
    
        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # # (u-l)/(2^n-1)
        # s = (self.upper_bound - self.lower_bound) / torch.pow(2, n_bits)

        # # clip(x,l,u)
        # c = self.clip(x, self.lower_bound, self.upper_bound)

        # # int value \in [0,2^n-1]
        # # r = self.clip(self.round((c - self.lower_bound) / s - 0.5, 0, torch.pow(2, n_bits)-1))
        # r = self.clip(
        #     self.round((c - self.lower_bound) / s - 0.5),
        #     0, torch.pow(2, n_bits)-1
        #     )

        # return s * (r+0.5) + self.lower_bound


class FakeQuantizerAct(FakeQuantizerBase):
    def __init__(self,bit=4,name='') -> None:
        """
        if dynamic, bound will be the minmax value of the input values.
        dynamic is only used for act.
        if running_stat, bound will be set by moving average.
        """
        super(FakeQuantizerAct, self).__init__(bit=bit)

        self.running_stat = False # initial boundary
        self.first_iter = False # initial boundary
        self.dynamic = False
        self.beta = 0.995
        self.identity = False
        self.name=name 

    def forward(self, x):
        if not self.calibrated:
            # DBAF: fold outliers if tensor matches normal+3sigma profile, else pass through
            _profile = is_like_normal_plus_3sigma_outliers(x)
            if _profile['is_like_c']:
                _T = float(3.0 * _profile['stats']['std'])
                folded_x, _ = fold_outliers(x, _T, DBAF_ALPHA)
                calib_x = folded_x
            else:
                calib_x = x
            lb, rb, bit_change = DOBI(calib_x, bit=self.n_bit, one_direction=self.one_direction_search)
            self.set_params_lb_manually(lb)
            self.set_params_ub_manually(rb)

            if bit_change == 3:
                self.set_n_bit_manually(bit_change)
            self.calibrated = True
            return x

        if self.size_of_input is None:
            self.size_of_input = x.numel()

        if self.identity:
            return x

        if self.dynamic:
            n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

            lb = torch.min(x).detach()
            ub = torch.max(x).detach()
            n_bits = n_bits.detach()

            s = (ub - lb) / (torch.pow(2, n_bits) - 1)
            c = self.clip(x, lb, ub)
            r = self.round((c - lb) / s)

            return s * r + lb

        if self.running_stat:
            if self.first_iter:
                self.lower_bound.data = torch.min(x).detach().clone()
                self.upper_bound.data = torch.max(x).detach().clone()
                self.first_iter = False
            else:
                self.lower_bound = self.beta * self.lower_bound + (1 - self.beta) * torch.min(x)
                self.upper_bound = self.beta * self.upper_bound + (1 - self.beta) * torch.max(x)
            return x

        n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
        if self.use_bit2bound:
            try:
                lb, ub = self.bit2bound[int(n_bits.item())]
                self.set_params_lb_manually(lb)
                self.set_params_ub_manually(ub)
            except Exception as e:
                print(f'use bit 2 bound.{int(n_bits.item())} not found.')

        s = (self.upper_bound - self.lower_bound) / (torch.pow(2, n_bits) - 1) + self.alpha

        # DBAF: fold outliers if tensor matches normal+3sigma profile, else quantize directly
        _profile = is_like_normal_plus_3sigma_outliers(x)
        _apply_dbaf = _profile['is_like_c']
        _dbaf_tag = None
        _T = None
        if _apply_dbaf:
            _T = float(3.0 * _profile['stats']['std'])
            x_q, _dbaf_tag = fold_outliers(x, _T, DBAF_ALPHA)
        else:
            x_q = x

        c = self.clip(x_q, self.lower_bound, self.upper_bound)
        Z = self.lower_bound + self.beta_Z
        r = self.round((c - Z) / s)
        out = s * r + Z

        if _apply_dbaf:
            out = unfold_outliers(out, _dbaf_tag, _T, DBAF_ALPHA)

        return out.float().to("cuda") 

        
        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # # (u-l)/(2^n-1)
        # s = (self.upper_bound - self.lower_bound) / torch.pow(2, n_bits)

        # # clip(x,l,u)
        # c = self.clip(x, self.lower_bound, self.upper_bound)

        # # int value \in [0,2^n-1]
        #     r = self.clip(
        #     self.round((c - self.lower_bound) / s - 0.5),
        #     0, torch.pow(2, n_bits)-1
        #     )

        # return s * (r+0.5) + self.lower_bound
    
##########################################################
##########################################################


class FakeQuantizerWeightParam2(FakeQuantizerBase):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):

        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / self.n_bit + self.alpha

        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound) 

        # int value \in [0,2^n-1]
        Z = self.lower_bound + self.beta_Z
        r = self.round((c - Z) / s)

        return s * r + Z


class FakeQuantizerActParam2(FakeQuantizerBase):
    def __init__(self) -> None:
        """
        if dynamic, bound will be the minmax value of the input values.
        dynamic is only used for act.
        if running_stat, bound will be set by moving average.
        """
        super().__init__()

        self.running_stat = True # initial boundary
        self.first_iter = True # initial boundary
        self.dynamic = False
        self.beta = 0.995

    def forward(self, x):
        if self.dynamic:
            '''
                use min,max as clip boundary
                TODO: use other oberver later.
            '''
            # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
            n_bits = self.n_bit
            lb = torch.min(x).detach()
            ub = torch.max(x).detach()
            n_bits = n_bits.detach()

            # (u-l)/(2^n-1)
            s = (ub - lb) / n_bits

            # clip(x,l,u)
            c = self.clip(x, lb, ub)

            # int value \in [0,2^n-1]
            r = self.round((c - lb) / s)

            return s * r + lb

        if self.running_stat:
            if self.first_iter:
                self.lower_bound.data = torch.min(x).detach().clone()
                self.upper_bound.data = torch.max(x).detach().clone()
                self.first_iter = False
            else:
                self.lower_bound =  self.beta * self.lower_bound + (1-self.beta) *  torch.min(x)
                self.upper_bound =  self.beta * self.upper_bound + (1-self.beta) *  torch.max(x)
            return x


        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / n_bits + self.alpha 

  
        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound) 

        # int value \in [0,2^n-1]
        Z = self.lower_bound + self.beta_Z 
        r = self.round((c - Z) / s)

        return s * r + Z

class QuantWeight(Module):
    """
    Standalone quantizer for a parameter tensor.
    Uses the SAME quantization logic as QuantLinear.
    """
    def __init__(self, weight: torch.Tensor, config: dict):
        super().__init__()
        self.weight_fp = torch.nn.Parameter(weight.detach().clone())
        self.quant = True
        self.bit = config['bit']
        self.name = config['name']
        self.pruning_keep_frac = config['pruning_keep_frac']
        self.weight_quantizer = FakeQuantizerWeight(self.bit,self.pruning_keep_frac,self.name)

    def set_quant_flag(self, flag: bool):
        self.quant = flag

    def forward(self):
        if not self.quant:
            return self.weight_fp
        
        quant_weight = self.weight_quantizer(self.weight_fp)
        return quant_weight


class QuantBase(Module):
    def __init__(self,config):
        super().__init__()
        self.quant = True
        self.bit = config['bit']
        self.name = config['name']
        self.pruning_keep_frac = config['pruning_keep_frac']
        self.weight_quantizer = FakeQuantizerWeight(self.bit,self.pruning_keep_frac,self.name)
        self.act_quantizer = FakeQuantizerAct(self.bit,self.name)
        # if config['param'] == 1:
        #     self.weight_quantizer = FakeQuantizerWeight()
        #     self.act_quantizer = FakeQuantizerAct()
        # elif config['param'] == 2:
        #     self.weight_quantizer = FakeQuantizerWeightParam2()
        #     self.act_quantizer = FakeQuantizerActParam2()

    def get_weight_quantizer(self):
        return self.weight_quantizer

    def get_act_quantizer(self):
        return self.act_quantizer

    def set_quant_flag(self, enable: bool):
        self.quant = enable

    def set_require_grad(self, enable: bool):
        # 似乎没有必要分别设置.
        self.weight_quantizer.set_require_grad(enable,enable, enable)
        self.act_quantizer.set_require_grad(enable,enable, enable)

    def set_weight_bias_grad(self, enable: bool):
        self.weight.requires_grad = enable
        if self.bias:
            self.bias.requires_grad = enable

    def get_quant_weight_bias(self):
        quant_weight = self.weight_quantizer(self.weight)

        return (quant_weight, self.bias)


class QuantLinear(QuantBase):
    def __init__(self,config):
        super().__init__(config)
    
    def load_values(self, value):
        min_value, max_value = value
        self.act_quantizer.set_params_lb_manually(min_value)
        self.act_quantizer.set_params_ub_manually(max_value)

    def set_param(self, linear: Linear):
        """
        must be called before forward.
        """

        self.in_feature = linear.in_features
        self.out_feature = linear.out_features

        self.weight = Parameter(linear.weight.data.clone())

        if linear.bias is not None:
            self.bias = Parameter(linear.bias.data.clone())
        else:
            self.bias = linear.bias

    def forward(self, x):
        if not self.quant:
            return F.linear(x, self.weight, self.bias)

        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)
        # bias don't need to be quant

        return F.linear(quant_act, quant_weight, self.bias)

class QuantLinearQKV(Module):
    def __init__(self,config):
        super().__init__()
        self.q = QuantLinear(config)
        self.k = QuantLinear(config)
        self.v = QuantLinear(config)
    
    def load_values(self, value):
        min_value, max_value = value
        self.q.act_quantizer.set_params_lb_manually(min_value)
        self.q.act_quantizer.set_params_ub_manually(max_value)
        self.k.act_quantizer.set_params_lb_manually(min_value)
        self.k.act_quantizer.set_params_ub_manually(max_value)
        self.v.act_quantizer.set_params_lb_manually(min_value)
        self.v.act_quantizer.set_params_ub_manually(max_value)
        
    def set_quant_flag(self, enable: bool):
        self.q.set_quant_flag(enable)
        self.k.set_quant_flag(enable)
        self.v.set_quant_flag(enable)
        
    def set_require_grad(self, enable: bool):
        # 似乎没有必要分别设置.
        self.q.set_require_grad(enable)
        self.k.set_require_grad(enable)
        self.v.set_require_grad(enable)


    def set_weight_bias_grad(self, enable: bool):
        self.q.set_weight_bias_grad(enable)
        self.k.set_weight_bias_grad(enable)
        self.v.set_weight_bias_grad(enable)
    

    def get_quant_weight_bias(self):
        w_q,b_q = self.q.get_quant_weight_bias()
        w_k,b_k = self.k.get_quant_weight_bias()
        w_v,b_v = self.v.get_quant_weight_bias()
        
        quant_weight = torch.cat([w_q, w_k, w_v],dim=0)
        if b_q is not None:
            bias = torch.cat([b_q,b_k,b_v])

        return (quant_weight, bias)

    def set_param(self, linear: Linear):
        """
        must be called before forward.
        """

        self.in_feature = linear.in_features 
        self.out_feature = linear.out_features // 3
        
        linear_q = Linear(self.in_feature, self.out_feature,bias=linear.bias is not None)
        linear_k = Linear(self.in_feature, self.out_feature,bias=linear.bias is not None)
        linear_v = Linear(self.in_feature, self.out_feature,bias=linear.bias is not None)
        
        
        linear_q.weight.data,linear_k.weight.data,linear_v.weight.data = linear.weight.data.clone().reshape(3,self.in_feature,self.out_feature)
        
        
        if linear.bias is not None:
            linear_q.bias.data, linear_k.bias.data, linear_v.bias.data = linear.bias.data.clone().reshape(3,self.out_feature)
    
        self.q.set_param(linear_q)
        self.k.set_param(linear_k)
        self.v.set_param(linear_v)


    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # print('qsize',q.size())
        # print('ksize',k.size())
        # print('vsize',v.size())
        return torch.cat([q,k,v], dim=-1)
    
        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)
        # bias don't need to be quant

        return F.linear(quant_act, quant_weight, self.bias)


class QuantConv2d(QuantBase):
    def __init__(self,config):
        super().__init__(config)

    def set_param(self, conv: Conv2d):
        """
        must be called before forward.
        """

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.conv_kwargs = {
            "stride": conv.stride,
            "padding": conv.padding,
            "dilation": conv.dilation,
            "groups": conv.groups,
        }
        self.weight = Parameter(conv.weight.data.clone())
        if conv.bias is not None:
            self.bias = Parameter(conv.bias.data.clone())
        else:
            self.bias = conv.bias

    def forward(self, x):
        if not self.quant:
            return F.conv2d(x, self.weight, self.bias, **self.conv_kwargs)

        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)
        # bias doesn't need to be quant

        return F.conv2d(quant_act, quant_weight, self.bias, **self.conv_kwargs)


if __name__ == "__main__":

    def differentiable_test():
        x = Tensor((1.0, 2.0, 3.0, 4.0, 5.0))
        lb = Tensor((2.0,))
        ub = Tensor((4.0,))

        n = Tensor((2.0,))

        x.requires_grad = True
        lb.requires_grad = True
        ub.requires_grad = True
        n.requires_grad = True

        s = (ub - lb) / (torch.pow(2, n) - 1)
        print(f"s:{s}")
        round = Differentiable_Round.apply
        clip = Differentiable_Clip.apply

        clipped = clip(x, lb, ub)
        subbed = clipped - lb
        normd = subbed / s
        roundded = round(normd)
        unnormed = s * roundded + lb

        x_hat_sum = unnormed.sum()
        x_hat_sum.backward()

        print("clipped", clipped)
        print("subbed", subbed)
        print(subbed[2] / s)
        print(type(s))
        print("normd", normd, float(normd[2]))
        print("roundded", roundded)
        print("unnormed", unnormed)
        print("=" * 10)
        print(x_hat_sum)
        print(x.grad)
        print(lb.grad)
        print(ub.grad)
        print(n.grad)

    def linear_test():
        l = Linear(10, 10, True)
        l_nb = Linear(10, 10, False)

        print(l.weight)
        print(l.bias)

        print(l_nb.weight)
        print(l_nb.bias)
        if l_nb.bias:
            print(1)
        else:
            print(2)

    # differentiable_test()
    # linear_test()

    def conv_test():
        c = Conv2d(2, 6, (3, 4), 2, 1, bias=True)
        c_nb = Conv2d(3, 6, 3, 2, 1, bias=False)

        print(c.weight.shape)
        # n_kernal(out_channels),in_channels, kerner_shape_H, kernel_shape_W
        print(c.bias.shape)  # n_kernal(out_channels)

        print(c_nb.weight.shape)
        print(c_nb.bias)

        # print(c.)

    def quant_sim_test():
        torch.manual_seed(1234)
        l = Linear(10, 10, True)

        l_q = QuantLinear()
        l_q.set_param(l)
        l_q.set_quant_flag(False)

        input = torch.randn((16, 3, 10))
        out1 = l(input)
        out2 = l_q(input)

        # No quant: tensor(0., grad_fn=<SumBackward0>)
        print("No quant:", (out1 - out2).sum())

        l_q.set_quant_flag(True)
        act_min_val, act_max_val = input.min(), input.max()
        weight_min_val, weight_max_val = l_q.weight.min(), l_q.weight.max()

        l_q.get_weight_quantizer().set_params_manually(
            weight_min_val, weight_max_val, 4
        )
        l_q.get_act_quantizer().set_params_manually(act_min_val, act_max_val, 4)

        out1 = l(input)
        out2 = l_q(input)

        # 4bit quant: tensor(0.7627, grad_fn=<SumBackward0>)
        print("4bit quant:", (out1 - out2).sum())

        l_q.get_weight_quantizer().set_params_manually(
            weight_min_val, weight_max_val, 8
        )
        l_q.get_act_quantizer().set_params_manually(act_min_val, act_max_val, 8)

        out1 = l(input)
        out2 = l_q(input)
        # 8bit quant: tensor(-0.1381, grad_fn=<SumBackward0>)
        print("8bit quant:", (out1 - out2).sum())

        ######################################
        ######################################
        c = Conv2d(3, 8, 3)

        c_q = QuantConv2d()
        c_q.set_param(c)
        c_q.set_quant_flag(False)

        input = torch.randn((16, 3, 15, 15))
        c_q.set_quant_flag(False)
        out1 = c(input)
        out2 = c_q(input)

        # No quant: tensor(0., grad_fn=<SumBackward0>)
        print("No quant:", (out1 - out2).sum())

        c_q.set_quant_flag(True)
        act_min_val, act_max_val = input.min(), input.max()
        weight_min_val, weight_max_val = c_q.weight.min(), c_q.weight.max()

        c_q.get_weight_quantizer().set_params_manually(
            weight_min_val, weight_max_val, 4
        )
        c_q.get_act_quantizer().set_params_manually(act_min_val, act_max_val, 4)

        out1 = c(input)
        out2 = c_q(input)

        # 4bit quant: tensor(0.7627, grad_fn=<SumBackward0>)
        print("4bit quant:", (out1 - out2).sum())

        c_q.get_weight_quantizer().set_params_manually(
            weight_min_val, weight_max_val, 8
        )
        c_q.get_act_quantizer().set_params_manually(act_min_val, act_max_val, 8)

        out1 = c(input)
        out2 = c_q(input)
        # 8bit quant: tensor(-0.1381, grad_fn=<SumBackward0>)
        print("8bit quant:", (out1 - out2).sum())

        # import matplotlib.pyplot as plt

        # plt.subplot(1, 2, 1)
        # plt.hist(c.weight.reshape(-1).detach().numpy(), bins=40)
        # plt.subplot(1, 2, 2)
        # plt.hist(c_q.get_quant_weight_bias()[0].reshape(-1).detach().numpy(), bins=40)
        # plt.show()

        c_q.get_weight_quantizer().set_params_manually(
            weight_min_val, weight_max_val, 16
        )
        c_q.get_act_quantizer().set_params_manually(act_min_val, act_max_val, 16)

        out1 = c(input)
        out2 = c_q(input)
        # 8bit quant: tensor(-0.1381, grad_fn=<SumBackward0>)
        print("16bit quant:", (out1 - out2).sum())

    # quant_sim_test()

    def replace_test():

        class test_module(Module):
            def __init__(self):
                super().__init__()

                self.c1 = Conv2d(3, 16, 3)
                self.act = ReLU()
                self.l = Linear(2704, 15)

            def forward(self, x):
                x = self.c1(x)
                x = self.act(x).view(x.size(0), -1)
                print(x.shape)
                x = self.l(x)

                return x

        input = torch.randn((16, 3, 15, 15))

        m = test_module()
        out1 = m(input)

        for name, module in m.named_modules():
            if isinstance(module, Linear):
                new_module = QuantLinear()
                new_module.set_param(module)
                module = new_module
                module.set_quant_flag(True)
            elif isinstance(module, Conv2d):
                new_module = QuantConv2d()
                new_module.set_param(module)
                module = new_module
                module.set_quant_flag(True)

        # for name, module in m.named_modules():
        #     if isinstance(module, Linear):
        #         print(1)
        #     elif isinstance(module, Conv2d):
        #         print(2)
        print(m)

        out2 = m(input)

        print("8bit quant:", (out1 - out2).sum())
    # replace_test()
    def linear_shape_test():
        l = Linear(2, 6, True)
        print(l.weight)
        print(l.weight.size())
        print(l.bias.size())
        print(l.weight.reshape(3,2,2))
        a,b,c = l.weight.reshape(3,2,2)
        print(a,b,c)
    # linear_shape_test()
    
    def qkv_equal_test():
        torch.random.manual_seed(3407)
        import torch.nn.functional as F
        config = {'param':1}
        in_feature = 30
        out_feature = 90
        l = Linear(3, 9, False)
        l_q = QuantLinearQKV(config)
        l_qq = QuantLinear(config)
        print(l.weight.size())
        
        l_qq.set_param(l)
        l_q.set_param(l)
        
        l_qq.set_quant_flag(False)
        l_q.q.set_quant_flag(False)
        l_q.k.set_quant_flag(False)
        l_q.v.set_quant_flag(False)
        
        input = torch.randn(5, 3)
        
        out1 = l(input)
        out2, q, k ,v = l_q(input)
        out3 = l_qq(input)
        
        q_1, k_1, v_1 = out1.reshape(5,3,3).permute(1,0,2)
        
        print(out1.size())
        print(out2.size())
        print(out3.size())
        print(F.l1_loss(out1, out2))
        print(F.l1_loss(out1, out3))
        print(F.l1_loss(q, q_1))
        print(F.l1_loss(q, k_1))
        print(F.l1_loss(q, v_1))
        print(F.l1_loss(k, k_1))
        print(F.l1_loss(v, v_1))
        
        weight_raw = l.weight.data
        weight_q = l_q.q.weight
        weight_k = l_q.k.weight
        weight_v = l_q.v.weight
        print('weight loss')
        print(F.l1_loss(weight_q, weight_raw[:3,:]))
        print(F.l1_loss(weight_k, weight_raw[3:6,:]))
        print(F.l1_loss(weight_v, weight_raw[6:,:]))
        
        out_ma_1 = input @ weight_q.transpose(0,1)
        print('mannul')
        print(F.l1_loss(out_ma_1, q))
        print(F.l1_loss(out_ma_1, q_1))
        print(F.l1_loss(out_ma_1, out1[:,:3]))
        print(F.l1_loss(out_ma_1, out1[:,3:6]))
        print(F.l1_loss(out_ma_1, out1[:,6:]))
        
        print('linear self')
        out_ma = input @ weight_raw.transpose(0,1) # 这就是linear的计算方式
        out_q_real = out_ma[:,:3] # linear得到的q
        out_q = input @ weight_raw[:3,:].transpose(0,1)
        out_q1 = input @ weight_raw.transpose(0,1)[:,:3]
        out_q2 = input @ weight_raw.reshape(3, 3,3 ).permute(1, 0, 2)[0].transpose(0,1)
        print(F.l1_loss(out_ma, out1))
        print(F.l1_loss(out_q_real, out_q))
        print(F.l1_loss(out_q_real, out_q1))
        print(F.l1_loss(out_q_real, out_q2))
        
        print('detail')
        
        qkv = l(input).reshape(5, 3, 3).permute(1, 0, 2)
        
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        print(F.l1_loss(q, out_q_real))
        print('weight show')
        print(weight_raw.transpose(0,1))
        print(weight_raw[:3,:].transpose(0,1))
        
        print(input@weight_raw.transpose(0,1))
        print((input@weight_raw.transpose(0,1))[:,:3])
        print(input@weight_raw[:3,:].transpose(0,1))
        
        w1 = weight_raw.transpose(0,1)[:,:3]
        w2 = weight_raw[:3,:].transpose(0,1)
        
        print(F.l1_loss(w1, w2))
        print(F.l1_loss(input@w1, input@w2))
        print(F.l1_loss(input@w1, q))
        print(F.l1_loss(input@w2, q))
        
        print(F.l1_loss((input@(weight_raw.transpose(0,1))[:,:3]), input@(weight_raw[:3,:].transpose(0,1))))
        
    # qkv_equal_test()
    
    def qkv_equal_gpt():
        import torch
        import torch.nn as nn

        # 定义一个函数来拆分线性层成三个线性层
        def split_linear_layer(linear_layer, split_size):
            in_features = linear_layer.in_features
            out_features = linear_layer.out_features
            q_linear = nn.Linear(in_features, split_size)
            k_linear = nn.Linear(in_features, split_size)
            v_linear = nn.Linear(in_features, split_size)
            q_linear.weight.data = linear_layer.weight.data[:split_size, :]
            k_linear.weight.data = linear_layer.weight.data[split_size:2*split_size, :]
            v_linear.weight.data = linear_layer.weight.data[2*split_size:, :]
            q_linear.bias.data = linear_layer.bias.data[:split_size]
            k_linear.bias.data = linear_layer.bias.data[split_size:2*split_size]
            v_linear.bias.data = linear_layer.bias.data[2*split_size:]
            return q_linear, k_linear, v_linear

        # 创建一个示例线性层
        linear_layer = nn.Linear(100, 300)

        # 将线性层拆分成三个线性层
        q_linear, k_linear, v_linear = split_linear_layer(linear_layer, 100)

        # 创建一些输入数据
        batch_size = 16
        input_data = torch.randn(batch_size, 100, dtype=torch.float64)  # 使用 torch.float64

        # 使用拆分前的线性层进行前向传播
        output_before_split = linear_layer(input_data)

        # 使用拆分后的线性层进行前向传播
        q_output, k_output, v_output = q_linear(input_data), k_linear(input_data), v_linear(input_data)
        output_after_split = torch.cat([q_output, k_output, v_output], dim=1)

        # 计算 L1 损失来验证结果一致性
        l1_loss = nn.L1Loss()
        loss = l1_loss(output_before_split, output_after_split)

        print("L1 损失:", loss.item())

    # qkv_equal_gpt()
    
    def param_test():
        torch.random.manual_seed(3407)
        import torch.nn.functional as F
        
        config = {'param':1}
        l_raw = Linear(10,20)
        l_q = QuantLinear(config)
        l_q.set_param(l_raw)
        
        input = torch.randn(5, 10)
        
        l_q.set_quant_flag(False)
        
        out1 = l_raw(input)
        out2 = l_q(input)
        
        print(F.l1_loss(out1, out2))
        
        # print(l_q.named_parameters())
        for name, param in l_q.named_parameters():
            print(name)
        
        
    param_test()