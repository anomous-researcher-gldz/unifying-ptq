from torch import nn
from .observer import * #MSEFastObserver, MinMaxObserver, AvgMinMaxObserver, MSEObserver, AvgMSEObserver, AvgMSEFastObserver, LogAvgMSEFastObserver, SignAvgMSEFastObserver, PCTObserver, AvgMinMaxGroupObserver, HybridParamObserver
from .fake_quant import *
import torch.nn.functional as F
from .fake_quant_blocks import * 
import torch
import math
import numpy as np
from .persistent_avg import get_avg, peek_after_add, update_avg
from .bitmapping import *  
ObserverDict = {
    'MinMaxObserver':           MinMaxObserver,                                    # noqa: E241
    'AvgMinMaxObserver':        AvgMinMaxObserver,                                 # noqa: E241
    'MSEObserver':              MSEObserver,                                       # noqa: E241
    'AvgMSEObserver':           AvgMSEObserver,                                    # noqa: E241
    'MSEFastObserver':          MSEFastObserver,                                   # noqa: E241
    'AvgMSEFastObserver':       AvgMSEFastObserver,                                # noqa: E241
    'LogAvgMSEFastObserver':    LogAvgMSEFastObserver,
    'SignAvgMSEFastObserver':    SignAvgMSEFastObserver,
    'PCTObserver':  PCTObserver,
    'AvgMinMaxGroupObserver': AvgMinMaxGroupObserver,
    'HybridParamObserver': HybridParamObserver,
      'BlockMSEObserver': BlockMSEObserver,
}

FakeQuantizeDict = {
    'FixedFakeQuantize':     FixedFakeQuantize,                                    # noqa: E241
    'LSQFakeQuantize':       LSQFakeQuantize,                                      # noqa: E241
    'LSQSignFakeQuantize':       LSQSignFakeQuantize,                              # noqa: E241
    'LSQPlusFakeQuantize':   LSQPlusFakeQuantize,                                  # noqa: E241
    'LSQPlusSignFakeQuantize':   LSQPlusSignFakeQuantize,                          # noqa: E241
    'AdaRoundFakeQuantize':  AdaRoundFakeQuantize,                                 # noqa: E241
    'AdaptiveGranularityQuantize': AdaptiveGranularityQuantize,                    # noqa: E241
    'GroupLSQFakeQuantize': GroupLSQFakeQuantize,
    'HybridQuantize': HybridQuantize,
    'AdaRoundFakeQuantizeBlock':  AdaRoundFakeQuantizeBlock,
    'AnchorAwareFakeQuantize': AnchorAwareFakeQuantize, 
}

import torch

@torch.no_grad()
def compute_M(x: torch.Tensor, q: float = 0.999):
    """
    Robust estimate of extreme magnitude M = quantile(|x|, q)

    Args:
        x: activation tensor
        q: quantile (e.g., 0.999 or 0.9999)

    Returns:
        scalar tensor M
    """
    x_abs = x.abs().reshape(-1)
    return torch.quantile(x_abs, q)

@torch.no_grad()
def compute_p_out(x: torch.Tensor, T: float):
    return (x.abs() > T).float().mean()

@torch.no_grad()
def compute_alpha_star(
    x: torch.Tensor,
    q_M: float = 0.999,
    alpha_min: float = 0.6,
    alpha_max: float = 1.0,
    eps: float = 1e-8,
):
    """
    Compute data-driven alpha* with stability guards.
    """
    T = compute_T(x)
    M = compute_M(x, q=q_M)
    if M <= T + eps:
        return alpha_max

    p_out = compute_p_out(x, T)
    p_in = 1.0 - p_out

    alpha_star = (p_out * T) / ((M - T) * p_in + eps)
    # print(p_in,p_out,alpha_star)
    # alpha_star = torch.clamp(alpha_star, alpha_min, alpha_max)
    return alpha_star

def compute_T(x: torch.Tensor, k: float = 3.0) -> float:
    """
    Compute T = k * std(x) for DBAF folding.
    Returns a Python float for convenience.
    """
    x = x.detach().float()
    std = x.std().clamp_min(1e-8)
    return float(k * std)
def calculate_mse(x,quantx): 
    return ((x-quantx)**2).mean() 
import os, json, time, fcntl

def _append_ordered(path: str, record: dict):
    """Append one JSON line to `path` with a POSIX file lock + fsync."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    line = json.dumps(record, separators=(",", ":")) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # serialize writers
        f.write(line)
        f.flush(); os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)

def profile_with_3sigma_outliers(
    x: Tensor,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Flatten x and compute basic stats plus fraction of |z| > 3 std outliers.
    """
    flat = x.detach().float().reshape(-1)
    n = flat.numel()
    if n == 0:
        raise ValueError("Empty tensor")

    mean = flat.mean()
    std = flat.std().clamp_min(eps)

    z = (flat - mean) / std  # standardized

    skew = (z**3).mean()
    kurt = (z**4).mean()  # normal ~ 3
    frac_out_3 = (z.abs() > 3.0).float().mean()

    return {
        "n": int(n),
        "mean": mean.item(),
        "std": std.item(),
        "skew": skew.item(),
        "kurtosis": kurt.item(),
        "frac_out_3": frac_out_3.item(),
    }

def compute_T(x: torch.Tensor, k: float = 3.0) -> float:
    """
    Compute T = k * std(x) for DBAF folding.
    Returns a Python float for convenience.
    """
    x = x.detach().float()
    std = x.std().clamp_min(1e-8)
    return float(k * std)
def compute_T_from_percentile(x: torch.Tensor, p: float = 99.9) -> float:
    # choose T so that only (100 - p)% of values exceed T
    x = x.detach().float().abs().reshape(-1)
    k = max(int(len(x) * p / 100.0), 1)
    topk, _ = torch.kthvalue(x, k)
    return float(topk)
def is_like_normal_plus_3sigma_outliers(
    x: Tensor,
    # how strict you want these can be tuned
    skew_thresh: float = 0.7,       # |skew| <= this was 0.7
    kurt_min: float = 3.0,          # heavier than Gaussian
    kurt_max: float = 30, # was 30
    frac3_min: float = 1e-4,        # at least this many 3σ outliers
    frac3_max: float =2e-2,        # but not too many
) -> Dict[str, Any]:
    """
    Heuristic: is x like case (c) = normal-ish core + rare 3σ outliers?

    Your definition of outlier is: |x - mean| > 3 * std.
    """
    stats = profile_with_3sigma_outliers(x)

    skew = abs(stats["skew"])
    kurt = stats["kurtosis"]
    frac3 = stats["frac_out_3"]

    cond_skew = skew <= skew_thresh
    cond_kurt = (kurt_min <= kurt <= kurt_max)
    cond_frac3 = (frac3_min <= frac3 <= frac3_max)

    is_like_c = bool(cond_skew and cond_kurt and cond_frac3)
        

    return {
        "is_like_c": is_like_c,
        "stats": stats,
        "conditions": {
            "cond_skew": cond_skew,
            "cond_kurt": cond_kurt,
            "cond_frac3": cond_frac3,
        },
    }

def set_bit_for_fq(fake_quantizer,inp): 

    # alpha_star = compute_alpha_star(inp) 
    # if isinstance(alpha_star, torch.Tensor): 
    #     alpha_star = alpha_star.item()
    # log_path="layer_wise_best_alpha.jsonl"
    # rec = {
    #     "best_alpha": alpha_star,
    # }
    # # print(alpha_star)
    # _append_ordered(log_path, rec)
    # orig_bit = fake_quantizer.bit 
    # best_error = calculate_mse(inp, fake_quantizer(inp))
    # print(best_error,flush=True)
    # log_path="original_errors_ahcptq.jsonl"
    # rec = {
    #     "orig_bit": int(orig_bit),
    #     "error": float(best_error),
    # }
    # normal_outputs = is_like_normal_plus_3sigma_outliers(inp)
    
    # if(normal_outputs['is_like_c']): 
    #    index,_ = get_avg() 
    # #    rank = get_rank_for_index("layer_bits.jsonl", index) 
    #    name = "/home/AHCPTQ/normal_like_c_yolox_samh/" + str(fake_quantizer) + str(index)+".pt"
    #    print(">>> ABOUT TO SAVE:", name)
    #    torch.save(inp, name)
    #    _,_= update_avg(4.0)
    # else: 
    #    _,_= update_avg(4.0)
       

    
    # _append_ordered(log_path, rec)
    # index,_ = get_avg() 
    # best_bit = get_bit_for_index("layer_bits.jsonl", index)

    # rank = get_rank_for_index("layer_bits.jsonl", index) 
    # if rank <=66: 
    #     name = "/home/AHCPTQ/highest_errors_wa/error_rank_"+str(rank)+'.pt'
        # torch.save(inp, name)
    # best_bit = orig_bit 
    # for b in range(5,9):
    #     print(b,flush=True)  
    #     fake_quantizer.set_bit(b)
    #     fake_quantizer.enable_observer()  
    #     error = calculate_mse(inp, fake_quantizer(inp))
    #     newavgbit = peek_after_add(b)[1]
    #     print(error,newavgbit, flush=True)
    #     if error<best_error and newavgbit<=4.34:   
    #         best_error=error 
    #         best_bit=b 

    # print("best combo",best_error,best_bit,flush=True)
    # print("bit to set", best_bit,flush=True)
    # print(fake_quantizer)
    # print(fake_quantizer.observer_enabled)
    # fake_quantizer.set_bit(best_bit)
    # fake_quantizer.enable_observer()  
    # fake_quantizer.disable_fake_quant() 
    # _= fake_quantizer(inp)
    # fake_quantizer.disable_observer() 
    # fake_quantizer.enable_fake_quant() 
    # error = calculate_mse(inp, fake_quantizer(inp))
    # print(best_bit,error,flush=True)
    # log_path="best_errors_higherbits.jsonl"
    # rec = {
    #     "best_bit": int(best_bit),
    #     "error": float(error),
    # }
    # _append_ordered(log_path, rec)
    # n2,avg2 = update_avg(best_bit)
    # print("new avgbit", avg2,flush=True)
    return True, fake_quantizer 

def update_specialized_quantizer_config(base_config, quantizer_name):
    import copy
    specialized_config = copy.deepcopy(base_config)

    update_keys = {
        'group': {'quantizer': 'GroupLSQFakeQuantize',
                  'observer': 'AvgMinMaxGroupObserver'},
        'hybrid': {'quantizer': 'HybridQuantize',
                   'observer': 'HybridParamObserver'}
    }[quantizer_name]
    specialized_config.update(update_keys)
    return specialized_config

def ActivationQuantizer(a_qconfig, detect_ch_axis=False):
    fq_cls = FakeQuantizeDict[a_qconfig.quantizer]
    obs_cls = ObserverDict[a_qconfig.observer]

    ch_axis = 'det' if detect_ch_axis else a_qconfig.ch_axis

    # base kwargs for all quantizers
    kwargs = dict(
        observer=obs_cls,
        bit=a_qconfig.bit,
        symmetric=a_qconfig.symmetric,
        ch_axis=ch_axis,
    )

    # special case: anchor-aware variant
    if a_qconfig.quantizer == 'AnchorAwareFakeQuantize':
        kwargs['num_anchors'] = getattr(a_qconfig, 'num_anchors', 8)
        # print(kwargs['num_anchors'] )

    quantizer = fq_cls(**kwargs)
    return quantizer

def SignActivationQuantizer(a_qconfig):
    return FakeQuantizeDict['LSQSignFakeQuantize'](ObserverDict[a_qconfig.observer], bit=a_qconfig.bit,
                                                 symmetric=a_qconfig.symmetric, ch_axis=a_qconfig.ch_axis)

def WeightQuantizer(w_qconfig):
    return FakeQuantizeDict[w_qconfig.quantizer](
            ObserverDict[w_qconfig.observer],
            bit=w_qconfig.bit,
            symmetric=w_qconfig.symmetric,
            ch_axis=w_qconfig.ch_axis)
#            block_size=4)


class QuantizedOperator():
    pass


class QConv2d(QuantizedOperator, nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 groups,
                 bias,
                 padding_mode,
                 w_qconfig):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)
        self.changed_bit=False 

    def forward(self, input, gamma=None):
        with torch.no_grad():
            if self.changed_bit==False: 
                cb,wfq = set_bit_for_fq(self.weight_fake_quant, self.weight)
                self.changed_bit = cb 
                self.weight_fake_quant = wfq 
        # print(input.device)
        weight = self.weight_fake_quant(self.weight)
        # if weight.device != input.device:
        #     weight = weight.to(input.device)
        # bias = self.bias
        # if bias is not None and bias.device != input.device:
        #     bias = bias.to(input.device)
        # if self.weight.device != input.device: 
        #     self.weight = self.weight.to("cuda")
        return self._conv_forward(input, weight, self.bias)


class QLinear(QuantizedOperator, nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias,
                 w_qconfig):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)
        self.changed_bit=False 
    def forward(self, input, gamma = None):
        if gamma is None:
            with torch.no_grad():
                if self.changed_bit==False: 
                    cb,wfq = set_bit_for_fq(self.weight_fake_quant, self.weight)
                    self.changed_bit = cb 
                    self.weight_fake_quant = wfq 
            return F.linear(input, self.weight_fake_quant(self.weight), self.bias)
        else:
            fused_weight = self.weight.mul(gamma.unsqueeze(1))
            fused_bias = self.bias.mul(gamma)
            with torch.no_grad():
                if self.changed_bit==False: 
                    cb,wfq = set_bit_for_fq(self.weight_fake_quant, fused_weight)
                    self.changed_bit = cb 
                    self.weight_fake_quant = wfq 
            return F.linear(input, self.weight_fake_quant(fused_weight), fused_bias)


class QEmbedding(QuantizedOperator, nn.Embedding):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx,
                 max_norm,
                 norm_type,
                 scale_grad_by_freq,
                 sparse,
                 _weight,
                 w_qconfig):
        super().__init__(num_embeddings=num_embeddings,
                         embedding_dim=embedding_dim,
                         padding_idx=padding_idx,
                         max_norm=max_norm,
                         norm_type=norm_type,
                         scale_grad_by_freq=scale_grad_by_freq,
                         sparse=sparse,
                         _weight=_weight)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)
        self.changed_bit=False 
    def forward(self, input):
        with torch.no_grad():
                if self.changed_bit==False: 
                    cb,wfq = set_bit_for_fq(self.weight_fake_quant, self.weight)
                    self.changed_bit = cb 
                    self.weight_fake_quant = wfq 
        return F.embedding(
            input, self.weight_fake_quant(self.weight), self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


module_type_to_quant_weight = {
    nn.Linear: QLinear,
    nn.Conv2d: QConv2d,
    nn.Embedding: QEmbedding,
}


def get_module_args(module):
    if isinstance(module, nn.Linear):
        return dict(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None
            )
    elif isinstance(module, nn.Conv2d):
        return dict(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            )
    elif isinstance(module, nn.Embedding):
        return dict(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
            _weight=None,
        )
    else:
        raise NotImplementedError


def Quantizer(module, config, detect_ch_axis=False, sign=False):
    if module is None:
        if detect_ch_axis:
            return ActivationQuantizer(a_qconfig=config, detect_ch_axis=detect_ch_axis)
        if sign:
            return SignActivationQuantizer(a_qconfig=config)
            # return LogSqrt2Quantize(a_qconfig=config)
        return ActivationQuantizer(a_qconfig=config)
    module_type = type(module)
    if module_type in module_type_to_quant_weight:
        kwargs = get_module_args(module)
        qmodule = module_type_to_quant_weight[module_type](**kwargs, w_qconfig=config)
        qmodule.weight.data = module.weight.data.clone()
        if getattr(module, 'bias', None) is not None:
            qmodule.bias.data = module.bias.data.clone()
        return qmodule
    return module


class QuantizedModule(nn.Module):
    def __init__(self):
        super().__init__()


class QuantizedLayer(QuantizedModule):
    def __init__(self, module, activation, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.module = Quantizer(module, w_qconfig)
        self.activation = activation
        self.changed_bit=False 
        if qoutput:
            self.layer_post_act_fake_quantize = Quantizer(None, a_qconfig)
        
    def forward(self, x):
        x = self.module(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.qoutput:
            with torch.no_grad():
                if self.changed_bit==False: 
                    # print("here2")
                    cb,wfq = set_bit_for_fq(self.layer_post_act_fake_quantize, x)
                    self.changed_bit = cb 
                    self.layer_post_act_fake_quantize = wfq 
            x = self.layer_post_act_fake_quantize(x)
        return x


class PreQuantizedLayer(QuantizedModule):
    def __init__(self, module, activation, w_qconfig, a_qconfig, type='normal', qinput=True):
        super().__init__()
        self.qinput = qinput
        self.module = Quantizer(module, w_qconfig)
        self.activation = activation
        detect_ch_axis = False
        if type == 'group':
            a_qconfig = update_specialized_quantizer_config(a_qconfig, 'group')
            detect_ch_axis = True
        elif type == 'hybrid':
            a_qconfig = update_specialized_quantizer_config(a_qconfig, 'hybrid')
        elif type == 'normal':
            a_qconfig = a_qconfig
        else:
            raise NotImplementedError
        if qinput:
            self.layer_pre_act_fake_quantize = Quantizer(None, a_qconfig, detect_ch_axis)
        if type == 'hybrid':
            self.layer_pre_act_fake_quantize.weight = module.weight
            self.layer_pre_act_fake_quantize.bias = module.bias
        self.changed_bit=False 

    def forward(self, x, gamma = None):
        if self.qinput:
            with torch.no_grad():
                if self.changed_bit==False: 
                    # print("here", flush=True)
                    cb,wfq = set_bit_for_fq(self.layer_pre_act_fake_quantize, x)
                    self.changed_bit = cb 
                    self.layer_pre_act_fake_quantize = wfq 

            
            x = self.layer_pre_act_fake_quantize(x)
        # print(self.module)
        x = self.module(x, gamma)
        if self.activation is not None:
            x = self.activation(x)
        return x

class QuantizedMatMul(QuantizedModule):
    def __init__(self, a_qconfig, qinput=True):
        super().__init__()
        self.qinput = qinput
        self.changed_bit_a=False 
        self.changed_bit_b=False 
        if qinput:
            self.a_layer_pre_act_fake_quantize = Quantizer(None, a_qconfig)
            self.b_layer_pre_act_fake_quantize = Quantizer(None, a_qconfig)
    
    def forward(self, inputs):
        a, b = inputs
        if self.qinput:
            with torch.no_grad():
                if self.changed_bit_a==False: 
                    cb,wfq = set_bit_for_fq(self.a_layer_pre_act_fake_quantize, a)
                    self.changed_bit_a = cb 
                    self.a_layer_pre_act_fake_quantize = wfq 
            with torch.no_grad():
                if self.changed_bit_b==False: 
                    cb,wfq = set_bit_for_fq(self.b_layer_pre_act_fake_quantize, b)
                    self.changed_bit_b = cb 
                    self.b_layer_pre_act_fake_quantize = wfq 
            a = self.a_layer_pre_act_fake_quantize(a)
            b = self.b_layer_pre_act_fake_quantize(b)
        x = a @ b
        return x


class QuantizedBlock(QuantizedModule):
    def __init__(self):
        super().__init__()
