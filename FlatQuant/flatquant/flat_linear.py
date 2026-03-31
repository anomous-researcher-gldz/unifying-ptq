import torch
import torch.nn as nn
import torch.nn.functional as F

from flatquant.quant_utils import (
    WeightQuantizer, ActivationQuantizer,
    fold_outliers, unfold_outliers, is_like_normal_plus_3sigma_outliers,
)
from flatquant.flat_utils import kronecker_matmul

class FlatQuantizedLinear(nn.Module):
    def __init__(self, args, linear: nn.Linear, dbaf_alpha: float = 0.99):
        super(FlatQuantizedLinear, self).__init__()
        self.args = args
        self.linear = linear

        self.weight_quantizer = WeightQuantizer()
        self.weight_quantizer.configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)
        self.act_quantizer = ActivationQuantizer(bits=args.a_bits, sym=not(args.a_asym), lac=args.lac, groupsize=args.a_groupsize, )

        self.lwc = args.lwc
        if self.lwc:
            lwc_dim = self.linear.weight.shape[0] if self.lwc else -1
            init_value = 4.
            self.clip_factor_w_max = nn.Parameter(torch.ones((lwc_dim, 1))*init_value, requires_grad=True)
            self.clip_factor_w_min = nn.Parameter(torch.ones((lwc_dim, 1))*init_value, requires_grad=True)
            self.sigmoid = nn.Sigmoid()

        self._eval_mode = False
        self.dbaf_alpha = dbaf_alpha

    def apply_wclip(self, weight):
        wmin, wmax = weight.min(1, keepdim=True)[0], weight.max(1, keepdim=True)[0]
        wmax *= self.sigmoid(self.clip_factor_w_max)
        wmin *= self.sigmoid(self.clip_factor_w_min)
        weight = torch.clamp(weight, min=wmin, max=wmax)
        return weight

    def apply_trans(self, weight, qa_trans):
        if isinstance(qa_trans, list):
            weight = kronecker_matmul(weight, qa_trans[0].to(weight), qa_trans[1].to(weight))
        else:
            weight = qa_trans(weight, inv_t=True)
        return weight

    def _ori_forward(self, hidden_states):
        return self.linear(hidden_states)

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
        _dbaf_result = is_like_normal_plus_3sigma_outliers(weight) if self.dbaf_alpha is not None else {'is_like_c': False}
        _dbaf_w = _dbaf_result['is_like_c']
        _w_tag = None
        _T_w = None
        if _dbaf_w:
            _T_w = float(3.0 * _dbaf_result['stats']['std'])
            weight, _w_tag = fold_outliers(weight, _T_w, self.dbaf_alpha)

        # quantize weight
        self.weight_quantizer.find_params(weight)
        weight = self.weight_quantizer(weight)

        if _dbaf_w:
            weight = unfold_outliers(weight, _w_tag, _T_w, self.dbaf_alpha)

        # quantize activation
        hidden_states = self.act_quantizer(hidden_states, anchor_id=anchor_id)

        if out_trans is not None and self.linear.bias is not None:
            bias = out_trans(self.linear.bias.data)
        else:
            bias = self.linear.bias
        output = F.linear(hidden_states, weight, bias)
        return output

    def forward(self, hidden_states, qa_trans=None, out_trans=None, anchor_id=None):
        if not self._eval_mode:
            return self._train_forward(hidden_states, qa_trans=qa_trans, out_trans=out_trans, anchor_id=anchor_id)
        else:
            return self._eval_forward(hidden_states, anchor_id=anchor_id)

    def _eval_forward(self, hidden_states, anchor_id=None):
        x_dtype = hidden_states.dtype
        hidden_states = self.act_quantizer(hidden_states, anchor_id=anchor_id).to(x_dtype)

        output = self.linear(hidden_states)
        return output

    def reparameterize(self, qa_trans=None, out_trans=None):
        weight = self.linear.weight.data
        ori_dtype = weight.dtype
        weight = weight.to(torch.float64)
        # quantization-adaptive transform
        if qa_trans is not None:
            weight = self.apply_trans(weight, qa_trans)
        if self.lwc:
            weight = self.apply_wclip(weight)
        if out_trans is not None:
            weight = out_trans(weight.T).T
        if out_trans is not None and self.linear.bias is not None:
            self.linear.bias.data = out_trans(self.linear.bias.data)
        
        self.linear.weight.data = weight.to(ori_dtype)
        self._eval_mode = True

