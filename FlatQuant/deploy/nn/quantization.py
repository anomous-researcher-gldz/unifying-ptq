import deploy
import torch

try:
    from ahcptq.quantization.fake_quant import (
        fold_outliers,
        unfold_outliers,
        is_like_normal_plus_3sigma_outliers,
    )
    _DBAF_AVAILABLE = True
except ImportError:
    _DBAF_AVAILABLE = False


class Quantizer(torch.nn.Module):
    def __init__(self, input_clip_ratio=1.0, lac=False, dbaf_alpha=None):
        super().__init__()
        self.input_clip_ratio = input_clip_ratio
        self.lac = lac
        self.dbaf_alpha = dbaf_alpha
        self.register_buffer("clip_factor_a_max", torch.tensor(4.0))
        self.register_buffer("clip_factor_a_min", torch.tensor(4.0))

    def forward(self, x):
        if not isinstance(x, deploy.PackedQuantizedTensor):
            # DBAF: fold outliers before quantization
            _dbaf_tag = None
            _dbaf_T = None
            if self.dbaf_alpha is not None and _DBAF_AVAILABLE:
                _profile = is_like_normal_plus_3sigma_outliers(x)
                if _profile['is_like_c']:
                    _dbaf_T = float(3.0 * x.detach().float().std().clamp_min(1e-8))
                    x, _dbaf_tag = fold_outliers(x, _dbaf_T, self.dbaf_alpha)

            if self.lac:
                reshaped_x = x.reshape((-1, x.shape[-1]))
                xmax, xmin = reshaped_x.amax(1, keepdim=True), reshaped_x.amin(1, keepdim=True)
                tmp = torch.zeros_like(xmax)
                xmax, xmin = torch.maximum(xmax, tmp), torch.minimum(xmin, tmp)

                xmax = xmax * torch.sigmoid(self.clip_factor_a_max.to(x.device))
                xmin = xmin * torch.sigmoid(self.clip_factor_a_min.to(x.device))

                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmax == 0
                scales_x = (xmax / 7)
                scales_x[tmp] = 1
                scales_x = scales_x.to(torch.float16)
            else:
                scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1)/7).to(torch.float16) * self.input_clip_ratio

            quantized_x = deploy.sym_quant(x, scales_x)
            packed_tensor = deploy.PackedQuantizedTensor(
                quantized_x, scales_x,
                dbaf_tag=_dbaf_tag, dbaf_T=_dbaf_T, dbaf_alpha=self.dbaf_alpha,
            )
            return packed_tensor
        else:
            return x
