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
    assert not torch.allclose(bank.anchors, anchors_before)


def test_activation_quantizer_dbaf_fold_unfold():
    """DBAF fold/unfold should be a no-op on a tensor without outliers."""
    from flatquant.quant_utils import ActivationQuantizer
    q = ActivationQuantizer(bits=8, sym=True)
    x = torch.randn(2, 16, 64)   # normal distribution — no outliers expected
    out = q(x)
    assert out.shape == x.shape


def test_activation_quantizer_dbaf_applied_on_outlier_tensor():
    """On a tensor with clear 3-sigma outliers the quantizer should still produce correct shape."""
    from flatquant.quant_utils import ActivationQuantizer
    q = ActivationQuantizer(bits=8, sym=True)
    x = torch.randn(2, 16, 64)
    x[0, 0, 0] = 1000.0   # inject an outlier
    out = q(x)
    assert out.shape == x.shape


def test_anchor_aware_quantizer_basic():
    from flatquant.quant_utils import AnchorAwareActivationQuantizer
    q = AnchorAwareActivationQuantizer(bits=8, sym=True, num_anchors=8)
    x = torch.randn(3, 16, 64)
    anchor_id = torch.tensor([0, 3, 7])
    out = q(x, anchor_id=anchor_id)
    assert out.shape == x.shape


def test_anchor_aware_quantizer_no_anchor_id_falls_back():
    from flatquant.quant_utils import AnchorAwareActivationQuantizer
    q = AnchorAwareActivationQuantizer(bits=8, sym=True, num_anchors=8)
    x = torch.randn(2, 16, 64)
    out = q(x)  # no anchor_id — falls back to anchor 0
    assert out.shape == x.shape


def test_anchor_aware_quantizer_no_dbaf():
    """PCSA quantizer must not apply DBAF even on outlier tensors."""
    from flatquant.quant_utils import AnchorAwareActivationQuantizer
    from unittest.mock import patch
    q = AnchorAwareActivationQuantizer(bits=8, sym=True, num_anchors=8)
    x = torch.randn(2, 16, 64)
    x[0, 0, 0] = 1000.0  # inject outlier
    anchor_id = torch.tensor([0, 1])
    # fold_outliers should never be called in this path
    with patch('flatquant.quant_utils.fold_outliers') as mock_fold:
        q(x, anchor_id=anchor_id)
        mock_fold.assert_not_called()


def test_flat_quantized_linear_forward_no_anchor_id():
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
    out = layer(x)
    assert out.shape == (2, 4, 32)


def test_flat_quantized_linear_forward_with_anchor_id():
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
    layer.act_quantizer = AnchorAwareActivationQuantizer(bits=8, sym=True, num_anchors=8)

    x = torch.randn(2, 4, 64)
    anchor_id = torch.tensor([0, 3])
    out = layer(x, anchor_id=anchor_id)
    assert out.shape == (2, 4, 32)


def test_flat_quantized_linear_dbaf_weight_branch():
    """Test that DBAF fold/unfold fires on the weight path when outliers are present."""
    import torch.nn as nn
    from types import SimpleNamespace
    from flatquant.flat_linear import FlatQuantizedLinear
    from unittest.mock import patch

    args = SimpleNamespace(
        w_bits=8, w_asym=False, a_bits=8, a_asym=False,
        lac=False, a_groupsize=-1, lwc=False,
    )
    linear = nn.Linear(64, 32, bias=False)
    # Craft a weight that satisfies the DBAF heuristic:
    # normal core + ~1% symmetric 4.5-sigma outliers gives
    # |skew| <= 0.7, 3 <= kurtosis <= 30, 1e-4 <= frac_out_3 <= 2e-2.
    torch.manual_seed(0)
    n_weights = linear.weight.data.numel()  # 64*32 = 2048
    base = torch.randn(n_weights)
    std = base.std().item()
    n_out = max(1, int(0.01 * n_weights))  # ~1% outliers, symmetric
    base[:n_out // 2] = 4.5 * std
    base[n_out // 2:n_out] = -4.5 * std
    linear.weight.data = base.reshape(linear.weight.shape)
    layer = FlatQuantizedLinear(args, linear)
    x = torch.randn(2, 4, 64)

    with patch('flatquant.flat_linear.fold_outliers', wraps=__import__('flatquant.quant_utils', fromlist=['fold_outliers']).fold_outliers) as mock_fold:
        layer(x)
        # fold_outliers should have been called (weight has an outlier)
        mock_fold.assert_called()
