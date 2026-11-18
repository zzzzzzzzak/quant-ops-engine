import pytest
import torch
import torch.nn.functional as F

from src.numerical_ops import (
    naive_layernorm,
    naive_softmax,
    rmsnorm,
    safe_online_softmax,
    stable_cross_entropy,
    stable_layernorm,
    stable_logsumexp,
)


def test_stable_logsumexp_parity_and_extremes():
    torch.manual_seed(0)
    x = torch.randn(4, 16, 32)
    assert torch.allclose(stable_logsumexp(x, dim=-1), torch.logsumexp(x, dim=-1), atol=1e-5)

    # Extreme overflow test + all -inf masked row
    x_extreme = torch.tensor(
        [
            [1000.0, 1001.0, 999.0],
            [-1000.0, -999.0, -1001.0],
            [float("-inf"), float("-inf"), float("-inf")],
        ],
        dtype=torch.float32,
    )
    out = stable_logsumexp(x_extreme, dim=-1)
    assert torch.isfinite(out[0]) and torch.isfinite(out[1])
    assert out[2].item() == float("-inf")
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("block_size", [1, 7, 32, 64])
def test_safe_online_softmax_block_invariance(block_size: int):
    torch.manual_seed(1)
    x = torch.randn(3, 5, 53)
    ref = torch.softmax(x, dim=-1)
    out = safe_online_softmax(x, dim=-1, block_size=block_size)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)


def test_safe_online_softmax_prevents_fp16_overflow():
    x_fp16 = torch.tensor([[20.0, 21.0, 19.0, 18.0]], dtype=torch.float16)
    naive = naive_softmax(x_fp16, dim=-1)
    assert torch.isnan(naive).any(), "Expected naive FP16 softmax to overflow to NaN"

    stable = safe_online_softmax(x_fp16, dim=-1, block_size=2)
    ref = torch.softmax(x_fp16.to(torch.float32), dim=-1).to(torch.float16)
    assert not torch.isnan(stable).any()
    assert torch.allclose(stable, ref, atol=1e-3)


def test_stable_cross_entropy_parity_and_ignore_index():
    torch.manual_seed(2)
    logits = torch.randn(16, 10) * 50.0  # Extreme logits
    targets = torch.randint(0, 10, (16,))
    targets[0] = -100
    targets[5] = -100

    out = stable_cross_entropy(logits, targets, ignore_index=-100, reduction="mean")
    ref = F.cross_entropy(logits, targets, ignore_index=-100, reduction="mean")
    assert torch.isfinite(out)
    assert torch.allclose(out, ref, atol=1e-4)


def test_stable_layernorm_catastrophic_cancellation():
    torch.manual_seed(3)
    x_offset = 1e5 + torch.randn(8, 128, dtype=torch.float32)
    ref_fp64 = F.layer_norm(x_offset.to(torch.float64), (128,)).to(torch.float32)

    stable_out = stable_layernorm(x_offset)
    naive_out = naive_layernorm(x_offset)

    assert not torch.isnan(stable_out).any()
    assert torch.allclose(stable_out, ref_fp64, atol=1e-4)
    # Naive one-pass suffers catastrophic cancellation or NaN
    naive_failed = torch.isnan(naive_out).any() or (naive_out - ref_fp64).abs().max().item() > 1e-2
    assert naive_failed


def test_rmsnorm_fp16_parity():
    torch.manual_seed(4)
    x = torch.randn(4, 32, 64, dtype=torch.float16) * 5.0
    w = torch.ones(64, dtype=torch.float16)

    out = rmsnorm(x, weight=w, eps=1e-6)
    x_fp64 = x.to(torch.float64)
    ref = (x_fp64 * torch.rsqrt(x_fp64.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(torch.float16)

    assert not torch.isnan(out).any()
    assert torch.allclose(out, ref, atol=1e-2)
