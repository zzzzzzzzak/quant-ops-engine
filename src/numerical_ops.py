"""Numerically stable core ML primitives with explicit failure-mode references.

Implements production-grade numerical operators designed to prevent IEEE 754
overflow, underflow, NaN propagation, and catastrophic cancellation across
FP32, FP16, and BF16 precisions.
"""

from typing import Optional
import torch


def naive_logsumexp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Reference naive log-sum-exp (vulnerable to exp overflow > 88.7 in FP32, > 11.0 in FP16)."""
    return torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=keepdim))


def stable_logsumexp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Numerically stable log-sum-exp using max-shift and FP32 accumulation.

    Handles extreme positive logits (prevents exp -> inf) and all -inf masked rows
    (prevents (-inf) - (-inf) -> NaN).
    """
    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)

    max_val = torch.amax(x_fp32, dim=dim, keepdim=True)
    # Guard against all -inf slices where max_val == -inf -> (-inf) - (-inf) = NaN
    safe_max = torch.where(torch.isfinite(max_val), max_val, torch.zeros_like(max_val))

    shifted = x_fp32 - safe_max
    lse = safe_max + torch.log(torch.sum(torch.exp(shifted), dim=dim, keepdim=True))
    # Restore -inf where the entire slice was -inf
    lse = torch.where(torch.isfinite(max_val), lse, max_val)

    if not keepdim:
        lse = lse.squeeze(dim=dim)
    return lse.to(orig_dtype)


def naive_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Reference naive softmax (overflows to NaN on moderate logits in FP16/FP32)."""
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def safe_online_softmax(
    x: torch.Tensor,
    dim: int = -1,
    block_size: int = 64,
) -> torch.Tensor:
    """FlashAttention-style chunked online safe softmax with FP32 running statistics.

    Computes normalization statistics (running max m_i and running normalizer d_i)
    in tiled blocks without materializing the full unnormalized exponentiated tensor
    at once, preventing FP16/BF16 overflow and reducing peak intermediate memory.
    """
    orig_dtype = x.dtype
    # Move target dimension to last axis for contiguous tiled iteration
    x_moved = x.movedim(dim, -1).to(torch.float32)
    seq_len = x_moved.shape[-1]

    if seq_len == 0:
        return x.clone()

    # Pass 1: Online tiled computation of global max (m) and partition function (d)
    m = torch.full(x_moved.shape[:-1] + (1,), float("-inf"), dtype=torch.float32, device=x.device)
    d = torch.zeros(x_moved.shape[:-1] + (1,), dtype=torch.float32, device=x.device)

    for start in range(0, seq_len, block_size):
        block = x_moved[..., start : start + block_size]
        block_max = torch.amax(block, dim=-1, keepdim=True)
        new_m = torch.maximum(m, block_max)

        # Safe rescale factor: when m == -inf and new_m == -inf, scale = 0
        m_valid = torch.isfinite(m)
        scale_old = torch.where(m_valid, torch.exp(m - new_m), torch.zeros_like(m))

        new_m_safe = torch.where(torch.isfinite(new_m), new_m, torch.zeros_like(new_m))
        block_exp = torch.where(
            torch.isfinite(new_m),
            torch.exp(block - new_m_safe),
            torch.zeros_like(block),
        )
        d = d * scale_old + torch.sum(block_exp, dim=-1, keepdim=True)
        m = new_m

    # Pass 2: Normalize output using final (m, d)
    m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    d_safe = torch.where(d > 0, d, torch.ones_like(d))
    probs = torch.where(
        torch.isfinite(m),
        torch.exp(x_moved - m_safe) / d_safe,
        torch.zeros_like(x_moved),
    )
    return probs.movedim(-1, dim).to(orig_dtype)


def stable_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Fused log-softmax + NLL cross-entropy in FP32 accumulation.

    Avoids computing log(softmax(x)) explicitly, preventing log(0) -> -inf
    and 0 * (-inf) -> NaN gradients on confident predictions.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits [N, C], got shape {tuple(logits.shape)}")

    logits_fp32 = logits.to(torch.float32)
    lse = stable_logsumexp(logits_fp32, dim=-1, keepdim=True)
    log_probs = logits_fp32 - lse

    valid_mask = targets != ignore_index
    safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))

    nll = -log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    nll = torch.where(valid_mask, nll, torch.zeros_like(nll))

    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        denom = valid_mask.sum().clamp(min=1)
        return nll.sum() / denom
    raise ValueError(f"Unsupported reduction: {reduction}")


def naive_layernorm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """One-pass E[x^2] - E[x]^2 LayerNorm (prone to catastrophic cancellation on large offsets)."""
    mean = x.mean(dim=-1, keepdim=True)
    mean_sq = (x * x).mean(dim=-1, keepdim=True)
    var = mean_sq - mean * mean  # Can become negative or lose precision when mean >> std
    normed = (x - mean) * torch.rsqrt(var + eps)
    if weight is not None:
        normed = normed * weight
    if bias is not None:
        normed = normed + bias
    return normed


def stable_layernorm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Shifted two-pass LayerNorm (Chan-Golub-LeVeque) with FP32 accumulation.

    Shifts activations by the slice anchor x[..., :1] prior to mean summation,
    eliminating both catastrophic cancellation in E[x^2] - E[x]^2 AND mantissa
    precision loss when summing large-offset vectors (e.g., 1e5 + N(0, 1)) in FP32.
    """
    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    anchor = x_fp32[..., :1]
    shifted = x_fp32 - anchor
    mean_shifted = shifted.mean(dim=-1, keepdim=True)
    centered = shifted - mean_shifted
    var = (centered * centered).mean(dim=-1, keepdim=True)
    normed = centered * torch.rsqrt(var + eps)

    if weight is not None:
        normed = normed * weight.to(torch.float32)
    if bias is not None:
        normed = normed + bias.to(torch.float32)
    return normed.to(orig_dtype)


def rmsnorm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Root Mean Square Layer Normalization (LLaMA/Mistral style) with FP32 upcasting."""
    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x_fp32 * rms
    if weight is not None:
        out = out * weight.to(torch.float32)
    return out.to(orig_dtype)
