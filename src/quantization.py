"""From-scratch Post-Training Quantization (PTQ) kernels and calibration algorithms.

Implements:
- Symmetric INT8 (per-tensor & per-channel) with MinMax, Percentile, and MSE calibration
- Asymmetric (Affine) INT8 (per-tensor & per-channel)
- True Bit-Packed Group-Wise INT4 (2x 4-bit weights packed per uint8 byte)
- Quantitative error metrics (SQNR dB, MSE, max abs error, cosine similarity)
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple
import torch


CalibrationMethod = Literal["minmax", "percentile", "mse"]


@dataclass
class QuantizedTensorINT8:
    """Container for INT8 quantized tensor and quantization metadata."""
    qweight: torch.Tensor        # torch.int8 or torch.uint8
    scale: torch.Tensor          # float32/float16 scale factor(s)
    zero_point: Optional[torch.Tensor]  # None for symmetric; int32/int8 for asymmetric
    symmetric: bool
    per_channel: bool
    axis: int = 0

    def nbytes(self) -> int:
        total = self.qweight.numel() * self.qweight.element_size()
        total += self.scale.numel() * self.scale.element_size()
        if self.zero_point is not None:
            total += self.zero_point.numel() * self.zero_point.element_size()
        return total


@dataclass
class QuantizedTensorINT4:
    """Container for bit-packed group-wise INT4 weights (2 weights per uint8 byte)."""
    packed_weight: torch.Tensor  # torch.uint8, shape [out_features, padded_in // 2]
    scales: torch.Tensor         # torch.float16, shape [out_features, n_groups]
    zeros: torch.Tensor          # torch.float16, shape [out_features, n_groups]
    orig_shape: Tuple[int, int]  # (out_features, in_features)
    group_size: int

    def nbytes(self) -> int:
        return (
            self.packed_weight.numel() * self.packed_weight.element_size()
            + self.scales.numel() * self.scales.element_size()
            + self.zeros.numel() * self.zeros.element_size()
        )


def _find_mse_optimal_amax(
    tensor: torch.Tensor,
    abs_max: torch.Tensor,
    qmax: int = 127,
    steps: int = 20,
    min_ratio: float = 0.75,
) -> torch.Tensor:
    """Grid-search clipping threshold minimizing L2 dequantization error."""
    best_amax = abs_max.clone()
    best_err = torch.full_like(abs_max, float("inf"))
    reduce_dims = tuple(i for i in range(tensor.ndim) if abs_max.shape[i] == 1)

    for ratio in torch.linspace(min_ratio, 1.0, steps, device=tensor.device):
        cand_amax = torch.clamp(abs_max * ratio, min=1e-8)
        scale = cand_amax / qmax
        q = torch.clamp(torch.round(tensor / scale), -qmax, qmax)
        recon = q * scale
        err = (tensor - recon).pow(2).mean(dim=reduce_dims, keepdim=True)
        improved = err < best_err
        best_err = torch.where(improved, err, best_err)
        best_amax = torch.where(improved, cand_amax, best_amax)

    return best_amax


def quantize_int8_symmetric(
    tensor: torch.Tensor,
    per_channel: bool = True,
    axis: int = 0,
    calibration: CalibrationMethod = "minmax",
    percentile: float = 0.999,
) -> QuantizedTensorINT8:
    """Symmetric INT8 quantization mapping [-amax, +amax] -> [-127, +127]."""
    x = tensor.detach().to(torch.float32)
    qmax = 127

    if per_channel:
        reduce_dims = tuple(i for i in range(x.ndim) if i != axis)
        abs_x = x.abs()
        if calibration == "minmax":
            amax = abs_x.amax(dim=reduce_dims, keepdim=True)
        elif calibration == "percentile":
            # Compute percentile per channel along flattened non-axis dimensions
            moved = abs_x.movedim(axis, 0).reshape(x.shape[axis], -1)
            q_vals = torch.quantile(moved, percentile, dim=1)
            view_shape = [1] * x.ndim
            view_shape[axis] = x.shape[axis]
            amax = q_vals.view(*view_shape)
        elif calibration == "mse":
            raw_amax = abs_x.amax(dim=reduce_dims, keepdim=True)
            amax = _find_mse_optimal_amax(x, raw_amax, qmax=qmax)
        else:
            raise ValueError(f"Unknown calibration method: {calibration}")
    else:
        abs_x = x.abs()
        if calibration == "minmax":
            amax = abs_x.amax().view(*([1] * x.ndim))
        elif calibration == "percentile":
            amax = torch.quantile(abs_x.reshape(-1), percentile).view(*([1] * x.ndim))
        elif calibration == "mse":
            raw_amax = abs_x.amax().view(*([1] * x.ndim))
            amax = _find_mse_optimal_amax(x, raw_amax, qmax=qmax)
        else:
            raise ValueError(f"Unknown calibration method: {calibration}")

    scale = torch.clamp(amax, min=1e-8) / qmax
    qweight = torch.clamp(torch.round(x / scale), -qmax, qmax).to(torch.int8)

    return QuantizedTensorINT8(
        qweight=qweight,
        scale=scale.to(torch.float32),
        zero_point=None,
        symmetric=True,
        per_channel=per_channel,
        axis=axis,
    )


def dequantize_int8(qtensor: QuantizedTensorINT8, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Dequantize symmetric or asymmetric INT8 tensor back to floating point."""
    if qtensor.symmetric or qtensor.zero_point is None:
        return (qtensor.qweight.to(torch.float32) * qtensor.scale.to(torch.float32)).to(dtype)
    return (
        (qtensor.qweight.to(torch.float32) - qtensor.zero_point.to(torch.float32))
        * qtensor.scale.to(torch.float32)
    ).to(dtype)


def quantize_int8_asymmetric(
    tensor: torch.Tensor,
    per_channel: bool = True,
    axis: int = 0,
) -> QuantizedTensorINT8:
    """Asymmetric (affine) UINT8 quantization mapping [min, max] -> [0, 255]."""
    x = tensor.detach().to(torch.float32)
    qmin, qmax = 0, 255

    if per_channel:
        reduce_dims = tuple(i for i in range(x.ndim) if i != axis)
        w_min = x.amin(dim=reduce_dims, keepdim=True)
        w_max = x.amax(dim=reduce_dims, keepdim=True)
    else:
        w_min = x.amin().view(*([1] * x.ndim))
        w_max = x.amax().view(*([1] * x.ndim))

    # Ensure zero is representable in range
    w_min = torch.minimum(w_min, torch.zeros_like(w_min))
    w_max = torch.maximum(w_max, torch.zeros_like(w_max))

    scale = torch.clamp(w_max - w_min, min=1e-8) / (qmax - qmin)
    zero_point = torch.clamp(torch.round(-w_min / scale), qmin, qmax).to(torch.int32)
    qweight = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax).to(torch.uint8)

    return QuantizedTensorINT8(
        qweight=qweight,
        scale=scale.to(torch.float32),
        zero_point=zero_point,
        symmetric=False,
        per_channel=per_channel,
        axis=axis,
    )


def pack_int4(unpacked_uint4: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit unsigned integers [0..15] along the last dim into uint8 bytes."""
    if unpacked_uint4.shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension must be even for INT4 packing, got {unpacked_uint4.shape[-1]}")
    u8 = unpacked_uint4.to(torch.uint8) & 0x0F
    low = u8[..., 0::2]
    high = u8[..., 1::2]
    return low | (high << 4)


def unpack_int4(packed_uint8: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 bytes back into pairs of 4-bit unsigned integers [0..15]."""
    low = packed_uint8 & 0x0F
    high = (packed_uint8 >> 4) & 0x0F
    out_shape = packed_uint8.shape[:-1] + (packed_uint8.shape[-1] * 2,)
    return torch.stack([low, high], dim=-1).reshape(out_shape)


def quantize_int4_groupwise(
    weight: torch.Tensor,
    group_size: int = 64,
) -> QuantizedTensorINT4:
    """Group-wise asymmetric INT4 weight quantization with physical uint8 bit-packing.

    Compresses a 2D weight matrix [out_features, in_features] by dividing each row
    into contiguous blocks of `group_size`, computing per-group FP16 scale & zero-point,
    and physically packing two 4-bit weights per uint8 byte.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix [out, in], got {tuple(weight.shape)}")
    if group_size <= 0 or group_size % 2 != 0:
        raise ValueError(f"group_size must be a positive even integer, got {group_size}")

    w = weight.detach().to(torch.float32)
    out_features, in_features = w.shape

    # Pad in_features to multiple of group_size
    remainder = in_features % group_size
    pad_len = (group_size - remainder) if remainder != 0 else 0
    if pad_len > 0:
        w = torch.nn.functional.pad(w, (0, pad_len), mode="constant", value=0.0)

    padded_in = w.shape[1]
    n_groups = padded_in // group_size
    grouped = w.view(out_features, n_groups, group_size)

    g_min = grouped.amin(dim=-1, keepdim=True)
    g_max = grouped.amax(dim=-1, keepdim=True)
    scales = torch.clamp(g_max - g_min, min=1e-8) / 15.0
    zeros = g_min

    # Quantize to [0, 15]
    q_int4 = torch.clamp(torch.round((grouped - zeros) / scales), 0, 15).to(torch.uint8)
    q_flat = q_int4.view(out_features, padded_in)
    packed = pack_int4(q_flat)

    return QuantizedTensorINT4(
        packed_weight=packed,
        scales=scales.squeeze(-1).to(torch.float16),
        zeros=zeros.squeeze(-1).to(torch.float16),
        orig_shape=(out_features, in_features),
        group_size=group_size,
    )


def dequantize_int4_groupwise(
    qtensor: QuantizedTensorINT4,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unpack and dequantize bit-packed INT4 weights back to floating point."""
    unpacked = unpack_int4(qtensor.packed_weight).to(torch.float32)
    out_features, in_features = qtensor.orig_shape
    n_groups = qtensor.scales.shape[1]

    grouped = unpacked.view(out_features, n_groups, qtensor.group_size)
    scales = qtensor.scales.to(torch.float32).unsqueeze(-1)
    zeros = qtensor.zeros.to(torch.float32).unsqueeze(-1)

    recon = (grouped * scales + zeros).view(out_features, -1)
    return recon[:, :in_features].to(dtype)


def compute_sqnr_db(reference: torch.Tensor, approx: torch.Tensor) -> float:
    """Compute Signal-to-Quantization-Noise Ratio in decibels (higher is better)."""
    ref_f = reference.detach().to(torch.float64)
    app_f = approx.detach().to(torch.float64)
    signal_power = ref_f.pow(2).sum().item()
    noise_power = (ref_f - app_f).pow(2).sum().item()
    if noise_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return 0.0
    return float(10.0 * torch.log10(torch.tensor(signal_power / noise_power)).item())


def compute_quantization_metrics(reference: torch.Tensor, approx: torch.Tensor) -> Dict[str, float]:
    """Compute comprehensive numerical fidelity metrics between reference and quantized tensors."""
    ref_f = reference.detach().to(torch.float64).reshape(-1)
    app_f = approx.detach().to(torch.float64).reshape(-1)
    diff = ref_f - app_f

    mse = float(diff.pow(2).mean().item())
    max_abs_err = float(diff.abs().max().item())
    cos_sim = float(
        torch.nn.functional.cosine_similarity(ref_f.unsqueeze(0), app_f.unsqueeze(0)).item()
    )
    sqnr = compute_sqnr_db(ref_f, app_f)

    return {
        "sqnr_db": round(sqnr, 3),
        "mse": mse,
        "max_abs_error": round(max_abs_err, 6),
        "cosine_similarity": round(cos_sim, 6),
    }
