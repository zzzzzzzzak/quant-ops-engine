"""Quantized PyTorch linear layers, recursive model converter, and memory profiling."""

import copy
from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantization import (
    CalibrationMethod,
    QuantizedTensorINT4,
    QuantizedTensorINT8,
    dequantize_int4_groupwise,
    dequantize_int8,
    quantize_int4_groupwise,
    quantize_int8_symmetric,
)
from src.numerical_ops import rmsnorm, safe_online_softmax


class QuantLinearINT8(nn.Module):
    """Drop-in replacement for nn.Linear using symmetric per-channel INT8 weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        calibration: CalibrationMethod = "minmax",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.calibration = calibration

        self.register_buffer("qweight", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.ones((out_features, 1), dtype=torch.float32))
        if has_bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        calibration: CalibrationMethod = "minmax",
    ) -> "QuantLinearINT8":
        q_mod = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            has_bias=linear.bias is not None,
            calibration=calibration,
        )
        qt = quantize_int8_symmetric(
            linear.weight.data,
            per_channel=True,
            axis=0,
            calibration=calibration,
        )
        q_mod.qweight.copy_(qt.qweight)
        q_mod.scale.copy_(qt.scale)
        if linear.bias is not None and q_mod.bias is not None:
            q_mod.bias.copy_(linear.bias.data.to(torch.float32))
        return q_mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.qweight.to(x.dtype) * self.scale.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class QuantLinearINT4(nn.Module):
    """Drop-in replacement for nn.Linear storing physically bit-packed INT4 weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 64,
        has_bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        remainder = in_features % group_size
        padded_in = in_features + ((group_size - remainder) if remainder != 0 else 0)
        n_groups = padded_in // group_size

        self.register_buffer(
            "packed_weight",
            torch.zeros((out_features, padded_in // 2), dtype=torch.uint8),
        )
        self.register_buffer("scales", torch.ones((out_features, n_groups), dtype=torch.float16))
        self.register_buffer("zeros", torch.zeros((out_features, n_groups), dtype=torch.float16))
        if has_bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear: nn.Linear, group_size: int = 64) -> "QuantLinearINT4":
        q_mod = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            group_size=group_size,
            has_bias=linear.bias is not None,
        )
        qt = quantize_int4_groupwise(linear.weight.data, group_size=group_size)
        q_mod.packed_weight.copy_(qt.packed_weight)
        q_mod.scales.copy_(qt.scales)
        q_mod.zeros.copy_(qt.zeros)
        if linear.bias is not None and q_mod.bias is not None:
            q_mod.bias.copy_(linear.bias.data.to(torch.float32))
        return q_mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qt = QuantizedTensorINT4(
            packed_weight=self.packed_weight,
            scales=self.scales,
            zeros=self.zeros,
            orig_shape=(self.out_features, self.in_features),
            group_size=self.group_size,
        )
        w = dequantize_int4_groupwise(qt, dtype=x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def quantize_model(
    model: nn.Module,
    mode: Literal["int8", "int4"] = "int8",
    calibration: CalibrationMethod = "mse",
    group_size: int = 64,
    inplace: bool = False,
) -> nn.Module:
    """Recursively convert all nn.Linear modules in a PyTorch model to quantized layers."""
    target = model if inplace else copy.deepcopy(model)

    def _replace(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                if mode == "int8":
                    setattr(module, name, QuantLinearINT8.from_float(child, calibration=calibration))
                elif mode == "int4":
                    setattr(module, name, QuantLinearINT4.from_float(child, group_size=group_size))
                else:
                    raise ValueError(f"Unsupported quantization mode: {mode}")
            else:
                _replace(child)

    _replace(target)
    return target


def compute_model_memory_bytes(model: nn.Module) -> int:
    """Compute exact physical memory footprint of all parameters and buffers in bytes."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return int(param_bytes + buffer_bytes)


class StableTransformerBlock(nn.Module):
    """Reference Transformer block using RMSNorm and Flash-style Online Safe Softmax."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, ffn_mult: int = 4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1_weight = nn.Parameter(torch.ones(d_model))
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2_weight = nn.Parameter(torch.ones(d_model))
        self.ffn_up = nn.Linear(d_model, d_model * ffn_mult, bias=True)
        self.ffn_down = nn.Linear(d_model * ffn_mult, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        h = rmsnorm(x, self.norm1_weight)

        q = self.q_proj(h).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = safe_online_softmax(scores, dim=-1, block_size=32)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        x = x + self.o_proj(context)

        h2 = rmsnorm(x, self.norm2_weight)
        ffn_out = self.ffn_down(F.silu(self.ffn_up(h2)))
        return x + ffn_out


class BenchmarkTransformer(nn.Module):
    """Multi-layer Transformer stack for quantization compression & latency benchmarking."""

    def __init__(self, n_layers: int = 4, d_model: int = 512, n_heads: int = 8, ffn_mult: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [StableTransformerBlock(d_model=d_model, n_heads=n_heads, ffn_mult=ffn_mult) for _ in range(n_layers)]
        )
        self.final_norm_weight = nn.Parameter(torch.ones(d_model))
        self.head = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = rmsnorm(x, self.final_norm_weight)
        return self.head(x)
