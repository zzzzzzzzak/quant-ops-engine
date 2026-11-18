"""Numerical stability diagnostics, outlier sensitivity auditor, and gradient monitor.

Designed for isolating numerical failures (FP16/FP32 overflow, catastrophic
cancellation, NaN gradients) and verifying objective numerical correctness.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List
import torch
import torch.nn as nn

from src.numerical_ops import (
    naive_layernorm,
    naive_logsumexp,
    naive_softmax,
    safe_online_softmax,
    stable_layernorm,
    stable_logsumexp,
)
from src.quantization import (
    compute_quantization_metrics,
    dequantize_int8,
    quantize_int8_symmetric,
)


@dataclass
class GradientAlert:
    layer_name: str
    issue: str
    grad_norm: float


class GradientHealthMonitor:
    """Context manager attaching backward hooks to isolate NaN/Inf/exploding gradients."""

    def __init__(self, model: nn.Module, max_grad_norm: float = 1e3):
        self.model = model
        self.max_grad_norm = max_grad_norm
        self.alerts: List[GradientAlert] = []
        self._handles: List[Any] = []

    def _make_hook(self, name: str):
        def _hook(grad: torch.Tensor):
            if torch.isnan(grad).any():
                self.alerts.append(GradientAlert(name, "NaN detected in gradient", float("nan")))
            elif torch.isinf(grad).any():
                self.alerts.append(GradientAlert(name, "Inf detected in gradient", float("inf")))
            else:
                norm = float(grad.detach().to(torch.float32).norm().item())
                if norm > self.max_grad_norm:
                    self.alerts.append(
                        GradientAlert(name, f"Exploding gradient (>{self.max_grad_norm})", round(norm, 3))
                    )
        return _hook

    def __enter__(self) -> "GradientHealthMonitor":
        self.alerts.clear()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(self._make_hook(name))
                self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def is_healthy(self) -> bool:
        return len(self.alerts) == 0


def run_numerical_stress_audit() -> Dict[str, Dict[str, Any]]:
    """Run adversarial numerical failure scenarios comparing naive vs. stable operators."""
    torch.manual_seed(42)
    results: Dict[str, Dict[str, Any]] = {}

    # 1. FP16 Softmax Overflow (exp(20) > 65504 max FP16)
    x_fp16 = torch.tensor([[20.0, 21.0, 19.0, 18.0]], dtype=torch.float16)
    naive_sm = naive_softmax(x_fp16, dim=-1)
    stable_sm = safe_online_softmax(x_fp16, dim=-1, block_size=2)
    ref_sm = torch.softmax(x_fp16.to(torch.float32), dim=-1).to(torch.float16)

    results["fp16_softmax_overflow"] = {
        "naive_has_nan_or_inf": bool(torch.isnan(naive_sm).any() or torch.isinf(naive_sm).any()),
        "stable_has_nan_or_inf": bool(torch.isnan(stable_sm).any() or torch.isinf(stable_sm).any()),
        "stable_max_err_vs_fp32_ref": round(float((stable_sm - ref_sm).abs().max().item()), 6),
        "status": "PASS" if not torch.isnan(stable_sm).any() else "FAIL",
    }

    # 2. Extreme FP32 LogSumExp Overflow (exp(1000) -> inf)
    x_ext = torch.tensor([[1000.0, 1001.0, 999.0]], dtype=torch.float32)
    naive_lse = naive_logsumexp(x_ext, dim=-1)
    stable_lse = stable_logsumexp(x_ext, dim=-1)
    ref_lse = torch.logsumexp(x_ext, dim=-1)

    results["fp32_logsumexp_overflow"] = {
        "naive_output": float(naive_lse.item()),
        "stable_output": round(float(stable_lse.item()), 4),
        "stable_err_vs_torch_ref": round(float((stable_lse - ref_lse).abs().item()), 6),
        "status": "PASS" if torch.isfinite(stable_lse).all() else "FAIL",
    }

    # 3. Catastrophic Cancellation in LayerNorm (large mean offset = 1e5 in FP32)
    x_offset = 1e5 + torch.randn(4, 256, dtype=torch.float32)
    naive_ln = naive_layernorm(x_offset)
    stable_ln = stable_layernorm(x_offset)
    ref_ln = torch.nn.functional.layer_norm(x_offset.to(torch.float64), (256,)).to(torch.float32)

    naive_ln_has_nan = bool(torch.isnan(naive_ln).any())
    naive_ln_err = float("nan") if naive_ln_has_nan else float((naive_ln - ref_ln).abs().max().item())
    stable_ln_err = float((stable_ln - ref_ln).abs().max().item())

    results["layernorm_catastrophic_cancellation"] = {
        "naive_has_nan": naive_ln_has_nan,
        "naive_max_err_vs_fp64_ref": "NaN" if naive_ln_has_nan else round(naive_ln_err, 6),
        "stable_max_err_vs_fp64_ref": round(stable_ln_err, 6),
        "status": "PASS" if (not torch.isnan(stable_ln).any() and stable_ln_err < 1e-4) else "FAIL",
    }

    # 4. Outlier Weight Quantization Sensitivity (emergent 25-sigma outliers)
    w_outlier = torch.randn(128, 512, dtype=torch.float32) * 0.05
    # Inject 0.1% heavy outliers (common in LLM transformer projections)
    w_outlier[0, 0] = 25.0
    w_outlier[17, 128] = -22.0

    q_minmax = quantize_int8_symmetric(w_outlier, per_channel=False, calibration="minmax")
    q_mse = quantize_int8_symmetric(w_outlier, per_channel=False, calibration="mse")
    q_per_ch = quantize_int8_symmetric(w_outlier, per_channel=True, calibration="mse")

    m_minmax = compute_quantization_metrics(w_outlier, dequantize_int8(q_minmax))
    m_mse = compute_quantization_metrics(w_outlier, dequantize_int8(q_mse))
    m_per_ch = compute_quantization_metrics(w_outlier, dequantize_int8(q_per_ch))

    results["outlier_quantization_sensitivity"] = {
        "per_tensor_minmax_sqnr_db": m_minmax["sqnr_db"],
        "per_tensor_mse_sqnr_db": m_mse["sqnr_db"],
        "per_channel_mse_sqnr_db": m_per_ch["sqnr_db"],
        "per_channel_gain_db": round(m_per_ch["sqnr_db"] - m_minmax["sqnr_db"], 2),
        "status": "PASS" if m_per_ch["sqnr_db"] > m_minmax["sqnr_db"] else "FAIL",
    }

    return results
