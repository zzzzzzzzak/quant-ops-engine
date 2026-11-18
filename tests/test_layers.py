import torch
import torch.nn as nn

from src.layers import (
    BenchmarkTransformer,
    QuantLinearINT4,
    QuantLinearINT8,
    compute_model_memory_bytes,
    quantize_model,
)
from src.quantization import compute_quantization_metrics


def test_quant_linear_int8_and_int4_forward_parity():
    torch.manual_seed(20)
    linear = nn.Linear(256, 128, bias=True).eval()
    x = torch.randn(8, 256)

    with torch.no_grad():
        ref = linear(x)
        q8_out = QuantLinearINT8.from_float(linear, calibration="mse")(x)
        q4_out = QuantLinearINT4.from_float(linear, group_size=64)(x)

    assert q8_out.shape == ref.shape
    assert q4_out.shape == ref.shape

    m8 = compute_quantization_metrics(ref, q8_out)
    m4 = compute_quantization_metrics(ref, q4_out)
    assert m8["cosine_similarity"] > 0.999
    assert m4["cosine_similarity"] > 0.99


def test_recursive_transformer_quantization_and_memory_savings():
    torch.manual_seed(21)
    model = BenchmarkTransformer(n_layers=2, d_model=256, n_heads=4).eval()
    x = torch.randn(2, 16, 256)

    base_bytes = compute_model_memory_bytes(model)
    m_int8 = quantize_model(model, mode="int8", calibration="mse").eval()
    m_int4 = quantize_model(model, mode="int4", group_size=128).eval()

    bytes_int8 = compute_model_memory_bytes(m_int8)
    bytes_int4 = compute_model_memory_bytes(m_int4)

    assert base_bytes / bytes_int8 > 3.8, f"Expected >3.8x INT8 compression, got {base_bytes / bytes_int8:.2f}x"
    assert base_bytes / bytes_int4 > 6.5, f"Expected >6.5x INT4 compression, got {base_bytes / bytes_int4:.2f}x"

    with torch.no_grad():
        ref = model(x)
        out8 = m_int8(x)
        out4 = m_int4(x)

    assert torch.isfinite(out8).all()
    assert torch.isfinite(out4).all()
    assert compute_quantization_metrics(ref, out8)["cosine_similarity"] > 0.999
    assert compute_quantization_metrics(ref, out4)["cosine_similarity"] > 0.97
