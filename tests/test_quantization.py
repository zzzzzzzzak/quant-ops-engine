import torch

from src.quantization import (
    compute_quantization_metrics,
    compute_sqnr_db,
    dequantize_int4_groupwise,
    dequantize_int8,
    pack_int4,
    quantize_int4_groupwise,
    quantize_int8_asymmetric,
    quantize_int8_symmetric,
    unpack_int4,
)


def test_int4_bit_packing_lossless_roundtrip():
    """Every 4-bit value [0..15] must survive pack -> unpack with exact equality."""
    torch.manual_seed(10)
    vals = torch.randint(0, 16, (32, 128), dtype=torch.uint8)
    packed = pack_int4(vals)
    assert packed.dtype == torch.uint8
    assert packed.shape == (32, 64)  # 2x physical reduction

    unpacked = unpack_int4(packed)
    assert torch.equal(vals, unpacked)


def test_int8_symmetric_fidelity_and_memory():
    torch.manual_seed(11)
    w = torch.randn(256, 512, dtype=torch.float32)
    q = quantize_int8_symmetric(w, per_channel=True, calibration="minmax")
    recon = dequantize_int8(q)

    metrics = compute_quantization_metrics(w, recon)
    assert metrics["sqnr_db"] > 35.0
    assert metrics["cosine_similarity"] > 0.999
    # Nearly 4x smaller than FP32
    compression = (w.numel() * w.element_size()) / q.nbytes()
    assert compression > 3.9


def test_int8_asymmetric_preserves_exact_zero():
    torch.manual_seed(12)
    w = torch.randn(64, 128) + 2.5  # Shifted non-symmetric distribution
    w[0, 0] = 0.0
    q = quantize_int8_asymmetric(w, per_channel=True)
    recon = dequantize_int8(q)

    assert abs(recon[0, 0].item()) < 1e-6
    assert compute_sqnr_db(w, recon) > 35.0


def test_mse_calibration_beats_minmax_on_outliers():
    torch.manual_seed(13)
    w = torch.randn(64, 256) * 0.05
    w[0, 0] = 20.0  # Heavy emergent outlier

    q_minmax = quantize_int8_symmetric(w, per_channel=False, calibration="minmax")
    q_mse = quantize_int8_symmetric(w, per_channel=False, calibration="mse")

    sqnr_minmax = compute_sqnr_db(w, dequantize_int8(q_minmax))
    sqnr_mse = compute_sqnr_db(w, dequantize_int8(q_mse))
    assert sqnr_mse > sqnr_minmax


def test_int4_groupwise_padding_and_compression():
    """Group-wise INT4 must handle non-divisible dimensions and compress > 6x."""
    torch.manual_seed(14)
    w = torch.randn(128, 300, dtype=torch.float32)  # 300 not divisible by 64
    q = quantize_int4_groupwise(w, group_size=64)
    recon = dequantize_int4_groupwise(q)

    assert recon.shape == w.shape
    metrics = compute_quantization_metrics(w, recon)
    assert metrics["sqnr_db"] > 17.0
    assert metrics["cosine_similarity"] > 0.99

    compression = (w.numel() * w.element_size()) / q.nbytes()
    assert compression > 6.0
