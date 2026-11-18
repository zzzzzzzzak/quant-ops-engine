# quant-ops-engine

[![CI Pipeline](https://github.com/zzzzzzzzak/quant-ops-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/zzzzzzzzak/quant-ops-engine/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-18%20Passed-brightgreen.svg)](tests/)

First-principles Post-Training Quantization (PTQ), physically bit-packed INT4/INT8 linear layers, numerically stable ML primitives, and an automated numerical failure debugger built in Python and PyTorch.

---

## Core Capabilities

1. **Physical Weight Compression (`src/quantization.py`, `src/layers.py`)**
   - **Symmetric & Asymmetric INT8**: Per-tensor and per-output-channel quantization with `minmax`, `percentile`, and `mse` grid-search calibration.
   - **Bit-Packed Group-Wise INT4**: Divides weight rows into contiguous groups (`group_size=64` or `128`) with FP16 per-group scales/zeros and packs two 4-bit integers per `uint8` byte (`packed = low | (high << 4)`). Achieves **7.47x** actual RAM reduction on Transformer weights.
   - **Drop-In Recursive Converter**: `quantize_model()` swaps every `nn.Linear` module in a PyTorch architecture into `QuantLinearINT8` or `QuantLinearINT4`.

2. **Numerically Stable Core Operators (`src/numerical_ops.py`)**
   - **Online Safe Softmax**: FlashAttention-style tiled streaming softmax maintaining running `(m_i, d_i)` in FP32 to prevent FP16/BF16 exponent overflow.
   - **Shifted Two-Pass LayerNorm**: Implements the Chan-Golub-LeVeque anchor-shifted variance algorithm (`x - x[..., :1]`), eliminating catastrophic cancellation and FP32 mantissa loss on large-offset activations (`1e5 + N(0, 1)`).
   - **Stable LogSumExp & Fused Cross-Entropy**: Max-shifted log-sum-exp with explicit all-`-inf` attention-mask guards (`(-inf) - (-inf) != NaN`).
   - **RMSNorm**: Low-precision-safe Root Mean Square Normalization with FP32 upcasting.

3. **Numerical Failure Debugger (`src/debugger.py`)**
   - **`run_numerical_stress_audit()`**: Automated adversarial suite contrasting naive implementations against stable ops under extreme inputs and emergent 25-sigma weight outliers.
   - **`GradientHealthMonitor`**: Backward-hook context manager that isolates exact layer names producing `NaN`, `Inf`, or exploding L2 gradient norms during training.

---

## Measured Benchmark Results

Evaluated on a 4-layer Transformer stack (`d_model=512`, `n_heads=8`, `ffn_mult=4`, **12.86M parameters**, batch `[4, 64, 512]`, CPU). Full output saved in [`benchmarks/results/benchmark_report.json`](benchmarks/results/benchmark_report.json).

### 1. Compression, Fidelity & Latency

| Configuration | Weight Memory | Compression | Output SQNR (dB) | Output Cosine Sim | CPU Latency (p50) |
|---|:---:|:---:|:---:|:---:|:---:|
| **FP32 Baseline** | 49.06 MB | 1.00x | `inf` | 1.00000 | 83.31 ms |
| **INT8 Per-Channel (MinMax)** | 12.38 MB | **3.96x** | 47.059 dB | 0.99999 | 118.37 ms |
| **INT8 Per-Channel (MSE)** | 12.38 MB | **3.96x** | 47.065 dB | 0.99999 | 100.33 ms |
| **INT4 Group-Wise (g=64, Bit-Packed)** | 6.95 MB | **7.06x** | 22.867 dB | 0.99743 | 146.98 ms |
| **INT4 Group-Wise (g=128, Bit-Packed)** | 6.56 MB | **7.47x** | 22.651 dB | 0.99730 | 121.43 ms |

> **Hardware Trade-off Note (Memory vs. Eager Latency):** While bit-packed INT4 cuts weight memory bandwidth by **7.47x**, running dequantization as a separate eager PyTorch op before `F.linear` introduces an extra tensor allocation and memory pass on CPU (121 ms vs. 83 ms). Realizing wall-clock latency speedups requires fusing bit-unpacking directly inside the GEMM inner loop (e.g., Triton/CUDA or AVX-512 C++ kernels), where memory-bound autoregressive decoding (`batch=1`) directly scales with weight footprint reduction.

### 2. Adversarial Numerical Stress Audit

| Stress Scenario | Naive Baseline | Stable / Calibrated Operator | Status |
|---|---|---|---|
| **FP16 Softmax Overflow** (`logits = [20, 21, 19, 18]`) | `NaN / Inf` (`exp(20) > 65504`) | `max_err = 0.0` vs. FP32 ref | **PASS** |
| **FP32 LogSumExp Overflow** (`logits = [1000, 1001, 999]`) | `inf` | `1001.4076` (`err = 0.0` vs. torch) | **PASS** |
| **LayerNorm Catastrophic Cancellation** (`1e5 + N(0,1)`) | `NaN` (negative variance) | `max_err = 0.0` vs. FP64 ref | **PASS** |
| **Emergent 25-Sigma Weight Outliers** (INT8 PTQ) | Per-Tensor MinMax: `9.55 dB` | Per-Channel MSE: `27.94 dB` (**+18.39 dB**) | **PASS** |

---

## Bit-Packing Memory Layout (INT4)

```text
Two unsigned 4-bit weights (w_even, w_odd in [0..15]) -> 1 uint8 byte:

  Bit:    7   6   5   4  |  3   2   1   0
        +----------------+----------------+
  Byte: |   high nibble  |   low nibble   |
        |   (w_odd << 4) | (w_even & 0xF) |
        +----------------+----------------+

Unpacking:
  w_even = byte & 0x0F
  w_odd  = (byte >> 4) & 0x0F
  w_fp32 = w_uint4 * scale_fp16 + zero_fp16
```

---

## Repository Structure

```text
quant-ops-engine/
├── .github/workflows/ci.yml       # Automated CPU PyTorch CI (tests + benchmarks)
├── benchmarks/
│   ├── run_benchmarks.py          # Transformer compression, fidelity & latency runner
│   └── results/
│       └── benchmark_report.json  # Serialized benchmark and stress audit outputs
├── src/
│   ├── __init__.py
│   ├── numerical_ops.py           # Online Softmax, Shifted LayerNorm, RMSNorm, LogSumExp
│   ├── quantization.py            # INT8 (MinMax/Percentile/MSE) & Bit-Packed INT4 kernels
│   ├── layers.py                  # QuantLinearINT8/INT4, recursive converter, Transformer block
│   └── debugger.py                # Stress audit suite & GradientHealthMonitor hooks
├── tests/
│   ├── test_numerical_ops.py      # Parity & adversarial overflow/cancellation tests
│   ├── test_quantization.py       # Lossless bit-packing, SQNR, outlier calibration tests
│   ├── test_layers.py             # Forward parity & physical memory compression assertions
│   └── test_debugger.py           # NaN/exploding gradient hook detection tests
├── requirements.txt
└── README.md
```

---

## Quickstart & Verification

```bash
git clone https://github.com/zzzzzzzzak/quant-ops-engine.git
cd quant-ops-engine

pip install -r requirements.txt

# 1. Run 18 objective correctness & adversarial stability tests
pytest tests/ -v

# 2. Reproduce compression, fidelity & numerical stress benchmarks
python benchmarks/run_benchmarks.py
```

---

## Author

**Siddiqui Zakir Ali**  
*ML Systems & Data Analytics | Numerical Computing, Optimization & Production Pipelines*  
* [LinkedIn](https://linkedin.com/in/zakir-siddiqui-182b9711a)
* [GitHub](https://github.com/zzzzzzzzak)
