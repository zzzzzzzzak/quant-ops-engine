"""End-to-end model compression, memory footprint, fidelity, and latency benchmark suite."""

import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.debugger import run_numerical_stress_audit
from src.layers import BenchmarkTransformer, compute_model_memory_bytes, quantize_model
from src.quantization import compute_quantization_metrics

RESULTS_DIR = os.path.join(ROOT_DIR, "benchmarks", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _measure_latency_ms(model: torch.nn.Module, x: torch.Tensor, warmup: int = 3, iters: int = 12) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        times_ms: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
    return {
        "p50_ms": round(float(np.percentile(times_ms, 50)), 2),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 2),
        "mean_ms": round(float(np.mean(times_ms)), 2),
    }


def run_all_benchmarks(
    n_layers: int = 4,
    d_model: int = 512,
    n_heads: int = 8,
    batch_size: int = 4,
    seq_len: int = 64,
    iters: int = 10,
) -> Dict[str, Any]:
    torch.manual_seed(42)
    device = torch.device("cpu")

    print("=" * 78)
    print(f"QUANT-OPS-ENGINE BENCHMARK | Transformer ({n_layers}L, d={d_model}, h={n_heads}) | CPU")
    print("=" * 78)

    base_model = BenchmarkTransformer(n_layers=n_layers, d_model=d_model, n_heads=n_heads).to(device).eval()
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    with torch.no_grad():
        ref_out = base_model(x)

    base_bytes = compute_model_memory_bytes(base_model)
    base_mb = base_bytes / (1024 ** 2)
    base_lat = _measure_latency_ms(base_model, x, iters=iters)

    configs = [
        ("FP32 Baseline", base_model),
        ("INT8 Per-Channel (MinMax)", quantize_model(base_model, mode="int8", calibration="minmax")),
        ("INT8 Per-Channel (MSE)", quantize_model(base_model, mode="int8", calibration="mse")),
        ("INT4 Group-Wise (g=64, Bit-Packed)", quantize_model(base_model, mode="int4", group_size=64)),
        ("INT4 Group-Wise (g=128, Bit-Packed)", quantize_model(base_model, mode="int4", group_size=128)),
    ]

    model_rows: List[Dict[str, Any]] = []
    for label, mod in configs:
        mod.eval()
        mem_bytes = compute_model_memory_bytes(mod)
        mem_mb = mem_bytes / (1024 ** 2)
        compression_ratio = base_bytes / mem_bytes

        with torch.no_grad():
            out = mod(x)

        if label == "FP32 Baseline":
            fidelity = {"sqnr_db": float("inf"), "max_abs_error": 0.0, "cosine_similarity": 1.0}
            lat = base_lat
        else:
            fidelity = compute_quantization_metrics(ref_out, out)
            lat = _measure_latency_ms(mod, x, iters=iters)

        row = {
            "config": label,
            "memory_mb": round(mem_mb, 2),
            "compression_ratio": round(compression_ratio, 2),
            "output_sqnr_db": "inf" if fidelity["sqnr_db"] == float("inf") else fidelity["sqnr_db"],
            "output_cosine_sim": fidelity["cosine_similarity"],
            "output_max_abs_err": fidelity["max_abs_error"],
            "latency_p50_ms": lat["p50_ms"],
            "latency_p95_ms": lat["p95_ms"],
        }
        model_rows.append(row)
        print(
            f"{label:<36} | {row['memory_mb']:>6.2f} MB ({row['compression_ratio']:>4.2f}x) | "
            f"SQNR: {str(row['output_sqnr_db']):>6} dB | CosSim: {row['output_cosine_sim']:.5f} | "
            f"p50: {row['latency_p50_ms']:>6.2f} ms"
        )

    print("\nRunning Numerical Stability Stress Audit...")
    audit = run_numerical_stress_audit()
    for test_name, detail in audit.items():
        print(f"  [{detail['status']}] {test_name}: {detail}")

    report = {
        "architecture": {
            "n_layers": n_layers,
            "d_model": d_model,
            "n_heads": n_heads,
            "total_fp32_params": sum(p.numel() for p in base_model.parameters()),
            "input_shape": [batch_size, seq_len, d_model],
        },
        "compression_benchmarks": model_rows,
        "numerical_stress_audit": audit,
    }

    out_path = os.path.join(RESULTS_DIR, "benchmark_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nBenchmark report written to: {out_path}")
    return report


if __name__ == "__main__":
    run_all_benchmarks()
