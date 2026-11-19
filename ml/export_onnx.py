"""
ONNX Model Export, INT8 Quantization & Numerical Parity Benchmark.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.train_arcface import MobileFaceNetBackbone


def verify_onnx_parity(
    torch_model: torch.nn.Module,
    onnx_path: str,
    tolerance: float = 1e-4
) -> bool:
    """
    Test numerical parity between PyTorch source model and ONNX Runtime execution.
    """
    import onnxruntime as ort

    torch_model.eval()
    dummy_input = torch.randn(1, 3, 112, 112, dtype=torch.float32)

    with torch.no_grad():
        torch_out = torch_model(dummy_input).cpu().numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = session.run(None, ort_inputs)[0]

    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    print(f"ONNX vs PyTorch Max Absolute Difference: {max_diff:.6e}")
    is_parity = max_diff < tolerance
    print(f"Parity Test (tol={tolerance}): {'[PASSED]' if is_parity else '[FAILED]'}")
    return is_parity


def benchmark_cpu_latency(onnx_path: str, iterations: int = 50) -> float:
    """
    Benchmark CPU inference latency per face in milliseconds. Target: < 150 ms.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 112, 112).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run(None, {input_name: dummy})

    start = time.perf_counter()
    for _ in range(iterations):
        session.run(None, {input_name: dummy})
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000.0
    print(f"CPU Latency ({iterations} runs): {avg_ms:.2f} ms / face (Target: < 150 ms)")
    return avg_ms


def quantize_to_int8(input_onnx_path: str, output_onnx_path: str):
    """
    Apply dynamic INT8 quantization to reduce model footprint and accelerate CPU inference.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        print(f"Quantizing {input_onnx_path} to INT8 -> {output_onnx_path}...")
        quantize_dynamic(
            input_onnx_path,
            output_onnx_path,
            weight_type=QuantType.QInt8,
        )
        orig_size = os.path.getsize(input_onnx_path) / (1024 * 1024)
        quant_size = os.path.getsize(output_onnx_path) / (1024 * 1024)
        print(f"Quantization complete: {orig_size:.2f} MB -> {quant_size:.2f} MB")
    except Exception as exc:
        print(f"Quantization skipped or failed: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="ml/weights/arcface_export.onnx")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.onnx), exist_ok=True)
    model = MobileFaceNetBackbone(embedding_size=512)
    model.eval()

    dummy = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        model,
        dummy,
        args.onnx,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"Exported to {args.onnx}")

    verify_onnx_parity(model, args.onnx)
    benchmark_cpu_latency(args.onnx)

    if args.quantize:
        quant_path = args.onnx.replace(".onnx", "_int8.onnx")
        quantize_to_int8(args.onnx, quant_path)
        benchmark_cpu_latency(quant_path)


if __name__ == "__main__":
    main()
