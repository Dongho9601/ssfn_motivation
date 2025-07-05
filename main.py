#!/usr/bin/env python3
import argparse
import numpy as np
from typing import Callable, Dict


def generate_data(size: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=size).astype(np.float32)


def round_to_FP16(x: np.ndarray, block: int) -> np.ndarray:
    return x.astype(np.float16).astype(np.float32)


def round_to_BFP16(x: np.ndarray, block: int) -> np.ndarray:
    """
    Quantize a float32 array to bfloat16 precision (round-to-nearest), by manipulating the bit pattern of the float32.
    We implement round-to-nearest by adding 0x00008000 before truncation.
    """
    bits = x.view(np.uint32)
    bits = bits + np.uint32(0x00008000)
    bits = bits & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    diff = original - reconstructed
    return float(np.mean(diff * diff))


def run_benchmark(
    method_id: int,
    data: np.ndarray,
    quantizers: Dict[str, Callable[[np.ndarray], np.ndarray]],
    block: int
) -> Dict[str, float]:
    results = {}
    for name, fn in quantizers.items():
        try:
            recon = fn(data, block)
            mse = compute_mse(data, recon)
            results[name] = mse
        except Exception as e:
            results[name] = float('nan')
            print(f"[Warning] {name} failed: {e}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark quantization methods by MSE against FP32 data."
    )
    parser.add_argument(
        "--size", "-n",
        type=int,
        default=100000,
        help="Number of random floats to generate (default: 100,000)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for data generation (default: 42)"
    )
    parser.add_argument(
        "--method", "-m",
        type=int,
        default=0,
        help="Qutization method to test (defalut: 0 - all)"
    )
    parser.add_argument(
        "--block", "-b",
        type=int,
        default=32,
        help="Block size for scaled numeric formats"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = generate_data(args.size, args.seed)

    quantizers = {
        "method_1": round_to_FP16,
        "method_2": round_to_BFP16
    }

    results = run_benchmark(args.method, data, quantizers, args.block)

    print("\nQuantization Benchmark Results (MSE):")
    print("------------------------------------")
    for name, mse in results.items():
        print(f"{name:>12}: {mse:.6e}")


if __name__ == "__main__":
    main()
