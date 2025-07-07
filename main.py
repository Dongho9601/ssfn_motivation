#!/usr/bin/env python3
import argparse
import math
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

def MXINT8(x: np.ndarray, block: int) -> np.ndarray:
    """
    Quantizes a float32 array to MXINT8 using power-of-two scaling,
    following the specified algorithm, and then dequantizes it.
    https://arxiv.org/pdf/2310.10537
    """
    dequantized_x = np.empty_like(x, dtype=np.float32)
    
    # For INT8, the maximum value is 127. The exponent of the largest
    # normal number in the element data format (emax_elem) is floor(log2(127)).
    emax_elem = np.floor(np.log2(127.0)) # 6.0

    # Iterate over the array in blocks of the specified size
    for i in range(0, x.size, block):
        current_block = x[i:i+block]
        abs_max = np.max(np.abs(current_block))

        if abs_max == 0:
            dequantized_x[i:i+block] = current_block
            continue

        # Algorithm Step 1: Calculate the shared exponent
        # shared_exp <- floor(log2(max_i(|V_i|))) - emax_elem
        shared_exp = np.floor(np.log2(abs_max)) - emax_elem

        # Algorithm Step 2: Calculate the scaling factor 'X'
        # X <- 2^shared_exp
        scale_x = 2.0**shared_exp

        # Algorithm Step 4 (for INT8): Quantize to element format
        # P_i = quantize_to_element_format(V_i / X)
        # This involves scaling, rounding, and clamping to the INT8 range.
        quantized_block = np.round(current_block / scale_x)
        quantized_block = np.clip(quantized_block, -127, 127)

        # Dequantize: scale the integer block back to the original float range
        dequantized_block = quantized_block * scale_x

        # Store the dequantized block in our output array
        dequantized_x[i:i+block] = dequantized_block

    return dequantized_x

def _mx_quantize_int(x: np.ndarray, block: int, bits: int) -> np.ndarray:
    """Block-wise symmetric integer quantization with dynamic scale."""
    if block <= 0:
        raise ValueError("block must be positive")
    max_int = 2 ** (bits - 1) - 1
    out = np.empty_like(x, dtype=np.float32)
    for i in range(0, len(x), block):
        chunk = x[i : i + block]
        amax = np.max(np.abs(chunk))
        if amax == 0:
            out[i : i + block] = 0
            continue
        scale = amax / max_int
        q = np.round(chunk / scale)
        q = np.clip(q, -max_int, max_int)
        out[i : i + block] = q * scale
    return out


def _mx_quantize_fp(
    x: np.ndarray, block: int, exp_bits: int, mant_bits: int
) -> np.ndarray:
    """Block-wise floating point quantization according to MXFP specs."""
    if block <= 0:
        raise ValueError("block must be positive")
    out = np.empty_like(x, dtype=np.float32)
    max_exp = 2 ** (exp_bits - 1) - 1
    min_exp = -max_exp
    step = 1 << mant_bits
    max_mant = 2.0 - math.ldexp(1.0, -mant_bits)
    for i in range(0, len(x), block):
        chunk = x[i : i + block]
        amax = np.max(np.abs(chunk))
        if amax == 0:
            out[i : i + block] = 0
            continue
        exp = int(math.floor(math.log2(amax)))
        exp = max(min(exp, max_exp), min_exp)
        scale = math.ldexp(1.0, -exp)
        scaled = chunk * scale
        scaled = np.clip(scaled, -max_mant, max_mant)
        quant = np.round(scaled * step) / step
        out[i : i + block] = quant / scale
    return out


def round_to_MXINT8(x: np.ndarray, block: int) -> np.ndarray:
    return _mx_quantize_int(x, block, 8)


def round_to_MXFP8(x: np.ndarray, block: int) -> np.ndarray:
    return _mx_quantize_fp(x, block, 4, 3)


def round_to_MXFP6(x: np.ndarray, block: int) -> np.ndarray:
    return _mx_quantize_fp(x, block, 3, 2)


def round_to_MXFP4(x: np.ndarray, block: int) -> np.ndarray:
    return _mx_quantize_fp(x, block, 2, 1)


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
<<<<<<< HEAD
        "method_1": round_to_FP16,
        "method_2": round_to_BFP16,
        "method_3": MXINT8
=======
        "FP16": round_to_FP16,
        "BFP16": round_to_BFP16,
        "MXINT8": round_to_MXINT8,
        "MXFP8": round_to_MXFP8,
        "MXFP6": round_to_MXFP6,
        "MXFP4": round_to_MXFP4,
>>>>>>> Add requirements file
    }

    results = run_benchmark(args.method, data, quantizers, args.block)

    print("\nQuantization Benchmark Results (MSE):")
    print("------------------------------------")
    for name, mse in results.items():
        print(f"{name:>12}: {mse:.6e}")


if __name__ == "__main__":
    main()
