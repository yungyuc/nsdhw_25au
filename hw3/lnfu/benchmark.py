#!/usr/bin/env python3

import time
import sys
import numpy as np
import _matrix
from typing import Callable, List, Tuple


def create_random_matrix(n: int, seed: int = None) -> _matrix.Matrix:
    if seed is not None:
        np.random.seed(seed)
    data = [np.random.randn(n).tolist() for _ in range(n)]
    return _matrix.Matrix(data)


def benchmark_multiply(
    func: Callable,
    A: _matrix.Matrix,
    B: _matrix.Matrix,
    warmup: int = 1,
    repeat: int = 3,
) -> float:
    # Warmup
    for _ in range(warmup):
        func(A, B)

    # Timing runs
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(A, B)
        end = time.perf_counter()
        times.append(end - start)

    return min(times)


def run_benchmarks(sizes: List[int]) -> List[Tuple[int, dict]]:
    results = []

    for n in sizes:
        # Progress to stderr to avoid polluting stdout
        print(f"Benchmarking {n}×{n}...", file=sys.stderr)

        # Create test matrices with fixed seed for reproducibility
        A = create_random_matrix(n, seed=42)
        B = create_random_matrix(n, seed=43)

        # Benchmark each method
        time_naive = benchmark_multiply(_matrix.multiply_naive, A, B)
        time_tile = benchmark_multiply(_matrix.multiply_tile, A, B)
        time_mkl = benchmark_multiply(_matrix.multiply_mkl, A, B)

        results.append((n, {"naive": time_naive, "tile": time_tile, "mkl": time_mkl}))

    return results


def print_report(results: List[Tuple[int, dict]]):
    print("=" * 80)
    print("MATRIX MULTIPLICATION PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()

    # Execution time table
    print("EXECUTION TIME (seconds)")
    print("-" * 80)
    print(f"{'Size':<10} {'Naive':<15} {'Tile':<15} {'MKL':<15}")
    print("-" * 80)

    for n, times in results:
        print(
            f"{n:<10} {times['naive']:<15.6f} {times['tile']:<15.6f} {times['mkl']:<15.6f}"
        )

    print()

    # Speedup analysis
    print("SPEEDUP RELATIVE TO NAIVE")
    print("-" * 80)
    print(f"{'Size':<10} {'Tile':<15} {'MKL':<15}")
    print("-" * 80)

    for n, times in results:
        speedup_tile = times["naive"] / times["tile"]
        speedup_mkl = times["naive"] / times["mkl"]
        print(f"{n:<10} {speedup_tile:<15.2f}x {speedup_mkl:<15.2f}x")

    print()

    # Performance metrics (GFLOPS)
    print("PERFORMANCE METRICS")
    print("-" * 80)
    print(f"{'Size':<10} {'Method':<10} {'GFLOPS':<15}")
    print("-" * 80)

    for n, times in results:
        # Matrix multiplication FLOPS: 2n³ - n²
        flops = 2 * n**3 - n**2

        for method in ["naive", "tile", "mkl"]:
            gflops = (flops / times[method]) / 1e9
            print(f"{n:<10} {method:<10} {gflops:<15.2f}")

    print()

    # Conclusion
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    largest_n, largest_times = results[-1]
    best_method = min(largest_times, key=largest_times.get)
    best_time = largest_times[best_method]

    print(f"\nFor {largest_n}×{largest_n} matrices:")
    print(f"  Best method: {best_method.upper()} ({best_time:.6f}s)")

    for method in ["naive", "tile", "mkl"]:
        if method != best_method:
            speedup = largest_times[method] / best_time
            print(f"  {method.capitalize()} is {speedup:.2f}x slower")


def main():
    sizes = [64, 128, 256, 512, 1024, 2048]

    results = run_benchmarks(sizes)
    print_report(results)


if __name__ == "__main__":
    main()
