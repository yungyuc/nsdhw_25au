import argparse
import os
import time
from statistics import median

import _matrix as M  # your pybind11 module: Matrix, multiply_naive, multiply_tile, multiply_mkl


def set_threads(threads: int | None):
    """
    Control BLAS thread count (useful for MKL/OpenBLAS). If None, keep env unchanged.
    Must be called before first BLAS call for consistent behavior.
    """
    if threads is None:
        return
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[k] = str(threads)


def fill_constant(mat: "M.Matrix", value: float):
    """Fill a Matrix with a constant. (Python-loop; OK for one-time setup.)"""
    nrow, ncol = mat.nrow, mat.ncol
    for i in range(nrow):
        for j in range(ncol):
            mat[i, j] = value


def verify_equal(Cref: "M.Matrix", C: "M.Matrix", name: str, tol=1e-9):
    """Use approx_equal if available; __eq__ may already be approx-equal per our binding."""
    try:
        ok = Cref.approx_equal(C, tol)  # exposed in the binding I gave you
    except AttributeError:
        ok = (Cref == C)  # fall back
    if not ok:
        raise AssertionError(f"Result mismatch for {name}")


def gflops(n: int, seconds: float) -> float:
    # Matmul ~ 2*n^3 floating ops
    return (2.0 * n * n * n) / (seconds * 1e9)


def time_once(fn, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    _ = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return t1 - t0


def timed_call(fn, *args, repeat: int = 5, warmup: int = 1, **kwargs):
    # warm-up
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
    # measure
    times = [time_once(fn, *args, **kwargs) for _ in range(repeat)]
    return median(times), times


def main():
    p = argparse.ArgumentParser(description="Benchmark naive/tiled/MKL matrix multiplication")
    p.add_argument("--n", type=int, default=1024, help="Matrix size n (n x n), must be >= 1000")
    p.add_argument("--block", type=int, default=64, help="Block size for tiled multiply")
    p.add_argument("--repeat", type=int, default=5, help="Number of timed runs (median is reported)")
    p.add_argument("--warmup", type=int, default=1, help="Warm-up runs per method")
    p.add_argument("--threads", type=int, default=None, help="Force BLAS threads (MKL/OPENBLAS)")
    p.add_argument("--no-assert", action="store_true", help="Do not assert the 20%% speedup requirement")
    args = p.parse_args()

    if args.n < 1000:
        raise SystemExit("n must be >= 1000 as per assignment requirement.")

    set_threads(args.threads)

    n = args.n
    A = M.Matrix(n, n)
    B = M.Matrix(n, n)

    fill_constant(A, 1.0)
    fill_constant(B, 1.0)

    C_naive = M.multiply_naive(A, B)
    C_tile = M.multiply_tile(A, B, args.block)
    C_mkl = M.multiply_mkl(A, B)
    verify_equal(C_naive, C_tile, "tiled vs naive")
    verify_equal(C_naive, C_mkl, "mkl vs naive")

    t_naive, raw_naive = timed_call(M.multiply_naive, A, B, repeat=args.repeat, warmup=args.warmup)
    t_tile, raw_tile = timed_call(M.multiply_tile, A, B, repeat=args.repeat, warmup=args.warmup, block=args.block)
    t_mkl, raw_mkl = timed_call(M.multiply_mkl, A, B, repeat=args.repeat, warmup=args.warmup)

    gf_naive = gflops(n, t_naive)
    gf_tile = gflops(n, t_tile)
    gf_mkl = gflops(n, t_mkl)

    speedup_tile_vs_naive = t_naive / t_tile if t_tile > 0 else float("inf")
    speedup_mkl_vs_naive = t_naive / t_mkl if t_mkl > 0 else float("inf")

    print(f"Start multiply_naive (repeat={args.repeat}), take min = {min(raw_naive):.15f} seconds")
    print(f"Start multiply_mkl   (repeat={args.repeat}), take min = {min(raw_mkl):.15f} seconds")

    print(f"MKL speed-up over naive: {t_naive / t_mkl:.3f} x")

    if not args.no_assert:
        assert t_tile <= 0.8 * t_naive, (
            f"Tiled version is not ≥20% faster than naive. "
            f"(tiled {t_tile:.4f}s vs naive {t_naive:.4f}s; speedup {speedup_tile_vs_naive:.2f}x)"
        )

if __name__ == "__main__":
    main()