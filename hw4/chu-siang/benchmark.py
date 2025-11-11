# benchmark.py — ≥1000×1000 runtime check (naive/tile/mkl), writes performance.txt
import os, time
import _matrix as M

N  = int(os.environ.get("MATRIX_SIZE", "1024"))
TS = int(os.environ.get("TILE_SIZE",   "64"))
assert N >= 1000, "Matrix size must be >= 1000"

def make_square(n):
    a, b = M.Matrix(n, n), M.Matrix(n, n)
    for i in range(n):
        for j in range(n):
            v = (i * n + j + 1) * 1.0
            a[i, j] = v
            b[i, j] = v
    return a, b

def tmin_sec(fn, repeat=5):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best: best = dt
    return best

def main():
    a, b = make_square(N)
    t_naive = tmin_sec(lambda: M.multiply_naive(a, b))
    t_tile  = tmin_sec(lambda: M.multiply_tile(a, b, TS))
    t_mkl   = tmin_sec(lambda: M.multiply_mkl(a, b))

    ratio = t_tile / t_naive
    lines = [
        f"Matrix size: {N} x {N}",
        f"Tile size: {TS}",
        f"multiply_naive: {t_naive:.6f} sec (best of 5)",
        f"multiply_tile : {t_tile:.6f} sec (best of 5)",
        f"multiply_mkl  : {t_mkl:.6f} sec (best of 5)",
        f"tile / naive  : {ratio:.4f}  (must be < 0.8000)",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    with open("performance.txt", "w") as f:
        f.write(text)

if __name__ == "__main__":
    main()

