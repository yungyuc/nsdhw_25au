import os, time
import numpy as np
import matrix_ops as mo

N = int(os.environ.get("SIZE","1500"))
TILE = int(os.environ.get("TILE","128"))
REPEAT = int(os.environ.get("REPEAT","5"))

rng = np.random.default_rng(42)
A = rng.standard_normal((N,N)).astype(np.double)
B = rng.standard_normal((N,N)).astype(np.double)

MA = mo.Matrix.from_numpy(A)
MB = mo.Matrix.from_numpy(B)

def timeit(fn, *args, repeat=REPEAT):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = fn(*args)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best

print(f"Benchmark: N={N}, TILE={TILE}, REPEAT={REPEAT}")

t_naive = timeit(mo.multiply_naive, MA, MB)
t_tiled = timeit(lambda X,Y: mo.multiply_tile(X,Y,TILE), MA, MB)

mkl_ok = True
try:
    t_mkl = timeit(mo.multiply_mkl, MA, MB)
except Exception:
    mkl_ok = False
    t_mkl = None

speedup = t_naive / t_tiled if t_tiled>0 else float("inf")
pass20 = t_tiled < 0.8 * t_naive

lines = [
    "NSD HW3 Performance Report",
    "==========================",
    f"Matrix size: {N} x {N}",
    f"TILE size:   {TILE}",
    "",
    f"Naive: {t_naive:.6f} s",
    f"Tiled: {t_tiled:.6f} s  (speedup x{speedup:.2f})",
    f"Criterion (tiled < 0.8 * naive): {'PASS' if pass20 else 'FAIL'}",
    f"DGEMM: {t_mkl:.6f} s" if mkl_ok else "DGEMM: unavailable",
]

with open("performance.txt","w",encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
print("\nWrote performance.txt")
