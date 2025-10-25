import time
import _matrix as mx

def bench(n: int = 1000, tile: int = 64):
    a = mx.Matrix(n, n)
    b = mx.Matrix(n, n)
    t0 = time.time(); c1 = mx.multiply_naive(a, b); t1 = time.time()
    t2 = time.time(); c2 = mx.multiple_tile(a, b, tile); t3 = time.time()
    return (t1 - t0), (t3 - t2)

if __name__ == "__main__":
    naive, tiled = bench()
    print(f"naive={naive:.6f}s tiled={tiled:.6f}s ratio={tiled/naive if naive else 0:.3f}")


