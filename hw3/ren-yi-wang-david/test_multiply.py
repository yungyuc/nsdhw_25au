import _matrix as mm
import pytest
import time


def random_matrix(n: int) -> mm.Matrix:
    """產生空白矩陣（目前無 setitem 綁定，僅用於測試運算效能）"""
    return mm.Matrix(n, n)


@pytest.mark.parametrize("n", [64, 128])
def test_naive_and_mkl_runs(n):
    """測試 naive 與 MKL 能正確執行且不報錯"""
    A = random_matrix(n)
    B = random_matrix(n)
    C1 = mm.multiply_naive(A, B)
    C2 = mm.multiply_mkl(A, B)
    assert C1.elapsed() > 0
    assert C2.elapsed() > 0


@pytest.mark.parametrize("n", [128])
def test_tiled_runs(n):
    """測試 tiled 版本執行正確"""
    A = random_matrix(n)
    B = random_matrix(n)
    C_tile = mm.multiply_tile(A, B, 64)
    assert C_tile.elapsed() > 0


def test_performance_benchmark():
    """效能測試，輸出 performance.txt 並記錄速度倍數"""
    n = 256
    A = random_matrix(n)
    B = random_matrix(n)

    C_naive = mm.multiply_naive(A, B)
    C_tile = mm.multiply_tile(A, B, 64)
    C_mkl = mm.multiply_mkl(A, B)

    t_naive = C_naive.elapsed()
    t_tile = C_tile.elapsed()
    t_mkl = C_mkl.elapsed()

    speedup_tile = t_naive / t_tile if t_tile > 0 else 0
    speedup_mkl = t_naive / t_mkl if t_mkl > 0 else 0

    with open("performance.txt", "w") as f:
        f.write("=== Matrix Multiplication Performance ===\n")
        f.write(f"Matrix size: {n} x {n}\n")
        f.write("-----------------------------------------\n")
        f.write(f"Naive: {t_naive:.6f}s\n")
        f.write(f"Tiled: {t_tile:.6f}s  (speedup {speedup_tile:.2f}x)\n")
        f.write(f"MKL:   {t_mkl:.6f}s  (speedup {speedup_mkl:.2f}x)\n")

    print("✅ performance.txt generated.")
    assert t_naive > 0 and t_mkl > 0 and t_tile > 0



if __name__ == "__main__":
    print(">>> Running full benchmark (1000x1000)...")
    t0 = time.time()
    n = 1000
    A = random_matrix(n)
    B = random_matrix(n)

    C_naive = mm.multiply_naive(A, B)
    C_tile = mm.multiply_tile(A, B, 64)
    C_mkl = mm.multiply_mkl(A, B)

    t_naive, t_tile, t_mkl = C_naive.elapsed(), C_tile.elapsed(), C_mkl.elapsed()
    s_tile, s_mkl = t_naive / t_tile, t_naive / t_mkl

    print(f"Naive: {t_naive:.6f}s")
    print(f"Tiled: {t_tile:.6f}s  →  {s_tile:.2f}x faster")
    print(f"MKL:   {t_mkl:.6f}s  →  {s_mkl:.2f}x faster")

    with open("performance.txt", "w") as f:
        f.write("=== Matrix Multiplication Performance ===\n")
        f.write(f"Matrix size: {n} x {n}\n")
        f.write("-----------------------------------------\n")
        f.write(f"Naive: {t_naive:.6f}s\n")
        f.write(f"Tiled: {t_tile:.6f}s  (speedup {s_tile:.2f}x)\n")
        f.write(f"MKL:   {t_mkl:.6f}s  (speedup {s_mkl:.2f}x)\n")

    print("✅ performance.txt generated.")
    print(f"✅ Done. Total elapsed {time.time() - t0:.2f}s")
