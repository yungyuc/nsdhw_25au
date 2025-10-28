import _matrix as mm
import pytest
import time


def random_matrix(n: int) -> mm.Matrix:
    """產生空白矩陣（目前無 setitem 綁定，僅用於效能測試）"""
    return mm.Matrix(n, n)


@pytest.mark.parametrize("n", [64, 128])
def test_naive_mkl_tiled_runs(n):
    """測試三種乘法版本是否能執行且返回時間"""
    A = random_matrix(n)
    B = random_matrix(n)
    C_naive = mm.multiply_naive(A, B)
    C_tile = mm.multiply_tile(A, B, 64)
    C_mkl = mm.multiply_mkl(A, B)
    assert C_naive.elapsed() > 0
    assert C_tile.elapsed() > 0
    assert C_mkl.elapsed() > 0


def test_performance_write_to_file():
    """執行三者效能比較並輸出 performance.txt"""
    n = 256
    A = random_matrix(n)
    B = random_matrix(n)

    # 執行三種版本
    C_naive = mm.multiply_naive(A, B)
    C_tile = mm.multiply_tile(A, B, 64)
    C_mkl = mm.multiply_mkl(A, B)

    # 取得時間
    t_naive = C_naive.elapsed()
    t_tile = C_tile.elapsed()
    t_mkl = C_mkl.elapsed()

    # 計算加速倍率
    s_tile = t_naive / t_tile if t_tile > 0 else 0
    s_mkl = t_naive / t_mkl if t_mkl > 0 else 0

    # 寫入檔案
    with open("performance.txt", "w") as f:
        f.write("=== Matrix Multiplication Performance ===\n")
        f.write(f"Matrix size: {n} x {n}\n")
        f.write("-----------------------------------------\n")
        f.write(f"Naive: {t_naive:.6f}s\n")
        f.write(f"Tiled: {t_tile:.6f}s  →  {s_tile:.2f}x faster\n")
        f.write(f"MKL:   {t_mkl:.6f}s  →  {s_mkl:.2f}x faster\n")

    print("✅ performance.txt generated.")
    assert all(t > 0 for t in [t_naive, t_tile, t_mkl])


if __name__ == "__main__":
    """獨立執行時進行大型 benchmark"""
    print(">>> Running full benchmark (1000x1000)...")
    t0 = time.time()

    n = 1000
    A = random_matrix(n)
    B = random_matrix(n)

    # 三種乘法
    C_naive = mm.multiply_naive(A, B)
    C_tile = mm.multiply_tile(A, B, 64)
    C_mkl = mm.multiply_mkl(A, B)

    # 取得時間
    t_naive, t_tile, t_mkl = (
        C_naive.elapsed(),
        C_tile.elapsed(),
        C_mkl.elapsed(),
    )
    s_tile, s_mkl = t_naive / t_tile, t_naive / t_mkl

    # 終端顯示
    print(f"Naive: {t_naive:.6f}s")
    print(f"Tiled: {t_tile:.6f}s  →  {s_tile:.2f}x faster")
    print(f"MKL:   {t_mkl:.6f}s  →  {s_mkl:.2f}x faster")

    # 寫入結果
    with open("performance.txt", "w") as f:
        f.write("=== Matrix Multiplication Performance ===\n")
        f.write(f"Matrix size: {n} x {n}\n")
        f.write("-----------------------------------------\n")
        f.write(f"Naive: {t_naive:.6f}s\n")
        f.write(f"Tiled: {t_tile:.6f}s  →  {s_tile:.2f}x faster\n")
        f.write(f"MKL:   {t_mkl:.6f}s  →  {s_mkl:.2f}x faster\n")

    print("✅ performance.txt generated.")
    print(f"✅ Done. Total elapsed {time.time() - t0:.2f}s")
