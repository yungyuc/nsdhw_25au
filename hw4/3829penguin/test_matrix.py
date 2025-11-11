import pytest
import _matrix as M
import numpy as np
import timeit

def test_int_multiply():
    start_bytes = M.bytes()
    start_alloc = M.allocated()
    start_dealloc = M.deallocated()

    A = M.Matrix(5, 10)
    a_value = np.random.randint(100, size=(5, 10))
    A.load(a_value)

    B = M.Matrix(10, 3)
    b_value = np.random.randint(100, size=(10, 3))
    B.load(b_value)
    expected = np.dot(a_value, b_value)

    result_naive = M.multiply_naive(A, B)
    result_tile = M.multiply_tile(A, B, 8)
    result_mkl = M.multiply_mkl(A, B)

    result_numpy = M.Matrix(5, 3)
    result_numpy.load(expected)

    assert result_numpy == result_naive, "Naive method mismatch"
    assert result_numpy == result_tile, "Tiled method mismatch"
    assert result_numpy == result_mkl, "MKL method mismatch"

    end_bytes = M.bytes()
    end_alloc = M.allocated()
    end_dealloc = M.deallocated()

    print(f"\nMemory tracking:")
    print(f"  bytes used:   {end_bytes - start_bytes}")
    print(f"  allocated:    {end_alloc - start_alloc}")
    print(f"  deallocated:  {end_dealloc - start_dealloc}")

    assert end_alloc >= end_dealloc, "Deallocated bytes should not exceed allocated bytes"
    assert end_bytes >= 0, "Bytes used should not be negative"


def test_performance():
    setup_code = """
import _matrix as M
import numpy as np
A = M.Matrix(256, 512)
a_value = np.random.randint(1000, size=(256, 512))
A.load(a_value)
B = M.Matrix(512, 128)
b_value = np.random.randint(1000, size=(512, 128))
B.load(b_value)
"""
    repeat = 20  

    naive_time = timeit.timeit("M.multiply_naive(A, B)", setup=setup_code, number=repeat)
    tile_time = timeit.timeit("M.multiply_tile(A, B, 16)", setup=setup_code, number=repeat)
    mkl_time = timeit.timeit("M.multiply_mkl(A, B)", setup=setup_code, number=repeat)

    report = (
        f"Matrix multiplication A(256,512) * B(512,128), repeated {repeat} times:\n"
        f" - Naive method: {naive_time:.4f} s\n"
        f" - Tiled method: {tile_time:.4f} s\n"
        f" - MKL method:   {mkl_time:.4f} s\n"
    )

    print(report)
    with open("performance.txt", "w", encoding="utf-8") as f:
        f.write(report)

    assert mkl_time < naive_time, "MKL should outperform naive multiplication"
