import pytest
import _matrix as M
import numpy as np
import timeit


def test_int_multiply():
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
    repeat = 100

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
