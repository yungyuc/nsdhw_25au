# test_gemm.py — unit tests for Matrix & multiply_* (pytest)
import _matrix as M
import pytest

def make_square(n: int):
    a, b, z = M.Matrix(n, n), M.Matrix(n, n), M.Matrix(n, n)
    for i in range(n):
        for j in range(n):
            v = i * n + j + 1
            a[i, j] = v
            b[i, j] = v
            z[i, j] = 0.0
    return a, b, z

def test_matrix_basic():
    n = 10
    a, b, z = make_square(n)
    assert a.nrow == n and a.ncol == n
    assert b.nrow == n and b.ncol == n
    assert z.nrow == n and z.ncol == n
    assert a[0, 1] == 2.0
    assert a[1, 1] == n + 2   # 1*n + 1 + 1
    assert a[1, n - 1] == 2 * n
    assert a[n - 1, n - 1] == n * n
    assert a == b
    assert a is not b
    assert a != z

def test_dim_and_bounds():
    n = 8
    a, b, _ = make_square(n)
    # out-of-range
    with pytest.raises(IndexError):
        _ = a[n, 0]
    with pytest.raises(IndexError):
        a[-1, 0] = 1.0
    # dimension mismatch: columns of A (n) != rows of B' (n+1)
    a_bad = M.Matrix(n+1, n)
    with pytest.raises(ValueError):
        _ = M.multiply_naive(a, a_bad)
    with pytest.raises(ValueError):
        _ = M.multiply_tile(a, a_bad, 16)
    with pytest.raises(ValueError):
        _ = M.multiply_mkl(a, a_bad)

def _assert_same(C1: M.Matrix, C2: M.Matrix):
    assert C1.nrow == C2.nrow and C2.ncol == C1.ncol
    for i in range(C1.nrow):
        for j in range(C1.ncol):
            assert C1[i, j] == C2[i, j]

@pytest.mark.parametrize("tsize", [16, 17, 19, 32, 64])
def test_correctness_small(tsize):
    n = 64
    a, b, _ = make_square(n)
    c_naive = M.multiply_naive(a, b)
    c_tile  = M.multiply_tile(a, b, tsize)
    c_mkl   = M.multiply_mkl(a, b)  # may fallback
    _assert_same(c_naive, c_tile)
    _assert_same(c_naive, c_mkl)

def test_zero_behavior():
    n = 32
    a, b, z = make_square(n)
    c1 = M.multiply_naive(a, z)
    c2 = M.multiply_mkl(a, z)
    for i in range(n):
        for j in range(n):
            assert c1[i, j] == 0.0
            assert c2[i, j] == 0.0

