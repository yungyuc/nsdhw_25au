import numpy as np
import _matrix as mm

def to_numpy(M):
    a = np.empty((M.nrow, M.ncol))
    for i in range(M.nrow):
        for j in range(M.ncol):
            a[i, j] = M[i, j]
    return a

def build(m, k, n):
    A = mm.Matrix(m, k)
    B = mm.Matrix(k, n)
    mm.populate(A); mm.populate(B)
    return A, B

def test_tile_matches_naive():
    A, B = build(5, 6, 4)
    Cn = mm.multiply_naive(A, B)
    Ct = mm.multiply_tile(A, B, 3)
    assert np.allclose(to_numpy(Cn), to_numpy(Ct), atol=1e-9, rtol=0.0)

def test_mkl_matches_naive():
    A, B = build(6, 7, 5)
    Cn = mm.multiply_naive(A, B)
    Cm = mm.multiply_mkl(A, B)
    assert np.allclose(to_numpy(Cn), to_numpy(Cm), atol=1e-8, rtol=0.0)